#!/usr/bin/env python3
"""
Fair laptop benchmarking script that uses the exact same setup as LUMI
but processes a subset of data and extrapolates total training time.

This script:
- Uses the same HDF5 dataset, ViT-B-16 model, and hyperparameters as LUMI
- Processes a configurable subset (default 1%) of the data
- Measures per-sample processing times accurately
- Extrapolates total training time for the full dataset
- Logs both actual subset metrics and extrapolated full-dataset metrics to Wandb
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split, Subset
import time
import os
import sys
import psutil
import platform
import numpy as np

# Add resources path to import the HDF5Dataset
sys.path.append('/home/takalaki/Projects/LUMI-AI-Guide-COPY/resources')
from hdf5_dataset import HDF5Dataset

import wandb

def get_system_info():
    """Get detailed system information for logging"""
    cpu_info = {
        'cpu_count': psutil.cpu_count(),
        'cpu_count_logical': psutil.cpu_count(logical=True), 
        'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
        'memory_total_gb': psutil.virtual_memory().total / (1024**3),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
    }
    
    # Try to get more detailed CPU info
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read()
            for line in cpuinfo.split('\n'):
                if line.startswith('model name'):
                    cpu_info['cpu_model'] = line.split(':')[1].strip()
                    break
    except:
        cpu_info['cpu_model'] = 'Unknown'
    
    return cpu_info

def print_timing_summary(phase_times, total_samples, subset_size):
    """Print a detailed timing breakdown"""
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Subset processed: {subset_size:,} samples ({subset_size/total_samples*100:.1f}% of full dataset)")
    print(f"Full dataset size: {total_samples:,} samples")
    print(f"")
    
    for phase, times in phase_times.items():
        avg_time = np.mean(times)
        total_time = sum(times)
        print(f"{phase.upper()}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per epoch: {avg_time:.2f}s")
        print(f"  Samples per second: {subset_size/avg_time:.1f}")
        print(f"  Time per sample: {avg_time/subset_size*1000:.2f}ms")
        
        # Extrapolation
        full_epoch_time = avg_time * (total_samples / subset_size)
        full_total_time = total_time * (total_samples / subset_size)
        print(f"  EXTRAPOLATED for full dataset:")
        print(f"    Time per epoch: {full_epoch_time/60:.1f} minutes ({full_epoch_time/3600:.2f} hours)")
        print(f"    Total training time: {full_total_time/60:.1f} minutes ({full_total_time/3600:.2f} hours)")
        print()

def train_model_with_timing(model, criterion, optimizer, train_loader, val_loader, epochs, device, subset_size, total_samples):
    """Train model with detailed timing measurements"""
    
    phase_times = {'training': [], 'validation': []}
    metrics = {'train_loss': [], 'val_accuracy': []}
    
    print(f"Starting training on {len(train_loader)} batches per epoch...")
    print(f"Processing {subset_size:,} samples ({subset_size/total_samples*100:.1f}% of full {total_samples:,} samples)")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        num_samples = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_samples += images.size(0)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_time = time.time() - epoch_start
        phase_times['training'].append(train_time)
        avg_loss = running_loss / len(train_loader)
        metrics['train_loss'].append(avg_loss)
        
        print(f"  Training completed in {train_time:.2f}s, Avg Loss: {avg_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_start = time.time()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_time = time.time() - val_start
        phase_times['validation'].append(val_time)
        accuracy = 100 * correct / total
        metrics['val_accuracy'].append(accuracy)
        
        print(f"  Validation completed in {val_time:.2f}s, Accuracy: {accuracy:.2f}%")
        
        # Log to Wandb with both actual and extrapolated metrics
        epoch_total_time = train_time + val_time
        samples_per_second = num_samples / train_time
        
        # Extrapolated metrics for full dataset
        full_train_time = train_time * (total_samples / subset_size)
        full_val_time = val_time * (total_samples / subset_size)
        full_epoch_time = full_train_time + full_val_time
        
        wandb.log({
            # Actual subset metrics
            "epoch": epoch + 1,
            "subset_train_loss": avg_loss,
            "subset_val_accuracy": accuracy,
            "subset_train_time_sec": train_time,
            "subset_val_time_sec": val_time,
            "subset_epoch_time_sec": epoch_total_time,
            "subset_samples_per_sec": samples_per_second,
            
            # Extrapolated full dataset metrics
            "extrapolated_train_time_min": full_train_time / 60,
            "extrapolated_val_time_min": full_val_time / 60,
            "extrapolated_epoch_time_min": full_epoch_time / 60,
            "extrapolated_epoch_time_hr": full_epoch_time / 3600,
            "extrapolated_total_time_hr": full_epoch_time * epochs / 3600,
        })
    
    return phase_times, metrics

def main():
    # Configuration
    SUBSET_PERCENTAGE = 1.0  # Process 1% of data for timing
    EPOCHS = 3  # Fewer epochs for laptop demo
    BATCH_SIZE = 16  # Smaller batch size for laptop
    LEARNING_RATE = 0.001  # Same as LUMI
    
    # Get system info
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Initialize Wandb
    wandb.init(
        project="LUMI-Laptop-Benchmark",
        name=f"Laptop-Benchmark-{SUBSET_PERCENTAGE}pct-subset",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "vit_b_16",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "subset_percentage": SUBSET_PERCENTAGE,
            "device_type": "CPU",
            **system_info
        },
        tags=["laptop", "benchmark", "subset", "extrapolation"]
    )
    
    # Setup device
    device = torch.device("cpu")  # Force CPU for fair laptop comparison
    print(f"\nUsing device: {device}")
    
    # Use all available CPU cores
    torch.set_num_threads(psutil.cpu_count())
    print(f"Using {torch.get_num_threads()} CPU threads")
    
    # Same transforms as LUMI
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load the exact same HDF5 dataset as LUMI
    print("\nLoading HDF5 dataset...")
    hdf5_path = "/home/takalaki/Projects/LUMI-AI-Guide-COPY/resources/train_images.hdf5"
    
    with HDF5Dataset(hdf5_path, transform=transform) as full_dataset:
        total_samples = len(full_dataset)
        subset_size = max(1, int(total_samples * SUBSET_PERCENTAGE / 100))
        
        print(f"Full dataset size: {total_samples:,} samples")
        print(f"Subset size: {subset_size:,} samples ({subset_size/total_samples*100:.1f}%)")
        
        # Create subset using first N samples for consistency
        subset_indices = list(range(subset_size))
        subset_dataset = Subset(full_dataset, subset_indices)
        
        # Split subset into train/val (same 80/20 ratio as LUMI)
        train_size = int(0.8 * len(subset_dataset))
        val_size = len(subset_dataset) - train_size
        train_dataset, val_dataset = random_split(subset_dataset, [train_size, val_size])
        
        print(f"Train subset: {len(train_dataset):,} samples")
        print(f"Val subset: {len(val_dataset):,} samples")
        
        # Create data loaders (no distributed sampler for laptop)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=min(4, psutil.cpu_count()//2)  # Reasonable number of workers
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=min(4, psutil.cpu_count()//2)
        )
        
        # Same model as LUMI
        print("\nInitializing ViT-B-16 model...")
        model = vit_b_16(weights="DEFAULT").to(device)
        
        # Same loss and optimizer as LUMI
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train with timing
        print(f"\nStarting training for {EPOCHS} epochs...")
        start_time = time.time()
        
        phase_times, metrics = train_model_with_timing(
            model, criterion, optimizer, train_loader, val_loader, 
            EPOCHS, device, subset_size, total_samples
        )
        
        total_time = time.time() - start_time
        
        # Print comprehensive timing summary
        print_timing_summary(phase_times, total_samples, subset_size)
        
        # Calculate and log final extrapolated metrics
        avg_epoch_time = np.mean([sum(times) for times in zip(phase_times['training'], phase_times['validation'])])
        full_epoch_time = avg_epoch_time * (total_samples / subset_size)
        full_training_time = full_epoch_time * EPOCHS
        
        print(f"\nFINAL EXTRAPOLATED RESULTS:")
        print(f"One epoch on full dataset: {full_epoch_time/3600:.2f} hours")
        print(f"Full training ({EPOCHS} epochs): {full_training_time/3600:.2f} hours")
        print(f"Speedup needed vs laptop: {full_training_time/3600:.1f}x for 1-hour training")
        
        # Log final summary metrics
        wandb.log({
            "final_extrapolated_total_time_hr": full_training_time / 3600,
            "final_extrapolated_epoch_time_hr": full_epoch_time / 3600,
            "final_speedup_needed_for_1hr": full_training_time / 3600,
            "subset_processing_efficiency": subset_size / total_time,  # samples per second overall
        })
        
        print(f"\nActual time spent: {total_time/60:.1f} minutes")
        print("Wandb logging complete!")

if __name__ == "__main__":
    main()
