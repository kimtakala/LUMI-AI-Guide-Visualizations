#!/usr/bin/env python3
"""
Fair multicore laptop benchmarking script that uses the exact same setup as LUMI
but processes a subset of data and extrapolates total training time.

This script:
- Uses the same HDF5 dataset, ViT-B-16 model, and hyperparameters as LUMI
- Optimizes for multi-core CPU usage
- Processes a configurable subset (default 2%) of the data
- Measures per-sample processing times accurately
- Extrapolates total training time for the full dataset
- Logs both actual subset metrics and extrapolated full-dataset metrics to Wandb
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import time
import os
import sys
import psutil
import platform
import numpy as np
import multiprocessing as mp

# Add resources path to import the HDF5Dataset
sys.path.append('/home/takalaki/Projects/LUMI-AI-Guide-COPY/resources')
from hdf5_dataset import HDF5Dataset

import wandb

# Set optimal number of threads for CPU training
num_cores = mp.cpu_count()
torch.set_num_threads(num_cores)
os.environ['OMP_NUM_THREADS'] = str(num_cores)

print(f"🚀 Using all {num_cores} CPU cores for training")

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

def train_epoch_with_timing(model, dataloader, criterion, optimizer, device, epoch, total_epochs, start_time, subset_size, total_samples):
    """Train epoch with detailed timing for extrapolation"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    num_samples = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        num_samples += targets.size(0)
        
        # Log every 10 batches for progress tracking
        if batch_idx % 10 == 0:
            current_time = time.time()
            elapsed_minutes = (current_time - start_time) / 60
            progress_percentage = ((epoch + (batch_idx / len(dataloader))) / total_epochs) * 100
            batch_accuracy = 100. * correct / total
            samples_per_second = num_samples / (current_time - start_time)
            
            # Extrapolated metrics
            full_epoch_time = elapsed_minutes * (total_samples / subset_size)
            
            wandb.log({
                "epoch": epoch,
                "batch": batch_idx,
                "subset_loss": running_loss / (batch_idx + 1),
                "subset_acc": batch_accuracy,
                "progress_percentage": progress_percentage,
                "subset_elapsed_minutes": elapsed_minutes,
                "subset_samples_per_sec": samples_per_second,
                "extrapolated_elapsed_minutes": full_epoch_time,
                "extrapolated_total_time_hr": full_epoch_time * total_epochs / 60,
            })
    
    epoch_time = time.time() - start_time
    return running_loss / len(dataloader), 100. * correct / total, epoch_time

def main():
    # Configuration - Fair benchmarking with same setup as LUMI
    SUBSET_PERCENTAGE = 2.0  # Process 2% of data for multicore timing
    EPOCHS = 3  # Fewer epochs for laptop demo
    BATCH_SIZE = 16  # Smaller batch size for laptop
    LEARNING_RATE = 0.001  # Same as LUMI
    
    # Get system info
    system_info = get_system_info()
    print("System Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Device setup (CPU optimized for multi-core)
    device = torch.device("cpu")  # Force CPU for fair laptop comparison
    print(f"\nUsing device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")
    
    # Initialize wandb
    wandb.init(
        project="LUMI-Laptop-Benchmark",
        name=f"Laptop-Multicore-{SUBSET_PERCENTAGE}pct-{num_cores}cores",
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "vit_b_16",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "subset_percentage": SUBSET_PERCENTAGE,
            "device_type": "CPU",
            "cpu_cores": num_cores,
            "pytorch_threads": torch.get_num_threads(),
            "scaling_strategy": "multi_core_cpu",
            **system_info
        },
        tags=["laptop", "multicore", "benchmark", "subset", "extrapolation"]
    )
    
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
        
        # Optimized data loaders for multi-core
        num_workers = min(num_cores//2, 8)  # Use half cores for data loading
        trainloader = DataLoader(
            train_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        valloader = DataLoader(
            val_dataset, 
            batch_size=BATCH_SIZE, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Same model as LUMI
        print("\nInitializing ViT-B-16 model...")
        model = vit_b_16(weights="DEFAULT").to(device)
        
        # Same loss and optimizer as LUMI  
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Data loader workers: {num_workers}")
        
        # Training loop with timing
        print(f"\nStarting training for {EPOCHS} epochs...")
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            epoch_start = time.time()
            
            # Training with timing
            train_loss, train_acc, train_time = train_epoch_with_timing(
                model, trainloader, criterion, optimizer, 
                device, epoch, EPOCHS, epoch_start, subset_size, total_samples
            )
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            val_start = time.time()
            with torch.no_grad():
                for data, targets in valloader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            val_time = time.time() - val_start
            val_acc = 100. * correct / total
            epoch_duration = time.time() - epoch_start
            total_elapsed = time.time() - start_time
            
            # Extrapolated metrics
            full_train_time = train_time * (total_samples / subset_size)
            full_val_time = val_time * (total_samples / subset_size)
            full_epoch_time = full_train_time + full_val_time
            
            # Log epoch summary with extrapolation
            wandb.log({
                "epoch": epoch,
                "subset_train_loss": train_loss,
                "subset_train_acc": train_acc,
                "subset_val_loss": val_loss / len(valloader),
                "subset_val_acc": val_acc,
                "subset_epoch_time_min": epoch_duration / 60,
                "subset_total_elapsed_min": total_elapsed / 60,
                "extrapolated_epoch_time_min": full_epoch_time / 60,
                "extrapolated_epoch_time_hr": full_epoch_time / 3600,
                "extrapolated_total_time_hr": full_epoch_time * EPOCHS / 3600,
                "progress_percentage": ((epoch + 1) / EPOCHS) * 100,
            })
            
            print(f"Epoch {epoch+1}/{EPOCHS}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {val_loss/len(valloader):.4f}, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_duration/60:.2f}min")
            print(f"  Extrapolated full epoch time: {full_epoch_time/60:.1f}min ({full_epoch_time/3600:.2f}hr)")
        
        total_time = time.time() - start_time
        
        # Final extrapolated results
        avg_epoch_time = total_time / EPOCHS
        full_training_time = avg_epoch_time * (total_samples / subset_size) * EPOCHS
        
        print(f"\n{'='*60}")
        print(f"MULTICORE BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"Subset processed: {subset_size:,} samples ({subset_size/total_samples*100:.1f}%)")
        print(f"Full dataset size: {total_samples:,} samples")
        print(f"CPU cores used: {num_cores}")
        print(f"Actual time: {total_time/60:.1f} minutes")
        print(f"")
        print(f"EXTRAPOLATED for full dataset:")
        print(f"  Total training time: {full_training_time/3600:.2f} hours")
        print(f"  Time per epoch: {full_training_time/EPOCHS/3600:.2f} hours")
        print(f"  Speedup needed for 1hr training: {full_training_time/3600:.1f}x")
        
        # Log final summary
        wandb.log({
            "final_extrapolated_total_time_hr": full_training_time / 3600,
            "final_extrapolated_epoch_time_hr": (full_training_time/EPOCHS) / 3600,
            "final_speedup_needed_for_1hr": full_training_time / 3600,
            "multicore_efficiency": subset_size / total_time,  # samples per second overall
        })
        
        wandb.finish()
        print(f"\nWandb logging complete! Check project: LUMI-Laptop-Benchmark")

if __name__ == "__main__":
    main()
