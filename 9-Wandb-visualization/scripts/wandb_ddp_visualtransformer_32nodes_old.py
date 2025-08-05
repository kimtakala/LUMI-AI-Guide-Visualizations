#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
import wandb
import time

# Add the resources directory to the path to import the custom dataset
sys.path.append('/opt/miniconda3/envs/pytorch/lib/python3.12/site-packages')
sys.path.append('../resources')
sys.path.append('/user-software/lib/python3.12/site-packages')

from hdf5_dataset import HDF5Dataset

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='nccl')
    
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    return local_rank

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_model():
    """Create a Vision Transformer model"""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, 10)  # 10 classes for CIFAR-10
    return model

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 10 == 0:
            accuracy = 100. * correct / total
            if dist.get_rank() == 0:
                wandb.log({
                    'epoch': epoch,
                    'batch': batch_idx,
                    'loss': loss.item(),
                    'accuracy': accuracy,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # Setup distributed training
    local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    # Initialize wandb only on rank 0
    if dist.get_rank() == 0:
        wandb.init(
            project="lumi-distributed-training-32nodes",
            name=f"vit-training-32nodes-{dist.get_world_size()}gpus",
            config={
                "model": "vision_transformer_b16",
                "dataset": "cifar10-hdf5",
                "batch_size_per_gpu": 16,  # Smaller batch per GPU for stability
                "effective_batch_size": 4096,  # 16 * 256 GPUs
                "base_learning_rate": 0.001,
                "learning_rate": 0.016,  # Scaled for 256 GPUs
                "epochs": 3,  # Very few epochs for extreme scale
                "world_size": dist.get_world_size(),
                "nodes": 32,
                "gpus_per_node": 8,
                "total_gpus": 256,
                "optimizer": "Adam",
                "scheduler": "OneCycleLR"
            },
            tags=["32-nodes", "256-gpus", "extreme-scale", "dev-g-max"]
        )
    
    # Create model and move to device
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = HDF5Dataset('/opt/miniconda3/envs/pytorch/lib/python3.12/site-packages/train_images.hdf5', transform=transform)
    sampler = DistributedSampler(dataset, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=2, pin_memory=True)
    
    # Training loop
    start_time = time.time()
    for epoch in range(5):
        sampler.set_epoch(epoch)
        
        avg_loss, accuracy = train_epoch(model, dataloader, criterion, optimizer, device, epoch)
        scheduler.step()
        
        if dist.get_rank() == 0:
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/5, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s')
            wandb.log({
                'epoch_loss': avg_loss,
                'epoch_accuracy': accuracy,
                'epoch_time': epoch_time,
                'lr': scheduler.get_last_lr()[0]
            })
        
        start_time = time.time()
    
    if dist.get_rank() == 0:
        wandb.finish()
    
    cleanup_distributed()

if __name__ == '__main__':
    main()
