import torch
import torch.distributed as dist
import os
import time
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from hdf5_dataset import HDF5Dataset

import wandb

# Distributed setup
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])

print(f"Rank {rank} (local {local_rank}) on GPU {local_rank}")

if rank == 0:
    print("Wandb init")
    wandb.init(
        # set the wandb project where this run will be logged
        project="Wandb-visualization",
        name="4 GPU 1 Node Run",
        # track hyperparameters and run metadata
        config={
            "base_learning_rate": 0.001,
            "learning_rate": 0.001,
            "batch_size_per_gpu": 32,
            "effective_batch_size": 32 * 4,  # 32 * 4 GPUs
            "architecture": "vit_b_16",
            "epochs": 10,
            "nodes": 1,
            "gpus": 4,
            "total_gpus": 4,
            "scaling_strategy": "distributeddataparallel_single_node",
        },
    )


# Define transformations
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


model = vit_b_16(weights="DEFAULT").to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    train_dataset,
    train_sampler,
    epochs=10,
):

    # Calculate dynamic steps_per_epoch based on actual dataset and configuration
    dataset_size = len(train_dataset)
    batch_size_per_gpu = train_loader.batch_size
    world_size = dist.get_world_size()
    effective_batch_size = batch_size_per_gpu * world_size
    steps_per_epoch = len(train_loader)

    if rank == 0:
        print(f"Dynamic scheduler calculation:")
        print(f"  Dataset size: {dataset_size}")
        print(f"  Batch size per GPU: {batch_size_per_gpu}")
        print(f"  World size: {world_size}")
        print(f"  Effective batch size: {effective_batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")

    # Create scheduler with dynamic steps_per_epoch
    warmup_epochs = 1
    total_epochs = epochs

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_epochs / total_epochs,
        anneal_strategy="cos",
    )

    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm

    training_start_time = time.time()

    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(local_rank), labels.to(local_rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Add scheduler step

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Log every 10 batches for monitoring
            if batch_idx % 10 == 0 and rank == 0:
                batch_acc = 100.0 * correct_predictions / total_samples
                elapsed_time = time.time() - training_start_time
                current_lr = scheduler.get_last_lr()[0]

                wandb.log(
                    {
                        "epoch": epoch,
                        "batch": batch_idx,
                        "batch_loss": loss.item(),
                        "running_accuracy": batch_acc,
                        "learning_rate": current_lr,
                        "total_elapsed_minutes": elapsed_time / 60,
                    }
                )

        avg_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        epoch_duration = time.time() - epoch_start_time
        progress_percentage = ((epoch + 1) / epochs) * 100

        if rank == 0:
            print(
                f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%"
            )

        # Validation step
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(local_rank), labels.to(local_rank)
                outputs = model(images)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total

        if rank == 0:
            print(f"Val Accuracy: {val_accuracy:.2f}%")
            print(f"Epoch time: {epoch_duration:.2f}s")

            wandb.log(
                {
                    # Original trackers for comparison with old runs
                    "acc": val_accuracy,
                    "loss": avg_loss,
                    # New detailed trackers
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "train_accuracy": epoch_accuracy,
                    "val_accuracy": val_accuracy,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "epoch_duration_minutes": epoch_duration / 60,
                    "progress_percentage": progress_percentage,
                    "total_elapsed_minutes": (time.time() - training_start_time) / 60,
                }
            )


with HDF5Dataset(
    "../resources/train_images.hdf5", transform=transform
) as full_train_dataset:

    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=32, num_workers=7
    )

    val_sampler = DistributedSampler(val_dataset)
    val_loader = DataLoader(
        val_dataset, sampler=val_sampler, batch_size=32, num_workers=7, shuffle=False
    )

    train_model(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        train_dataset,
        train_sampler,
    )

dist.destroy_process_group()

if rank == 0:
    torch.save(
        model.module.state_dict(), "lumi_vit_b16_quad_gpu_1node_4gpu_trained_model.pth"
    )
