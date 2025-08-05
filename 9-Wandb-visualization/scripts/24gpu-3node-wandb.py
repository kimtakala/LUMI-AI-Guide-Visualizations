import torch
import os
import time
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel
from hdf5_dataset import HDF5Dataset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import psutil
import wandb


# The performance of the CPU mapping needs to be tested
def set_cpu_affinity(local_rank, rank):
    LUMI_GPU_CPU_map = {
        # A mapping from GCD to the closest CPU cores in a LUMI-G node
        # Note that CPU cores 0, 8, 16, 24, 32, 40, 48, 56 are reserved for the
        # system and not available for the user
        # See https://docs.lumi-supercomputer.eu/hardware/lumig/
        0: [49, 50, 51, 52, 53, 54, 55],
        1: [57, 58, 59, 60, 61, 62, 63],
        2: [17, 18, 19, 20, 21, 22, 23],
        3: [25, 26, 27, 28, 29, 30, 31],
        4: [1, 2, 3, 4, 5, 6, 7],
        5: [9, 10, 11, 12, 13, 14, 15],
        6: [33, 34, 35, 36, 37, 38, 39],
        7: [41, 42, 43, 44, 45, 46, 47],
    }
    cpu_list = LUMI_GPU_CPU_map[local_rank]
    print(f"Rank {rank} (local {local_rank}) binding to cpus: {cpu_list}")
    psutil.Process().cpu_affinity(cpu_list)


dist.init_process_group(backend="nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
rank = int(os.environ["RANK"])
set_cpu_affinity(local_rank, rank)

if rank == 0:
    print("Wandb init")
    wandb.init(
        # set the wandb project where this run will be logged
        project="Wandb-visualization",
        name="24 GPU 3 Node Run",
        # track hyperparameters and run metadata
        config={
            "base_learning_rate": 0.001,
            "learning_rate": 0.003,
            "batch_size_per_gpu": 32,
            "effective_batch_size": 32 * 24,  # 32 * 24 GPUs
            "architecture": "vit_b_16",
            "epochs": 10,
            "nodes": 3,
            "gpus": 8,
            "total_gpus": 24,
            "scaling_strategy": "distributeddataparallel_multi_node",
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)  # 3x scaling for 24 GPUs

# Learning rate scheduler with warmup for stable multi-node training
warmup_epochs = 1
total_epochs = 10
steps_per_epoch = 40  # Approximate based on dataset size and distributed sampling

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.003,
    epochs=total_epochs,
    steps_per_epoch=steps_per_epoch,
    pct_start=warmup_epochs / total_epochs,
    anneal_strategy="cos",
)


def train_model(
    model, criterion, optimizer, train_loader, val_loader, train_dataset, epochs=10
):
    # Calculate dynamic steps_per_epoch based on actual dataset and distributed configuration
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
        max_lr=0.003,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=warmup_epochs / total_epochs,
        anneal_strategy="cos",
    )

    training_start_time = time.time()

    for epoch in range(epochs):
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
            scheduler.step()  # Step the scheduler after each batch

            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            # Log every 10 batches for monitoring (only rank 0)
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

        # Gather metrics from all processes with error handling
        correct_tensor = torch.tensor(correct, device=local_rank)
        total_tensor = torch.tensor(total, device=local_rank)

        dist.barrier()
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

        if rank == 0:
            val_accuracy = 100 * correct_tensor.item() / total_tensor.item()
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
        val_dataset, sampler=val_sampler, batch_size=32, num_workers=7
    )

    train_model(model, criterion, optimizer, train_loader, val_loader, train_dataset)

    dist.destroy_process_group()

torch.save(
    model.state_dict(), "lumi_vit_b16_triple_node_full_3node_24gpu_trained_model.pth"
)
