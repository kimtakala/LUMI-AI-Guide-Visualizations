from torch.utils.data import DataLoader, random_split
import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resources.hdf5_dataset import HDF5Dataset

# Define transformations for dataset
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

model = vit_b_16(weights="DEFAULT")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10):
    start_time = time.time()
    # note that "cuda" is used as a general reference to GPUs,
    # even when running on AMD GPUs that use ROCm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    print(f"Starting training with {len(train_loader)} batches per epoch...")
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch+1}.")
        model.train()
        running_loss = 0.0
        batch_count = 0
        epoch_start = time.time()
        
        for images, labels in train_loader:
            batch_start = time.time()
            batch_count += 1
            
            if batch_count == 1:
                print(f"  First batch loaded, starting training...")
            elif batch_count % 100 == 0:
                elapsed = time.time() - epoch_start
                print(f"  Batch {batch_count}/{len(train_loader)} - {elapsed:.1f}s elapsed")
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            batch_time = time.time() - batch_start
            if batch_count == 1:
                print(f"  First batch completed in {batch_time:.2f}s")
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s, Loss: {running_loss/len(train_loader):.4f}")
        # Validation step
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}%")
    
    end_time = time.time()
    total_time = end_time - start_time
    # Calculate samples trained (train_size * epochs)
    samples_trained = len(train_loader.dataset) * epochs
    time_per_100_samples = (total_time / samples_trained) * 100
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Time per 100 samples: {time_per_100_samples:.2f} seconds")


with HDF5Dataset(
    "../resources/train_images.hdf5", transform=transform
) as full_train_dataset:
    # Splitting the dataset into train and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Configuration parameters
    batch_size = 32
    num_workers = 7
    print(f"Using full dataset of {len(full_train_dataset)} samples")
    print(f"Using {num_workers} workers for data loading")
    print(f"Batch size: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_model(model, criterion, optimizer, train_loader, val_loader)

torch.save(model.state_dict(), "vit_b_16_imagenet.pth")
