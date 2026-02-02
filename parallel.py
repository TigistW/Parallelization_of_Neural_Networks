import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from model import SimpleCNN

def setup(rank, world_size):
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    transform = transforms.ToTensor()
    dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        sampler=sampler
    )

    model = SimpleCNN()
    model = DDP(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()

    loss_history = []

    for epoch in range(5):
        sampler.set_epoch(epoch)
        total_loss = 0.0

        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if rank == 0:
            loss_history.append(avg_loss)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    if rank == 0:
        print("Parallel training time:", time.time() - start)
        torch.save(loss_history, f"parallel_loss_{world_size}.pt")
        torch.save(model.module.state_dict(), "parallel_model.pt")
    
    cleanup()

if __name__ == "__main__":
    world_size = 2  # number of CPU processes
    mp.spawn(train, args=(world_size,), nprocs=world_size)
