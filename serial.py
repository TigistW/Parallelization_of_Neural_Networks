import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import SimpleCNN

def train():
    device = torch.device("cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    model = SimpleCNN().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.time()
    loss_history = []

    for epoch in range(5):
        total_loss = 0.0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

    print("Training time:", time.time() - start)
    torch.save(loss_history, "serial_loss.pt")
    torch.save(model.state_dict(), "serial_model.pt")

if __name__ == "__main__":
    train()
