import torch
import matplotlib.pyplot as plt

serial_loss = torch.load("serial_loss.pt")
parallel_loss = torch.load("parallel_loss_4.pt")

epochs = range(1, len(serial_loss) + 1)

plt.plot(epochs, serial_loss, label="Serial")
plt.plot(epochs, parallel_loss, label="Parallel (4 processes)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.show()
