import torch
from model import SimpleCNN

serial = SimpleCNN()
parallel = SimpleCNN()

serial.load_state_dict(torch.load("serial_model.pt"))
parallel.load_state_dict(torch.load("parallel_model.pt"))

diff = 0.0
norm = 0.0

for p_s, p_p in zip(serial.parameters(), parallel.parameters()):
    diff += torch.norm(p_s - p_p).item()
    norm += torch.norm(p_s).item()

relative_error = diff / norm
print("Relative parameter difference:", relative_error)
