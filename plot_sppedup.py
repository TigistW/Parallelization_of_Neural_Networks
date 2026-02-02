import torch
import matplotlib.pyplot as plt

serial_time = 342.3532180786133  # sec
parallel_times = {
    2: 215.94809222221375,
    4: 227.26946568489075
}

processes = list(parallel_times.keys())
speedups = [serial_time / parallel_times[p] for p in processes]

plt.plot(processes, speedups, marker='o')
plt.xlabel("Number of Processes")
plt.ylabel("Speedup")
plt.title("Parallel Speedup vs Number of Processes")
plt.show()
