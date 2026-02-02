# Parallelization of Deep Learning Models

## Overview

This project implements and evaluates serial and parallel versions of a deep learning training algorithm as part of a take-home programming assignment on parallelization. The goal is to compare a serial baseline with a data-parallel implementation executed on a shared-memory multicore CPU system, and to analyze performance, scalability, and correctness.

The parallel implementation uses **data parallelism** via **PyTorch Distributed Data Parallel (DDP)** on CPU.

---

## Project Structure

```
parallel-dl-assignment/
│
├── model.py                 # CNN model definition
├── serial_train.py          # Serial training baseline
├── parallel_train.py        # Parallel training using DDP
├── plot_loss.py             # Loss curve visualization
├── plot_speedup.py          # Speedup visualization
├── check_correctness.py     # Parameter difference check
├── requirements.txt         # Python dependencies
├── Parallelization of DNNs.pdf               # Technical report
└── README.md                # This file
```

---

## Requirements

* Python 3.8 or later
* PyTorch
* torchvision
* matplotlib

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Dataset

The **MNIST handwritten digit dataset** is used for all experiments.

* 60,000 training images
* 10,000 test images
* Image size: 28×28 grayscale
* 10 output classes (digits 0–9)

The dataset is automatically downloaded when running the training scripts.

---

## Running the Experiments

### 1. Serial Training

Run the serial baseline implementation:

```bash
python serial_train.py
```

This script:

* Trains the CNN using a single CPU process
* Prints training loss per epoch
* Records total training time
* Saves loss history and final model parameters

---

### 2. Parallel Training

Run the parallel data-parallel implementation:

```bash
python parallel_train.py
```

By default, the script launches **4 parallel CPU processes**. You can change the number of processes by modifying the `world_size` variable in `parallel_train.py`.

This script:

* Partitions the dataset across processes
* Synchronizes gradients using all-reduce
* Records training time and loss values (rank 0)
* Saves the final parallel model parameters

---

## Visualization

### Loss Curves

To compare convergence between serial and parallel training:

```bash
python plot_loss.py
```

This generates a plot of training loss versus epoch for both implementations.

### Speedup Plot

After recording training times, generate the speedup plot:

```bash
python plot_speedup.py
```

Speedup is computed as:

```
Speedup = Serial Training Time / Parallel Training Time
```

---

## Correctness Verification

Correctness is verified by comparing model parameters from serial and parallel runs.

Run:

```bash
python check_correctness.py
```

This script computes the relative L2 norm difference between corresponding parameters. Small differences are expected due to non-deterministic update ordering, but overall similarity confirms correctness.

---

## Notes

* All experiments are performed on a shared-memory CPU system.
* GPU acceleration is not required.
* Results may vary depending on the number of CPU cores and system load.

---

## Reproducibility

To reproduce results:

1. Install dependencies
2. Run serial training
3. Run parallel training with the desired number of processes
4. Generate plots and correctness checks
5. Refer to `Parallelization of DNNs.pdf ` for analysis and discussion

---

## Author

Tigist Wondimneh
