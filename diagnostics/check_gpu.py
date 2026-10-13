import torch
# Standalone matrix multiplication diagnostic, not classifier benchmarking.
import time

print("=" * 50)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available!")

device = torch.device("cuda")

print("GPU:", torch.cuda.get_device_name(0))

# Tạo hai ma trận trên GPU
a = torch.randn(4096, 4096, device=device)
b = torch.randn(4096, 4096, device=device)

# Warm-up
for _ in range(5):
    c = torch.matmul(a, b)

torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()

for _ in range(20):
    c = torch.matmul(a, b)

torch.cuda.synchronize()

elapsed = time.perf_counter() - start

print("=" * 50)
print(f"Device: {c.device}")
print(f"Matrix shape: {a.shape}")
print(f"20 matrix multiplications: {elapsed:.4f} seconds")
print(f"Average: {elapsed / 20 * 1000:.2f} ms")
print("=" * 50)
