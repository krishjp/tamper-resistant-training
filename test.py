import torch

# Check if XPU is available
if torch.xpu.is_available():
    device = torch.device("xpu")
    print("XPU is available. Using:", device)
elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple M1 Pro GPU (MPS)")
else:
    device = torch.device("cpu")
    print("XPU is not available. Falling back to CPU.")

# Example tensor operation on device
x = torch.tensor([1.0, 2.0, 3.0], device=device)
y = torch.tensor([4.0, 5.0, 6.0], device=device)
z = x + y
print("Result:", z)