import torch
from torchmetrics.functional import structural_similarity_index_measure as ssim

def mse(a, b): return torch.mean((a-b)**2).item()
def ssim_img(a, b): return ssim(a, b).item()
