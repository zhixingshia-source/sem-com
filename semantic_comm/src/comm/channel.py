import torch

def add_awgn(x, snr_db):
    signal_power = torch.mean(x ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise

def drop_tokens(x, drop_rate=0.2):
    if x.dim() == 2: mask = (torch.rand(x.shape[0], device=x.device) > drop_rate).float().unsqueeze(1)
    elif x.dim() == 3: mask = (torch.rand(x.shape[0], device=x.device) > drop_rate).float().unsqueeze(1).unsqueeze(2)
    else: raise ValueError("Unexpected token shape")
    return x * mask
