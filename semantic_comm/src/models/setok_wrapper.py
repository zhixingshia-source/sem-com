from pathlib import Path
import torch
from .tokenizer import SetokTokenizer
from .detokenizer import SetokDeTokenizer

class SeTokWrapper:
    def __init__(self, cfg_paths, cfg_setok):
        ckpt_dir = Path(cfg_paths["checkpoints_dir"])
        self.device = cfg_paths.get("device", "cuda")
        self.tokenizer = SetokTokenizer(**{k:v for k,v in cfg_setok.items() if k not in ["weights"]}).to(self.device).eval()
        self.detokenizer = SetokDeTokenizer(**{k:v for k,v in cfg_setok.items() if k not in ["weights"]}).to(self.device).eval()

        # 可选：加载你训练的权重
        tok_w = ckpt_dir / cfg_setok["weights"]["tokenizer"]
        detok_w = ckpt_dir / cfg_setok["weights"]["detokenizer"]
        if tok_w.exists(): self.tokenizer.load_state_dict(torch.load(tok_w, map_location=self.device), strict=False)
        if detok_w.exists(): self.detokenizer.load_state_dict(torch.load(detok_w, map_location=self.device), strict=False)

    @torch.no_grad()
    def encode(self, img_tensor):
        return self.tokenizer(img_tensor)

    @torch.no_grad()
    def decode(self, tokens, attention_masks=None):
        return self.detokenizer(tokens, attention_masks)
