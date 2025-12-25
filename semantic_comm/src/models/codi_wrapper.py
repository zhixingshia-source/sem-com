from pathlib import Path
import torch
from semantic_comm.core.models.model_module_infer import model_module

class CoDiWrapper:
    def __init__(self, cfg_paths, cfg_codi):
        ckpt_dir = Path(cfg_paths["checkpoints_dir"])
        pths = [str(ckpt_dir / p) for p in cfg_codi["weights"]]
        self.model = model_module(data_dir=str(ckpt_dir), pth=pths, fp16=cfg_codi.get("fp16", True)).to(cfg_paths.get("device","cuda")).eval()
        self.cfg = cfg_codi

    @torch.no_grad()
    def text2image(self, prompt):
        o = self.model.inference(
            ['image'], condition=[prompt], condition_types=['text'],
            n_samples=1, image_size=self.cfg["image"]["size"],
            ddim_steps=self.cfg["image"]["ddim_steps"], scale=self.cfg["image"]["scale"]
        )
        return o[0][0]

    @torch.no_grad()
    def text2audio(self, prompt):
        o = self.model.inference(
            xtype=['audio'], condition=[prompt], condition_types=['text'],
            n_samples=1, ddim_steps=self.cfg["audio"]["ddim_steps"], scale=self.cfg["audio"]["scale"]
        )
        return o[0]

    @torch.no_grad()
    def text2video(self, prompt):
        o = self.model.inference(
            ['video'], condition=[prompt], condition_types=['text'],
            n_samples=1, image_size=self.cfg["video"]["size"],
            ddim_steps=self.cfg["video"]["ddim_steps"], num_frames=self.cfg["video"]["num_frames"],
            scale=self.cfg["video"]["scale"]
        )
        return o[0][0]
