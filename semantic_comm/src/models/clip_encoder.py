import torch
import torch.nn as nn
from transformers import (
    CLIPModel,
    CLIPProcessor,
    CLIPTokenizer,
    CLIPImageProcessor
)


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",  # ✅ 比 patch16 小很多
        device=None,
        unfreeze=False,
        use_fp16=True,  # ✅ 自动半精度节省显存
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.unfreeze = unfreeze
        self.use_fp16 = use_fp16

        # ✅ 分步初始化，防止 transformer 在 CPU 上加载时炸内存
        print(f"🔄 Initializing CLIPVisionTower with {model_name}...")

        # --- 初始化 tokenizer 与 image_processor ---
        tokenizer = CLIPTokenizer.from_pretrained(model_name)
        image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.processor = CLIPProcessor(tokenizer=tokenizer, image_processor=image_processor)
        self.image_processor = image_processor

        # --- 初始化模型 ---
        dtype = torch.float16 if (str(self.device).startswith("cuda") and self.use_fp16) else torch.float32
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            ignore_mismatched_sizes=True
        ).vision_model
        
        # 确保模型和权重都在正确的设备和数据类型上
        self.model = self.model.to(device=self.device, dtype=dtype)


        # --- 冻结参数（除非你要 finetune）---
        if not self.unfreeze:
            for p in self.model.parameters():
                p.requires_grad = False

        print(f"✅ CLIPVisionTower initialized ({model_name}), "
              f"unfreeze={self.unfreeze}, fp16={self.use_fp16}, device={self.device}")

    def forward(self, images):
        """
        Args:
            images: PIL.Image or torch.Tensor
        Returns:
            Tensor [B, hidden_dim] - vision embeddings
        """
        if isinstance(images, torch.Tensor):
            pixel_values = images.to(self.device)
        else:
            inputs = self.processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device)

        # 确保输入数据在正确的设备和数据类型上
        if self.use_fp16 and str(self.device).startswith("cuda"):
            pixel_values = pixel_values.to(device=self.device, dtype=torch.float16)
        else:
            pixel_values = pixel_values.to(device=self.device, dtype=torch.float32)
        
        outputs = self.model(pixel_values=pixel_values)

        return outputs.pooler_output
