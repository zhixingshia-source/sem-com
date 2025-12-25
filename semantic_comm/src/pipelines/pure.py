# --- Must be placed before pyplot ---
import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
import math
import matplotlib
matplotlib.use("Agg")  # Only keep once, and before pyplot
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from diffusers import AutoencoderKL
from transformers import CLIPModel, CLIPProcessor
from io import BytesIO
import requests

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.manual_seed(0); np.random.seed(0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
#   VAE Tokenizer / DeTokenizer
# ======================
class VQTokenizer:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device=DEVICE, torch_dtype=None):
        print("📥 Loading AutoencoderKL (VAE Encoder)...")
        kwargs = {}
        if "sd-vae" not in model_name:  # 例如 CompVis/stable-diffusion-v1-4 需要 subfolder
            kwargs["subfolder"] = "vae"
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.vae = AutoencoderKL.from_pretrained(model_name, **kwargs).to(device).eval()
        self.device = device
        self.dtype = torch_dtype if torch_dtype is not None else torch.float32

    def __call__(self, image_tensor):
        """
        image_tensor: [B,3,H,W] in [0,1]
        returns tokens [B,N,D], grid size (h,w)
        """
        # 确保输入张量与模型使用相同的数据类型
        image_tensor = image_tensor.to(dtype=self.dtype)
        
        x = 2 * image_tensor - 1  # [0,1] -> [-1,1]
        with torch.no_grad():
            # Encode 时添加 scaling_factor
            latent = self.vae.encode(x).latent_dist.mean * self.vae.config.scaling_factor

        b, c, h, w = latent.shape
        tokens = latent.flatten(2).permute(0, 2, 1)  # [B,N,D]  N=h*w, D=4
        return tokens, h, w


class VQDeTokenizer:
    def __init__(self, model_name="CompVis/stable-diffusion-v1-4", device=DEVICE, torch_dtype=None):
        print("📥 Loading AutoencoderKL (VAE Decoder)...")
        kwargs = {}
        if "sd-vae" not in model_name:  # 例如 CompVis/stable-diffusion-v1-4 需要 subfolder
            kwargs["subfolder"] = "vae"
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.vae = AutoencoderKL.from_pretrained(model_name, **kwargs).to(device).eval()
        self.device = device
        self.dtype = torch_dtype if torch_dtype is not None else torch.float32

    def __call__(self, tokens, h, w, pad_to_grid=True):
        """
        tokens: [B,N,D]   h,w: grid size
        pad_to_grid=True 时：如果 N < h*w，就自动 0 填充到 h*w，再 reshape 为 [B,4,h,w] 解码
        """
        b, n, d = tokens.shape
        target_n = h * w
        
        # 调试信息
        #print(f"🔍 Detokenizer debug: tokens.shape={tokens.shape}, target_n={target_n}, h={h}, w={w}")
        
        if pad_to_grid:
            if n < target_n:
                pad = target_n - n
                pad_tok = tokens.new_zeros(b, pad, d)
                tokens = torch.cat([tokens, pad_tok], dim=1)
                print(f"🔍 Padded tokens to shape: {tokens.shape}")
            elif n > target_n:
                tokens = tokens[:, :target_n, :]
                print(f"🔍 Truncated tokens to shape: {tokens.shape}")

        # 确保 reshape 操作有效
        try:
            latents = tokens.permute(0, 2, 1).reshape(b, d, h, w)  # d 应该=4
            print(f"🔍 Reshaped latents to: {latents.shape}")
        except RuntimeError as e:
            print(f"❌ Reshape error: {e}")
            print(f"   tokens shape: {tokens.shape}")
            print(f"   target shape: [{b}, {d}, {h}, {w}]")
            raise
        
        # 确保 latents 与模型使用相同的数据类型
        latents = latents.to(dtype=self.dtype)
            
        with torch.no_grad():
            # Decode 时取消 scaling_factor
            
            #recon = self.vae.decode(latents).sample
            recon = self.vae.decode(latents / self.vae.config.scaling_factor).sample


        #print("🧠 recon range:", recon.min().item(), recon.max().item())
        recon = recon.clamp(-1, 1)
        return (recon + 1) / 2  # [-1,1] -> [0,1]


# ======================
#   主 Pipeline
# ======================
class VQImageSemanticPipeline:
    def __init__(
        self,
        use_clip=True,
        vae_repo="CompVis/stable-diffusion-v1-4",
        half_if_cuda=True
    ):
        self.device = DEVICE
        torch_dtype = torch.float16 if (half_if_cuda and self.device.type == "cuda") else None

        self.tokenizer = VQTokenizer(model_name=vae_repo, device=self.device, torch_dtype=torch_dtype)
        self.detokenizer = VQDeTokenizer(model_name=vae_repo, device=self.device, torch_dtype=torch_dtype)

        if use_clip:
            print("📥 Loading CLIP for semantic similarity...")
            # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
            # self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            # __init__ 里把 CLIP 换成 L/14@336（更强）
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(self.device).eval()
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

        else:
            self.clip_model = None
            self.clip_processor = None

    # ---- IO ----
    def load_image(self, image_path=None, size=512):
        if image_path and (image_path.startswith("http://") or image_path.startswith("https://")):
            print(f"🌐 Downloading image from URL: {image_path}")
            resp = requests.get(image_path, timeout=15)
            img = Image.open(BytesIO(resp.content)).convert("RGB").resize((size, size))
        elif image_path and Path(image_path).exists():
            img = Image.open(image_path).convert("RGB").resize((size, size))
            print(f"📷 Loaded local image: {image_path}")
        else:
            try:
                print("📷 Downloading test image from picsum...")
                resp = requests.get(f"https://picsum.photos/{size}", timeout=10)
                img = Image.open(BytesIO(resp.content)).convert("RGB").resize((size, size))
            except Exception as e:
                print(f"⚠️ Download failed: {e}. Generating a random image instead.")
                arr = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
        tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        return img, tensor.to(self.device)

    # ---- 压缩策略 A：latent 下采样（推荐）----
    def compress_latent_downsample(self, image_tensor, ratio=0.5):
        """
        先得到 latent，再在 latent 空间做双线性下采样
        ratio 指 "像素数"压缩比（面积），因此边长按 sqrt(ratio)
        修复网格扭曲：先定 N′=round(ratio*h*w) 再分解成最接近 √N′ 的 (h′, w′)
        """
        tokens, h, w = self.tokenizer(image_tensor)  # tokens=[B,N,D], D=4
        latents = tokens.permute(0, 2, 1).reshape(1, 4, h, w)  # [1,4,h,w]
        
        # 修复网格扭曲：精确计算目标尺寸
        target_n = round(h * w * ratio)  # 目标 token 数
        if target_n <= 0:
            target_n = 1
        
        # 找到最接近 sqrt(target_n) 的 h', w' 组合
        sqrt_n = (target_n ** 0.5)
        new_h = max(1, round(sqrt_n))
        new_w = max(1, round(target_n / new_h))
        
        # 确保 h' * w' 接近 target_n
        if new_h * new_w < target_n:
            if new_h <= new_w:
                new_h += 1
            else:
                new_w += 1
        
        # 关键修复：调整为8的倍数，减少反卷积伪影
        def round_to(x, m=8):
            return max(m, int(round(x / m)) * m)
        new_h = round_to(new_h, 8)
        new_w = round_to(new_w, 8)
        
        #print(f"🔍 Downsample: {h}x{w} -> {new_h}x{new_w} (target_n={target_n}, actual_n={new_h*new_w})")
        
        # 改进下采样：优先使用avg_pool2d（当缩放因子接近整数时）
        scale_h = h / new_h
        scale_w = w / new_w
        if abs(scale_h - round(scale_h)) < 1e-6 and abs(scale_w - round(scale_w)) < 1e-6:
            # 整数缩放因子，使用平均池化
            kernel_h, kernel_w = int(round(scale_h)), int(round(scale_w))
            latent_small = F.avg_pool2d(latents, kernel_size=(kernel_h, kernel_w))
        else:
            # 非整数缩放因子，使用双线性插值
            latent_small = F.interpolate(latents, size=(new_h, new_w), mode="bicubic", align_corners=False, antialias=True)
        
        # 关键修复：统计量配准，让下采样后的latent回到原分布
        mu_orig, sigma_orig = latents.mean((2,3), keepdim=True), latents.std((2,3), keepdim=True).clamp_min(1e-6)
        mu_small, sigma_small = latent_small.mean((2,3), keepdim=True), latent_small.std((2,3), keepdim=True).clamp_min(1e-6)
        latent_small = (latent_small - mu_small) / sigma_small * sigma_orig + mu_orig
        tokens_small = latent_small.flatten(2).permute(0, 2, 1)  # [B, N', 4]
        return {"tokens": tokens_small, "h": new_h, "w": new_w, "ratio": ratio, "mode": "latent_downsample"}

    # ---- 压缩策略 B：token 子采样 + 插值回网格（修复版）----
    def compress_token_subsample(self, image_tensor, ratio=0.5):
        tokens, h, w = self.tokenizer(image_tensor)  # tokens=[B,N,4]
        n = tokens.shape[1]
        target_n = max(1, int(n * ratio))
        # 等间距抽样
        idx = torch.linspace(0, n - 1, steps=target_n, device=self.device).round().long()
        tokens_sel = tokens[:, idx, :]  # [B, N', 4]
        
        # 修复：把子采样 token 先还原为粗网格再上采样，而不是 0 填充
        b, n_small, d = tokens_sel.shape
        h_small = max(1, int(h * (ratio ** 0.5)))
        w_small = max(1, int(w * (ratio ** 0.5)))
        
        # 将子采样后的 token 重塑为小网格
        lat_small = tokens_sel.permute(0, 2, 1).reshape(b, d, h_small, w_small)  # [B, 4, h_small, w_small]
        # 上采样回原始网格大小
        lat_full = F.interpolate(lat_small, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
        tokens_full = lat_full.flatten(2).permute(0, 2, 1)  # [B, N, 4]
        
        return {"tokens": tokens_full, "h": h, "w": w, "ratio": ratio, "mode": "token_subsample_fixed"}
    # ---- 压缩策略 C：阈值合并 + 位置还原（带宽按代表token数K计，解码按N=h*w还原）----
    def compress_merge_with_assignment(
        self,
        image_tensor,
        threshold: float = 0.75,  # 降低阈值，允许更多合并
        ratio: float = 1.0,
        neighborhood_radius: int = 2,  # 默认添加邻域约束，只允许局部合并
    ):
        """
        先在 latent 空间按 ratio 下采样得到预算网格 h×w（统一横轴），
        再做余弦阈值合并；簇代表使用“簇均值”；解码前按位置回填到 N=h*w。
        返回：
        tokens: [1, N, D]（用于解码，已保留空间位置）
        unique_tokens: [1, K, D]（真实传输的代表token）
        assignment: [N]（每个位置对应代表集下标 0..K-1）
        budget_hw: N=h*w；K: 代表数
        """
        # 1) 预算网格
        comp = self.compress_latent_downsample(image_tensor, ratio=ratio)
        flat = comp["tokens"][0]                     # [N, D]
        N, D = flat.shape
        h, w = comp["h"], comp["w"]

        # 预计算每个位置的(y,x)坐标，供邻域约束使用
        if neighborhood_radius is not None and neighborhood_radius > 0:
            idxs = torch.arange(N, device=flat.device)
            ys = torch.div(idxs, w, rounding_mode="floor")
            xs = torch.remainder(idxs, w)

        # 2) 贪心代表选择 + 位置指派（先写入“原始代表下标”）
        t_norm = F.normalize(flat, dim=-1)           # [N, D]
        keep_mask = torch.ones(N, dtype=torch.bool, device=flat.device)
        assignment_orig = torch.full((N,), -1, dtype=torch.long, device=flat.device)
        reps_orig = []  # 保存被选为代表的“原始下标”

        for i in range(N):
            if keep_mask[i]:
                reps_orig.append(i)
                sim_i = t_norm @ t_norm[i]           # [N]
                if neighborhood_radius is not None and neighborhood_radius > 0:
                    yi, xi = divmod(i, w)
                    near = (ys - yi).abs() <= neighborhood_radius
                    near &= (xs - xi).abs() <= neighborhood_radius
                    dup = (sim_i >= threshold) & keep_mask & near
                else:
                    dup = (sim_i >= threshold) & keep_mask

                assignment_orig[dup] = i             # 先记录“原始代表下标”
                keep_mask[dup] = False               # 这些位置已被覆盖

        reps = torch.tensor(reps_orig, device=flat.device, dtype=torch.long)  # [K]（原始下标）
        K = int(reps.numel())

        # 3) 原始代表下标 -> 代表集下标（0..K-1）
        rep_id_map = torch.full((N,), -1, dtype=torch.long, device=flat.device)
        rep_id_map[reps] = torch.arange(K, device=flat.device, dtype=torch.long)
        assignment_ids = rep_id_map[assignment_orig]                             # [N] in 0..K-1

        # 兜底：若仍有 -1，指派到最近代表
        if (assignment_ids < 0).any():
            missing = assignment_ids < 0
            # 用当前已有 reps 的 token 作为代表临时值
            reps_tokens = flat[reps]                                             # [K, D]
            dist = torch.cdist(flat[missing], reps_tokens)                       # [M, K]
            nearest = dist.argmin(dim=1)
            assignment_ids[missing] = nearest

        # 4) 用“簇均值”作为代表（而不是 reps 的原值）
        sums = torch.zeros((K, D), device=flat.device, dtype=flat.dtype)         # [K, D]
        sums.index_add_(0, assignment_ids, flat)                                  # Σ x_j
        counts = torch.bincount(assignment_ids, minlength=K).unsqueeze(1).clamp_min(1).to(flat.dtype)
        unique_tokens = sums / counts                                             # [K, D]

        # 5) 展开回 N=h*w 供解码（保持空间位置）
        expanded = unique_tokens[assignment_ids].unsqueeze(0)                     # [1, N, D]

        return {
            "tokens": expanded,                     # 解码真正使用（已保位）
            "h": h, "w": w,
            "mode": "expanded",
            "unique_tokens": unique_tokens.unsqueeze(0),  # [1, K, D] —— 真实传输
            "assignment": assignment_ids,                 # [N] —— 代表集下标（0..K-1）
            "budget_hw": int(h * w),
            "K": K,
        }

    # ---- 解压 ----
    def decompress(self, compressed):
        mode = compressed.get("mode", "")
        if mode == "expanded":
            # tokens 已经是 [1, N, D] 且 N=h*w，保持空间位置，直接解码即可
            return self.detokenizer(compressed["tokens"], compressed["h"], compressed["w"], pad_to_grid=False)
        elif mode == "token_subsample_fixed":
            # 修复版：tokens 已经是完整网格，不需要 pad
            return self.detokenizer(compressed["tokens"], compressed["h"], compressed["w"], pad_to_grid=False)
        # 其余保持原逻辑
        pad_to_grid = (mode == "token_subsample")
        return self.detokenizer(compressed["tokens"], compressed["h"], compressed["w"], pad_to_grid=pad_to_grid)

    # ---- 图像增强 ----
    def unsharp_mask(self, img_tensor, radius=1.0, amount=1.0, threshold=0):
        """
        Unsharp Mask 锐化：高斯模糊后回加残差，恢复边缘细节
        radius: 模糊半径，amount: 锐化强度，threshold: 锐化阈值
        """
        with torch.no_grad():
            # 高斯模糊
            kernel_size = max(3, int(2 * radius + 1))
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            # 创建高斯核
            sigma = radius / 3.0
            x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
            gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
            gauss = gauss / gauss.sum()
            
            # 2D 高斯核
            kernel = gauss[:, None] * gauss[None, :]
            kernel = kernel / kernel.sum()
            
            # 扩展到 3 通道
            kernel = kernel[None, None, :, :].repeat(3, 1, 1, 1)
            
            # 应用高斯模糊
            blurred = F.conv2d(img_tensor, kernel, padding=kernel_size//2, groups=3)
            
            # 计算锐化：原图 + amount * (原图 - 模糊图)
            sharpened = img_tensor + amount * (img_tensor - blurred)
            
            # 应用阈值：只对差异大于阈值的像素进行锐化
            if threshold > 0:
                mask = torch.abs(img_tensor - blurred) > threshold
                sharpened = torch.where(mask, sharpened, img_tensor)
            
            return sharpened.clamp(0, 1)

    def gaussian_blur_test(self, img_tensor, sigma_range=(0.5, 3.0), steps=5):
        """
        测试原图 vs 高斯模糊原图的 CLIP 相似度极限
        """
        print("🔍 测试高斯模糊对 CLIP 相似度的影响...")
        
        sigmas = torch.linspace(sigma_range[0], sigma_range[1], steps)
        similarities = []
        
        with torch.no_grad():
            for sigma in sigmas:
                # 高斯模糊
                kernel_size = max(3, int(2 * sigma + 1))
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # 创建高斯核
                x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
                gauss = torch.exp(-(x ** 2) / (2 * sigma ** 2))
                gauss = gauss / gauss.sum()
                kernel = gauss[:, None] * gauss[None, :]
                kernel = kernel / kernel.sum()
                kernel = kernel[None, None, :, :].repeat(3, 1, 1, 1)
                
                blurred = F.conv2d(img_tensor, kernel, padding=kernel_size//2, groups=3)
                blurred = blurred.clamp(0, 1)
                
                # 计算 CLIP 相似度
                sim = self.similarity(img_tensor, blurred)
                similarities.append(float(sim))    
                print(f"   σ={sigma:.2f}: CLIP 相似度 = {sim:.4f}")
        
        print(f"📊 模糊极限：最高 {max(similarities):.4f}，最低 {min(similarities):.4f}")
        return dict(zip(sigmas.tolist(), similarities))

    def bandwidth_match(self, ref_tensor, target_h, target_w, factor=8):
        """
        将参考图低通至与重建等效的图像分辨率，然后再回到原大小
        实现频带对齐，让评估更公平
        
        Args:
            ref_tensor: [B, C, H, W] 参考图像张量
            target_h, target_w: 目标 latent 尺寸
            factor: VAE 下采样因子 (默认8)
        """
        # 计算等效的图像分辨率
        H_eff, W_eff = target_h * factor, target_w * factor
        
        # 先下采样到等效分辨率
        low = F.interpolate(ref_tensor, size=(H_eff, W_eff), mode="bicubic",
                           align_corners=False, antialias=True)
        
        # 再上采样回原尺寸
        up = F.interpolate(low, size=ref_tensor.shape[2:], mode="bicubic",
                          align_corners=False, antialias=True)
        
        return up.clamp(0, 1)

    def _five_crops_336(self, pil):
        """
        生成五裁剪：四角 + 中心，每个 336×336
        """
        W, H = pil.size
        s = 336
        
        # 如果图像太小，直接缩放到 336×336
        if W < s or H < s:
            pil = ImageOps.fit(pil, (s, s), method=Image.BICUBIC, centering=(0.5, 0.5))
            return [pil]
        
        # 五裁剪：左上、右上、左下、右下、中心
        crops = [
            (0, 0, s, s),                    # 左上
            (W-s, 0, W, s),                  # 右上
            (0, H-s, s, H),                  # 左下
            (W-s, H-s, W, H),                # 右下
            ((W-s)//2, (H-s)//2, (W+s)//2, (H+s)//2)  # 中心
        ]
        
        return [pil.crop(box) for box in crops]

    def _clip_feats_multi(self, pil):
        """
        多裁剪 CLIP 特征提取：五裁剪 + 整图
        """
        imgs = self._five_crops_336(pil) + [pil]  # 加上整图一张
        feats = []
        
        for im in imgs:
            inputs = self.clip_processor(images=im, return_tensors="pt").to(self.device)
            f = self.clip_model.get_image_features(**inputs).float()
            feats.append(F.normalize(f, dim=-1))
        
        return torch.cat(feats, dim=0)  # [C, D]

    # ---- 相似度 ----
    def similarity(self, img1_tensor, img2_tensor, enhance_recon=False, unsharp_params=None):
        """
        计算两张图像的 CLIP 相似度
        使用五裁剪 max-pool 策略，对模糊/对齐偏差更稳健
        
        Args:
            img1_tensor: [B, C, H, W] 参考图像
            img2_tensor: [B, C, H, W] 重建图像
            enhance_recon: 是否对重建图进行锐化增强
            unsharp_params: 锐化参数 (radius, amount, threshold)
        """
        if self.clip_model is None:
            return float(1.0 / (1.0 + F.mse_loss(img1_tensor, img2_tensor).item()))
        
        with torch.no_grad():
            # 对重建图进行锐化增强
            if enhance_recon:
                if unsharp_params is None:
                    unsharp_params = (1.0, 1.0, 0)  # (radius, amount, threshold)
                img2_tensor = self.unsharp_mask(img2_tensor, *unsharp_params)
            
            def pil_from_tensor(t):
                return Image.fromarray((t[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))

            # 使用五裁剪 max-pool 策略
            pil1 = pil_from_tensor(img1_tensor)
            pil2 = pil_from_tensor(img2_tensor)
            
            F1 = self._clip_feats_multi(pil1)  # [C1, D] 已归一化
            F2 = self._clip_feats_multi(pil2)  # [C2, D] 已归一化
            
            # max-over-crops: 取所有裁剪对的最大相似度
            sim_matrix = torch.mm(F1, F2.t())  # [C1, C2]
            max_sim = sim_matrix.max()
            
            return float(max_sim.item())

    # ---- 零压缩回传自检 ----
    def self_check_vae_clip(self, img_tensor):
        """
        零压缩回传自检：验证 VAE 和 CLIP 是否正常工作
        期望：PSNR ≈ 30–33 dB（自然图），CLIP 余弦 ≈ 0.98+
        """
        print("🔍 开始零压缩回传自检...")
        
        with torch.no_grad():
            # VAE 编码-解码回传
            x = img_tensor * 2 - 1  # [0,1] -> [-1,1]
            z = self.tokenizer.vae.encode(x).latent_dist.mean * self.tokenizer.vae.config.scaling_factor
            y = self.tokenizer.vae.decode(z / self.tokenizer.vae.config.scaling_factor).sample
            y = (y.clamp(-1,1)+1)/2  # [-1,1] -> [0,1]
            
            # 计算 PSNR
            mse = F.mse_loss(img_tensor, y).item()
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # 计算 CLIP 相似度
            if self.clip_model is not None:
                clip_sim = self.similarity(img_tensor, y)
            else:
                clip_sim = 1.0 / (1.0 + mse)
            
            print(f"✅ VAE 自检结果:")
            print(f"   PSNR: {psnr:.2f} dB")
            print(f"   CLIP 相似度: {clip_sim:.4f}")
            
            # 判断是否正常
            if psnr >= 30 and clip_sim >= 0.95:
                print("✅ VAE 和 CLIP 工作正常")
                return True
            else:
                print("⚠️ VAE 或 CLIP 可能有问题，请检查配置")
                return False


    # ---- 可视化 ----
    def visualize_grid(self, orig_img, recon_imgs, ratios, title="Reconstruction Comparison", save_path=None):
        cols = len(recon_imgs) + 1
        plt.figure(figsize=(3.2 * cols, 3.2))
        # 原图
        plt.subplot(1, cols, 1)
        plt.imshow(np.array(orig_img)); plt.axis("off"); plt.title("Original")
        # 重建
        for i, (img, r) in enumerate(zip(recon_imgs, ratios), start=2):
            plt.subplot(1, cols, i)
            plt.imshow(np.array(img)); plt.axis("off"); plt.title(f"ratio={r:g}")
        plt.suptitle(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {save_path}")
        plt.show()

    def plot_curve(self, xs, ys, xlabel, ylabel, title, save_path=None):
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, "o-")
        plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
        plt.grid(True, alpha=0.3); plt.ylim(0,1)
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches="tight")
            print(f"💾 Saved: {save_path}")
        plt.show()
# ====== 新增：token 网格与合并边界可视化工具 ======

def _draw_grid_on_axes(ax, img_h, img_w, h, w, color='white', lw=0.6, alpha=0.8):
    cell_h = img_h / h
    cell_w = img_w / w
    # 画水平网格线
    for i in range(1, h):
        y = i * cell_h
        ax.plot([0, img_w], [y, y], color=color, linewidth=lw, alpha=alpha)
    # 画垂直网格线
    for j in range(1, w):
        x = j * cell_w
        ax.plot([x, x], [0, img_h], color=color, linewidth=lw, alpha=alpha)

def _draw_assignment_boundaries(ax, assignment_hw, img_h, img_w, color='lime', lw=1.4, alpha=0.95):
    """
    assignment_hw: [h, w]，每个位置的簇ID（合并后）
    在相邻token簇ID不相等的地方画边界线
    """
    h, w = assignment_hw.shape
    cell_h = img_h / h
    cell_w = img_w / w
    # 横向边界
    for i in range(h - 1):
        for j in range(w):
            if assignment_hw[i, j] != assignment_hw[i + 1, j]:
                y = (i + 1) * cell_h
                x0 = j * cell_w
                x1 = (j + 1) * cell_w
                ax.plot([x0, x1], [y, y], color=color, linewidth=lw, alpha=alpha)
    # 纵向边界
    for j in range(w - 1):
        for i in range(h):
            if assignment_hw[i, j] != assignment_hw[i, j + 1]:
                x = (j + 1) * cell_w
                y0 = i * cell_h
                y1 = (i + 1) * cell_h
                ax.plot([x, x], [y0, y1], color=color, linewidth=lw, alpha=alpha)

def make_uniform_assignment(h, w, K):
    """
    生成均匀划分到 ~K 个块的 assignment（baseline）
    思路：把 h×w 粗分成 kh×kw 个大块（kh*kw≈K），每块一个簇ID
    """
    kh = max(1, min(h, int(round(math.sqrt(K)))))
    kw = max(1, min(w, int(math.ceil(K / kh))))
    block_h = int(math.ceil(h / kh))
    block_w = int(math.ceil(w / kw))
    assign = np.zeros((h, w), dtype=np.int32)
    cur = 0
    for i0 in range(0, h, block_h):
        for j0 in range(0, w, block_w):
            i1 = min(h, i0 + block_h); j1 = min(w, j0 + block_w)
            assign[i0:i1, j0:j1] = cur
            cur += 1
    return assign, cur  # 返回 assignment 以及实际块数

def merge_to_target_K(pipeline, img_tensor, ratio, K_target, tau_low=0.50, tau_high=0.98, max_iter=12, neighborhood_radius=2):
    """
    二分搜索阈值tau，使 compress_merge_with_assignment 的 K 尽量接近目标 K
    经验：阈值↑ => 更难合并 => K↑；阈值↓ => 更容易合并 => K↓
    """
    best = None
    for _ in range(max_iter):
        tau = (tau_low + tau_high) / 2
        merged = pipeline.compress_merge_with_assignment(
            img_tensor, threshold=tau, ratio=ratio, neighborhood_radius=neighborhood_radius
        )
        K = int(merged["K"])
        best = (merged, tau)
        if K > K_target:
            # token太多，需要更多合并 => 降低阈值
            tau_high = tau
        else:
            # token太少或刚好，尝试提高阈值
            tau_low = tau
    return best  # (merged_dict, tau_found)

def visualize_token_merging_panel(pipeline, orig_img, img_tensor, ratio=0.25, K_target=60, save_path="token_merging_panel.png"):
    """
    生成4行面板：
    1) Original
    2) Budget tokenized (h×w 网格)
    3) Uniform baseline (~K tokens) + 边界
    4) Ours (≈K tokens) + 边界
    """
    # 预算栅格（不合并）
    budget = pipeline.compress_latent_downsample(img_tensor, ratio=ratio)
    h, w = int(budget["h"]), int(budget["w"])

    # baseline: 均匀合并到 ~K
    assign_uniform, K_uniform = make_uniform_assignment(h, w, K_target)

    # ours: 搜阈值得到 ≈K 的自适应合并
    (merged, tau) = merge_to_target_K(pipeline, img_tensor, ratio=ratio, K_target=K_target, neighborhood_radius=2)
    assign_ours = merged["assignment"].view(h, w).detach().cpu().numpy()
    K_ours = int(merged["K"])

    # 画图
    fig, axs = plt.subplots(4, 1, figsize=(5.2, 10.0))
    for ax in axs:
        ax.axis('off')

    img_np = np.array(orig_img)
    H, W = img_np.shape[0], img_np.shape[1]

    # 1) Original
    axs[0].imshow(img_np)
    axs[0].set_title("Original", fontsize=12)

    # 2) Budget tokenized
    axs[1].imshow(img_np)
    _draw_grid_on_axes(axs[1], H, W, h, w, color='white', lw=0.6, alpha=0.9)
    axs[1].set_title(f"Budget tokenized (h×w = {h}×{w}, N={h*w})", fontsize=12)

    # 3) Uniform (~K)
    axs[2].imshow(img_np)
    _draw_assignment_boundaries(axs[2], assign_uniform, H, W, color='yellow', lw=1.4, alpha=0.95)
    axs[2].set_title(f"Uniform baseline (K≈{K_uniform})", fontsize=12)

    # 4) Ours (≈K)
    axs[3].imshow(img_np)
    _draw_assignment_boundaries(axs[3], assign_ours, H, W, color='lime', lw=1.6, alpha=0.95)
    axs[3].set_title(f"Ours: Adaptive merging (K≈{K_ours}, τ≈{tau:.2f})", fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"💾 Saved: {save_path}")


# ======================
#   运行
# ======================
def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--image", type=str, default="", help="指定本地图片路径；不填则随机下载一张")
        parser.add_argument("--ratios", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.125], help="压缩比例列表（面积比）")
        parser.add_argument("--mode", type=str, default="latent", choices=["latent","token","enhanced","adaptive"], help="压缩策略")
        parser.add_argument("--no-clip", action="store_true", help="不加载CLIP，使用MSE相似度")
        parser.add_argument("--vae", type=str, default="stabilityai/sd-vae-ft-ema", help="VAE权重来源")
        parser.add_argument("--outdir", type=str, default="outputs_vq", help="输出目录")
        parser.add_argument("--unsharp_radius", type=float, default=1.0, help="锐化半径")
        parser.add_argument("--unsharp_amount", type=float, default=1.0, help="锐化强度")
        parser.add_argument("--unsharp_threshold", type=float, default=0, help="锐化阈值")
        parser.add_argument("--viz_token_panel", action="store_true", help="生成token合并可视化面板")
        parser.add_argument("--panel_ratio", type=float, default=0.25, help="面板预算比例（面积比）")
        parser.add_argument("--panel_K", type=int, default=60, help="面板目标token数")
        args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    pipe = VQImageSemanticPipeline(use_clip=not args.no_clip, vae_repo=args.vae, half_if_cuda=False)

    # 1) 读图
    orig_img, img_tensor = pipe.load_image(args.image, size=512)
    
    # 1.5) 零压缩回传自检
    pipe.self_check_vae_clip(img_tensor)
    
    # 1.6) 高斯模糊极限测试
    pipe.gaussian_blur_test(img_tensor)

    # 2) 遍历压缩比
    recons = []
    sims = []
    recon_tensors = []
    k_values = []  # 实际 token 数

    for r in args.ratios:
        r = float(r)
        print(f"\n🔹 Compressing with ratio={r:g} using mode={args.mode} ...")
        if args.mode == "latent":
            comp = pipe.compress_latent_downsample(img_tensor, ratio=r)
        elif args.mode == "token":
            comp = pipe.compress_token_subsample(img_tensor, ratio=r)
        elif args.mode == "adaptive":
            comp = pipe.compress_merge_with_assignment(img_tensor, ratio=r)
        else:  # enhanced mode
            comp = pipe.compress_latent_downsample(img_tensor, ratio=r)

        recon_tensor = pipe.decompress(comp)
        # 只在尺寸不一致时才插值，避免不必要的质量损失
        if recon_tensor.shape[2:] != img_tensor.shape[2:]:
            recon_tensor = F.interpolate(recon_tensor, size=img_tensor.shape[2:], mode="bicubic", align_corners=False, antialias=True)

        # 频带对齐：把参考图低通到同等带宽
        ref_matched = pipe.bandwidth_match(img_tensor, comp["h"], comp["w"])

        # 根据模式选择相似度计算方式
        if args.mode == "enhanced":
            # 增强模式：使用频带对齐 + 锐化增强的相似度
            sim = pipe.similarity(ref_matched, recon_tensor, enhance_recon=True, 
                                unsharp_params=(args.unsharp_radius, args.unsharp_amount, args.unsharp_threshold))
            sim_original = pipe.similarity(img_tensor, recon_tensor)  # 原始对比
            sim_aligned = pipe.similarity(ref_matched, recon_tensor)  # 频带对齐对比
            print(f"✅ similarity={sim:.4f} (original={sim_original:.4f}, aligned={sim_aligned:.4f}, enhanced)")
        else:
            # 普通模式：显示频带对齐效果
            sim_original = pipe.similarity(img_tensor, recon_tensor)
            sim_aligned = pipe.similarity(ref_matched, recon_tensor)
            sim_enhanced = pipe.similarity(ref_matched, recon_tensor, enhance_recon=True, 
                                         unsharp_params=(args.unsharp_radius, args.unsharp_amount, args.unsharp_threshold))
            sim = sim_aligned  # 使用频带对齐的结果作为主要指标
            print(f"✅ similarity={sim:.4f} (original={sim_original:.4f}, aligned={sim_aligned:.4f}, enhanced={sim_enhanced:.4f})")
        
        sims.append(sim)
        recon_tensors.append(recon_tensor)
        
        # 记录实际 token 数 K
        k = comp["h"] * comp["w"]
        k_values.append(k)

        recon_img = Image.fromarray((recon_tensor[0].permute(1,2,0).clamp(0,1).cpu().numpy()*255).astype(np.uint8))
        recons.append(recon_img)
    

    # 3) 可视化对比 & 曲线
    grid_path = os.path.join(args.outdir, f"recon_grid_{args.mode}.png")
    pipe.visualize_grid(orig_img, recons, args.ratios, title=f"Reconstruction ({args.mode})", save_path=grid_path)

    curve_path = os.path.join(args.outdir, f"similarity_curve_{args.mode}.png")
    # 使用 K（实际 token 数）作为横轴，更直观
    pipe.plot_curve(k_values, sims, xlabel="Token Count (K)", ylabel="Semantic Similarity", title=f"Similarity vs Token Count ({args.mode})", save_path=curve_path)
    if args.viz_token_panel:
        panel_path = os.path.join(args.outdir, f"token_merging_panel_ratio{args.panel_ratio}_K{args.panel_K}.png")
        visualize_token_merging_panel(pipe, orig_img, img_tensor,
                                    ratio=args.panel_ratio, K_target=args.panel_K,
                                    save_path=panel_path)
    # 4) 保存 JSON
    out_json = {
        "mode": args.mode,
        "ratios": [float(x) for x in args.ratios],
        "k_values": [int(k) for k in k_values],
        "similarities": [float(s) for s in sims],
        "use_clip": not args.no_clip,
        "image": args.image if args.image else "picsum/random_or_random_generated",
        "vae": args.vae,
    }
    json_path = os.path.join(args.outdir, f"results_{args.mode}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved: {json_path}")

    # 5) 控制台总结
    best_idx = int(np.argmax(sims))
    worst_idx = int(np.argmin(sims))
    print("\n=== Summary ===")
    print(f"Best: K={k_values[best_idx]} (ratio={args.ratios[best_idx]:.3f}) | similarity={sims[best_idx]:.4f}")
    print(f"Worst: K={k_values[worst_idx]} (ratio={args.ratios[worst_idx]:.3f}) | similarity={sims[worst_idx]:.4f}")
    print(f"Token range: {min(k_values)} → {max(k_values)}")
    print("Done ✅")


def simulate_channel_noise(tokens, noise_level=0.1):
    """模拟信道噪声"""
    if isinstance(tokens, dict) and 'tokens' in tokens:
        # 对token添加噪声
        noisy_tokens = tokens.copy()
        noise = torch.randn_like(tokens['tokens']) * noise_level
        noisy_tokens['tokens'] = tokens['tokens'] + noise
        return noisy_tokens
    else:
        # 直接对tensor添加噪声
        noise = torch.randn_like(tokens) * noise_level
        return tokens + noise

def save_comparison_images(original_tensor, recon_tensor, ratio, outdir="test_results"):
    """保存原始图像和重建图像的对比"""
    import os
    os.makedirs(outdir, exist_ok=True)
    
    # 转换为PIL图像
    def tensor_to_pil(tensor):
        # 从 [1, 3, H, W] 转换为 [H, W, 3]
        img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    original_img = tensor_to_pil(original_tensor)
    recon_img = tensor_to_pil(recon_tensor)
    
    # 创建对比图像
    width, height = original_img.size
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original_img, (0, 0))
    comparison.paste(recon_img, (width, 0))
    
    # 保存图像
    filename = f"comparison_ratio_{ratio:.1f}.png"
    filepath = os.path.join(outdir, filename)
    comparison = comparison.convert("RGB")

    comparison.save(filepath)
    return filepath

def test_compression_ratios(image_path="", vae_repo="stabilityai/sd-vae-ft-ema"):
    """测试不同压缩比例的效果"""
    print("🧪 开始测试不同压缩比例的效果...")
    
    # 创建 pipeline
    pipe = VQImageSemanticPipeline(use_clip=False, vae_repo=vae_repo, half_if_cuda=False)
    print("✅ Pipeline 创建成功 (使用更好的VAE + float32)")
    
    # 使用你已有的下载逻辑；不给路径就在线下载，失败才会随机
    _, img_tensor = pipe.load_image(image_path, size=256)
    print("✅ 测试图像创建成功 (256x256，更容易还原)")
    
    # 测试不同的压缩比例
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print("\n=== 压缩效果测试 ===")
    print("压缩比例 | Token数量 | 压缩率 | 重建质量 | 信道噪声 | 效果评价")
    print("-" * 70)
    
    for ratio in ratios:
        try:
            # 压缩
            comp = pipe.compress_latent_downsample(img_tensor, ratio=ratio)
            
            # 计算token数量 (使用N而不是numel())
            if isinstance(comp, dict) and 'tokens' in comp:
                token_count = comp['tokens'].shape[1]  # = N
            else:
                token_count = comp.shape[1] if len(comp.shape) > 1 else 1
            
            # 解压（无噪声）
            recon_tensor = pipe.decompress(comp)
            
            # 计算重建质量 (MSE)
            # 只在尺寸不一致时才插值
            if recon_tensor.shape[2:] != img_tensor.shape[2:]:
                recon_tensor = torch.nn.functional.interpolate(
                        recon_tensor, size=img_tensor.shape[2:], mode="bicubic", align_corners=False, antialias=True
                    )
            recon_tensor = recon_tensor.clamp(0, 1)  # 确保在[0,1]范围内
            mse = torch.nn.functional.mse_loss(img_tensor, recon_tensor).item()

            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            
            # 计算压缩率 (latent像素 / 原图像素)
            latent_pixels = comp['tokens'].shape[1]  # = N
            orig_pixels = img_tensor.shape[-2] * img_tensor.shape[-1]
            compression_ratio = latent_pixels / orig_pixels
            
            # 信道噪声模拟
            noise_levels = [0.0, 0.05, 0.1, 0.2]
            noise_results = []
            
            for noise_level in noise_levels:
                if noise_level == 0.0:
                    # 无噪声情况
                    noisy_comp = comp
                    noisy_recon = recon_tensor
                else:
                    # 添加信道噪声
                    noisy_comp = simulate_channel_noise(comp, noise_level)
                    noisy_recon = pipe.decompress(noisy_comp)
                
                # 计算噪声下的重建质量
                # 只在尺寸不一致时才插值
                if noisy_recon.shape[2:] != img_tensor.shape[2:]:
                    noisy_recon = torch.nn.functional.interpolate(
                            noisy_recon, size=img_tensor.shape[2:], mode="bicubic", align_corners=False, antialias=True
                        )
                noisy_recon = noisy_recon.clamp(0, 1)  # 确保在[0,1]范围内
                noisy_mse = torch.nn.functional.mse_loss(img_tensor, noisy_recon).item()

                noisy_psnr = 20 * np.log10(1.0 / np.sqrt(noisy_mse)) if noisy_mse > 0 else float('inf')
                noise_results.append(noisy_psnr)
            
            # 效果评价
            if psnr > 30:
                quality = "优秀"
            elif psnr > 25:
                quality = "良好"
            elif psnr > 20:
                quality = "一般"
            else:
                quality = "较差"
            
            # 保存对比图像（选择几个代表性的比例）
            if ratio in [0.2, 0.5, 0.8]:
                comparison_path = save_comparison_images(img_tensor, recon_tensor, ratio)
                print(f"📸 已保存对比图像: {comparison_path}")
            
            results.append({
                'ratio': ratio,
                'tokens': token_count,
                'compression_ratio': compression_ratio,
                'psnr': psnr,
                'noise_psnr': noise_results,
                'quality': quality
            })
            
            # 显示噪声影响
            noise_info = f"无噪声:{noise_results[0]:.1f}dB, 5%:{noise_results[1]:.1f}dB, 10%:{noise_results[2]:.1f}dB, 20%:{noise_results[3]:.1f}dB"
            print(f"{ratio:8.1f} | {token_count:8d} | {compression_ratio:6.3f} | {psnr:6.1f}dB | {noise_info} | {quality}")
            
        except Exception as e:
            print(f"{ratio:8.1f} | 错误: {str(e)}")
    
    # 分析结果
    print("\n=== 压缩效果分析 ===")
    if results:
        best_quality = max(results, key=lambda x: x['psnr'])
        best_compression = max(results, key=lambda x: x['compression_ratio'])
        
        print(f"🏆 最佳质量: 压缩比例 {best_quality['ratio']:.1f} (PSNR: {best_quality['psnr']:.1f}dB)")
        print(f"📦 最高压缩: 压缩比例 {best_compression['ratio']:.1f} (压缩率: {best_compression['compression_ratio']:.3f})")
        
        # 噪声鲁棒性分析
        print("\n🔊 噪声鲁棒性分析:")
        for result in results:
            noise_degradation = result['psnr'] - result['noise_psnr'][-1]  # 20%噪声下的质量下降
            print(f"   压缩比例 {result['ratio']:.1f}: 20%噪声下质量下降 {noise_degradation:.1f}dB")
        
        # 推荐设置
        print("\n💡 推荐设置:")
        for result in results:
            if result['psnr'] > 25 and result['compression_ratio'] < 0.1:
                noise_robustness = result['psnr'] - result['noise_psnr'][2]  # 10%噪声下的鲁棒性
                print(f"   - 压缩比例 {result['ratio']:.1f}: 平衡质量与压缩率，10%噪声鲁棒性 {noise_robustness:.1f}dB")
    
    print("\n✅ 压缩效果测试完成")
    print("📁 对比图像已保存到 test_results/ 目录")

def add_awgn(latent, snr_db):
    """
    添加加性白高斯噪声(AWGN)
    Args:
        latent: 输入latent张量
        snr_db: 信噪比(dB)
    Returns:
        添加噪声后的latent
    """
    power = latent.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = power / snr_linear
    noise = torch.randn_like(latent) * noise_power.sqrt()
    return latent + noise

def measure_latency(func, *args, **kwargs):
    """
    测量函数执行时间
    """
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def measure_latency_gpu(func, *args, **kwargs):
    """
    测量GPU函数执行时间（带同步）
    """
    import time, torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    out = func(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.time()
    return out, t1 - t0

def tx_latency_bits(K, r_bits=32, R_net_mbps=10.0,  # 网络 Mbps
                    payload_bytes=1200, header_bytes=28, fec=1.0, prop_ms=0.0):
    """
    计算基于token数量的传输延迟
    
    Args:
        K: token数量
        r_bits: 每个token的平均比特数（量化+熵编码后）
        R_net_mbps: 网络速率 (Mbps)
        payload_bytes: 每包有效载荷字节数
        header_bytes: 每包头开销字节数
        fec: FEC/冗余系数
        prop_ms: 传播延迟 (ms)
    
    Returns:
        传输延迟 (秒)
    """
    import numpy as np
    R = R_net_mbps * 1e6                # bit/s
    B = int(np.ceil(fec * K * r_bits))  # 要发的净荷 bit
    P = payload_bytes * 8               # 每包净荷 bit
    H = header_bytes * 8                # 每包头部 bit
    N = int(np.ceil(B / P))             # 包数
    total_bits = B + N * H              # 加上包头
    t_tx = total_bits / R               # 传输时延 (s)
    t_prop = prop_ms / 1000.0           # 传播时延 (s)
    return t_tx + t_prop

def generate_comprehensive_analysis(pipeline, input_path, output_dir):
    """
    生成综合分析图表
    """
    print("开始生成综合分析图表...")
    
    # 使用pipeline的load_image方法加载图片
    orig_img, img_tensor = pipeline.load_image(input_path)
    
    # 获取原始latent
    vae = pipeline.detokenizer.vae  # 或 pipeline.tokenizer.vae
    scale = vae.config.scaling_factor
    with torch.no_grad():
        latent = vae.encode(img_tensor * 2 - 1).latent_dist.mean * scale
    
    # 实验1: 不同压缩比例下的准确率
    print("实验1: 不同压缩比例下的准确率...")
    ratios = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
    similarities_ratio = []
    token_counts_ratio = []
    
    for ratio in ratios:
        # 压缩
        compressed = pipeline.compress_latent_downsample(img_tensor, ratio)
        token_count = compressed['tokens'].shape[1]  # N
        
        # 重建
        recon_img = pipeline.decompress(compressed)
        
        # 计算相似度 - 关键修复：先对齐参考图带宽再评估
        # 计算相似度 - 用原始latent的尺寸对齐参考图带宽
        H0, W0 = int(latent.shape[2]), int(latent.shape[3])
        ref_matched = pipeline.bandwidth_match(img_tensor, H0, W0)
        similarity = pipeline.similarity(ref_matched, recon_img)

        
        similarities_ratio.append(similarity)
        token_counts_ratio.append(token_count)
    
    # ========= Experiment 2: Accuracy under cosine thresholds (unified x-axis: budget token = h*w) =========
    print("Experiment 2: accuracy under different cosine thresholds...")

    cosine_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    similarities_threshold = {th: [] for th in cosine_thresholds}
    unique_token_counts = {th: [] for th in cosine_thresholds}   # 实际传输的K（代表token数）

    # 1) Compute budget tokens (before merging) for each ratio
    budget_tokens_per_ratio = []
    for ratio in ratios:
        comp = pipeline.compress_latent_downsample(img_tensor, ratio)
        budget_tokens_per_ratio.append(int(comp["h"] * comp["w"]))

    print("DEBUG budgets (h*w per ratio):", budget_tokens_per_ratio)  # e.g. [4096, 3249, 2401, 1600, 784, 400, 196]

    # 2) For each threshold, run position-preserving merging and compute similarity
    for th in cosine_thresholds:
        sims_this_th = []
        ks_this_th = []
        for ratio in ratios:
            # 合并但保留空间位置；带宽按K计，解码按N=h*w展开
            merged = pipeline.compress_merge_with_assignment(img_tensor, threshold=th, ratio=ratio)
            recon  = pipeline.decompress(merged)
            sim    = pipeline.similarity(img_tensor, recon)

            sims_this_th.append(float(sim))
            ks_this_th.append(int(merged["K"]))  # 真实传输token数

        similarities_threshold[th] = sims_this_th
        unique_token_counts[th]    = ks_this_th

    # ========= Figure 2: Token merging vs. semantic similarity (unified x-axis) =========
    import numpy as np

    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(cosine_thresholds)))

    # sort by x-axis to get monotonic lines
    order = np.argsort(budget_tokens_per_ratio)
    xs_sorted = np.array(budget_tokens_per_ratio)[order]

    for i, th in enumerate(cosine_thresholds):
        ys = np.array(similarities_threshold[th])[order]
        # 在图例里显示平均带宽占比（K / budget）
        avg_ratio = (np.array(unique_token_counts[th]) / np.array(budget_tokens_per_ratio)).mean()
        label = f"τ={th} (~{avg_ratio*100:.0f}% tokens)"
        plt.plot(xs_sorted, ys, 'o-', color=colors[i], linewidth=2, markersize=6, label=label)

    plt.xlabel('Budget Tokens (h×w)', fontsize=12)
    plt.ylabel('CLIP Semantic Similarity', fontsize=12)
    plt.title('Token Merging vs Semantic Similarity (Unified X-Axis)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left', borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosine_threshold_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()



    # 实验3: 不同SNR下的准确率
    print("实验3: 不同SNR下的准确率...")
    snr_values = [0, 5, 10, 15, 20, 25, 30]
    similarities_snr = []
    H0, W0 = int(latent.shape[2]), int(latent.shape[3])

    for snr in snr_values:
        # 添加噪声
        noisy_latent = add_awgn(latent, snr)
        
        # 重建
        with torch.no_grad():
            recon_img = vae.decode(noisy_latent / scale).sample
            recon_img = torch.clamp((recon_img + 1.0) / 2.0, 0.0, 1.0)
        
        # 计算相似度 - 关键修复：先对齐参考图带宽再评估
        
        #ref_matched = pipeline.bandwidth_match(img_tensor, compressed["h"], compressed["w"])
        ref_matched = pipeline.bandwidth_match(img_tensor, H0, W0)

        similarity = pipeline.similarity(ref_matched, recon_img)
        similarities_snr.append(similarity)
    
    # 实验4: 端到端延迟测量
    print("实验4: 端到端延迟测量...")
    latencies_e2e = []
    encs, txs, decs = [], [], []
    
    # 网络参数配置
    LINK_MBPS = 10.0       # 网络速率 10 Mbps
    BITS_PER_TOKEN = 32    # 每个token的平均比特数（4通道×8bit）
    FEC = 1.0              # FEC冗余系数
    PAYLOAD = 1200         # UDP负载字节数
    HEADER = 28            # UDP/IP头开销字节数
    PROP_MS = 0.0          # 传播延迟（近端可忽略）
    
    for ratio in ratios:
        # 压缩一次，得到token数量K
        comp = pipeline.compress_latent_downsample(img_tensor, ratio)
        K = comp['tokens'].shape[1]
        
        # 本地编解码计时（带GPU同步）
        _, t_enc = measure_latency_gpu(pipeline.compress_latent_downsample, img_tensor, ratio)
        _, t_dec = measure_latency_gpu(pipeline.decompress, comp)
        
        # 网络传输延迟（按K线性增长）
        t_tx = tx_latency_bits(K, r_bits=BITS_PER_TOKEN, R_net_mbps=LINK_MBPS,
                               payload_bytes=PAYLOAD, header_bytes=HEADER,
                               fec=FEC, prop_ms=PROP_MS)
        
        # 端到端延迟（非流式）
        t_e2e = t_enc + t_tx + t_dec
        
        encs.append(t_enc)
        txs.append(t_tx)
        decs.append(t_dec)
        latencies_e2e.append(t_e2e)
    
    # 图1: Token数量 vs 图片准确率
    plt.figure(figsize=(10, 6))
    plt.plot(token_counts_ratio, similarities_ratio, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Tokens', fontsize=12)
    plt.ylabel('CLIP Semantic Similarity', fontsize=12)
    plt.title('Number of Tokens vs Image Reconstruction Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    


    
    # 图3: SNR vs 准确率
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, similarities_snr, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12)
    plt.ylabel('CLIP Semantic Similarity', fontsize=12)
    plt.title('Impact of Channel Noise on Image Reconstruction Accuracy', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'snr_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图4: Token数量 vs 端到端延迟（堆叠图）
    plt.figure(figsize=(12, 8))
    x = np.arange(len(token_counts_ratio))
    
    # 堆叠条形图显示各个组成部分
    plt.bar(x, encs, label='Encode', alpha=0.8, color='skyblue')
    plt.bar(x, txs, bottom=encs, label='Transmit', alpha=0.8, color='lightcoral')
    plt.bar(x, decs, bottom=np.array(encs)+np.array(txs), label='Decode', alpha=0.8, color='lightgreen')
    
    # 添加总延迟线
    plt.plot(x, latencies_e2e, 'ro-', linewidth=3, markersize=8, label='Total E2E Latency')
    
    plt.xticks(x, [str(k) for k in token_counts_ratio])
    plt.xlabel('Number of Tokens (K)', fontsize=12)
    plt.ylabel('End-to-End Latency [seconds]', fontsize=12)
    plt.title(f'E2E Latency vs Token Count @ {LINK_MBPS} Mbps', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'token_vs_e2e_latency.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 图5: 综合对比图
    plt.figure(figsize=(15, 10))
    
    # 子图1: 压缩比例效果
    plt.subplot(2, 3, 1)
    plt.plot(ratios, similarities_ratio, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Compression Ratio')
    plt.ylabel('CLIP Similarity')
    plt.title('Compression Ratio vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 子图2: 余弦阈值效果
    plt.subplot(2, 3, 2)
    for i, threshold in enumerate(cosine_thresholds):
        plt.plot(ratios, similarities_threshold[threshold], 'o-', 
                color=colors[i], linewidth=1.5, markersize=4, 
                label=f'Threshold={threshold}')
    plt.xlabel('Compression Ratio')
    plt.ylabel('CLIP Similarity')
    plt.title('Cosine Threshold Comparison')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 子图3: SNR效果
    plt.subplot(2, 3, 3)
    plt.plot(snr_values, similarities_snr, 'ro-', linewidth=2, markersize=6)
    plt.xlabel('SNR [dB]')
    plt.ylabel('CLIP Similarity')
    plt.title('Channel Noise Impact')
    plt.grid(True, alpha=0.3)
    
    # 子图4: 端到端延迟对比
    plt.subplot(2, 3, 4)
    plt.plot(ratios, latencies_e2e, 'go-', linewidth=2, markersize=6)
    plt.xlabel('Compression Ratio')
    plt.ylabel('E2E Latency [seconds]')
    plt.title('Compression Ratio vs E2E Latency')
    plt.grid(True, alpha=0.3)
    
    # 子图5: Token数量分布
    plt.subplot(2, 3, 5)
    plt.bar(range(len(ratios)), token_counts_ratio, color='skyblue', alpha=0.7)
    plt.xlabel('Compression Ratio Index')
    plt.ylabel('Number of Tokens')
    plt.title('Token Count Distribution')
    plt.grid(True, alpha=0.3)
    
    # 子图6: 效率对比
    plt.subplot(2, 3, 6)
    efficiency = [sim / (count / 1000) for sim, count in zip(similarities_ratio, token_counts_ratio)]
    plt.plot(ratios, efficiency, 'mo-', linewidth=2, markersize=6)
    plt.xlabel('Compression Ratio')
    plt.ylabel('Efficiency (Similarity/Token Count)')
    plt.title('Compression Efficiency Comparison')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"综合分析图表已保存到 {output_dir}")
    
    # 保存数据
    analysis_data = {
        'ratios': ratios,
        'similarities_ratio': similarities_ratio,
        'token_counts_ratio': token_counts_ratio,
        'cosine_thresholds': cosine_thresholds,
        'similarities_threshold': similarities_threshold,
        'budget_tokens_per_ratio': budget_tokens_per_ratio,  # 新增：统一横轴
        'snr_values': snr_values,
        'similarities_snr': similarities_snr,
        'latencies_e2e': latencies_e2e,
        'unique_token_counts': unique_token_counts,
    }
    
    with open(os.path.join(output_dir, 'comprehensive_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

# ========= 一键运行入口（无需任何命令行参数） =========
if __name__ == "__main__":
    IMAGE = "/path/to/your_image.jpg"   # 设为空字符串 "" 会自动下载测试图
    MODE = "latent"                     # "latent" / "token" / "enhanced" / "adaptive"
    RATIOS = [1.0, 0.5, 0.25, 0.125]
    VAE = "stabilityai/sd-vae-ft-ema"
    OUTDIR = "outputs_vq"

    VIZ_TOKEN_PANEL = True
    PANEL_RATIO = 0.25
    PANEL_K = 60

    UNSHARP_RADIUS = 1.0
    UNSHARP_AMOUNT = 1.0
    UNSHARP_THRESHOLD = 0

    import argparse as _argparse
    args = _argparse.Namespace(
        image=IMAGE,
        ratios=RATIOS,
        mode=MODE,
        no_clip=False,
        vae=VAE,
        outdir=OUTDIR,
        unsharp_radius=UNSHARP_RADIUS,
        unsharp_amount=UNSHARP_AMOUNT,
        unsharp_threshold=UNSHARP_THRESHOLD,
        viz_token_panel=VIZ_TOKEN_PANEL,
        panel_ratio=PANEL_RATIO,
        panel_K=PANEL_K,
    )
    main(args)  # 主流程（会保存网格图、曲线以及可选面板图）
    test_compression_ratios(IMAGE, VAE)  # 压缩比例测试
    os.makedirs("comprehensive_output", exist_ok=True)
    print("🚀 启动综合分析实验 ...")
    pipeline = VQImageSemanticPipeline(use_clip=True, vae_repo=VAE, half_if_cuda=False)
    input_path = IMAGE if IMAGE else "https://picsum.photos/512"
    generate_comprehensive_analysis(pipeline, input_path, "comprehensive_output")
    print("✅ 全流程完成")
