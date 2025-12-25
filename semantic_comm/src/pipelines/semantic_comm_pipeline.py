"""
semantic_comm_pipeline.py
Author: zhixingshia

功能：
融合 SeTok + CoDi 实现多模态语义通信：
1. 输入文本 → 生成图像/音频/视频 (CoDi)
2. 对生成图像进行 SeTok 语义分词压缩
3. 模拟信道传输 (丢包 / 加噪)
4. 使用 DeTokenizer 重建图像
5. 输出原图、重建图与对比指标
"""
import torch
import os
import sys

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ✅ 关键：强制半精度、释放缓存
torch.set_float32_matmul_precision("medium")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32,expandable_segments:True"

import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
import soundfile as sf
from torchvision import transforms
from torchmetrics.functional import structural_similarity_index_measure as ssim

# =====================================================
# 1️⃣ 模型加载
# =====================================================

print("🚀 Loading models ...")

# ---- CoDi ----
from semantic_comm.core.models.model_module_infer import model_module

# execution toggles
LIGHT_RUN = False
ENABLE_CODI = False            # set True to enable CoDi
ENABLE_SETOK = True            # set True to enable SeTok tokenizer
ENABLE_DETOKENIZER = False     # set True to enable SeTok detokenizer + eval
DEVICE = torch.device('cpu') if LIGHT_RUN else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

pth_list = [
    'CoDi_encoders.pth',
    'CoDi_text_diffuser.pth'
] if LIGHT_RUN else [
    'CoDi_encoders.pth',
    'CoDi_text_diffuser.pth',
    'CoDi_video_diffuser_8frames.pth',
    'CoDi_audio_diffuser_m.pth'
]
if ENABLE_CODI:
    codi = model_module(
        data_dir='checkpoints/',
        pth=pth_list,
        fp16=True
    ).eval()
else:
    codi = None
    print("[SKIP] CoDi disabled, skip initialization")

# ---- SeTok ----
if ENABLE_SETOK:
    from semantic_comm.src.models.tokenizer import SetokTokenizer
    # lighter tokenizer config to reduce RAM
    tokenizer = SetokTokenizer(
        vision_tower='openai/clip-vit-base-patch16',
        delay_load=True,
        hidden_dim=1024,
        token_feat_dim=1024,
        min_cluster_num=32,
        threshold=0.6,
        nheads=4,
        dim_feedforward=2048,
        inner_cluster_layers=1,
        intra_cluster_layers=1,
        proj_drop=0.1,
        drop_path=0.0,
    ).to(DEVICE).eval()
    if ENABLE_DETOKENIZER:
        from semantic_comm.src.models.detokenizer import SetokDeTokenizer
        detokenizer = SetokDeTokenizer(
            token_feat_dim=1024,
            hidden_dim=1024,
            decoder_embed_dim=512,
            decoder_nheads=8,
            decoder_depth=4,
            num_hidden_layers=2,
            cross_attention_freq=2,
        ).to(DEVICE).eval()
    else:
        detokenizer = None
else:
    tokenizer = None
    detokenizer = None

print("✅ Models loaded successfully")

# =====================================================
# 2️⃣ 输入文本与输出目录
# =====================================================
prompt = "夜晚繁星下的城市"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_dir = Path(f"outputs/{timestamp}")
out_dir.mkdir(parents=True, exist_ok=True)

# =====================================================
# 3️⃣ 使用 CoDi 生成图像
# =====================================================
print("\n🖼️ Preparing image...")
# light mode to avoid OOM/killed on constrained GPUs
gen_image_size = 192 if LIGHT_RUN else 512
gen_steps = 10 if LIGHT_RUN else 50
gen_scale = 4.0 if LIGHT_RUN else 7.5
if codi is not None:
    with torch.inference_mode():
        image_outputs = codi.inference(
            ['image'],
            condition=[prompt],
            condition_types=['text'],
            n_samples=1,
            image_size=gen_image_size,
            ddim_steps=gen_steps,
            scale=gen_scale
        )
    image = image_outputs[0][0]
else:
    # fallback: create a placeholder image to keep pipeline running
    image = Image.new('RGB', (gen_image_size, gen_image_size), color=(30, 30, 30))
img_path = out_dir / "original_image.png"
image.save(img_path)
print(f"✅ Image saved to {img_path}")

if not ENABLE_SETOK:
    print("[SKIP] SeTok disabled.")
    print("\n🎉 Semantic communication process completed!")
    print(f"📁 All outputs saved in: {out_dir.resolve()}")
    sys.exit(0)

# =====================================================
# 4️⃣ 图像转Tensor并送入 SeTok 语义压缩
# =====================================================
print("\n🔍 Performing semantic tokenization (SeTok)...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
img_tensor = transform(image).unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

with torch.inference_mode():
    tokens, idx_cluster, score = tokenizer(img_tensor)
print(f"Tokens extracted: {tokens.shape}")

# =====================================================
# 5️⃣ 信道模拟（丢包 + 噪声）
# =====================================================
def add_awgn(x, snr_db):
    signal_power = torch.mean(x ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise

def drop_tokens(x, drop_rate=0.2):
    mask = (torch.rand(x.shape[0], device=x.device) > drop_rate).float().unsqueeze(1)
    return x * mask

if ENABLE_DETOKENIZER:
    print("\n📡 Simulating communication channel...")
    with torch.inference_mode():
        tokens_noisy = add_awgn(tokens, snr_db=15)
        tokens_dropped = drop_tokens(tokens_noisy, drop_rate=0.2)

# =====================================================
# 6️⃣ 解码重建
# =====================================================
if ENABLE_DETOKENIZER:
    print("\n🔄 Reconstructing image from semantic tokens...")
    with torch.inference_mode():
        recon = detokenizer(tokens_dropped, attention_masks=None)
    # recon: (B, H*W, C) -> 需 reshape 为图像
    recon_img = recon[0].detach().cpu()
    recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
    recon_img = recon_img.view(128, 128, 3).numpy()
    recon_img = Image.fromarray((recon_img * 255).astype(np.uint8))
    recon_path = out_dir / "reconstructed_image.png"
    recon_img.save(recon_path)
    print(f"✅ Reconstructed image saved to {recon_path}")

# =====================================================
# 7️⃣ 评估指标
# =====================================================
if ENABLE_DETOKENIZER:
    print("\n📊 Evaluating reconstruction quality...")
    orig = transform(Image.open(img_path)).unsqueeze(0)
    recon_eval = transform(Image.open(recon_path)).unsqueeze(0)
    mse = torch.mean((orig - recon_eval) ** 2).item()
    ssim_val = ssim(orig, recon_eval).item()
    print(f"MSE: {mse:.6f}")
    print(f"SSIM: {ssim_val:.4f}")

# =====================================================
# 8️⃣ 音频与视频（CoDi 直接生成）
# =====================================================
# Skip heavy audio/video generation by default to reduce memory/time
if not LIGHT_RUN:
    print("\n🎵 Generating audio ...")
    with torch.inference_mode():
        audio_wave = codi.inference(
            xtype=['audio'],
            condition=[prompt],
            condition_types=['text'],
            scale=gen_scale,
            n_samples=1,
            ddim_steps=gen_steps
        )[0]
    audio_data = np.array(audio_wave.squeeze(), dtype=np.float32)
    sf.write(out_dir / "city_night_audio.wav", audio_data, 16000)

    print("\n🎬 Generating video ...")
    with torch.inference_mode():
        video_outputs = codi.inference(
            ['video'],
            condition=[prompt],
            condition_types=['text'],
            n_samples=1,
            image_size=gen_image_size,
            ddim_steps=gen_steps,
            num_frames=8,
            scale=gen_scale
        )
    video = video_outputs[0][0]
    gif_path = out_dir / "city_night_video.gif"
    video[0].save(
        gif_path,
        format="GIF",
        append_images=video[1:],
        save_all=True,
        duration=2000 / len(video),
        loop=0
    )

# =====================================================
# 9️⃣ 输出结果
# =====================================================
print("\n🎉 Semantic communication process completed!")
print(f"📁 All outputs saved in: {out_dir.resolve()}")
