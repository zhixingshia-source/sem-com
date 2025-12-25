#!/usr/bin/env python3
"""
简化版图到图语义通信管线
专注于核心功能：图像压缩、语义相似度评估、性能分析
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
# 在 SimpleImageSemanticPipeline.__init__ 或模型初始化结束后添加
import torch.nn as nn

class MiniDecoder(nn.Module):
    """轻量级上采样重建模块"""
    def __init__(self, token_dim=768):
        super().__init__()
        self.fc = nn.Linear(token_dim, 512)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, tokens):
        b, n, d = tokens.shape
        h = int(np.ceil(np.sqrt(n)))
        w = h
        pad = h * w - n
        if pad > 0:
            pad_tok = tokens.new_zeros(b, pad, d)
            tokens = torch.cat([tokens, pad_tok], dim=1)
        x = self.fc(tokens).transpose(1, 2).reshape(b, 512, h, w)
        return self.up(x)


# 初始化


# 设置 PyTorch CUDA 内存管理优化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 添加项目根目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class SimpleImageSemanticPipeline:
    """简化版图到图语义通信管线"""
    
    def __init__(self, use_setok=True, use_clip=True, memory_fraction=0.8):
        """初始化管线
        
        Args:
            use_setok: 是否使用Setok tokenizer
            use_clip: 是否使用CLIP模型
            memory_fraction: GPU内存使用比例 (0.1-0.9)
        """
        import random
        
        # —— 关随机，保确定性 ——
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(0); np.random.seed(0); random.seed(0)
        
        self.device = DEVICE
        self.use_setok = use_setok
        self.use_clip = use_clip
        self.memory_fraction = memory_fraction
        self.clip_model = None
        self.tokenizer = None
        self.detokenizer = None
        self.simple_decoder = MiniDecoder().to(self.device)

        # GPU内存配置
        if torch.cuda.is_available():
            # 清理缓存
            torch.cuda.empty_cache()
            # 设置GPU内存使用比例
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
            print(f"🔧 GPU memory fraction set to: {memory_fraction*100:.0f}%")
        
        # 初始化模型
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化模型"""
        import time
        import threading
        import psutil
        import torch
        
        print("🔄 Initializing models...")
        start_time = time.time()
        
        # 显示系统资源信息
        print(f"💻 System resources:")
        print(f"   - CPU: {psutil.cpu_count()} cores")
        print(f"   - RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB total, {psutil.virtual_memory().available / (1024**3):.1f} GB available")
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"   - GPU: {gpu_memory:.1f} GB total, {gpu_allocated:.1f} GB allocated, {gpu_cached:.1f} GB cached")
        else:
            print(f"   - GPU: Not available")
        
        # 初始化CLIP模型用于语义相似度计算 - 使用本地模型，无需下载
        if self.use_clip:
            print("📥 Loading CLIP model...")
            clip_start = time.time()
        else:
            print("⏭️ Skipping CLIP model (disabled)")
            self.clip_model = None
        
        def load_clip():
            try:
                from src.models.clip_encoder import CLIPVisionTower
                self.clip_model = CLIPVisionTower(
                    model_name="openai/clip-vit-base-patch32",  # 使用较小的模型
                    device=self.device,
                    unfreeze=False,
                    use_fp16=True
                ).to(self.device).eval()
                
                # 显示内存使用情况
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"   📊 GPU memory after CLIP: {gpu_allocated:.2f} GB")
                
                return True, None
            except Exception as e:
                return False, str(e)
        
        if self.use_clip:
            # 使用线程加载CLIP，设置超时
            clip_result = [None, None]
            clip_thread = threading.Thread(target=lambda: clip_result.__setitem__(0, load_clip()))
            clip_thread.start()
            clip_thread.join(timeout=60)  # 60秒超时
            
            if clip_thread.is_alive():
                print("⏰ CLIP model loading timeout (60s) - this might indicate a network issue or model download")
                print("💡 Try running with use_clip=False to skip CLIP model")
                self.clip_model = None
            elif clip_result[0] is not None:
                success, error = clip_result[0]
                if success:
                    clip_time = time.time() - clip_start
                    print(f"✅ CLIP model loaded successfully in {clip_time:.2f}s")
                else:
                    print(f"❌ Failed to load CLIP model: {error}")
                    self.clip_model = None
            else:
                print("❌ CLIP model loading failed - no result returned")
                self.clip_model = None
        
        # 初始化Setok tokenizer
        if self.use_setok:
            print("📥 Loading Setok tokenizer...")
            setok_start = time.time()
        else:
            print("⏭️ Skipping Setok tokenizer (disabled)")
            self.tokenizer = None
        
        def load_setok():
            try:
                # 基本的内存清理
                if torch.cuda.is_available():
                    print("🧹 Performing basic memory cleanup before Setok loading...")
                    torch.cuda.empty_cache()
                    
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_free = gpu_total - gpu_allocated
                    print(f"   📊 GPU memory before Setok: {gpu_allocated:.2f} GB used, {gpu_free:.2f} GB free")
                    
                    if gpu_free < 0.5:  # 只需要500MB空闲内存
                        return False, f"Insufficient GPU memory: {gpu_free:.2f} GB free, need at least 0.5 GB"
                
                from src.models.tokenizer import SetokTokenizer
                
                # 使用极简配置来减少内存使用
                print("   🔧 Using ultra-minimal configuration for memory efficiency...")
                config = {
                    'vision_tower': 'openai/clip-vit-base-patch32',  # 使用最小的CLIP模型
                    'delay_load': True,
                    'hidden_dim': 768,        # 匹配CLIP输出维度
                    'token_feat_dim': 768,    # 匹配CLIP输出维度
                    'min_cluster_num': 4,     # 减少聚类数量
                    'threshold': 0.6,
                    'nheads': 2,              # 减少注意力头数
                    'dim_feedforward': 1024,  # 适当的前馈网络维度
                    'inner_cluster_layers': 1,
                    'intra_cluster_layers': 1,
                    'proj_drop': 0.1,
                    'drop_path': 0.0,
                }
                
                # 分步加载，避免一次性占用过多内存
                print("   📦 Loading Setok tokenizer components...")
                
                # 创建模型
                self.tokenizer = SetokTokenizer(**config)
                print("   🔄 Initializing on GPU...")
                
                # 将模型移动到GPU
                self.tokenizer = self.tokenizer.to(self.device)
                self.tokenizer.eval()  # ✅ 关键：关掉dropout等
                
                # 在GPU上执行初始化forward，确保模型和输入在同一设备上
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 224, 224).to(self.device)
                    _ = self.tokenizer(test_input)
                
                # 清理测试输入
                del test_input
                print("   ✅ GPU initialization completed")
                
                # 显示加载后的内存使用
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"   📊 GPU memory after Setok: {gpu_allocated:.2f} GB")
                
                return True, None
            except Exception as e:
                return False, str(e)
        
        if self.use_setok:
            # 使用线程加载Setok，设置超时
            setok_result = [None, None]
            setok_thread = threading.Thread(target=lambda: setok_result.__setitem__(0, load_setok()))
            setok_thread.start()
            setok_thread.join(timeout=120)  # 120秒超时
            
            if setok_thread.is_alive():
                print("⏰ Setok tokenizer loading timeout (120s)")
                print("💡 Try running with --no-use_setok to skip Setok tokenizer")
                self.tokenizer = None
            elif setok_result[0] is not None:
                success, error = setok_result[0]
                if success:
                    setok_time = time.time() - setok_start
                    print(f"✅ Setok tokenizer loaded successfully in {setok_time:.2f}s")
                else:
                    print(f"❌ Failed to load Setok tokenizer: {error}")
                    self.tokenizer = None
            else:
                print("❌ Setok tokenizer loading failed - no result returned")
                self.tokenizer = None
        
        # 初始化解token化器
        if self.use_setok:
            print("📥 Loading Setok detokenizer...")
            detokenizer_start = time.time()
        else:
            print("⏭️ Skipping Setok detokenizer (disabled)")
            self.detokenizer = None
        
        def load_detokenizer():
            try:
                # 检查依赖
                try:
                    import diffusers
                    print(f"   ✓ diffusers version: {diffusers.__version__}")
                except ImportError as e:
                    return False, f"Missing dependency: {e}. Please install with: pip install diffusers"
                
                # 检查内存
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    gpu_free = gpu_total - gpu_allocated
                    print(f"   📊 GPU memory before detokenizer: {gpu_allocated:.2f} GB used, {gpu_free:.2f} GB free")
                    
                    if gpu_free < 0.5:  # 只需要500MB空闲内存
                        return False, f"Insufficient GPU memory: {gpu_free:.2f} GB free, need at least 0.5 GB"
                
                from src.models.detokenizer import SetokDeTokenizer
                
                # 使用与tokenizer匹配的极简配置
                print("   🔧 Using memory-optimized detokenizer configuration...")
                config = {
                    'token_feat_dim': 768,      # 与tokenizer匹配
                    'hidden_dim': 768,          # 与tokenizer匹配
                    'decoder_embed_dim': 512,   # 适中的解码器维度
                    'decoder_nheads': 2,        # 减小
                    'decoder_depth': 2,         # 减小
                    'num_hidden_layers': 1,     # 减小
                    'cross_attention_freq': 1,
                    'patch_size': 14,
                    'image_size': 224,          # 使用较小的图像尺寸
                }
                self.detokenizer = SetokDeTokenizer(**config).to(self.device).eval()
                
                # 显示加载后的内存使用
                if torch.cuda.is_available():
                    gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    print(f"   📊 GPU memory after detokenizer: {gpu_allocated:.2f} GB")
                
                return True, None
            except Exception as e:
                return False, str(e)
        
        if self.use_setok:
            # 使用线程加载detokenizer，设置超时
            detokenizer_result = [None, None]
            detokenizer_thread = threading.Thread(target=lambda: detokenizer_result.__setitem__(0, load_detokenizer()))
            detokenizer_thread.start()
            detokenizer_thread.join(timeout=60)  # 60秒超时
            
            if detokenizer_thread.is_alive():
                print("❌ Setok detokenizer loading timeout (60s)")
                self.detokenizer = None
            elif detokenizer_result[0] is not None:
                success, error = detokenizer_result[0]
                detokenizer_time = time.time() - detokenizer_start
                if success:
                    print(f"✓ Setok detokenizer loaded successfully in {detokenizer_time:.2f}s")
                else:
                    print(f"✗ Failed to load Setok detokenizer: {error}")
                    self.detokenizer = None
            else:
                print("❌ Setok detokenizer loading failed - no result returned")
                self.detokenizer = None
        
        total_time = time.time() - start_time
        print(f"🏁 Model initialization completed in {total_time:.2f}s")
        
        # 显示最终状态
        print(f"📊 Final status:")
        print(f"   - CLIP model: {'✅ Ready' if self.clip_model else '❌ Not available'}")
        print(f"   - Setok tokenizer: {'✅ Ready' if self.tokenizer else '❌ Not available'}")
        print(f"   - Setok detokenizer: {'✅ Ready' if self.detokenizer else '❌ Not available'}")
        
        # 显示最终内存使用情况
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            gpu_cached = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"📊 Final GPU memory: {gpu_allocated:.2f} GB allocated, {gpu_cached:.2f} GB cached")
            
            # 如果模型加载失败，清理内存
            if self.tokenizer is None or self.detokenizer is None:
                print("🧹 Cleaning up GPU memory due to failed model loading...")
                torch.cuda.empty_cache()
                gpu_allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_cached_after = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"📊 GPU memory after cleanup: {gpu_allocated_after:.2f} GB allocated, {gpu_cached_after:.2f} GB cached")
        
        if self.clip_model is None and self.tokenizer is None:
            print("⚠️  Warning: No models loaded successfully. Pipeline will use fallback methods.")
        
        # 测试模型是否正常工作
        self._test_models()
    
    def _test_models(self):
        """测试模型是否正常工作"""
        print("🧪 Testing models...")
        print(f"🔍 Detokenizer type: {type(self.detokenizer)}")
        print(f"🔍 Detokenizer is None: {self.detokenizer is None}")
        
        # 测试CLIP模型
        if self.clip_model:
            try:
                # 创建测试图像
                test_image = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    features = self.clip_model(test_image)
                print(f"   ✓ CLIP model test passed - output shape: {features.shape}")
            except Exception as e:
                print(f"   ✗ CLIP model test failed: {e}")
                self.clip_model = None
        
        # 测试Setok tokenizer
        if self.tokenizer:
            try:
                # 创建测试图像
                test_image = torch.randn(1, 3, 224, 224).to(self.device)
                with torch.no_grad():
                    tokens, idx_cluster, score = self.tokenizer(test_image)
                print(f"   ✓ Setok tokenizer test passed - output shape: {tokens.shape}")
            except Exception as e:
                print(f"   ✗ Setok tokenizer test failed: {e}")
                self.tokenizer = None
        
        # 测试Setok detokenizer
        if self.detokenizer:
            try:
                # 创建测试tokens和attention_masks，使用与detokenizer匹配的维度
                # detokenizer期望 16x16=256 个tokens (height=16, width=16)
                test_tokens = torch.randn(1, 256, 768).to(self.device)  # 使用256个tokens匹配16x16
                test_attention_masks = torch.ones(1, 256).to(self.device)  # 与token数量匹配
                with torch.no_grad():
                    # 使用正确的参数调用detokenizer
                    image = self.detokenizer(test_tokens, test_attention_masks)
                print(f"   ✓ Setok detokenizer test passed - output shape: {image.shape}")
            except Exception as e:
                print(f"   ✗ Setok detokenizer test failed: {e}")
                # 尝试不同的参数组合
                try:
                    print("   🔄 Trying alternative detokenizer test...")
                    test_tokens = torch.randn(1, 256, 768).to(self.device)
                    # 尝试不传递attention_masks
                    with torch.no_grad():
                        image = self.detokenizer(test_tokens, None)
                    print(f"   ✓ Alternative detokenizer test passed - output shape: {image.shape}")
                except Exception as e2:
                    print(f"   ✗ Alternative detokenizer test also failed: {e2}")
                    # 最后尝试：检查detokenizer的forward方法签名
                    try:
                        print("   🔄 Checking detokenizer forward method...")
                        import inspect
                        sig = inspect.signature(self.detokenizer.forward)
                        print(f"   📋 Detokenizer forward signature: {sig}")
                        # 尝试使用默认参数
                        test_tokens = torch.randn(1, 256, 768).to(self.device)
                        test_attention_masks = torch.ones(1, 256).to(self.device)
                        with torch.no_grad():
                            image = self.detokenizer(test_tokens, test_attention_masks)
                        print(f"   ✓ Direct forward call test passed - output shape: {image.shape}")
                    except Exception as e3:
                        print(f"   ✗ All detokenizer tests failed: {e3}")
                        self.detokenizer = None
        
        print("🧪 Model testing completed")
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """加载和预处理图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(target_size)
            
            # 转换为tensor
            image_array = np.array(image) / 255.0
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float()
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    # def create_test_image(self, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    #     """创建测试图像"""
    #     # 创建一个简单的测试图像
    #     img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
    #     img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    #     img_tensor = img_tensor.unsqueeze(0).to(self.device)
    #     return img_tensor
    
    def create_test_image(self, size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """从网络下载测试图像（若失败则随机生成）"""
        import requests
        from io import BytesIO
        
        url = "https://picsum.photos/256"  # 随机风景图片，每次都会变
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            image = image.resize(size)
            img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            print(f"✅ Downloaded test image from {url}")
            return img_tensor
        except Exception as e:
            print(f"⚠️ Failed to download test image: {e}")
            return super().create_test_image(size)



    def compress_image(self, image_tensor: torch.Tensor, compression_ratio: float = 0.5) -> Dict:
        """改进版图像压缩函数 — 支持基于score筛选的重要token"""
        if self.tokenizer is None:
            # 没有Setok就用简单下采样
            return self._simple_compress(image_tensor, compression_ratio)
        
        try:
            with torch.no_grad():
                tokens, idx_cluster, score = self.tokenizer(image_tensor)

                # 确保 tokens 维度正确
                if tokens.dim() == 2:
                    tokens = tokens.unsqueeze(0)  # [N, D] → [1, N, D]

                num_tokens = tokens.shape[1]
                target_tokens = max(1, int(num_tokens * compression_ratio))

                # ✅ 按score筛选最重要的token（若score可用）
                if score is not None and score.numel() == num_tokens:
                    topk_idx = torch.topk(score.squeeze(), target_tokens, largest=True).indices
                    selected_tokens = tokens[:, topk_idx, :]
                else:
                    selected_tokens = tokens[:, :target_tokens, :]

                # 生成 attention mask
                selected_mask = torch.ones(1, target_tokens, dtype=torch.bool, device=self.device)

                return {
                    'tokens': selected_tokens,
                    'attention_mask': selected_mask,
                    'num_tokens': target_tokens,
                    'compression_ratio': compression_ratio,
                    'idx_cluster': idx_cluster,
                    'score': score
                }

        except Exception as e:
            print(f"⚠️ Tokenizer compression failed: {e}")
            return self._simple_compress(image_tensor, compression_ratio)

    
    def _simple_compress(self, image_tensor: torch.Tensor, compression_ratio: float) -> Dict:
        """简单的图像压缩（下采样）"""
        # 计算目标尺寸
        _, _, h, w = image_tensor.shape
        target_h = max(1, int(h * np.sqrt(compression_ratio)))
        target_w = max(1, int(w * np.sqrt(compression_ratio)))
        
        # 下采样
        compressed = F.interpolate(image_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return {
            'compressed_image': compressed,
            'num_tokens': target_h * target_w,
            'compression_ratio': compression_ratio
        }
    
    def _align_tokens_for_detok(
        self,
        tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_n: int = 256,
        score: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 [B, N, D] 的 tokens 对齐到 target_n（默认 256）
        - N < target_n: 零填充，mask=1...0
        - N > target_n: 按 score 选 top-k；没 score 就均匀采样
        返回: (tokens_aligned [B, target_n, D], mask [B, target_n])
        """
        b, n, d = tokens.shape
        device = tokens.device

        if attention_mask is None:
            attention_mask = torch.ones(b, n, device=device)

        if n == target_n:
            return tokens, attention_mask

        if n < target_n:
            # ✅ 重复采样填满，而不是零填充；mask 直接全 1
            repeats = int(np.ceil(target_n / n))
            idx = torch.arange(n, device=device).repeat(repeats)[:target_n]
            tokens_aligned = tokens[:, idx, :]            # [B, target_n, D]
            mask_aligned = torch.ones(b, target_n, device=device, dtype=attention_mask.dtype)
            return tokens_aligned, mask_aligned

        # n > target_n
        if score is not None and score.numel() == n:
            idx = torch.topk(score.squeeze(), target_n, largest=True).indices
        else:
            idx = torch.linspace(0, n - 1, steps=target_n, device=device).round().long()
        tokens_sel = tokens[:, idx, :]
        mask_sel = torch.ones(b, target_n, device=device, dtype=attention_mask.dtype)
        return tokens_sel, mask_sel


    def decompress_image(self, compressed_data: Dict, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """保持 detokenizer，不做 256 对齐；直接用变长 tokens + bool mask。"""
        if 'tokens' not in compressed_data:
            return self._simple_decompress(compressed_data, target_size)

        tokens = compressed_data['tokens']                # [B, N, D]
        attn = compressed_data.get('attention_mask', None)

        # 统一到 bool mask，且跟 tokens 同设备
        if attn is None:
            attn = torch.ones(tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=self.device)
        else:
            attn = attn.to(dtype=torch.bool, device=self.device)

        try:
            with torch.no_grad():
                if self.detokenizer is not None:
                    reconstructed = self.detokenizer(tokens, attn)   # 不做任何重复/填充
                else:
                    reconstructed = self.simple_decoder(tokens)

                if reconstructed.shape[2:] != target_size:
                    reconstructed = F.interpolate(reconstructed, size=target_size, mode='bilinear', align_corners=False)
                return reconstructed

        except Exception as e:
            print(f"⚠️ Detokenizer failed ({e}), using MiniDecoder fallback.")
            with torch.no_grad():
                reconstructed = self.simple_decoder(tokens)
                reconstructed = F.interpolate(reconstructed, size=target_size, mode='bilinear', align_corners=False)
                return reconstructed


    
    def _simple_decompress(self, compressed_data: Dict, target_size: Tuple[int, int]) -> torch.Tensor:
        """简单的图像解压缩（上采样）"""
        if 'compressed_image' in compressed_data:
            compressed = compressed_data['compressed_image']
            # 上采样到目标尺寸
            reconstructed = F.interpolate(compressed, size=target_size, mode='bilinear', align_corners=False)
            return reconstructed
        else:
            # 如果没有压缩图像，返回随机图像
            return self.create_test_image(target_size)
    
    def calculate_semantic_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """最终稳定版：固定 224 输入 + 只取 CLIP 全局向量，避免 token 维度不一致"""
        if self.clip_model is None:
            return self._pixel_similarity(img1, img2)

        try:
            with torch.no_grad():
                size = 224  # 关掉多尺度，彻底规避 37 vs 50
                i1 = F.interpolate(img1, size=(size, size), mode='bilinear', align_corners=False)
                i2 = F.interpolate(img2, size=(size, size), mode='bilinear', align_corners=False)

                def get_global_feat(x: torch.Tensor) -> torch.Tensor:
                    out = self.clip_model(x)

                    # 1) 解包各种返回类型
                    if isinstance(out, dict):
                        # 优先取常见全局键；没有就选第一个张量值
                        for k in ("pooled", "image_embeds", "global", "cls", "image_features"):
                            if k in out and torch.is_tensor(out[k]):
                                out = out[k]
                                break
                        else:
                            out = next(v for v in out.values() if torch.is_tensor(v))
                    elif isinstance(out, (list, tuple)):
                        out = next(t for t in out if torch.is_tensor(t))

                    # 2) 规范成 [B, D]
                    if out.dim() == 3:
                        # [B, N, D] -> 取 CLS（第 0 个 token），若没有 CLS 语义也 OK
                        out = out[:, 0, :]
                    elif out.dim() == 2:
                        # [B, D] -> 不动；如果是 [B, N]（像 token 分数），再兜底处理
                        if out.size(1) in (37, 49, 50):   # 典型 token 数
                            out = out.mean(dim=1, keepdim=True)  # 压成 [B,1]，再线性抬到 D
                            # 用 CLIP 维度 768 作为目标维度（不训练，仅避免形状不匹配）
                            out = out.expand(out.size(0), 768)
                    else:
                        # 其他形状，直接展平到 [B, D]
                        out = out.view(out.size(0), -1)

                    # 3) 转 float32 + 归一化（fp16 有时会让相似度数值不稳）
                    out = out.float()
                    out = F.normalize(out, dim=1)
                    return out

                f1 = get_global_feat(i1)
                f2 = get_global_feat(i2)
                sim = F.cosine_similarity(f1, f2, dim=1).mean().item()
                return float(sim)
        except Exception as e:
            print(f"⚠️ Error in CLIP similarity (final stable): {e}")
            return self._pixel_similarity(img1, img2)

    
    def _pixel_similarity(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """简单的像素级相似度"""
        # 计算MSE
        mse = F.mse_loss(img1, img2)
        # 转换为相似度 (0-1)
        similarity = 1.0 / (1.0 + mse.item())
        return similarity
    
    def analyze_compression_performance(
        self,
        original_image: torch.Tensor,
        compression_ratios: List[float] = None,
        token_counts: List[int] = None,
    ) -> Dict:
        if compression_ratios is None:
            compression_ratios = [0.1, 0.3, 0.5, 0.7, 1.0]
        if token_counts is None:
            token_counts = [4, 8, 16, 32, 64, 128, 256]

        results = {
            "compression_ratios": [],
            "token_counts": [],
            "similarities_ratio": [],
            "similarities_count": [],
            "original_similarity": 0.0,
        }

        print("Analyzing compression performance...")

        # 1) 先 token 一次，后面都用这批 tokens
        with torch.no_grad():
            base_tokens, base_idx_cluster, base_score = self.tokenizer(original_image)
        if base_tokens.dim() == 2:
            base_tokens = base_tokens.unsqueeze(0)  # [N,D] -> [1,N,D]
        total_tokens = base_tokens.shape[1]
        print(f"✅ Fixed total_tokens = {total_tokens}")

        # 原图自相似度（CLIP 全局）
        results["original_similarity"] = self.calculate_semantic_similarity(original_image, original_image)

        def select_idx(n_target: int):
            n_target = max(1, min(total_tokens, n_target))
            if base_score is not None and base_score.numel() == total_tokens:
                idx = torch.topk(base_score.squeeze(), n_target, largest=True).indices
                idx, _ = torch.sort(idx)   # ✅ 选完保持原始顺序，空间更连贯
                return idx
            else:
                # 更稳的等间距抽样（避免 float round 抖动）
                step = total_tokens / n_target
                pos = torch.floor(torch.arange(n_target, device=base_tokens.device) * step)
                idx = pos.clamp_max(total_tokens - 1).long()
                return idx

        # 2) Ratio 模式
        print("\n--- Compression Ratio Mode ---")
        for ratio in compression_ratios:
            n_target = max(1, int(total_tokens * ratio))
            idx = select_idx(n_target)
            tokens_sel = base_tokens[:, idx, :]
            compressed = {
                "tokens": tokens_sel,
                "attention_mask": torch.ones(1, n_target, device=self.device),
                "num_tokens": int(n_target),
                "compression_ratio": float(ratio),
                "score": base_score,
            }
            recon = self.decompress_image(compressed, original_image.shape[2:])
            sim = self.calculate_semantic_similarity(original_image, recon)
            results["compression_ratios"].append(ratio)
            results["token_counts"].append(int(n_target))
            results["similarities_ratio"].append(sim)
            print(f"  Ratio={ratio:.2f}  Tokens={n_target}  Sim={sim:.4f}")

        # 3) 绝对 Token 数模式
        print("\n--- Token Count Mode ---")
        for n_tokens in token_counts:
            n_target = max(1, min(total_tokens, n_tokens))
            idx = select_idx(n_target)
            tokens_sel = base_tokens[:, idx, :]
            compressed = {
                "tokens": tokens_sel,
                "attention_mask": torch.ones(1, n_target, device=self.device),
                "num_tokens": int(n_target),
                "compression_ratio": float(n_target / total_tokens),
                "score": base_score,
            }
            recon = self.decompress_image(compressed, original_image.shape[2:])
            sim = self.calculate_semantic_similarity(original_image, recon)
            results["similarities_count"].append(sim)
            print(f"  Tokens={n_target}  Sim={sim:.4f}")

        return results

    
    def plot_results(self, results: Dict, save_path: str = None):
        """绘制结果图表"""
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # ========== 图1: 压缩比例 vs 语义相似度 ==========
        if "similarities_ratio" in results:
            ax1.plot(
                results["compression_ratios"],
                results["similarities_ratio"],
                "bo-",
                linewidth=2,
                markersize=8,
                label="By Ratio",
            )
            ax1.set_xlabel("Compression Ratio")
            ax1.set_ylabel("Semantic Similarity")
            ax1.set_title("Compression Ratio vs Semantic Similarity")
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            ax1.legend()

        # ========== 图2: Token 数 vs 语义相似度 ==========
        if "similarities_count" in results:
            token_counts = [4, 8, 16, 32, 64, 128, 256][: len(results["similarities_count"])]
            ax2.plot(
                token_counts,
                results["similarities_count"],
                "ro-",
                linewidth=2,
                markersize=8,
                label="By Token Count",
            )
            ax2.set_xlabel("Number of Tokens")
            ax2.set_ylabel("Semantic Similarity")
            ax2.set_title("Token Count vs Semantic Similarity")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Results plot saved to: {save_path}")

        plt.show()

    
    def save_results(self, results: Dict, filepath: str):
        """保存结果到 JSON 文件（兼容新版结果结构）"""
        save_data = {
            "compression_ratios": [float(x) for x in results.get("compression_ratios", [])],
            "token_counts": [int(x) for x in results.get("token_counts", [])],
            "similarities_ratio": [float(x) for x in results.get("similarities_ratio", [])],
            "similarities_count": [float(x) for x in results.get("similarities_count", [])],
            "original_similarity": float(results.get("original_similarity", 0.0)),
        }

        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {filepath}")



def main():
    """主函数"""
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description='Simple Image-to-Image Semantic Communication Pipeline')
    parser.add_argument('--no-use_setok', action='store_true', default=False,
                        help='Skip Setok tokenizer and detokenizer')
    parser.add_argument('--no-use_clip', action='store_true', default=False,
                        help='Skip CLIP model for semantic similarity')
    parser.add_argument('--memory-fraction', type=float, default=0.8,
                        help='GPU memory fraction to use (0.1-0.9, default: 0.8)')
    args = parser.parse_args()

    print("=== Simple Image-to-Image Semantic Communication Pipeline ===")
    print(f"🔧 Memory configuration: {args.memory_fraction*100:.0f}% of GPU memory")

    # 创建管线
    pipeline = SimpleImageSemanticPipeline(
        use_setok=not args.no_use_setok,
        use_clip=not args.no_use_clip,
        memory_fraction=args.memory_fraction
    )

    # 创建测试图像
    print("\nCreating test image...")
    test_image = pipeline.create_test_image()
    print(f"Test image shape: {test_image.shape}")

    # 分析压缩性能
    print("\nAnalyzing compression performance...")
    results = pipeline.analyze_compression_performance(test_image)

    # 绘制结果
    print("\nPlotting results...")
    pipeline.plot_results(results, "compression_analysis.png")

    # 保存结果
    pipeline.save_results(results, "compression_results.json")

    # === 输出汇总 ===
    print("\n=== Analysis Summary ===")
    print(f"Original similarity: {results.get('original_similarity', 0.0):.4f}")

    # 收集所有相似度数据
    all_sims = []
    if "similarities_ratio" in results:
        all_sims.extend(results["similarities_ratio"])
    if "similarities_count" in results:
        all_sims.extend(results["similarities_count"])

    if all_sims:
        best_sim = max(all_sims)
        worst_sim = min(all_sims)
        print(f"Best compressed similarity: {best_sim:.4f}")
        print(f"Worst compressed similarity: {worst_sim:.4f}")
    else:
        print("⚠️ No similarity data found!")

    # 计算 ratio 模式的最佳点
    if "similarities_ratio" in results and results["similarities_ratio"]:
        best_idx_ratio = int(np.argmax(results["similarities_ratio"]))
        best_ratio = results["compression_ratios"][best_idx_ratio]
        best_similarity = results["similarities_ratio"][best_idx_ratio]
        best_tokens = results["token_counts"][best_idx_ratio]
        print(f"\n📊 Best (Ratio Mode):")
        print(f"  • Compression ratio: {best_ratio:.2f}")
        print(f"  • Similarity: {best_similarity:.4f}")
        print(f"  • Token count: {best_tokens}")

    # 计算 token 模式的最佳点
    if "similarities_count" in results and results["similarities_count"]:
        best_idx_count = int(np.argmax(results["similarities_count"]))
        token_counts = [4, 8, 16, 32, 64, 128, 256][:len(results["similarities_count"])]
        best_token_count = token_counts[best_idx_count]
        best_token_similarity = results["similarities_count"][best_idx_count]
        print(f"\n📊 Best (Token Count Mode):")
        print(f"  • Token count: {best_token_count}")
        print(f"  • Similarity: {best_token_similarity:.4f}")

    print("\n✅ All results successfully saved and summarized!")


if __name__ == "__main__":
    main()

