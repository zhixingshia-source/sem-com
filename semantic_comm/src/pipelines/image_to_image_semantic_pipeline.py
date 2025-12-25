#!/usr/bin/env python3
"""
Image-to-Image Semantic Communication Pipeline
图到图语义通信管线

功能：
1. 图像压缩和token化
2. 语义通信传输
3. 图像解压缩和重建
4. 语义相似度评估
5. 不同压缩token数下的性能分析
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 启用功能配置
ENABLE_CODI = True            # 启用CoDi图像处理
ENABLE_SETOK = True           # 启用Setok语义token化
ENABLE_DETOKENIZER = True     # 启用解token化

class ImageToImageSemanticPipeline:
    """图到图语义通信管线"""
    
    def __init__(self, 
                 codi_model_path: str = None,
                 setok_config: Dict = None,
                 detokenizer_config: Dict = None):
        """
        初始化管线
        
        Args:
            codi_model_path: CoDi模型路径
            setok_config: Setok配置
            detokenizer_config: 解token化器配置
        """
        self.device = DEVICE
        self.codi_model = None
        self.tokenizer = None
        self.detokenizer = None
        
        # 默认配置
        self.default_setok_config = {
            'vision_tower': 'openai/clip-vit-base-patch16',
            'delay_load': True,
            'hidden_dim': 1024,
            'token_feat_dim': 1024,
            'min_cluster_num': 32,
            'threshold': 0.6,
            'nheads': 4,
            'dim_feedforward': 2048,
            'inner_cluster_layers': 1,
            'intra_cluster_layers': 1,
            'proj_drop': 0.1,
            'drop_path': 0.0,
        }
        
        self.default_detokenizer_config = {
            'token_feat_dim': 1024,
            'hidden_dim': 1024,
            'decoder_embed_dim': 512,
            'decoder_nheads': 8,
            'decoder_depth': 4,
            'num_hidden_layers': 2,
            'cross_attention_freq': 2,
        }
        
        # 合并配置
        self.setok_config = {**self.default_setok_config, **(setok_config or {})}
        self.detokenizer_config = {**self.default_detokenizer_config, **(detokenizer_config or {})}
        
        # 初始化模型
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化所有模型"""
        print("Initializing models...")
        
        # 初始化CoDi模型
        if ENABLE_CODI:
            try:
                from semantic_comm.src.models.codi import CoDiModel
                self.codi_model = CoDiModel().to(self.device).eval()
                print("✓ CoDi model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load CoDi model: {e}")
                ENABLE_CODI = False
        
        # 初始化Setok tokenizer
        if ENABLE_SETOK:
            try:
                from semantic_comm.src.models.tokenizer import SetokTokenizer
                self.tokenizer = SetokTokenizer(**self.setok_config).to(self.device).eval()
                print("✓ Setok tokenizer loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load Setok tokenizer: {e}")
                ENABLE_SETOK = False
        
        # 初始化解token化器
        if ENABLE_DETOKENIZER and ENABLE_SETOK:
            try:
                from semantic_comm.src.models.detokenizer import SetokDeTokenizer
                self.detokenizer = SetokDeTokenizer(**self.detokenizer_config).to(self.device).eval()
                print("✓ Setok detokenizer loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load Setok detokenizer: {e}")
                ENABLE_DETOKENIZER = False
        
        print("Model initialization completed!")
    
    def load_image(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        加载和预处理图像
        
        Args:
            image_path: 图像路径
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像tensor
        """
        try:
            # 加载图像
            image = Image.open(image_path).convert('RGB')
            
            # 调整尺寸
            image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # 转换为tensor
            image_tensor = torch.from_numpy(np.array(image)).float()
            image_tensor = image_tensor.permute(2, 0, 1) / 255.0  # HWC -> CHW, normalize
            image_tensor = image_tensor.unsqueeze(0)  # 添加batch维度
            
            return image_tensor.to(self.device)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def compress_image(self, image_tensor: torch.Tensor, compression_ratio: float = 0.5) -> Dict:
        """
        压缩图像
        
        Args:
            image_tensor: 输入图像tensor
            compression_ratio: 压缩比例 (0-1)
            
        Returns:
            压缩结果字典
        """
        result = {
            'original_image': image_tensor,
            'compressed_tokens': None,
            'token_count': 0,
            'compression_ratio': compression_ratio
        }
        
        if not ENABLE_SETOK or self.tokenizer is None:
            print("Setok tokenizer not available, using simple compression")
            # 简单的下采样压缩
            h, w = image_tensor.shape[-2:]
            new_h, new_w = int(h * compression_ratio), int(w * compression_ratio)
            compressed = F.interpolate(image_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
            result['compressed_tokens'] = compressed
            result['token_count'] = new_h * new_w * 3
            return result
        
        try:
            with torch.no_grad():
                # 使用Setok进行语义token化
                tokens = self.tokenizer(image_tensor)
                
                # 根据压缩比例选择token数量
                if isinstance(tokens, dict):
                    token_features = tokens.get('token_features', tokens.get('tokens'))
                else:
                    token_features = tokens
                
                if token_features is not None:
                    num_tokens = token_features.shape[1] if len(token_features.shape) > 1 else len(token_features)
                    target_tokens = max(1, int(num_tokens * compression_ratio))
                    
                    # 选择最重要的tokens
                    if len(token_features.shape) > 1 and token_features.shape[1] > target_tokens:
                        # 简单的选择策略：选择前N个tokens
                        selected_tokens = token_features[:, :target_tokens, :]
                    else:
                        selected_tokens = token_features
                    
                    result['compressed_tokens'] = selected_tokens
                    result['token_count'] = target_tokens
                else:
                    print("Warning: No tokens generated")
                    result['compressed_tokens'] = image_tensor
                    result['token_count'] = image_tensor.numel()
        
        except Exception as e:
            print(f"Error in compression: {e}")
            result['compressed_tokens'] = image_tensor
            result['token_count'] = image_tensor.numel()
        
        return result
    
    def decompress_image(self, compressed_data: Dict, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
        """
        解压缩图像
        
        Args:
            compressed_data: 压缩数据字典
            target_size: 目标输出尺寸
            
        Returns:
            重建的图像tensor
        """
        if not ENABLE_DETOKENIZER or self.detokenizer is None:
            print("Detokenizer not available, using simple decompression")
            # 简单的上采样解压缩
            compressed_tokens = compressed_data['compressed_tokens']
            if len(compressed_tokens.shape) == 4:  # 图像tensor
                return F.interpolate(compressed_tokens, size=target_size, mode='bilinear', align_corners=False)
            else:
                # 如果是token，尝试重建
                return compressed_data['original_image']
        
        try:
            with torch.no_grad():
                compressed_tokens = compressed_data['compressed_tokens']
                
                # 使用detokenizer重建图像
                if len(compressed_tokens.shape) == 4:  # 已经是图像tensor
                    return F.interpolate(compressed_tokens, size=target_size, mode='bilinear', align_corners=False)
                
                # 从tokens重建
                reconstructed = self.detokenizer(compressed_tokens)
                
                if isinstance(reconstructed, dict):
                    reconstructed = reconstructed.get('reconstructed_image', reconstructed.get('output'))
                
                if reconstructed is not None:
                    # 确保输出尺寸正确
                    if reconstructed.shape[-2:] != target_size:
                        reconstructed = F.interpolate(reconstructed, size=target_size, mode='bilinear', align_corners=False)
                    return reconstructed
                else:
                    print("Warning: Failed to reconstruct from tokens")
                    return compressed_data['original_image']
        
        except Exception as e:
            print(f"Error in decompression: {e}")
            return compressed_data['original_image']
    
    def calculate_semantic_similarity(self, image1: torch.Tensor, image2: torch.Tensor) -> float:
        """
        计算两幅图像的语义相似度
        
        Args:
            image1: 第一幅图像
            image2: 第二幅图像
            
        Returns:
            语义相似度分数 (0-1)
        """
        try:
            # 使用CLIP计算语义相似度
            if ENABLE_SETOK and self.tokenizer is not None:
                with torch.no_grad():
                    # 提取特征
                    feat1 = self.tokenizer.extract_features(image1)
                    feat2 = self.tokenizer.extract_features(image2)
                    
                    if feat1 is not None and feat2 is not None:
                        # 计算余弦相似度
                        feat1_norm = F.normalize(feat1, p=2, dim=-1)
                        feat2_norm = F.normalize(feat2, p=2, dim=-1)
                        similarity = torch.cosine_similarity(feat1_norm, feat2_norm, dim=-1)
                        return similarity.mean().item()
            
            # 备用方法：使用像素级相似度
            mse = F.mse_loss(image1, image2)
            similarity = 1.0 / (1.0 + mse.item())
            return similarity
        
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            # 简单的像素级相似度
            mse = F.mse_loss(image1, image2)
            return 1.0 / (1.0 + mse.item())
    
    def process_image_pair(self, 
                          image1_path: str, 
                          image2_path: str,
                          compression_ratios: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) -> Dict:
        """
        处理图像对，测试不同压缩比例下的语义相似度
        
        Args:
            image1_path: 第一幅图像路径
            image2_path: 第二幅图像路径
            compression_ratios: 压缩比例列表
            
        Returns:
            处理结果字典
        """
        print(f"Processing image pair: {image1_path} vs {image2_path}")
        
        # 加载图像
        image1 = self.load_image(image1_path)
        image2 = self.load_image(image2_path)
        
        if image1 is None or image2 is None:
            print("Failed to load images")
            return None
        
        results = {
            'image1_path': image1_path,
            'image2_path': image2_path,
            'compression_ratios': compression_ratios,
            'similarities': [],
            'token_counts': [],
            'original_similarity': 0.0
        }
        
        # 计算原始相似度
        results['original_similarity'] = self.calculate_semantic_similarity(image1, image2)
        print(f"Original similarity: {results['original_similarity']:.4f}")
        
        # 测试不同压缩比例
        for ratio in compression_ratios:
            print(f"Testing compression ratio: {ratio}")
            
            # 压缩第一幅图像
            compressed1 = self.compress_image(image1, ratio)
            
            # 解压缩
            reconstructed1 = self.decompress_image(compressed1)
            
            # 计算相似度
            similarity = self.calculate_semantic_similarity(reconstructed1, image2)
            
            results['similarities'].append(similarity)
            results['token_counts'].append(compressed1['token_count'])
            
            print(f"  Ratio: {ratio:.1f}, Tokens: {compressed1['token_count']}, Similarity: {similarity:.4f}")
        
        return results
    
    def plot_compression_analysis(self, results: Dict, save_path: str = None):
        """
        绘制压缩分析图表
        
        Args:
            results: 处理结果
            save_path: 保存路径
        """
        plt.figure(figsize=(12, 8))
        
        # 子图1：压缩比例 vs 语义相似度
        plt.subplot(2, 2, 1)
        plt.plot(results['compression_ratios'], results['similarities'], 'bo-', linewidth=2, markersize=6)
        plt.axhline(y=results['original_similarity'], color='r', linestyle='--', alpha=0.7, label='Original')
        plt.xlabel('Compression Ratio')
        plt.ylabel('Semantic Similarity')
        plt.title('Compression Ratio vs Semantic Similarity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 子图2：Token数量 vs 语义相似度
        plt.subplot(2, 2, 2)
        plt.plot(results['token_counts'], results['similarities'], 'go-', linewidth=2, markersize=6)
        plt.axhline(y=results['original_similarity'], color='r', linestyle='--', alpha=0.7, label='Original')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Semantic Similarity')
        plt.title('Token Count vs Semantic Similarity')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 子图3：压缩效率分析
        plt.subplot(2, 2, 3)
        compression_efficiency = [s / r for s, r in zip(results['similarities'], results['compression_ratios'])]
        plt.plot(results['compression_ratios'], compression_efficiency, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Compression Ratio')
        plt.ylabel('Compression Efficiency (Similarity/Ratio)')
        plt.title('Compression Efficiency Analysis')
        plt.grid(True, alpha=0.3)
        
        # 子图4：Token效率分析
        plt.subplot(2, 2, 4)
        token_efficiency = [s / t for s, t in zip(results['similarities'], results['token_counts'])]
        plt.plot(results['token_counts'], token_efficiency, 'co-', linewidth=2, markersize=6)
        plt.xlabel('Number of Tokens')
        plt.ylabel('Token Efficiency (Similarity/Token)')
        plt.title('Token Efficiency Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Analysis plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, results: Dict, save_path: str):
        """
        保存结果到JSON文件
        
        Args:
            results: 结果字典
            save_path: 保存路径
        """
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(results)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {save_path}")


def main():
    """主函数 - 演示管线使用"""
    print("=== Image-to-Image Semantic Communication Pipeline ===")
    
    # 创建管线实例
    pipeline = ImageToImageSemanticPipeline()
    
    # 示例图像路径（请替换为实际路径）
    image1_path = "test_images/image1.jpg"  # 请替换为实际路径
    image2_path = "test_images/image2.jpg"  # 请替换为实际路径
    
    # 检查图像文件是否存在
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("Please provide valid image paths in the main() function")
        print("Example usage:")
        print("  pipeline = ImageToImageSemanticPipeline()")
        print("  results = pipeline.process_image_pair('path/to/image1.jpg', 'path/to/image2.jpg')")
        print("  pipeline.plot_compression_analysis(results)")
        return
    
    # 处理图像对
    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    results = pipeline.process_image_pair(image1_path, image2_path, compression_ratios)
    
    if results is not None:
        # 绘制分析图表
        pipeline.plot_compression_analysis(results, "compression_analysis.png")
        
        # 保存结果
        pipeline.save_results(results, "compression_results.json")
        
        print("\n=== Analysis Summary ===")
        print(f"Original similarity: {results['original_similarity']:.4f}")
        print(f"Best compressed similarity: {max(results['similarities']):.4f}")
        print(f"Worst compressed similarity: {min(results['similarities']):.4f}")
        
        # 找到最佳压缩比例
        best_idx = np.argmax(results['similarities'])
        best_ratio = results['compression_ratios'][best_idx]
        best_similarity = results['similarities'][best_idx]
        best_tokens = results['token_counts'][best_idx]
        
        print(f"Best compression ratio: {best_ratio:.1f}")
        print(f"Best similarity: {best_similarity:.4f}")
        print(f"Token count at best ratio: {best_tokens}")


if __name__ == "__main__":
    main()