# Image Semantic Retrieval Pipeline

这个文件夹是给“第一次接触项目的人”准备的最小主线。

原仓库里的老脚本名比较随意，这里保留了一份更容易理解的命名版本。原始文件没有删除。

## 运行顺序

1. `01_train_semantic_extractor.py`
   第一阶段，训练语义提取器。

2. `02_build_semantic_payloads.py`
   第二阶段，把图像/文本样本压成可传输的 semantic payload。

3. `03_build_image_semantic_database.py`
   用 CLIP 给图像库建立语义向量数据库。

4. `04_train_payload_to_clip_adapter.py`
   训练一个 adapter，把 payload 映射到 CLIP 检索空间。

5. `05_run_image_retrieval.py`
   用单个 payload 做最终图像检索，输出 top-k 结果。

## 对应原文件

- `01_train_semantic_extractor.py` <- `stage1_reworked.py`
- `02_build_semantic_payloads.py` <- `second_ssss.py`
- `03_build_image_semantic_database.py` <- `build_image_sem_db_(11-22).py`
- `04_train_payload_to_clip_adapter.py` <- `kreps2clip_train.py`
- `05_run_image_retrieval.py` <- `kreps2sem_infer_(11-22).py`

## 默认项目路径

- 数据目录：`data_coco`
- CoDi 目录：`i-Code-V3`
- CoDi 权重：`i-Code-V3/checkpoints`
- Stage-1 输出：`snapshots`
- Payload 输出：`comprehensive_output/payloads`
- 图像语义库：`runs/sem_db`
- Adapter 输出：`runs/kreps2clip_exp*`

## 最常用的运行方式

先进入项目根目录：

```bash
cd /root/autodl-tmp/sem-com
export CODI_ROOT=/root/autodl-tmp/sem-com/i-Code-V3
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
```

### 1. 训练语义提取器

```bash
python paper_retrieval_pipeline/01_train_semantic_extractor.py \
  --steps 300 \
  --K 16 \
  --img_take 4 \
  --lr 1e-4 \
  --save_every 100 \
  --slot_dropout 0.25 \
  --lambda_slot_usage 1.0 \
  --lambda_col_entropy 0.5 \
  --lambda_slot_infonce 0.5
```

### 2. 批量生成 payload

```bash
python paper_retrieval_pipeline/02_build_semantic_payloads.py \
  --data_root data_coco \
  --all \
  --sel 0.60 \
  --clu 0.70 \
  --floor 2 \
  --topk_per_modality 32 \
  --rep_per_cluster 2 \
  --ckpt snapshots/reworked_step_0300.pth \
  --out comprehensive_output/payloads
```

### 3. 建图像语义库

```bash
python paper_retrieval_pipeline/03_build_image_semantic_database.py \
  --image_root data_coco/image \
  --out_path runs/sem_db/image_sem_db_val2017.pt \
  --clip_model openai/clip-vit-large-patch14 \
  --batch_size 16 \
  --device cuda
```

### 4. 训练 payload 检索 adapter

```bash
python paper_retrieval_pipeline/04_train_payload_to_clip_adapter.py \
  --payload_root comprehensive_output/payloads \
  --image_root data_coco/image \
  --outdir runs/kreps2clip_exp1 \
  --clip_model openai/clip-vit-large-patch14 \
  --target_size 256 \
  --max_tokens 32 \
  --steps 1000 \
  --batch_size 8 \
  --hidden 4096 \
  --layers 2 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --lambda_mse 1.0 \
  --lambda_cos 0.5
```

### 5. 做单样本检索

```bash
python paper_retrieval_pipeline/05_run_image_retrieval.py \
  --payload_root comprehensive_output/payloads \
  --stem coco_000139 \
  --adapter_ckpt runs/kreps2clip_exp2/adapter_clip_best.pth \
  --sem_db runs/sem_db/image_sem_db_val2017.pt \
  --image_root data_coco/image \
  --outdir runs/kreps2clip_infer \
  --topk 5
```

## 建议

如果只是第一次验证环境，不要直接长时间训练。

建议先按下面顺序做 smoke test：

1. 先跑 `05_run_image_retrieval.py --help`
2. 再跑单样本 retrieval
3. 确认主线可用后，再回头训练或批量评估
