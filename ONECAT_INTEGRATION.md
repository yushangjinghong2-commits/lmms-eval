# OneCAT Integration for lmms-eval

## ✅ 已完成

1. **OneCAT 模型集成** (`/home/xinjiezhang/data/lei/lmms-eval/lmms_eval/models/simple/onecat.py`)
   - 实现了图像理解功能
   - 支持 OneCAT 的特性（智能调整大小、缩略图生成等）
   - 集成了 continual mode 缓存机制

2. **OneCAT Visual CoT 集成** (`/home/xinjiezhang/data/lei/lmms-eval/lmms_eval/models/simple/onecat_visual_cot.py`)
   - 实现了两阶段 Visual Chain-of-Thought 推理
   - 第一阶段：使用 OneCAT 的 generate_t2i() 生成辅助图像
   - 第二阶段：结合原图和辅助图进行理解
   - 支持中间结果保存和自定义生成参数

3. **模型注册** (已在 `models/__init__.py` 中注册 `onecat` 和 `onecat_visual_cot`)

4. **数据集配置** (illusionbench 数据集路径已更新)

5. **脚本工具**:
   - `download_onecat.sh` - 下载 OneCAT-3B 模型和 Infinity VAE
   - `run_onecat_illusionbench.sh` - 运行 illusionbench 评估
   - `test_onecat_integration.sh` - 测试脚本（4个测试用例）

## 🚀 使用指南

### 步骤 1: 下载 OneCAT 模型

```bash
cd /home/xinjiezhang/data/lei/lmms-eval

# 下载 OneCAT-3B 和 Infinity VAE
./download_onecat.sh
```

这将下载：
- OneCAT-3B 模型 → `/home/xinjiezhang/data/lei/lmms-eval/models/OneCAT-3B`
- Infinity VAE → `/home/xinjiezhang/data/lei/lmms-eval/models/infinity_vae/infinity_vae_d32reg.pth`

### 步骤 2: 运行 illusionbench 评估

```bash
cd /home/xinjiezhang/data/lei/lmms-eval

# 运行完整的 illusionbench_arshia_test (6个子任务)
./run_onecat_illusionbench.sh
```

### 手动运行示例

#### 基础图像理解

如果模型在其他位置，可以手动指定路径：

```bash
cd /home/xinjiezhang/data/lei/lmms-eval

python -m lmms_eval \
    --model onecat \
    --model_args pretrained=/path/to/OneCAT-3B,max_new_tokens=64,do_sample=false \
    --tasks illusionbench_arshia_test \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/onecat_illusionbench/
```

#### Visual Chain-of-Thought 推理

```bash
cd /home/xinjiezhang/data/lei/lmms-eval

python -m lmms_eval \
    --model onecat_visual_cot \
    --model_args pretrained=/path/to/OneCAT-3B,vae_path=/path/to/infinity_vae_d32reg.pth,save_intermediate=True \
    --tasks illusionbench_arshia_icon_shape_visual_cot \
    --batch_size 1 \
    --log_samples \
    --output_path ./logs/onecat_visual_cot/
```

## 📋 支持的任务

OneCAT 集成支持所有 illusionbench 测试任务：

### 基础测试任务（onecat）
- `illusionbench_arshia_icon_shape_test`
- `illusionbench_arshia_icon_scene_test`
- `illusionbench_arshia_logo_shape_test`
- `illusionbench_arshia_logo_scene_test`
- `illusionbench_arshia_in_shape_test`
- `illusionbench_arshia_in_scene_test`
- `illusionbench_arshia_test` (组任务，包含以上所有)

### Visual CoT 任务（onecat_visual_cot）
- `illusionbench_arshia_icon_shape_visual_cot`
- `illusionbench_arshia_icon_scene_visual_cot`
- `illusionbench_arshia_logo_shape_visual_cot`
- `illusionbench_arshia_logo_scene_visual_cot`
- `illusionbench_arshia_in_shape_visual_cot`
- `illusionbench_arshia_in_scene_visual_cot`

## ⚙️ 模型参数

### onecat 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained` | **必填** | OneCAT-3B 模型路径 |
| `max_new_tokens` | 1000 | 最大生成 token 数 |
| `do_sample` | False | 是否使用采样 |
| `num_beams` | 1 | Beam search 数量 |
| `top_k` | None | Top-k 采样 |
| `top_p` | None | Top-p 采样 |
| `dtype` | "bfloat16" | 模型精度 |

### onecat_visual_cot 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained` | **必填** | OneCAT-3B 模型路径 |
| `vae_path` | **必填** | Infinity VAE 模型路径 |
| `max_new_tokens` | 1000 | 第二阶段最大生成 token 数 |
| `do_sample` | False | 第二阶段是否使用采样 |
| `stage1_cfg` | 1.5 | 第一阶段 CFG scale |
| `stage1_top_k` | 2000 | 第一阶段 top-k 采样 |
| `stage1_top_p` | 1.0 | 第一阶段 top-p 采样 |
| `stage1_h_div_w` | 1.0 | 第一阶段图像高宽比 |
| `save_intermediate` | False | 是否保存中间生成的图像 |
| `dtype` | "bfloat16" | 模型精度 |

## 📊 关于 OneCAT

**OneCAT** (Decoder-Only Auto-Regressive Model) 是一个统一的多模态模型，特点：

- **纯解码器架构**：推理时无需外部 Vision Encoder 或 VAE tokenizer
- **Mixture-of-Experts (MoE)**：包含三个专门的 FFN 专家
  - Text FFN：语言理解
  - Understanding FFN：视觉理解
  - Generation FFN：图像生成
- **多尺度自回归**：Next Scale Prediction 范式，大幅减少生成步骤

### 功能支持

OneCAT 支持三大功能：
1. ✅ **视觉理解** (Visual Understanding) - 已集成到 lmms-eval
2. ✅ **Visual Chain-of-Thought** - 已集成到 lmms-eval
3. **文本生成图像** (Text-to-Image) - 未集成
4. **图像编辑** (Image Editing) - 未集成

当前集成专注于视觉理解任务的评估，包括基础理解和 Visual CoT 推理。

## 📁 文件结构

```
/home/xinjiezhang/data/lei/
├── lmms-eval/
│   ├── lmms_eval/
│   │   ├── models/
│   │   │   ├── __init__.py              # ✅ 已注册 onecat
│   │   │   └── simple/
│   │   │       └── onecat.py            # ✅ OneCAT 集成
│   │   └── tasks/
│   │       └── illusionbench/
│   │           └── *.yaml               # ✅ 数据集路径已更新
│   ├── datasets/
│   │   └── illusionbench/               # ✅ 数据集已解压
│   ├── models/
│   │   ├── OneCAT-3B/                   # ⬇️ 需要下载
│   │   └── infinity_vae/                # ⬇️ 需要下载
│   ├── download_onecat.sh               # ✅ 下载脚本
│   ├── run_onecat_illusionbench.sh      # ✅ 测试脚本
│   └── ONECAT_INTEGRATION.md            # ✅ 本文件
└── OneCAT/                               # ✅ OneCAT 源代码
    ├── onecat/
    │   ├── modeling_onecat.py
    │   ├── smart_resize.py
    │   └── util.py
    └── generate_understanding.py
```

## 🔧 故障排除

### 问题 1: "OneCAT repository not found"

**解决方案**: 确保 OneCAT 源代码在正确位置：
```bash
ls /home/xinjiezhang/data/lei/OneCAT/onecat/
```

### 问题 2: "Model not found"

**解决方案**: 运行下载脚本：
```bash
./download_onecat.sh
```

### 问题 3: CUDA Out of Memory

**解决方案**: OneCAT-3B 较小，通常不会有内存问题。如果遇到，可以：
- 确保 `batch_size=1`
- 使用 `dtype="float16"`

## 📚 参考资料

- [OneCAT Paper](https://arxiv.org/abs/2509.03498)
- [OneCAT GitHub](https://github.com/onecat-ai/OneCAT)
- [OneCAT Model on HuggingFace](https://huggingface.co/onecat-ai/OneCAT-3B)
- [OneCAT Homepage](https://onecat-ai.github.io/)

## 🎯 下一步

1. **下载模型**: `./download_onecat.sh`
2. **运行评估**: `./run_onecat_illusionbench.sh`
3. **查看结果**: `cat ./logs/onecat_illusionbench_arshia_test/results.json`

---

**集成完成！** 现在可以使用 OneCAT 模型在 lmms-eval 框架下进行 illusionbench 评估了。🎉
