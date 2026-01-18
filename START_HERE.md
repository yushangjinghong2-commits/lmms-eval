# 🎉 UniPic2 集成完成 - 立即开始使用

## ✅ 配置验证成功

所有模型已找到并验证：

### 已配置的模型

1. **Qwen2.5-VL-7B-Instruct** (基础理解模型)
   ```
   ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5
   ```

2. **UniPic2-Metaquery-9B** (统一模型，包含 SD3.5M-Kontext 组件)
   ```
   ~/.cache/huggingface/hub/models--Skywork--UniPic2-Metaquery-9B/snapshots/37a2f17d28578b89d38aebd79515ba5610e75cad
   ```

   包含组件：
   - ✓ transformer (SD3Transformer2DKontextModel)
   - ✓ vae (AutoencoderKL)
   - ✓ scheduler (FlowMatchEulerDiscreteScheduler)
   - ✓ conditioner (StableDiffusion3Conditioner)

---

## 🚀 快速开始 (3 步)

### 步骤 1: 激活环境并进入目录

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate
cd /home/xinjiezhang/data/lei/lmms-eval
```

### 步骤 2: 验证配置（可选）

```bash
./verify_model_paths.sh
```

预期输出：`✓ All model paths are valid and loadable!`

### 步骤 3: 运行测试

选择以下任意一个测试：

```bash
# 快速测试 - unipic2 模型 (limit=5 个样本)
./test_unipic2_integration.sh 1

# 快速测试 - unipic2_visual_cot 模型 (limit=5 个样本)
./test_unipic2_integration.sh 2

# 完整评估 - illusionbench icon shape 测试
./test_unipic2_integration.sh 3

# 完整评估 - illusionbench icon shape Visual CoT
./test_unipic2_integration.sh 4
```

---

## 📖 使用示例

### 示例 1: 标准图像理解 (unipic2)

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate
cd /home/xinjiezhang/data/lei/lmms-eval

accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5,max_new_tokens=64,temperature=0.0 \
    --tasks illusionbench_arshia_icon_shape_test \
    --batch_size 1 \
    --output_path ./logs/unipic2_icon_shape/
```

### 示例 2: Visual Chain-of-Thought (unipic2_visual_cot)

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate
cd /home/xinjiezhang/data/lei/lmms-eval

accelerate launch -m lmms_eval \
    --model unipic2_visual_cot \
    --model_args pretrained=$HOME/.cache/huggingface/hub/models--Skywork--UniPic2-Metaquery-9B/snapshots/37a2f17d28578b89d38aebd79515ba5610e75cad,qwen_model=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5,save_intermediate=True \
    --tasks illusionbench_arshia_icon_shape_visual_cot \
    --batch_size 1 \
    --output_path ./logs/unipic2_visual_cot_icon_shape/
```

### 使用更简洁的命令（推荐）

使用配置文件后，可以这样运行：

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate
cd /home/xinjiezhang/data/lei/lmms-eval
source model_paths.sh

# 示例 1
accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=$QWEN_MODEL_PATH \
    --tasks illusionbench_arshia_icon_shape_test \
    --batch_size 1 \
    --output_path ./logs/

# 示例 2
accelerate launch -m lmms_eval \
    --model unipic2_visual_cot \
    --model_args pretrained=$UNIPIC2_SD35M_PATH,qwen_model=$QWEN_MODEL_PATH \
    --tasks illusionbench_arshia_icon_shape_visual_cot \
    --batch_size 1 \
    --output_path ./logs/
```

---

## 📋 可用任务列表

### 标准测试任务 (使用 `unipic2`)

```bash
# Icon 子集
illusionbench_arshia_icon_shape_test
illusionbench_arshia_icon_scene_test

# Logo 子集
illusionbench_arshia_logo_shape_test
illusionbench_arshia_logo_scene_test

# ImageNet 子集
illusionbench_arshia_in_shape_test
illusionbench_arshia_in_scene_test

# 组合任务
illusionbench_arshia_test  # 包含所有上述任务
```

### Visual CoT 任务 (使用 `unipic2_visual_cot`)

```bash
# Icon 子集
illusionbench_arshia_icon_shape_visual_cot
illusionbench_arshia_icon_scene_visual_cot
illusionbench_arshia_icon_visual_cot  # 同时评估 shape 和 scene

# Logo 子集
illusionbench_arshia_logo_shape_visual_cot
illusionbench_arshia_logo_scene_visual_cot
illusionbench_arshia_logo_visual_cot

# ImageNet 子集
illusionbench_arshia_in_shape_visual_cot
illusionbench_arshia_in_scene_visual_cot
illusionbench_arshia_in_visual_cot
```

---

## ⚙️ 模型参数说明

### unipic2 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained` | **必填** | Qwen2.5-VL 模型路径 |
| `max_new_tokens` | 512 | 最大生成 token 数 |
| `temperature` | 0.0 | 采样温度（0.0 = 确定性）|
| `do_sample` | False | 是否使用采样 |
| `dtype` | "bfloat16" | 模型精度 |

### unipic2_visual_cot 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained` | **必填** | UniPic2-SD3.5M 路径 |
| `qwen_model` | **必填** | Qwen2.5-VL 路径 |
| `stage1_num_inference_steps` | 50 | 图像生成的推理步数 |
| `stage1_guidance_scale` | 3.5 | 引导强度 |
| `stage1_height` | 1024 | 生成图像高度 |
| `stage1_width` | 1024 | 生成图像宽度 |
| `save_intermediate` | False | 保存中间生成的图像 |
| `seed` | 0 | 随机种子 |

---

## 📊 预期输出

运行测试后，会在指定的 `output_path` 下生成：

```
./logs/
├── <model_name>_<task_name>/
│   ├── results.json           # 评估结果
│   ├── samples.jsonl          # 详细样本输出
│   └── <task>_<timestamp>/
│       ├── <doc_id>_stage1_generated.png  # 生成的辅助图像（Visual CoT）
│       └── <doc_id>_metadata.json         # 元数据
```

### 评估指标

- **shape_recall**: 形状识别召回率（0-1）
- **scene_recall**: 场景识别召回率（0-1）

---

## 🔧 故障排除

### 问题 1: CUDA Out of Memory

**解决方案**:
```bash
# 对于 Visual CoT，降低生成图像的分辨率
--model_args ...,stage1_height=512,stage1_width=512
```

### 问题 2: 生成速度慢

**解决方案**:
```bash
# 减少推理步数
--model_args ...,stage1_num_inference_steps=20
```

### 问题 3: 想要查看中间生成的图像

**解决方案**:
```bash
# 启用中间产物保存
--model_args ...,save_intermediate=True,intermediate_dir=./intermediate/
```

---

## 📁 项目文件结构

```
/home/xinjiezhang/data/lei/lmms-eval/
├── lmms_eval/
│   ├── models/
│   │   ├── __init__.py                    # ✅ 已注册模型
│   │   └── simple/
│   │       ├── unipic2.py                 # ✅ 图像理解模型
│   │       └── unipic2_visual_cot.py      # ✅ Visual CoT 模型
│   └── tasks/
│       └── illusionbench/
│           ├── arshia_utils.py            # 任务工具函数
│           └── *.yaml                     # 17 个任务配置
├── model_paths.sh                         # ✅ 模型路径配置
├── verify_model_paths.sh                  # ✅ 路径验证脚本
├── test_unipic2_integration.sh            # ✅ 测试脚本
├── check_unipic2_integration.py           # ✅ 集成检查脚本
├── UNIPIC2_INTEGRATION.md                 # 详细文档
├── UNIPIC2_QUICKSTART.md                  # 快速开始（中文）
└── START_HERE.md                          # ✅ 本文件
```

---

## ✨ 下一步行动

1. ✅ **环境验证完成** - 所有模型路径已验证
2. ✅ **集成测试完成** - 配置检查全部通过
3. **运行第一个测试**:
   ```bash
   cd /home/xinjiezhang/data/lei/lmms-eval
   ./test_unipic2_integration.sh 1
   ```
4. **查看结果并调整参数**
5. **运行完整评估以获得基准结果**

---

## 📚 参考文档

- **详细集成文档**: `UNIPIC2_INTEGRATION.md`
- **快速开始指南**: `UNIPIC2_QUICKSTART.md`
- **配置检查**: `python check_unipic2_integration.py`
- **模型路径验证**: `./verify_model_paths.sh`

---

## 🎯 总结

✅ 所有准备工作已完成：
- [x] 模型实现完成
- [x] 模型注册完成
- [x] 模型路径已找到并配置
- [x] 配置验证通过
- [x] 测试脚本就绪

**现在就可以开始使用了！** 🚀

运行以下命令开始第一个测试：
```bash
cd /home/xinjiezhang/data/lei/lmms-eval
./test_unipic2_integration.sh 1
```
