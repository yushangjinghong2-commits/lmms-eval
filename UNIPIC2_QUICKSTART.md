# UniPic2 集成完成总结

## ✅ 已完成的工作

### 1. 模型实现文件

已创建两个模型实现：

- **`lmms_eval/models/simple/unipic2.py`**
  基于 Qwen2.5-VL 的图像理解模型，用于标准的图像理解任务

- **`lmms_eval/models/simple/unipic2_visual_cot.py`**
  两阶段 Visual Chain-of-Thought 模型：
  - Stage 1: 使用 SD3.5M-Kontext 生成辅助可视化图像
  - Stage 2: 使用 Qwen2.5-VL 结合原图和辅助图回答问题

### 2. 模型注册

已在 `lmms_eval/models/__init__.py` 中注册：
- `unipic2`: UniPic2 图像理解模型
- `unipic2_visual_cot`: UniPic2 Visual CoT 模型

### 3. 文档和工具

- **`UNIPIC2_INTEGRATION.md`**: 详细的集成文档和使用指南
- **`check_unipic2_integration.py`**: 配置检查脚本，验证集成是否正确
- **`test_unipic2_integration.sh`**: 测试脚本，包含 4 个示例测试

### 4. 支持的任务

集成支持所有 illusionbench 任务：
- 6 个标准测试任务（icon/logo/in × shape/scene）
- 9 个 Visual CoT 任务

---

## 🚀 快速开始

### 1. 验证集成配置

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate
cd /home/xinjiezhang/data/lei/lmms-eval
python check_unipic2_integration.py
```

**预期输出**: 所有 5 项检查都应该通过 ✓

### 2. 运行第一个测试

在运行测试之前，需要：

1. **下载模型权重**（如果还没有）：
   - UniPic2-MetaQuery（Qwen2.5-VL based）
   - UniPic2-SD3.5M-Kontext（用于 Visual CoT）
   - Qwen2.5-VL-7B-Instruct（基础模型）

   模型下载地址: https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd

2. **更新测试脚本中的模型路径**：
   ```bash
   # 编辑 test_unipic2_integration.sh
   # 将这些路径更新为你的实际模型位置：
   UNIPIC2_METAQUERY_PATH="/path/to/UniPic2-MetaQuery"
   UNIPIC2_SD35M_PATH="/path/to/UniPic2-SD3.5M-Kontext"
   QWEN_MODEL_PATH="/path/to/Qwen2.5-VL-7B-Instruct"
   ```

3. **运行测试**：
   ```bash
   # 测试 unipic2 图像理解模型（快速测试，limit=5）
   ./test_unipic2_integration.sh 1

   # 测试 unipic2_visual_cot 模型（快速测试，limit=5）
   ./test_unipic2_integration.sh 2

   # 运行完整的 illusionbench icon shape 评估
   ./test_unipic2_integration.sh 3

   # 运行完整的 illusionbench icon shape Visual CoT 评估
   ./test_unipic2_integration.sh 4
   ```

---

## 📖 使用示例

### 示例 1: 标准图像理解任务

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

accelerate launch -m lmms_eval \
    --model unipic2 \
    --model_args pretrained=/path/to/UniPic2-MetaQuery,max_new_tokens=64,temperature=0.0 \
    --tasks illusionbench_arshia_icon_shape_test \
    --batch_size 1 \
    --output_path ./logs/unipic2_icon_shape/
```

### 示例 2: Visual CoT 任务

```bash
source /home/xinjiezhang/data/lei/UniPic/UniPic-2/.venv/bin/activate

accelerate launch -m lmms_eval \
    --model unipic2_visual_cot \
    --model_args pretrained=/path/to/UniPic2-SD3.5M-Kontext,qwen_model=/path/to/Qwen2.5-VL-7B-Instruct,save_intermediate=True \
    --tasks illusionbench_arshia_icon_shape_visual_cot \
    --batch_size 1 \
    --output_path ./logs/unipic2_visual_cot_icon_shape/
```

当 `save_intermediate=True` 时，会保存生成的辅助图像和元数据，方便检查。

---

## 🔧 配置参数说明

### unipic2 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `pretrained` | 必填 | UniPic2-MetaQuery 模型路径 |
| `max_new_tokens` | 512 | 最大生成 token 数 |
| `temperature` | 0.0 | 采样温度 |
| `do_sample` | False | 是否使用采样 |
| `top_p` | 1.0 | Top-p 采样参数 |
| `dtype` | "bfloat16" | 模型精度 |
| `attn_implementation` | "flash_attention_2" | 注意力实现 |
| `continual_mode` | True | 启用响应缓存 |

### unipic2_visual_cot 模型参数

**必需参数：**
- `pretrained`: UniPic2-SD3.5M-Kontext 路径
- `qwen_model`: Qwen2.5-VL-7B-Instruct 路径

**Stage 1 参数（图像生成）：**
- `stage1_num_inference_steps`: 50（推理步数）
- `stage1_guidance_scale`: 3.5（引导强度）
- `stage1_height`: 1024（生成图像高度）
- `stage1_width`: 1024（生成图像宽度）

**Stage 2 参数（图像理解）：**
- `stage2_max_new_tokens`: 512
- `stage2_temperature`: 0.0
- `stage2_do_sample`: False
- `stage2_top_p`: 1.0

**其他参数：**
- `save_intermediate`: False（保存中间产物）
- `intermediate_dir`: 自动设置（中间产物保存目录）
- `seed`: 0（随机种子）

---

## 📁 目录结构

```
/home/xinjiezhang/data/lei/
├── lmms-eval/
│   ├── lmms_eval/
│   │   ├── models/
│   │   │   ├── __init__.py                    # ✅ 已更新（注册新模型）
│   │   │   └── simple/
│   │   │       ├── unipic2.py                 # ✅ 新创建
│   │   │       └── unipic2_visual_cot.py      # ✅ 新创建
│   │   └── tasks/
│   │       └── illusionbench/
│   │           ├── arshia_utils.py            # ✅ 已存在
│   │           └── *.yaml                     # ✅ 17 个任务配置
│   ├── UNIPIC2_INTEGRATION.md                 # ✅ 新创建（详细文档）
│   ├── check_unipic2_integration.py           # ✅ 新创建（配置检查）
│   └── test_unipic2_integration.sh            # ✅ 新创建（测试脚本）
└── UniPic/
    └── UniPic-2/
        ├── .venv/                             # ✅ 已存在（Python 环境）
        ├── unipicv2/                          # ✅ 已存在（自定义模块）
        └── scripts/                           # ✅ 已存在
```

---

## ✨ 关键特性

### 1. unipic2 模型
- 基于 Qwen2.5-VL 架构
- 支持单图像理解任务
- 自动处理图像和文本输入
- 支持响应缓存（continual_mode）

### 2. unipic2_visual_cot 模型
- 两阶段推理流程：
  1. 生成辅助可视化图像
  2. 结合原图和辅助图回答问题
- 自动处理 `[GEN_PROMPT]` 和 `[QUESTION]` 标记
- 支持保存中间产物用于调试
- 错误容忍机制（fail_gracefully）

### 3. 与 illusionbench 任务完美集成
- 支持所有 6 个标准测试任务
- 支持所有 9 个 Visual CoT 任务
- 自动解析任务特定的提示词格式

---

## 🔍 故障排除

### 问题 1: "UniPic2 modules not found"
**解决方案**: 确保 UniPic-2 仓库在正确位置：
```bash
ls /home/xinjiezhang/data/lei/UniPic/UniPic-2/unipicv2/
```

### 问题 2: CUDA Out of Memory
**解决方案**:
- 降低 `stage1_height` 和 `stage1_width`
- 使用 `dtype="float16"`
- 减少 `max_new_tokens`

### 问题 3: Flash Attention 不可用
**解决方案**: 模型会自动回退到标准注意力机制

---

## 📊 预期结果

运行测试后，会生成以下输出：

1. **日志文件**: `./logs/<model_name>_<task_name>/`
2. **评估结果**: JSON 格式的评估指标
3. **中间产物**（Visual CoT）: 生成的辅助图像和元数据

### 评估指标

对于 illusionbench 任务，主要指标是：
- **shape_recall**: 形状识别召回率
- **scene_recall**: 场景识别召回率

---

## 📝 下一步

1. **下载模型权重**并更新路径
2. **运行配置检查**确保一切就绪
3. **运行快速测试**（limit=5）验证功能
4. **运行完整评估**获取基准结果
5. **调整参数**以优化性能

---

## 🙏 参考资料

- [UniPic2 论文](https://arxiv.org/abs/2509.04548)
- [UniPic2 GitHub](https://github.com/SkyworkAI/UniPic)
- [模型下载](https://huggingface.co/collections/Skywork/skywork-unipic2-6899b9e1b038b24674d996fd)
- [lmms-eval 框架](https://github.com/EvolvingLMMs-Lab/lmms-eval)

---

## ✅ 集成状态

- [x] 创建 unipic2.py 实现图像理解功能
- [x] 创建 unipic2_visual_cot.py 实现两阶段 Visual CoT
- [x] 更新 models/__init__.py 注册新模型
- [x] 创建配置检查脚本
- [x] 创建测试脚本
- [x] 创建集成文档
- [x] 验证配置（所有检查通过 ✓）

**集成已完成并通过验证！** 🎉
