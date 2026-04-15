# Local-Global Sparse Language Field 改造计划（fine_v1）

本文档用于在新的对话中快速接手当前工作，目标是在 `LangSplatV2` 基础上实现 **Local-Global Sparse Language Field**，主要聚焦 **静态场景**。

---

## 1. 当前目标

当前目标不是继续做论文阅读，而是进入 **代码改造设计与实现阶段**。

要实现的核心思想：

- 当前 `LangSplatV2` 使用的是 **单一 global codebook**；
- 下一步要把它扩展为 **global codebook + local codebook** 的双分支语言字段；
- 目标是在尽量保持 `LangSplatV2` 推理效率的前提下，提升静态复杂场景中的 **细粒度语义表达、小目标 query、边界质量和局部区分能力**。

建议的最小版本（MVP）不是一步做到最复杂，而是：

- 先做 **Python 主链路可训练、可 eval 的版本**；
- 暂时不优先改极速 quick-render / CUDA；
- 先验证方法有效，再继续下沉到 CUDA 加速路径。

---

## 2. 当前仓库中已有的核心实现方式

### 2.1 当前 LangSplatV2 的语言场结构

当前语言字段主链路是：

- 每个 Gaussian 保存一组 `language_feature_logits`
- 通过 `softmax + top-k` 得到稀疏系数
- 渲染阶段 rasterize 出像素级 weight map
- 再通过 `codebook × weight map` 恢复 512 维 language feature
- 再和文本 embedding 做相似度计算

也就是说，当前不是直接给每个 Gaussian 存最终 512 维特征，而是存：

- 稀疏系数的 logits
- 一套全局 codebook

### 2.2 需要明确的一点

当前的 `vq_layer_num` **不是** 你要做的 `local-global` 双分支。

它更接近 residual / additive VQ 的多层编码，不应直接把 `vq_layer_num=2` 当作 local-global 实现。

---

## 3. 建议的 Local-Global 版本设计

### 3.1 推荐先做的最小设计

推荐先实现下面这个版本：

- 一套 `global logits`
- 一套 `global codebook`
- 一套 `local logits`
- 一套 `local codebook`
- 每个 Gaussian 还要有一个 `local assignment`（说明它属于哪个 local 区域）

建议的局部定义先不要做对象级，而是先做 **空间块级 local region**：

- 按 xyz 空间划分固定网格，或按 scene bounding box 划分 local block
- 每个 Gaussian 归属一个 local block
- 每个 local block 共享自己的 local codebook

这是最容易先跑通的实现方式。

### 3.2 建议的表示方式

可以先从下面两种融合方式里选一种：

#### 方案 A：特征级加权融合（最直观）

对于第 `i` 个 Gaussian：

`f_i = alpha_i * (w_i^g S^g) + (1 - alpha_i) * (w_i^l S^l_{r(i)})`

其中：

- `S^g` 是 global codebook
- `S^l_{r(i)}` 是第 `r(i)` 个 local region 对应的 local codebook
- `w_i^g` 是 global 稀疏系数
- `w_i^l` 是 local 稀疏系数
- `alpha_i` 是 global / local 融合系数，可先设为常数，也可学习

#### 方案 B：先分别渲染，再在像素级融合（更利于后续保留高效路径）

- 分别 rasterize global sparse weight map 和 local sparse weight map
- 分别做 feature reconstruction
- 最后在像素级把 global/local feature map 融合

如果你后续希望保留 `LangSplatV2` 风格的高效 query，**方案 B 更自然**。

### 3.3 我建议的实现顺序

优先建议：

- 第一阶段先用 **方案 A 或 B 的 Python 版本跑通**
- 第二阶段再考虑如何把 global/local 融合压缩进 CUDA 快速路径

---

## 4. 需要改的文件总表

下面按优先级分成三类：

- **第一阶段必须改**：不改就没法完成 Local-Global 主链路
- **第二阶段高概率要改**：为支持完整训练/评估/快速查询而改
- **第三阶段可选改**：为了进一步优化性能、工程组织或扩展实验而改

---

## 5. 第一阶段必须改的文件（先做这些）

### 5.1 `scene/gaussian_model.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/scene/gaussian_model.py`

#### 为什么必须改

这是整个 Gaussian 模型的核心状态容器，当前所有语言字段主参数都定义在这里：

- `_language_feature_logits`
- `_language_feature_codebooks`
- `_language_feature_weights`
- `_language_feature_indices`

此外，它还负责：

- checkpoint 保存 / 恢复
- render 前的 sparse weight 生成
- feature map 重建

#### 你在这个文件里的任务

你需要把当前“单一 global codebook”的参数结构，扩展成 “global + local” 双分支。

#### 建议修改点

1. 新增语言字段参数

建议新增以下成员（命名可调整）：

- `_global_language_feature_logits`
- `_global_language_feature_codebooks`
- `_local_language_feature_logits`
- `_local_language_feature_codebooks`
- `_local_region_ids` 或 `_gaussian_local_assignments`
- 可选：`_global_local_alpha`

2. 修改 `training_setup(...)`

当前这里会初始化：

- `_language_feature_logits`
- `_language_feature_codebooks`

你需要改成：

- 初始化 global 分支
- 初始化 local 分支
- 把它们都加入 optimizer param group

3. 修改 checkpoint 相关逻辑

当前 `capture(...)` / `restore(...)` 只保存单一语言字段。

你需要把以下内容也加进去：

- global logits
- global codebook
- local logits
- local codebook
- local region assignment（如果属于模型状态）

4. 修改稀疏权重导出逻辑

当前 `get_render_weights(...)` 返回的是单一分支的渲染权重。

你需要决定：

- 返回 global 和 local 两套权重
- 或者在这里直接融合后再返回

建议第一阶段先返回两套，避免过早耦合。

5. 修改 feature reconstruction 函数

当前关键函数：

- `compute_layer_feature_map(...)`
- `compute_final_feature_map(...)`

你需要改成支持：

- global codebook 重建
- local codebook 重建
- 二者融合

建议拆成更清晰的函数，例如：

- `compute_global_feature_map(...)`
- `compute_local_feature_map(...)`
- `compute_fused_feature_map(...)`

#### 第一阶段完成标准

这个文件改完后，应当满足：

- 模型能持有 global/local 两套语言字段参数
- checkpoint 能正确保存恢复
- 给定 weight map 后能正确重建 global/local/fused feature map

---

### 5.2 `train.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/train.py`

#### 为什么必须改

这是训练入口。当前这里负责：

- 初始化 codebook
- 调用 `render(...)`
- 获取 GT language feature
- 调用 `gaussians.compute_layer_feature_map(...)`
- 计算 loss
- 保存 checkpoint

#### 你在这个文件里的任务

把训练流程从 “单 codebook 单分支” 改为 “global/local 双分支训练”。

#### 建议修改点

1. 修改初始化逻辑

当前会对单一 codebook 做聚类初始化。

你需要决定：

- global codebook 如何初始化
- local codebook 如何初始化

建议最小版本：

- global codebook：沿用当前聚类初始化策略
- local codebook：对每个 local region 内的 2D 语言特征做局部聚类初始化；如果工程负担大，先随机初始化也可以，但效果可能不稳

2. 增加 local region 构造逻辑

训练时需要知道每个 Gaussian 属于哪个 local region。

建议先在训练启动时构造：

- 根据 Gaussian 的 xyz 或场景包围盒划分网格
- 为每个 Gaussian 分配 local region id

3. 修改 render 后的语言重建路径

当前训练里拿到的是单一语言 feature map。

你需要改成：

- reconstruct global feature map
- reconstruct local feature map
- 融合成 fused feature map
- 用 fused map 与 GT language feature 做监督

4. 修改 loss 设计

第一阶段建议只保留简单的 feature supervision：

- 主损失：`fused_feature_map` vs GT feature

可选地再加辅助损失：

- `global_feature_map` vs GT
- `local_feature_map` vs GT

但建议不要一开始加太多正则，先确保主线可跑。

5. 修改 checkpoint 输出说明

需要保证保存时带有新的 global/local 参数，便于后续 eval 读取。

#### 第一阶段完成标准

- 能正常训练一个 global-local 版本模型
- loss 正常下降
- checkpoint 可保存并恢复

---

### 5.3 `arguments/__init__.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/arguments/__init__.py`

#### 为什么必须改

所有训练/推理参数都在这里定义，当前参数只适配单 codebook。

#### 你在这个文件里的任务

新增 local-global 相关配置项。

#### 建议新增参数

- `global_codebook_size`
- `local_codebook_size`
- `global_topk`
- `local_topk`
- `num_local_regions`
- `local_region_mode`（如 `grid`）
- `global_local_alpha`
- `local_codebook_init_mode`

#### 第一阶段完成标准

- train/eval 都可以通过命令行指定 local-global 配置

---

### 5.4 `gaussian_renderer/__init__.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/gaussian_renderer/__init__.py`

#### 为什么必须改

这里是 Python 侧 render 入口，目前默认只处理单一语言分支的 sparse weights。

#### 你在这个文件里的任务

把当前的 render 接口扩展成支持：

- global weights
- local weights
- 可选的 local assignment 或 local region 索引

#### 建议修改策略

第一阶段建议不要试图在这里一次性改得太复杂。

可以先实现下面任意一种：

1. **分两次 render**
   - 一次渲染 global weights
   - 一次渲染 local weights
   - Python 中融合 feature map

2. **一次 render 输出拼接权重图**
   - 输出 `[global_dim + local_dim]`
   - 后续在 `gaussian_model.py` 中切片重建

对于第一阶段，我更推荐 **方案 1：分两次 render**，因为改动最小、最容易 debug。

#### 第一阶段完成标准

- Python 训练时能够拿到 global / local 两条渲染结果

---

### 5.5 `eval_lerf.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_lerf.py`

#### 为什么必须改

你后面需要看 query 和 segmentation 效果，至少要先打通一个 eval 入口。

在静态场景研究里，建议优先把 `LERF` 路径先改通，而不是同时改三个 benchmark。

#### 你在这个文件里的任务

把当前 eval/query 流程改为支持：

- 加载 global-local checkpoint
- 渲染 global/local feature 或 weight map
- 重建 fused feature map
- 与文本 embedding 计算相似度

#### 第一阶段完成标准

- 能在 `LERF` 上完成一次正常 query
- 能看到 global-local 改造后的可视化结果

---

## 6. 第二阶段高概率要改的文件

这些文件建议在第一阶段主链路跑通后再动。

### 6.1 `utils/vq_utils.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/utils/vq_utils.py`

#### 为什么可能要改

这里实现了：

- `softmax_to_topk_soft_code(...)`
- `get_weights_and_indices(...)`

如果你希望：

- global 和 local 使用不同 top-k
- global 和 local 使用不同归一化方式
- 后续做 adaptive-K

那这里就要改。

#### 任务

- 支持 global/local 双分支 top-k 导出
- 可选支持不同的稀疏规则

---

### 6.2 `eval_3d_ovs.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_3d_ovs.py`

#### 为什么可能要改

当你在 `LERF` 上验证通过后，必须同步到第二个 benchmark 才能形成完整实验。

#### 任务

- 让 3D-OVS 的 eval 路径支持 global-local checkpoint 和 feature reconstruction

---

### 6.3 `eval_mip_nerf360.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_mip_nerf360.py`

#### 为什么可能要改

和 `eval_3d_ovs.py` 一样，是主 benchmark 之一。

#### 任务

- 同步 global-local eval 逻辑

---

### 6.4 `eval/openclip_encoder.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/eval/openclip_encoder.py`

#### 为什么可能要改

如果你以后只是做 fused feature 与 text 的匹配，这里未必需要改。

但如果你想实验：

- 先 global coarse query，再 local refine
- 不同 query 对 global/local feature 的不同加权

那这里会变成 query 策略的关键文件。

#### 任务（可选）

- 支持 coarse-to-fine query
- 支持 global/local 双路相似度融合

---

### 6.5 `train.sh` / `eval_lerf.sh` / `eval_3d_ovs.sh` / `eval_mip_nerf360.sh`

文件：

- `/Users/bytedance/demo/bytedance/work/LangSplatV2/train.sh`
- `/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_lerf.sh`
- `/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_3d_ovs.sh`
- `/Users/bytedance/demo/bytedance/work/LangSplatV2/eval_mip_nerf360.sh`

#### 为什么可能要改

一旦新增参数，这些脚本需要同步透传。

#### 任务

- 增加新的命令行参数
- 组织新的输出目录

---

## 7. 第三阶段再改的文件（主要是高性能路径）

这些文件都跟 **CUDA 加速 / quick render** 强相关。建议在 Python 主链路验证方法有效后再动。

### 7.1 `submodules/efficient-langsplat-rasterization/diff_gaussian_rasterization/__init__.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/diff_gaussian_rasterization/__init__.py`

#### 为什么要改

这是 Python 和 C++/CUDA 的桥接层。一旦 rasterizer 输入输出 tensor shape 变了，这里必须同步。

#### 任务

- 扩展接口以支持 global/local 两路参数或拼接通道

---

### 7.2 `submodules/efficient-langsplat-rasterization/rasterize_points.h`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/rasterize_points.h`

#### 为什么要改

CUDA/C++ 侧接口声明文件；如果渲染输入输出变了，函数签名必须同步。

#### 任务

- 修改前向/后向接口参数定义

---

### 7.3 `submodules/efficient-langsplat-rasterization/rasterize_points.cu`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/rasterize_points.cu`

#### 为什么要改

这里负责 Torch tensor 到底层 rasterizer 的桥接和输出 shape 构造。

#### 任务

- 支持 global/local 两路输出
- 或支持拼接后的更大语言通道数

---

### 7.4 `submodules/efficient-langsplat-rasterization/cuda_rasterizer/config.h`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/cuda_rasterizer/config.h`

#### 为什么要改

当前这里写死：

- `NUM_CHANNELS_language_feature = 64`
- `NUM_CHANNELS_quick_render = 12`

这与单一 global codebook 假设强绑定。做 local-global 后，这个假设大概率会失效。

#### 任务

- 改成适配新通道结构
- 最好避免过度硬编码

---

### 7.5 `submodules/efficient-langsplat-rasterization/cuda_rasterizer/forward.cu`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/cuda_rasterizer/forward.cu`

#### 为什么要改

这里是语言特征 alpha-blending 的核心 CUDA 前向逻辑。

#### 任务

- 定义 global/local 双路系数如何在 CUDA 中累加
- 或定义新的拼接/融合逻辑

---

### 7.6 `submodules/efficient-langsplat-rasterization/cuda_rasterizer/backward.cu`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/cuda_rasterizer/backward.cu`

#### 为什么要改

训练路径改了，梯度回传路径必然要对应修改。

#### 任务

- 让 global/local 分支都能正确回传梯度

---

### 7.7 `submodules/efficient-langsplat-rasterization/cuda_rasterizer/rasterizer.h`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/cuda_rasterizer/rasterizer.h`

#### 为什么要改

前后向接口头文件，必须和实现一致。

#### 任务

- 同步新的参数和数据结构

---

### 7.8 `submodules/efficient-langsplat-rasterization/cuda_rasterizer/rasterizer_impl.cu`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/submodules/efficient-langsplat-rasterization/cuda_rasterizer/rasterizer_impl.cu`

#### 为什么要改

底层调度 forward/backward kernel 的实现文件；改 forward/backward 后它通常也要同步修改。

#### 任务

- 同步新的渲染调用参数

---

## 8. 可以先不改的文件

### 8.1 `preprocess.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/preprocess.py`

#### 为什么暂时可以不改

如果 local 的定义先采用 **3D 空间网格划分**，而不是依赖新的 2D 局部监督，那么预处理阶段可以先不动。

### 8.2 `scene/cameras.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/scene/cameras.py`

#### 为什么暂时可以不改

如果训练监督依然是原来的 GT language feature，而不是额外引入 local supervision，则暂时不需要改。

### 8.3 `utils/loss_utils.py`

文件：`/Users/bytedance/demo/bytedance/work/LangSplatV2/utils/loss_utils.py`

#### 为什么暂时可以不改

第一阶段先复用现有 `cos_loss` / `l1_loss` 即可。

---

## 9. 第一阶段建议的最小实现路线

为了尽快验证你的方法有没有效果，建议严格按下面的顺序推进：

### 第 1 步：只改 Python 主链路

先只动这 5 个文件：

- `scene/gaussian_model.py`
- `train.py`
- `arguments/__init__.py`
- `gaussian_renderer/__init__.py`
- `eval_lerf.py`

目标：

- 能训练
- 能保存 checkpoint
- 能在 `LERF` 上完成 query
- 能输出可视化结果

### 第 2 步：先用最简单的 local 区域定义

不要一开始就上对象级 local codebook。

建议先做：

- 按 xyz 空间固定网格划分 local region
- 每个 Gaussian 分配一个 `local_region_id`

这样最容易 debug。

### 第 3 步：先不用 quick render

第一阶段不要急着改 CUDA 快速路径。

原因：

- `LangSplatV2` 当前 quick path 有较多硬编码
- 一开始就改 CUDA 容易把“方法验证”和“底层工程问题”混在一起

### 第 4 步：先跑一个数据集

建议先只跑 `LERF`。

不要同时追三个 benchmark。

### 第 5 步：确认方法是否有效

优先看：

- 小目标 query 是否更稳定
- 边界是否更干净
- cluttered static scene 是否更好
- overall IoU 是否提升

如果主线有效，再下沉到 CUDA。

---

## 10. 第二阶段建议的增强方向

当第一阶段的 global-local MVP 已经能跑通并有一定效果后，可以考虑继续加下面两类增强：

### 10.1 自适应稀疏度

把固定 top-k 改成：

- global 用一个 `k_g`
- local 用一个 `k_l`
- 或根据 region / Gaussian 的复杂度自适应分配

### 10.2 几何一致性正则

对静态场景，后续可考虑加：

- 邻域 Gaussian 的语义平滑
- 边界保留的几何正则
- 多视角一致性约束

但这些都建议在第一阶段跑通后再做。

---

## 11. 推荐的下一次对话接手方式

在新的对话里，建议直接从下面这句话开始：

> 我已经在 `LangSplatV2` 目录下准备好了 `fine_v1.md`，请按文档中的第一阶段方案，先帮我设计 `scene/gaussian_model.py` 的 local-global 参数结构，并列出需要新增的成员变量、shape 和 checkpoint 改法。

这样可以直接进入实现层面，而不是再重复前面的上下文。

---

## 12. 最后总结：当前真正要做的事情

当前最重要的不是同时修改全部文件，而是分阶段推进：

### 第一阶段（最重要）

- 改 `scene/gaussian_model.py`
- 改 `train.py`
- 改 `arguments/__init__.py`
- 改 `gaussian_renderer/__init__.py`
- 改 `eval_lerf.py`

目标：先得到一个 **能训练、能 query 的 Python 版 Local-Global Sparse Language Field**。

### 第二阶段

- 同步 `eval_3d_ovs.py`、`eval_mip_nerf360.py`
- 视情况改 `utils/vq_utils.py`

### 第三阶段

- 下沉到 `submodules/efficient-langsplat-rasterization/**`
- 恢复或重建高性能 quick render 路径

如果时间有限，**第一阶段一定优先于后两阶段**。

