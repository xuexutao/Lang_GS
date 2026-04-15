# fine_v1 落地记录（fine_v1_true）

本文档记录我**按照 `fine_v1.md` 第一阶段方案**在 `LangSplatV2` 中实际做的代码改动点（以“能训练、能保存、能在 LERF 跑 query 的 Python 主链路”为目标），并解释关键取舍。

---

## 1. 已落地的核心创新点

### 1.1 Local-Global 双分支语言字段（MVP）

- 在 Gaussian 模型中新增 **global 分支 + local 分支** 两套 logits/codebook。
- local 分支按 **3D 空间网格 local region** 共享 codebook（每个 region 一套 codebook）。
- 训练/评估阶段使用 **fused feature** 做监督与 query：
  - `fused = alpha * global + (1 - alpha) * local_sum`
  - `alpha` 由参数 `global_local_alpha` 控制（MVP 先用常数）。

### 1.2 MVP 渲染取舍：local 采用“按 region 多次 render”

原因：当前 rasterizer 的语言通道数与 `vq_layer_num * codebook_size` 强绑定（默认 64），不改 CUDA 无法直接把 `region` 维度拼进通道。

因此第一阶段采用：

- global：一次 render 得到 `global_weight_map`，再用 global codebook 重建
- local：对每个 local region `rid` 取该 region 的 Gaussian 子集，**多次 render** 得到 `local_weight_map[rid]`，再用 `local_codebook[rid]` 重建并累加
- 最后做 fused

这符合 `fine_v1.md` “先跑通 Python 主链路、暂不改 CUDA quick path” 的策略。

---

## 2. 修改的文件与关键变更

### 2.1 `scene/gaussian_model.py`

- 新增成员（local-global）：
  - `_global_language_feature_logits`
  - `_global_language_feature_codebooks`
  - `_local_language_feature_logits`
  - `_local_language_feature_codebooks`（shape: `[R, L, K, 512]`）
  - `_local_region_ids`（shape: `[N]`）
- 新增 local region 分配：`compute_local_region_ids(num_local_regions, mode='grid')`
  - `num_local_regions` 为立方数时使用 3D 均匀网格；否则退化为沿 x 的 1D 分桶
- 新增重建函数：
  - `compute_global_layer_feature_map(...)`
  - `compute_local_layer_feature_map(..., region_id)`
  - `compute_global_final_feature_map(...)`
  - `compute_local_final_feature_map(..., region_id)`
- `training_setup(...)`：初始化并把 global/local 两套参数加入同一 optimizer group
- checkpoint：
  - `capture(include_feature=True)` 输出新的 17-tuple（包含 global/local 参数与 `_local_region_ids`）
  - `restore(...)` 兼容旧 14-tuple（自动映射到 global 分支），并兼容无 feature 的 12-tuple
- 兼容性：保留 `_language_feature_logits/_language_feature_codebooks` 作为 global 分支别名，避免其它 eval/quick 脚本立刻崩
- 额外健壮性：`simple_knn` CUDA 扩展不可用时，`create_from_pcd` 使用 `torch.cdist` 作为慢速 fallback

### 2.2 `gaussian_renderer/__init__.py`

- render 接口扩展（保持默认行为不变）：
  - 新增参数 `language_branch='global'|'local'`
  - 新增参数 `gaussian_mask`（用于 local 的 per-region 子集渲染）
- `include_feature=True` 时根据 `language_branch` 调用 `pc.get_render_weights(topk, branch=...)`
  - `topk` 使用 `global_topk/local_topk`（无则回退 `topk`）
- 增加 sys.path fallback：允许不 `pip install -e submodules/...` 时也能从仓库内找到 `diff_gaussian_rasterization`
  - 若底层 `_C` 未编译，会抛出更明确的 ImportError 指引

### 2.3 `arguments/__init__.py`

在 `OptimizationParams` 中新增第一阶段参数（含默认值）：

- `global_codebook_size=64`
- `local_codebook_size=64`
- `global_topk=1`
- `local_topk=1`
- `num_local_regions=8`
- `local_region_mode='grid'`
- `global_local_alpha=0.5`
- `local_codebook_init_mode='copy_global'`（稳定起步；也支持 `random`）

注：为避免 CUDA 改动，MVP 强制 `global_codebook_size/local_codebook_size` 与原 `codebook_size` 一致。

### 2.4 `train.py`

- feature 训练阶段（已升级为 CUDA 方案1的一次 render）：
  - 只做一次 render 输出 packed weight map（global + R*local），不再 per-region 多次 render
  - Python 侧对 packed weight map 做切片重建 global/local，再融合成 fused：`alpha * global + (1 - alpha) * local`
- codebook 初始化：
  - global codebook：沿用原 `ResidualVectorQuantizationWithClustering` 初始化
  - local codebook：默认 `copy_global` 到每个 region（或 `random` 不处理）
- `--topk` 兼容：若用户只提供 `--topk`，会自动同步到 `global_topk/local_topk`

### 2.5 `eval_lerf.py`

- `render_language_feature_map(...)` 改为 fused feature map（一次 render 的 packed weight map 切片重建）

### 2.6 CUDA 方案1（关键提速改动）

- `submodules/efficient-langsplat-rasterization/cuda_rasterizer/config.h`：
  - 新增 `MAX_LOCAL_REGIONS`（默认 8）
  - 区分 `NUM_CHANNELS_language_feature_BASE=64` 与 `NUM_CHANNELS_language_feature_PACKED=64*(1+MAX_LOCAL_REGIONS)`
  - quick_render 仍使用 BASE（输出 3*64），include_feature 使用 PACKED
- `submodules/efficient-langsplat-rasterization/cuda_rasterizer/forward.cu`：
  - kernel 模板参数拆分 BASE vs PACKED
  - quick_render 的 WT 长度保持 `3*BASE`
  - include_feature 的 F 长度改为 `PACKED`
- `submodules/efficient-langsplat-rasterization/rasterize_points.cu`：
  - include_feature 输出改为 `[PACKED, H, W]`
  - backward 的 `dL_dlanguage_feature` 改为 `[P, PACKED]`
- `submodules/efficient-langsplat-rasterization/cuda_rasterizer/backward.cu`：
  - backward kernel 的 F 维度改为 `PACKED`

使用约束：运行时 `--num_local_regions <= MAX_LOCAL_REGIONS`，否则需要改 `MAX_LOCAL_REGIONS` 重新编译 rasterizer。

---

## 3. 验证情况（本机环境）

- 已通过 `python -m py_compile` 对以下文件的语法检查：
  - `scene/gaussian_model.py`
  - `gaussian_renderer/__init__.py`
  - `arguments/__init__.py`
  - `train.py`
  - `eval_lerf.py`
- 由于当前机器缺少 CUDA（`CUDA_HOME` 未配置），`diff_gaussian_rasterization` 的 `_C` 扩展无法在此环境编译，因此无法在本机完成端到端渲染训练/评估跑通。

---

## 4. 使用说明（建议）

在具备可用 CUDA + 已编译 rasterizer 的环境下：

- 训练（feature 训练，基于已有 checkpoint 的流程不变）：
  - 通过参数控制：`--num_local_regions`、`--global_local_alpha`、`--global_topk`、`--local_topk`
- 训练提速（推荐，避免 per-region 全量渲染带来的倍数开销）：
  - `--local_region_sample_num <K>`：每次迭代只随机渲染 K 个非空 region（其余 region 通过跨迭代覆盖），并做期望尺度修正
    - `K<0` 表示全量（默认行为）
    - `K=0` 表示关闭 local 分支（退化为 global-only）
    - 推荐从 `K=1/2` 开始
  - `--local_render_interval <N>`：每 N 次迭代才计算一次 local 分支，其余迭代只用 global 分支训练（`N=1` 表示每次都算）
    - 推荐从 `N=2/4` 开始
- LERF 评估：
  - 使用训练输出目录的 `cfg_args` 会自动携带 local-global 配置
