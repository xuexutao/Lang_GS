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

原因：当前 rasterizer 的语言通道数与 `vq_layer_num * codebook_size` 强绑定（默认 64），单纯 Python 侧无法把 `region` 维度拼进一次 render 的输出。

落地时做了两步取舍：

1) 最早的纯 Python MVP：local 走“按 region 多次 render”（易 debug，但耗时约随 `R=num_local_regions` 线性放大）。

2) 为避免训练耗时爆炸，最终升级为 **CUDA packed 方案1**：一次 render 输出 packed weight map（`global + R*local`），Python 侧切片重建并融合。

备注：`--quick_render` 路径仍保持 BASE 输出长度，不做 local-global 融合（见 2.6）。

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

### 2.5 `scene/cameras.py`

为减少训练每 iter 的额外开销，对 `Camera.get_language_feature(...)` 做了两点优化：

- 使用 `np.load(..., mmap_mode='r')` 做懒加载 + OS page cache，避免每次完整读入 npy。
- 去掉 `torch.meshgrid` 与大规模高级索引，改为直接用 `seg_map[level]` 在 GPU 上 gather 对应的 512 维特征。

这不改变数值语义（仍对 `seg==-1` 的像素做 mask）。

### 2.5 `eval_lerf.py`

- `render_language_feature_map(...)` 改为 fused feature map（一次 render 的 packed weight map 切片重建）

### 2.7 性能与排查（训练变慢的常见原因）

当启用 local-global（尤其是 packed 方案1）后，训练每 iter 变慢通常来自三部分叠加：

- packed 渲染输出通道数从 `64` 变为 `64*(1+R)`，例如 `R=8` 时是 `576` 通道，rasterizer 前向/反向的带宽与写出显著增加。
- local 重建在 Python 侧是按 region 做 `512×64 @ 64×(H·W)` 的矩阵乘，默认 `R=8` 会额外做 8 次。
- GT language feature 如果每 iter 都从磁盘读取 npy，会产生明显 IO 抖动（已在 2.5 中优化）。

为方便快速定位耗时热点，在 `train.py` 增加了可开关的分段计时：

- `--time_breakdown`：开启分段计时打印
- `--time_breakdown_first N`：前 N 次迭代都打印（用于 warmup/首屏定位）
- `--time_breakdown_every N`：之后每 N 次迭代打印一次

同时，针对 local 分支计算量，提供两组“降耗时旋钮”（无需 CUDA 改动）：

- `--local_region_sample_num K`：每次只计算 K 个非空 region（并做期望尺度修正），推荐从 `K=2` 起。
- `--local_render_interval N`：每 N 次迭代才计算一次 local，其余迭代退化为 global-only（`alpha_eff=1.0`），推荐从 `N=2/4` 起。

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
  - `scene/cameras.py`
  - `eval_lerf.py`
- 由于当前机器缺少 CUDA（`CUDA_HOME` 未配置），`diff_gaussian_rasterization` 的 `_C` 扩展无法在此环境编译，因此无法在本机完成端到端渲染训练/评估跑通。

---

## 4. 使用说明（建议）

在具备可用 CUDA + 已编译 rasterizer 的环境下：

### 4.1 编译扩展（必须）

在你的 Python/conda 环境中（保证 `torch` 是 CUDA 版本，且 `nvcc` 可用）：

- `pip install -v -e submodules/efficient-langsplat-rasterization`
- `pip install -v -e submodules/simple-knn`

快速自检：

- `python -c "import diff_gaussian_rasterization, simple_knn; print('ext ok')"`

### 4.2 训练（CUDA packed 方案1）

关键约束：`--num_local_regions <= MAX_LOCAL_REGIONS`，默认 `MAX_LOCAL_REGIONS=8`。

示例（单层 feature_level=1，带耗时分解与降耗时参数）：

- `python train.py -s <DATASET_ROOT>/<SCENE> -m output/<SCENE>_<IDX> --start_checkpoint <RGB_CKPT> --feature_level 1 --vq_layer_num 1 --codebook_size 64 --cos_loss --global_topk 4 --local_topk 4 --num_local_regions 8 --global_local_alpha 0.5 --local_region_mode grid --local_codebook_init_mode copy_global --time_breakdown --time_breakdown_first 5 --time_breakdown_every 50 --local_region_sample_num 2 --local_render_interval 1`

三层训练可直接复用 `train.sh`，只需追加 local-global 参数即可。we

### 4.3 LERF 评估

第一阶段建议先不用 `--quick_render`（quick 路径尚未做 local-global 融合）。

- `python eval_lerf.py -s <DATASET_ROOT>/lerf_ovs/<SCENE> -m output/<SCENE>_<IDX>_1 --dataset_name <SCENE> --index <IDX> --ckpt_root_path ./output --output_dir ./eval_result --mask_thresh 0.4 --json_folder <GT_LABEL_ROOT> --checkpoint 10000 --include_feature --global_topk 4 --local_topk 4 --num_local_regions 8 --global_local_alpha 0.5`

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
