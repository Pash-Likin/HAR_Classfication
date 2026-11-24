# 跨人员人体活动识别（HAR）：仅 PCA 降维与五分类器对比实验

本项目基于 UCI Human Activity Recognition Using Smartphones（HAR）官方 **561 维时频域特征**，开展**仅使用 PCA 降维**的跨人员人体活动识别对比研究。主运行脚本版本为 `pca-subject-only-highest-cv-2.2`（核心包 `har_study` 版本为 `2.0.0`）。

核心设计要点：

- **两类特征路线**：保留 561 维全特征基准（`full`）与仅 PCA 降维（`pca`，候选维数 `32 / 64 / 96 / 128`），不包含 ANOVA 特征筛选及从原始传感器序列重提取特征的步骤。
- **五种分类模型**：`RBF_SVM`、`RandomForest`、`LogisticRegression`、`KNN` 和 `FisherLDA`（`lsqr` 求解器，仅作为分类器）。
- **人员隔离验证**：训练集 5 折交叉验证严格采用 `StratifiedGroupKFold` 按人员分组（`subject_grouped_only`），不运行随机窗口对比协议（`random_window_comparison=false`）。
- **最高分选维规则**：按调参后平均训练人员分组交叉验证 Macro-F1 最高的维数选择 PCA 维度及超参数（`pca_selection="highest_cv_macro_f1"`），完全同分时优先较小维数。
- **一体化流水线**：集成断点续跑网格搜索、测试前选择锁定（`selection_lock.json`）、最终模型分阶段训练计时、固定超参数控制变量拟合耗时分析、测试评估与人员配对 Bootstrap、整体及多批量分阶段推理计时、全链路一致性核验与 300 DPI 中文图表绘制。

> **仓库文件范围说明**：根据 [`.gitignore`](.gitignore) 配置，原始数据集 `data/UCI HAR Dataset/` 与实验运行输出目录 `outputs/` 不纳入 Git 版本控制（数据集准备方式见 [data/README.md](data/README.md)）；[`validation/`](validation/README.md) 目录保留早期的历史验证归档记录。请在项目根目录（与 [`run_experiment.py`](run_experiment.py)、[`config.json`](config.json)、[`src/`](src) 同级）下执行所有命令。

---

## 1. 项目结构

| 路径 | 说明 |
| --- | --- |
| [`run_experiment.py`](run_experiment.py) | 主实验入口：支持 `--plan`、`--data`、`--out`、`--workers`、`--font`、`--resume`、`--verify-only` |
| [`verify_results.py`](verify_results.py) | 独立复算指标与文件一致性核验入口（`--run`、`--data`） |
| [`redraw_figures.py`](redraw_figures.py) | 基于已完成运行目录的 CSV/JSON 重绘全部中文图表（不重训） |
| [`measure_stages.py`](measure_stages.py) | 对已完成并锁定的 10 条模型单独补测/重测多批量推理阶段耗时并重绘阶段图 |
| [`predict.py`](predict.py) | 加载已保存的 `.joblib` Pipeline 对无表头、空格分隔的 561 维特征文本进行离线预测 |
| [`config.json`](config.json) | 主实验配置：种子、折数、PCA 维数、五分类器参数网格、整体计时及 Bootstrap 重复次数 |
| [`timing_config.json`](timing_config.json) | 多批量分阶段推理计时配置：`batch_sizes=[1, 32, 256, 2947]`、`repeats=15`、`warmups=3`、`seed=42` |
| [`requirements.txt`](requirements.txt) | 固定版本的 Python 依赖清单 |
| [`src/har_study/data.py`](src/har_study/data.py) | 官方数据目录/压缩包解析、安全解压、官网下载（失败不回退合成数据）与划分审计 |
| [`src/har_study/study.py`](src/har_study/study.py) | 配置校验、Pipeline 与分类器构建、人员分组折、选模、分阶段训练保存、控制变量耗时、测试评估、Bootstrap 与整体推理基准 |
| [`src/har_study/search.py`](src/har_study/search.py) | 折内共享 `StandardScaler` 与完整 SVD `PCA` 预处理的精确网格搜索模块 |
| [`src/har_study/audit.py`](src/har_study/audit.py) | 生成综合决策对照摘要表（`decision_summary.csv`） |
| [`src/har_study/diagnostics.py`](src/har_study/diagnostics.py) | 固定参数对照与测试后描述性资源权衡汇总（`resource_tradeoff.csv`） |
| [`src/har_study/stage_timing.py`](src/har_study/stage_timing.py) | 多批量分阶段推理计时、分阶段与完整管线预测零差异校验及阶段耗时绘图 |
| [`src/har_study/plots.py`](src/har_study/plots.py) | 本机中文字体检测、画布文字防越界检查与 42 张主实验图表生成 |
| [`src/har_study/verify.py`](src/har_study/verify.py) | 验证人员隔离、折分配、CV 均值/标准差、选模规则、选择锁、事件顺序、测试指标及模型哈希 |
| [`tests/`](tests) | 基于小型合成夹具的单元测试套件（含可选真实训练折核验脚本 [`tests/check_real_fisher.py`](tests/check_real_fisher.py)） |
| [`data/README.md`](data/README.md) | UCI HAR 官方数据集目录结构与引用信息说明 |
| [`validation/`](validation/README.md) | 历史结构化验证记录归档 |

---

## 2. 实验设计与协议

### 2.1 输入特征与管线路线

程序直接读取 UCI HAR 官方提供的 561 维特征向量，并进行严格规模与人员隔离检查（训练集 `21` 名受试者、共 `7352` 个时间窗口；测试集 `9` 名互不重叠受试者、共 `2947` 个时间窗口；涵盖 6 类活动：`WALKING`、`WALKING_UPSTAIRS`、`WALKING_DOWNSTAIRS`、`SITTING`、`STANDING`、`LAYING`）：

- **全特征基准（`full`）**：`561 维特征 → StandardScaler → 分类器`（管线中间步骤 `reduce="passthrough"`）。
- **PCA 降维路线（`pca`）**：`561 维特征 → StandardScaler → PCA(n_components=k, svd_solver="full", whiten=False) → 分类器`，候选维数为 `32、64、96、128`。
- **FisherLDA 定位**：`LinearDiscriminantAnalysis(solver="lsqr", priors=None)` 仅作为末端分类器使用，不作为替代 PCA 的降维器。

### 2.2 人员分组验证与维数联合选择

1. **严格人员隔离**：训练集按 `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` 划分 5 折，确保同一受试者不同窗口不会同时出现在训练折与验证折，且每折训练/验证集均完整覆盖 6 个活动类别。本版本不执行随机窗口切分对比。
2. **折内共享预处理**：在每一折的训练集上单独拟合 `StandardScaler` 与最大维数 `128` 的完整 SVD `PCA`（固定 SVD 解下，截取前 `k` 列与单独拟合 `k` 维等价，无随机近似）。验证集仅做 `.transform()`，测试集在搜索阶段完全不加载。
3. **无偏差联合选维**：四种 PCA 维数与全特征各自遍历相同的分类器参数网格（总计 `160` 组候选 × `5` 折 = `800` 次独立分类器拟合；`ConvergenceWarning` 视为错误）。每个维数先选出自身网格内交叉验证 Macro-F1 最优的参数组合，再比较 `32 / 64 / 96 / 128` 四个维数的最优得分（以未四舍五入的浮点值比较），选中平均训练人员分组 CV Macro-F1 最高的维数及其参数。若出现精确平局，优先选择较小维数，其次按 `candidate_id` 字典序稳定选择。
4. **如实记录与测试前锁定**：即使某模型所有 PCA 维数的 CV 分数均低于全特征基准，仍按上述规则选出最佳 PCA 维数并如实记录差值（`cv_delta_from_full_pp`）。`fixed_parameter_comparison.csv`、`pca_dimension_comparison.csv` 与 `old_vs_extended_cv.csv` 均从同一次网格搜索直接提取，用于开发期描述，不构成独立测试收益。锁定结果在读取测试集前写入 `selection_lock.json` 并固化 SHA-256 哈希。

### 2.3 分类器实现与参数网格

所有模型配置见 [`config.json`](config.json)，`FIXED_PARAMS` 对照定义于 [`run_experiment.py`](run_experiment.py) 与 [`src/har_study/diagnostics.py`](src/har_study/diagnostics.py)：

| 分类器 | 固定实现设置（[`study.py`](src/har_study/study.py)） | 参数搜索网格 | 每表示组合数 | 固定参数对照（`FIXED_PARAMS`） |
| --- | --- | --- | ---: | --- |
| `RBF_SVM` | `SVC(kernel="rbf", probability=False, cache_size=512, random_state=42)` | `C=[1, 10, 100, 1000]`；`gamma=["scale", 0.001]` | 8 | `C=10.0, gamma=0.001` |
| `RandomForest` | `RandomForestClassifier(n_estimators=200, max_features="sqrt", random_state=42, n_jobs=1)` | `max_depth=[None, 10, 20]`；`min_samples_leaf=[1, 3]` | 6 | `max_depth=None, min_samples_leaf=1` |
| `LogisticRegression` | `LogisticRegression(solver="lbfgs", max_iter=8000, tol=1e-4, random_state=42)` | `C=[0.01, 0.1, 1, 10, 100]` | 5 | `C=1.0` |
| `KNN` | `KNeighborsClassifier(p=2, algorithm="brute", n_jobs=1)` | `n_neighbors=[3, 5, 9, 15]`；`weights=["uniform", "distance"]` | 8 | `n_neighbors=5, weights="uniform"` |
| `FisherLDA` | `LinearDiscriminantAnalysis(solver="lsqr", priors=None)` | `shrinkage=[None, "auto", 0.05, 0.2, 0.5]` | 5 | `shrinkage=None` |

**并发与线程限制**：网格搜索通过 `joblib` `loky` 后端在进程级并行（由 `--workers` 控制，默认 `2`），同时显式锁定底层 `OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS`、`MKL_NUM_THREADS`、`NUMEXPR_NUM_THREADS` 及 `threadpool_limits(limits=1)` 与分类器 `n_jobs=1`；最终训练、控制变量计时及全部推理计时均在单线程串行条件下执行。

### 2.4 主流程执行顺序

执行 `run_experiment.py` 时，程序严格按以下顺序运行并向 `events.jsonl` 追加时间戳事件：

1. **冻结上下文与划分训练集**：校验配置与数据哈希，生成 `config_frozen.json`、`timing_config_frozen.json`、`search_plan.json`、`protocol_frozen.json`、`resume_context.json`，并保存 5 折人员划分 `splits/training_cv.csv`。
2. **断点网格搜索**：按折共享标准化与 PCA 缓存，执行 800 次独立拟合，每完成一折即原子保存至 `checkpoints/` 并周期更新 `tables/cv_folds.csv`，最终汇总 `tables/cv_candidates.csv`。
3. **衍生对比与锁定配置**：生成 `fixed_parameter_comparison.csv`、`old_vs_extended_cv.csv`、`boundary_diagnostics.csv`、`compression_curve.csv`、`selected_configs.csv`、`pca_dimension_comparison.csv`，并在读取测试集前生成 `selection_lock.json`。
4. **最终管线拟合与训练阶段计时**：在全部 7352 条训练样本上分别训练 `10` 条最终 Pipeline，实测并记录分阶段训练耗时与模型大小，导出 `models/*.joblib`、`models/*.json`、`final_fit_and_size.csv` 与 `pca_variance.csv`。
5. **固定超参数控制变量计时**：在加载测试集之前，仅基于官方训练集执行 `180` 次控制变量 `classifier.fit()` 计时，输出 `controlled_fit_timing_raw.csv`、`controlled_fit_timing_summary.csv` 与 `controlled_fit_protocol.json`。
6. **加载与评估测试集**：读取官方测试集（`data_audit.json`、`splits/official_test.csv`），执行测试评估（`test_metrics.csv`、`test_subjects.csv`、`test_activities.csv`、`test_predictions.csv`、`confusion_*.csv`）及 2000 次人员配对 Bootstrap（`subject_bootstrap.csv`）。
7. **推理基准与分阶段分解**：执行整体推理基准（`timing_raw.csv`、`timing_summary.csv`）、决策/帕累托摘要（`decision_summary.csv`、`resource_tradeoff.csv`）及多批量分阶段推理分解（`stage_timing_*.csv`、`PCA_vs_full_with_timing_breakdown.csv`）。
8. **清理临时缓存、全局核验与绘图**：移除 `.extended_cache`，依次执行 `verify_extended`、`verify_stage_timing`、`draw_all`（42 张）与 `draw_stage_figures`（10 张），输出 `verification.json`、`visualization_manifest.json` 与 `stage_figure_manifest.json`。

---

## 3. 训练耗时分解与控制变量实验

### 3.1 双栏训练耗时对比（`figures/training_cost/combined_{model}.png`）

每个模型生成一张左右双栏分组柱状图（共 5 张），两栏使用**独立的纵轴刻度**，数值不可直接相加：

- **左栏 ①（五折 CV 调参与验证累计耗时）**：汇总 `full` 路线与该模型最终选定 PCA 维数（`chosen_k`）在 5 折交叉验证中的所有候选参数拟合累计耗时（`classifier_fit_seconds`）、验证预测累计耗时（`classifier_predict_seconds`）及两项合计。其余未被选中的 3 个 PCA 维数仍正常参与网格搜索与选维，但不计入该图展示；左栏耗时**不含**折内共享标准化/PCA 预处理时间与进程调度开销，不代表程序整体墙钟时间。
- **右栏 ②（最终单次训练分阶段耗时）**：在 `fit_selected()` 对全部 `7352` 条训练样本拟合最终管线时直接拆解并记录 `scale_fit_seconds`（标准化）、`reduce_fit_seconds`（PCA 拟合与变换，`full` 路线为 `0`）、`classifier_fit_seconds`（分类器拟合）以及完整 `final_fit_seconds`（已涵盖上述全部步骤耗时，勿再将各分项与完整耗时重复相加）。
- 若对缺失最终分阶段训练字段的旧目录执行重绘，`plots.py` 会跳过右栏并退化生成单栏 `figures/training_cost/cv_{model}.png`。

### 3.2 固定超参数控制变量拟合实验（`controlled_fit_costs`）

该实验在最终管线拟合之后、加载官方测试集之前自动执行（仅使用官方训练集，不修改锁定的 10 条模型及测试预测结果，也不计入 `--plan` 候选统计）：

- **固定超参数**：每个分类器始终固定使用本轮 CV 选出的**全特征（`full`）最优超参数**，不在不同子集或维度下重新调参。
- **样本量控制变量（`sample_size` → `control_sample_size_{model}.png`）**：以 `seed=42` 按 `(subject, activity)` 分层近似比例随机排序，在 `21` 名训练人员、`7352` 条样本中构造严格嵌套的子集：`25%`（`1838` 条）、`50%`（`3676` 条）、`75%`（`5514` 条）、`100%`（`7352` 条）。每个子集内部独立重新拟合 `StandardScaler` 与最大候选维数 `PCA`（默认 `128` 维），对比 `full`（`561` 维）与 `pca`（`128` 维）的分类器拟合耗时。
- **维数控制变量（`input_dimension` → `control_dimension_{model}.png`）**：固定全部 `7352` 条训练样本，先在标准化后的全量训练集上拟合一次 `128` 维完整 SVD `PCA`，依次截取前 `32 / 64 / 96 / 128` 个主成分（同一 PCA 基底）作为分类器输入测量 `fit()` 耗时，并以 `100%` 样本 `full`（`561` 维）的中位拟合耗时作为不同特征基底的虚线参考。
- **计时范围与重复次数**：单线程下仅计时 `classifier.fit()`，不包含样本准备、标准化、PCA 拟合/变换、CV 与预测；每组条件每次重新实例化分类器，正式执行 `3` 轮且每轮随机打乱全部 `60` 个任务的执行顺序，总计 `5 模型 × (4 样本比例 × 2 表示 + 4 PCA 维数) × 3 轮 = 180` 次拟合，输出中位数与第 25/75 四分位数。
- **解释边界**：仅用于分析训练计算开销，不能由拟合时间曲线直接推断分类准确率变化；对于 `RandomForest`（`max_features="sqrt"`）与 `RBF_SVM`（`gamma="scale"`）等有效参数依赖特征维数或表示尺度的模型，不可将耗时变化单一归因于维数本身。

---

## 4. 推理耗时评估与分阶段耗时分解

1. **整体推理基准（[`study.py`](src/har_study/study.py) `benchmark`，参数源自 [`config.json`](config.json)）**：
   - 在单线程下对已锁定的 10 条完整 Pipeline 预热 `3` 轮后，随机交错模型顺序正式测量 `15` 轮，分别统计全测试集批量预测（`batch`，`2947` 个窗口）与单窗口逐条调用（`single`，从测试集均匀等距抽取 `64` 个窗口循环调用 `predict`，包含 Python 循环调度开销）的每样本耗时中位数与 Q25/Q75。
2. **多批量分阶段推理分解（[`stage_timing.py`](src/har_study/stage_timing.py)，参数源自 [`timing_config.json`](timing_config.json)）**：
   - 单线程串行测量 `batch_sizes = [1, 32, 256, 2947]`，每种批量等距取测试样本（`B=1` 对应测试集第 0 行固定窗口），预热 `3` 轮并随机交错模型与阶段执行 `15` 轮，分别记录 `scale`、`reduce`（`full` 路线记为结构性 `0` 且 `measured=False`）、`classifier` 与完整管线 `whole` 的纳秒级耗时（共 `2400` 条原始记录、`160` 行统计汇总）。
   - 同时核验各阶段拆解串联预测与完整 `Pipeline.predict` 在全部批量和模型上的 `32360` 次输出零差异（`tables/stage_prediction_parity.csv`）。
3. **计时边界**：两种推理计时均不涵盖传感器采样、原始信号 561 维特征构造、磁盘 I/O、`.joblib` 模型反序列化及设备功耗；分阶段计时的各阶段输入在计时前已预先准备，独立测量的各阶段中位数之和不要求严格等于 `whole` 的中位数。

---

## 5. 环境准备与运行命令

### 5.1 安装依赖

项目原生依赖版本记录于 [`requirements.txt`](requirements.txt)：

```text
numpy==2.3.5
scipy==1.17.0
pandas==2.2.3
scikit-learn==1.8.0
matplotlib==3.10.8
joblib==1.5.3
threadpoolctl==3.6.0
```

在项目根目录执行：

```powershell
python -m pip install -r requirements.txt
```

> **版本约束**：[`predict.py`](predict.py) 与 [`measure_stages.py`](measure_stages.py) 在反序列化 `.joblib` 模型时会检查 `scikit-learn` 版本是否与训练保存时完全一致，请保持相同环境版本。

### 5.2 查看搜索计划

```powershell
python run_experiment.py --plan
```

该命令不加载数据或执行训练，固定配置应输出如下规模：

| 计划项目 | 数量 | 说明 |
| --- | ---: | --- |
| 每模型表示方案 | 5 | 1 个全特征（561 维）+ 4 个 PCA（32 / 64 / 96 / 128 维） |
| 主搜索候选总数 | 160 | `(8 + 6 + 5 + 8 + 5) × 5` |
| 主 5 折 CV 分类器拟合 | 800 | `160 × 5`（仅人员分组验证，无随机窗口对照） |
| 最终锁定模型 | 10 | 5 个分类器 × (`full` + `pca`) |
| 正式运行时额外控制变量拟合 | 180 | 不计入 `--plan`，测试前自动在训练集上单线程执行 |

### 5.3 运行完整实验

- **数据查找优先级**：传入 `--data`（支持已解压目录或官方 ZIP 包）时直接使用指定数据；省略 `--data` 时优先检索 [`data/`](data/README.md) 目录；若本地不存在则尝试从 UCI 官网下载（[`data.py`](src/har_study/data.py)），下载失败立即停止，绝不使用合成数据替代正式实验。
- **输出目录规则**：通过 `--out` 指定全新或空输出目录（省略时默认为 `outputs/YYYYMMDD`）；若目标目录非空，程序拒绝覆盖并报错退出。

```powershell
# 若已将数据放在 data/UCI HAR Dataset/ 下，可省略 --data
python run_experiment.py --out outputs/pca_run --workers 2

# 或显式指定本地数据目录 / 官方 ZIP，并可选指定中文字体
python run_experiment.py --data "D:\path\to\UCI HAR Dataset" --out outputs/pca_run --workers 2
python run_experiment.py --data "D:\path\to\UCI HAR Dataset.zip" --out outputs/pca_run --workers 2 --font "C:\Windows\Fonts\msyh.ttc"
```

### 5.4 中断恢复

```powershell
python run_experiment.py --resume outputs/pca_run
```

数据未放在 [`data/`](data/README.md) 目录时需同步传入 `--data`。恢复时会校验 `resume_context.json` 中的 `context_sha256`（涵盖脚本版本号、`config.json`、`timing_config.json`、搜索计划、数据文件哈希、源码哈希、依赖版本与运行平台），任意一项发生变化均拒绝混用断点；对已标为 `completed` 的目录执行 `--resume` 仅触发验证与绘图，不重复训练。

### 5.5 核验、重绘、补测阶段耗时与离线预测

```powershell
# 1. 复核已完成目录（含主实验与分阶段计时检查）并重绘全部中文图表
python run_experiment.py --verify-only outputs/pca_run --font "C:\Windows\Fonts\msyh.ttc"

# 2. 独立脚本复算核验（仅核验，不重绘）
python verify_results.py --run outputs/pca_run

# 3. 仅根据已存在的 CSV 重绘全部中文图表（不重训，不会补跑未执行的控制变量/阶段计时）
python redraw_figures.py --run outputs/pca_run --font "C:\Windows\Fonts\msyh.ttc"

# 4. 对已锁定模型单独补测/重测多批量推理阶段耗时并重绘 figures/stage_timing/
python measure_stages.py --run outputs/pca_run --font "C:\Windows\Fonts\msyh.ttc"

# 5. 使用已保存的 Pipeline 对外部 561 维特征文本进行离线预测
python predict.py --model outputs/pca_run/models/FisherLDA__pca.joblib --input features_561.txt --output predictions.csv
```

> **图表字体说明**：[`plots.py`](src/har_study/plots.py) 会自动依次查找 `Microsoft YaHei`、`SimHei`、`PingFang SC`、`Noto Sans CJK SC`、`Source Han Sans SC`、`Noto Sans CJK JP`、`WenQuanYi Micro Hei` 并校验中文字形完整性；若未命中或需指定特定字体，请通过 `--font` 传入本机字体路径。`--verify-only` 下若绘图失败会写入 `extended_figure_error.json`，已生成的数值结果依然有效。  
> **离线预测输入与安全要求**：[`predict.py`](predict.py) 要求输入为**无表头、空格分隔的 561 列有限数值文本**，且列顺序与官方 `features.txt` 一致（不支持手机原始三轴传感器时序信号）；由于 `joblib` 反序列化可能执行代码，仅加载自己训练或确认来源可信的模型文件。

---

## 6. 输出目录与主要产物

一次完整运行会在输出目录（例如 `outputs/pca_run/`）下生成以下主要文件与子目录：

### 6.1 元数据、审计与锁定文件

- `config_frozen.json`、`timing_config_frozen.json`、`search_plan.json`、`protocol_frozen.json`、`source_hashes.json`、`resume_context.json`：实验启动时冻结的配置、源码与协议记录
- `data_audit.json`、`fold_transform_audit.json`、`search_engine.json`、`events.jsonl`、`manifest.json`：数据/折变换审计、搜索引擎信息、完整过程事件流与运行清单
- `selection_lock.json`：读取测试集之前生成的 10 条模型配置锁及校验哈希
- `controlled_fit_protocol.json`、`timing_environment.json`、`stage_timing_environment.json`：控制变量计时与推理计时的环境及实验边界记录
- `verification.json`、`stage_timing_verification.json`、`visualization_manifest.json`、`stage_figure_manifest.json`：复算核验报告与图表清单

### 6.2 模型、划分与结果表

- `splits/`：`training_cv.csv`（5 折人员分组样本分配）、`official_test.csv`（官方测试集顺序与标签）
- `models/`：10 组最终模型文件（`<Model>__full.joblib`、`<Model>__pca.joblib`）及对应元数据 `<selection_id>.json`
- `tables/` 核心表格：
  - **搜索与选模**：`cv_candidates.csv`（160 候选汇总）、`cv_folds.csv`（800 条逐折记录）、`selected_configs.csv`（最终 10 条管线及规则）、`compression_curve.csv`、`pca_dimension_comparison.csv`（5 模型 × 4 维度，20 行）、`fixed_parameter_comparison.csv`、`old_vs_extended_cv.csv`、`boundary_diagnostics.csv`
  - **训练、方差与控制变量**：`final_fit_and_size.csv`（最终训练各阶段秒数、模型字节数与 SVM 支持向量数）、`pca_variance.csv`（主成分解释方差比）、`controlled_fit_timing_raw.csv`（180 次拟合明细）、`controlled_fit_timing_summary.csv`（中位数与 Q25/Q75）
  - **测试评价与推理计时**：`test_metrics.csv`、`test_subjects.csv`、`test_activities.csv`、`test_predictions.csv`、`confusion_<selection_id>.csv`（10 份）、`subject_bootstrap.csv`、`decision_summary.csv`、`resource_tradeoff.csv`、`timing_raw.csv`、`timing_summary.csv`、`stage_timing_raw.csv`（2400 条）、`stage_timing_summary.csv`（160 行）、`stage_timing_inputs.csv`、`stage_prediction_parity.csv`、`PCA_vs_full_with_timing_breakdown.csv`

### 6.3 生成图表（共计 52 张，300 DPI）

| 图表目录 | 数量 | 内容说明 |
| --- | ---: | --- |
| `figures/overview/` | 7 | `01_activity_distribution.png`（类别分布）、`02_subject_grouped_cv.png`（人员分组 CV 对比）、`03_test_metrics.png`（测试准确率与 Macro-F1）、`04_subject_heatmap.png`（逐受试者热力图）、`05_timing_batch.png`、`05_timing_single.png`、`06_model_size.png` |
| `figures/compression/` | 5 | 各模型在 32 / 64 / 96 / 128 维下的调参后 CV 曲线及最高分维度星标（`<Model>.png`） |
| `figures/confusion/` | 5 | 各模型全特征与 PCA 双联测试混淆矩阵，统一使用 `Blues` 原始计数色标（`confusion_<Model>.png`） |
| `figures/parameters/` | 10 | 最终 10 个方案对应的网格搜索参数 CV Macro-F1 热力图（`parameters_<selection_id>.png`） |
| `figures/training_cost/` | 15 | 5 张双栏训练耗时图（`combined_<Model>.png`）、5 张样本量控制拟合耗时图（`control_sample_size_<Model>.png`）、5 张 PCA 维数控制拟合耗时图（`control_dimension_<Model>.png`） |
| `figures/stage_timing/` | 10 | 5 张最大批量下推理阶段分解柱状图（`<Model>_stage_timing.png`）与 5 张多批量缩放对数曲线（`<Model>_batch_size_comparison.png`） |

---

## 7. 软件测试、结果边界与参考来源

### 7.1 单元测试与可选训练折核验

```powershell
# 运行全部软件单元测试（使用小型合成夹具，无需官方数据集）
python -m unittest discover -s tests -v

# 可选：在真实 UCI HAR 训练集第 0 折上核验折内缓存与完整 Pipeline 的 FisherLDA 预测一致性
python tests/check_real_fisher.py
```

单元测试仅用于验证软件接口、折内隔离、选维逻辑、计时计算与画布检查，**不提供**真实 HAR 性能证据；`tests/check_real_fisher.py` 仅读取官方训练集第 0 折并将核验报告写至 `validation/fisher_training_fold_check.json`，不涉及测试集评估。

### 7.2 解释与协议边界

1. **探索性研究范围**：官方测试集在先前探索迭代中已被查看过（`prior_test_exposure=true`），因此通过新目录重跑仍属于扩展探索性实验，不可宣称为从未接触测试数据的独立盲测确认，也不得依据测试集表现回改参数网格或选择逻辑。
2. **误差线与重采样含义**：CV 曲线误差线为 5 折间总体标准差（`ddof=0`），并非独立重复实验的置信区间；`subject_bootstrap.csv` 以 9 名测试人员为单位做 2000 次重采样，仅反映固定已训练模型在该 9 人样本上的描述性配对差异区间，不包含训练不确定性。
3. **日志与校验范围**：`protocol_frozen.json`、`selection_lock.json`、`events.jsonl` 与 SHA-256 校验用于保障流程可追溯与文件一致性，不等价于第三方预注册，也不能防止人工同时修改全部本地记录。
4. **协议向后兼容性**：[`verify_results.py`](verify_results.py)、[`redraw_figures.py`](redraw_figures.py)、`--resume` 及 `--verify-only` 仅支持 `full` / `pca`（`highest_cv_macro_f1` + `subject_grouped_only`）协议，若检测到旧版 `anova` 路线、旧容忍阈值字段（`cv_threshold` / `meets_cv_tolerance`）或随机窗口协议输出（`protocol_comparison.csv` / `protocol_predictions.csv`），将主动报错拒绝。

### 7.3 数据与协议参考

- 数据集来源：[UCI Machine Learning Repository — Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)（DOI: `10.24432/C54S4K`）
- 联合搜索协议参考：[scikit-learn 1.8 — Pipelining: chaining a PCA and a logistic regression](https://scikit-learn.org/1.8/auto_examples/compose/plot_digits_pipe.html)
