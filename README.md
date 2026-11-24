# 跨人员人体活动识别：仅使用 PCA 降维

本版本按 2026-11-03 的修改要求，仅保留 **PCA 降维**，候选维数为 **32、64、96、128**，并按用户最新决定选择**调参后交叉验证 Macro-F1 最高的维度**。版本为 `pca-subject-only-highest-cv-2.2`。保留 561 维未降维基准，删除 ANOVA 特征筛选及其缓存、模型和绘图生成分支。五种分类器仍为 RBF-SVM、RandomForest、LogisticRegression、KNN 和 FisherLDA。

**交付状态：此完整项目已附带官方 HAR 数据、2026-11-03 的真实运行归档和10条已有模型。** 新增的训练分阶段耗时及样本量/维数控制变量实验是后续代码改进，历史 `outputs/20261103/` 不含这些新字段；须使用新输出目录重新运行才能得到完整的双栏训练耗时图。旧的 `outputs/20261027/`、`report/` 等资料仅作为历史对照。

## 本次修复版：训练耗时、CV对比与控制变量实验

- 项目统一放在 `HAR_Integrated_Rerun/` 文件夹内。命令需在该目录执行，不要在外层目录运行。
- `run_experiment.py` **原入口**负责160组候选、800次五折CV拟合、锁定配置后训练10条最终管线，并在同一轮新运行中记录训练阶段耗时；控制变量实验只用官方训练集。
- `src/har_study/plots.py` 在 `figures/training_cost/` 自动生成**5张双栏图**：左侧①为CV调参与验证累计耗时，仅对比全特征561维和选定的PCA维数（本轮记录为128维）；右侧②为同样两条最终管线的分阶段训练耗时。
- 同目录另生成**5张样本量控制图**（25% / 50% / 75% / 100%训练样本）和**5张维数控制图**（固定样本量，PCA32 / 64 / 96 / 128维），共15张新图。
- CV阶段依旧搜索四种PCA维数，**图中的PCA128一栏只筛选128维的CV拟合及验证预测记录**；已记录的两项累计值不含共享的折内标准化和PCA预处理，不能当作整个调参流程的墙钟时间。
- 单次最终训练图中完整训练耗时已经包含标准化、PCA（如有）及分类器拟合；不能把完整耗时与其分项再次相加。
- **新运行命令：** `python run_experiment.py --out outputs/training_timing_new --workers 2`。如果已有同名且非空的输出目录，需换新名字。
- **新结果重画：** `python redraw_figures.py --run outputs/training_timing_new`。历史 `outputs/20261103/` 缺少新增分阶段字段，只会生成旧结果可支持的CV图，无法补画缺失的训练阶段记录。
- **检查：** `python run_experiment.py --plan`、`python -m unittest discover -s tests -v`。

## 1. 本次“去掉特征提取”的具体范围

现有程序直接读取 UCI 官方提供的 561 维时域与频域特征，不包含从原始传感器信号重新提取特征的步骤。原程序的 ANOVA 是特征选择，本次删除的正是这条额外筛选路线。

- PCA 路线：官方 561 维特征 → StandardScaler → PCA（32 / 64 / 96 / 128）→ 分类器。
- 对照路线：官方 561 维特征 → StandardScaler → 分类器。
- FisherLDA 使用 `lsqr`，作为分类器保留，没有新增 LDA 降维阶段。

输入仍需满足官方 561 列的定义与顺序。PCA 会构造主成分，因此“无额外特征筛选或提取阶段”比“完全没有特征变换”更准确；本版没有把输入改为原始传感器序列。

## 2. 维数在调参时一起选择

四个维数分别使用同一分类器参数网格和同一组人员验证折。程序先取得各维数的最佳训练交叉验证 Macro-F1，再选择维数；不会用固定参数先淘汰其他维数。

对于每种分类器，选择 **32、64、96、128 中调参后平均训练人员分组 CV Macro-F1 最高的维度及其参数**。使用未四舍五入的分数比较；完全同分时优先较小维度，再按稳定的候选编号决定参数。不同分类器可以选择不同维数。

本版配置明确记录 `pca_selection="highest_cv_macro_f1"`。已移除原来的 1 个百分点阈值和最小达标规则。全特征基准只用于比较，即使所有 PCA 候选都低于全特征，也选择其中 CV 分数最高者并如实报告差值。测试集成绩不参与选择。

固定参数和调参后的维数表现会从**同一次网格搜索**提取，便于比较，无需再跑一遍。两条曲线是训练验证结果，不能把其差值当作独立测试收益。完整说明见 [docs/PCA_ONLY_CHANGELOG.md](docs/PCA_ONLY_CHANGELOG.md)。

## 3. Windows 运行方式

解压到一个新文件夹，打开终端并进入含 `run_experiment.py` 的项目根目录。可继续使用已有的 `har_project` 环境；项目原运行使用 Python 3.11；本次修复也已在 Python 3.13 环境通过单元测试。

```powershell
python -m pip install -r requirements.txt
python run_experiment.py --plan
```

`--plan` 不下载数据、不训练。正式配置应显示：

| 项目 | 数量 |
| --- | ---: |
| 每模型表示方案 | 5（全特征 + 4 个 PCA 维数） |
| 主候选 | 160 |
| 主 5 折分类器拟合 | 800 |
| 保留的固定参数验证协议对照拟合 | 50 |
| 最终模型 | 10（五模型各 full / PCA） |

用已经解压的本地官方数据运行：

```powershell
python run_experiment.py --data "D:\你的数据目录\UCI HAR Dataset" --out outputs/pca_best_cv_32_64_96_128 --workers 2
```

也可以指定官方 ZIP：

```powershell
python run_experiment.py --data "D:\你的数据目录\UCI HAR Dataset.zip" --out outputs/pca_best_cv_32_64_96_128 --workers 2
```

如果已将数据放回项目的 `data/UCI HAR Dataset/`，可省略 `--data`：

```powershell
python run_experiment.py --out outputs/pca_best_cv_32_64_96_128 --workers 2
```

没有本地数据时，入口会尝试官网下载。下载失败会停止，不会用合成数据冒充正式实验。也可把已有 `HAR_Local_Assets.zip` 中的数据恢复到本项目；旧模型只属于旧运行。

### 中断恢复

只恢复本版新建的实验：

```powershell
python run_experiment.py --data "D:\你的数据目录\UCI HAR Dataset" --resume outputs/pca_best_cv_32_64_96_128
```

恢复时数据、代码、配置、软件版本和平台必须一致。不要恢复旧 `outputs/20261027/` 的断点，也不要把旧结果 CSV 拷入新目录。修改网格后必须新建输出目录。

### 中文图表字体

Windows 通常可以自动找到微软雅黑。如果字体检测失败，可指定本机存在的中文字体：

```powershell
python run_experiment.py --verify-only outputs/pca_best_cv_32_64_96_128 --font "C:\Windows\Fonts\msyh.ttc"
```

该命令验证已完成的新实验并重新画图，不重新训练。图表失败会留下错误记录，数值结果不会被伪造为绘图成功。实际字体路径以本机为准；数据未放在项目 `data/` 时仍需加 `--data`。

## 4. 运行后优先查看哪些表

以下文件由新实验运行生成；交付时不预填新成绩。

| 新运行目录下的文件 | 用途 |
| --- | --- |
| `tables/pca_dimension_comparison.csv` | 五模型 × 四维度，共 20 行；固定参数与调参分数、各维最佳参数、CV 排名、选择维度、相对全特征差值 |
| `tables/compression_curve.csv` | 每种分类器在全特征及各 PCA 维数下的最佳参数与 CV 分数 |
| `tables/fixed_parameter_comparison.csv` | 包含全特征的固定参数 / 调参对照 |
| `tables/selected_configs.csv` | 最终 10 条管线及最高分选择规则、主指标、相对全特征差值 |
| `tables/cv_candidates.csv`、`cv_folds.csv` | 160 个候选及 800 条主 CV 逐折记录 |
| `tables/test_metrics.csv` | 维数与参数锁定后的测试指标 |
| `tables/PCA_vs_full_with_timing_breakdown.csv` | PCA 相对全特征的表现与阶段耗时 |
| `selection_lock.json` | 查看本轮测试数组之前写入的选择锁 |
| `figures/` | PCA 维数曲线、双联混淆矩阵、参数热图及资源比较 |

`fixed_cv_macro_f1` 是事先固定的那组参数的验证分数；`tuned_cv_macro_f1` 是同一维数上搜索后的最高验证分数。两者的差值属于开发期描述。误差线为验证折间标准差，不是独立重复实验的置信区间。

## 5. 保留的分类器参数网格

| 分类器 | 参数候选 | 每个表示的参数组合数 |
| --- | --- | ---: |
| RBF-SVM | C=1、10、100、1000；gamma=scale、0.001 | 8 |
| RandomForest | max_depth=None、10、20；min_samples_leaf=1、3；200 棵树 | 6 |
| LogisticRegression | C=0.01、0.1、1、10、100；lbfgs | 5 |
| KNN | n_neighbors=3、5、9、15；uniform / distance | 8 |
| FisherLDA | shrinkage=None、auto、0.05、0.2、0.5；lsqr | 5 |

## 6. 验证与结果边界

```powershell
python -m unittest discover -s tests -v
python verify_results.py --run outputs/pca_best_cv_32_64_96_128
python measure_stages.py --run outputs/pca_best_cv_32_64_96_128
```

第一条使用小型合成夹具检查软件正确性，不需要下载 HAR；后两条需要已完成的新实验、官方数据和对应模型。补测阶段耗时会更新计时文件。

最初的选维规则检查保存在 `validation/pca_highest_cv_20261103/`；2026-11-03 原始完整运行已归档于 `outputs/20261103/`。当前新增的训练分阶段、CV双栏图及控制变量实验尚未在官方数据上整体重跑；它们必须由更新后代码在新输出目录中生成，不能把旧归档误当作新增实验结果。该修复版已经通过软件单元测试和现有结果重绘检查；用户 Windows 设备上的新一轮完整训练尚未执行。

历史官方测试集已被查看过，因此本轮仍属于后续探索性实验。不能通过新建输出目录将它宣称为从未使用过的独立测试集，也不能按测试成绩反复选择维数。

历史文件仍保留：`outputs/20261027/`、`report/`、原有 `validation/`，以及 `docs/legacy_readme_20261027.md`、`docs/legacy_protocol_20261027.md`。其中旧维数和 ANOVA 记录没有改名伪装为新结果；旧报告需要在新实验实际完成后另行更新。**本版验证、恢复与绘图入口只针对最高 CV 分数的新协议；旧三路线快照及上一版最小达标规则运行均需使用各自原版代码及模型。**

数据和协议来源：[UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)，[scikit-learn PCA 与分类器联合搜索示例](https://scikit-learn.org/1.8/auto_examples/compose/plot_digits_pipe.html)。更多评价约定见 [docs/protocol.md](docs/protocol.md)。


## 新增：固定超参数控制变量训练耗时与双层对比图（原代码自动运行）

**本版没有新增独立 Python 脚本**，直接修改原来的 `run_experiment.py`、`src/har_study/study.py`、`src/har_study/plots.py`。用新目录重新运行原实验即可，在最终配置锁定并完成最终模型训练后，自动运行控制变量计时，随后测试与常规作图流程照旧。

```powershell
python run_experiment.py --data "data/UCI HAR Dataset" --out outputs/new_with_controls --workers 2
```

`--data` 可以照原有说明替换为Windows上真实的官方数据目录。已修改源代码的哈希，**必须使用全新的输出目录，不能用修改后的代码恢复旧版断点**。`--plan` 仅列出主CV计划，不实际执行新控制变量分析。

### 控制变量实验如何设计

- **保持超参数固定**：每个分类器使用当轮训练集五折CV选出的**全特征方案参数**，在两组控制实验中始终不变；这不是再次调参。原项目的分类器和PCA最高CV分数选维规则不变。
- **样本量变量**：按人员与活动分层、固定随机种子，在官方21名训练人员中构造严格嵌套的25%（1838条）、50%（3676条）、75%（5514条）、100%（7352条）样本子集；每个子集分别对比原561维和PCA128维。子集内重新拟合标准化及PCA，避免从外部数据拟合预处理。仅测分类器本身的 `fit()` 时间。
- **维数变量**：固定全部7352条训练样本，先在这批数据上仅拟合一次128维PCA，分别取前32、64、96、128个主成分做分类器输入。统一分类器超参数，测量各维度 `fit()` 耗时。全561维的原坐标表示在图中以**不同表示下的参考线**出现；32～128维的对照共享一个PCA基底。
- **计时范围和次数**：仅在单线程条件下记录分类器 `fit()` 执行时间，不计入准备样本、标准化、PCA拟合/变换、CV或测试预测。每种组合从头创建分类器、正式执行3轮，打乱各轮任务顺序，统计中位数和第25/75百分位。5模型×（4样本规模×2表示+4输入维数）×3轮，共**180次额外拟合**，不增加CV的800次候选拟合，不改变已锁定模型和最终测试预测。
- **研究边界**：样本量与维数测试用于观察计算量变化，不能由时间曲线直接得出分类准确率的变化。随机森林固定`max_features=sqrt`等配置后，其实际每节点候选列数仍随输入维数变化；SVM采用`gamma=scale`时，有效数值也依赖特征表示，这些都是配置本身的算法行为，不能将所有差异归因于维数一个因素。

### 自动生成的结果

| 输出目录 | 文件与含义 |
| --- | --- |
| `tables/controlled_fit_timing_raw.csv` | 180次分类器拟合的逐次耗时、方案、参数、样本比例和实际样本数 |
| `tables/controlled_fit_timing_summary.csv` | 每组重复测量的中位数、25%和75%分位数 |
| `controlled_fit_protocol.json` | 数据来源、抽样种子、样本集合指纹、固定参数说明及计时边界 |
| `figures/training_cost/combined_*.png` | **5张**，每模型一张，左侧为全特征与已选PCA128维的五折CV拟合和验证预测累计耗时，右侧为最终单次训练分步骤耗时；两个子图采用独立的秒数坐标轴 |
| `figures/training_cost/control_sample_size_*.png` | **5张**，每模型在25%、50%、75%、100%样本量下的全特征/PCA128分类器拟合耗时 |
| `figures/training_cost/control_dimension_*.png` | **5张**，每模型固定7352条样本，比较PCA32/64/96/128维的拟合耗时及561维全特征参考 |

**两层耗时口径：**左侧CV累计耗时来自分类器拟合与验证预测之和，分别统计全特征和已选定的PCA128维；原实验仍搜索PCA32/64/96/128四种维数，但本图不计入另外三种维数的任务。左侧**不包含折内共享标准化/PCA的预处理成本**，也不等于程序墙钟时间。右侧最终训练耗时是使用全部7352条训练样本拟合一次，包含标准化、PCA（如有）和分类器拟合。两侧的数值不要直接相加。控制变量实验在独立循环中对分类器重新拟合，仅作为训练复杂度诊断，不重新训练或替换10个锁定后的最终模型。

如果你只执行 `redraw_figures.py`，它只能利用已存在的CSV作图，不会补做未运行的控制变量计时。第一次生成两类控制变量图，需要使用上述正式运行命令完成实验。
