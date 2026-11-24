# 跨人员人体活动识别与特征压缩

基于 UCI HAR 的课程实验，比较 RBF-SVM、RandomForest、LogisticRegression、KNN 和 FisherLDA 在全特征、PCA、ANOVA 三种特征设置下的分类表现。
采用人员隔离划分，在训练人员中选择参数和维数，并测量完整预测流程及各阶段的计算开销。

仓库保留源码、配置、测试、一次已完成实验的结果表与图片，以及课程报告初稿。
原始数据和训练好的二进制模型不放在普通 Git 仓库里；它们仍在单独的本地资源包中。

## 查看已有结果

已有实验目录是 **`outputs/20261027/`**，不是旧说明中的 `outputs/integrated_rerun/`。
这个目录名称不参与算法计算。下面所有命令均从项目根目录执行。

| 路径 | 内容 |
| --- | --- |
| `outputs/20261027/tables/test_metrics.csv` | 15条最终管线的测试指标 |
| `outputs/20261027/tables/selected_configs.csv` | 选定参数、维数和验证达标状态 |
| `outputs/20261027/tables/cv_candidates.csv`、`cv_folds.csv` | 全部候选及逐折验证记录 |
| `outputs/20261027/tables/test_predictions.csv` | 每条测试窗口的真实类别和预测 |
| `outputs/20261027/tables/test_subjects.csv` | 各测试人员的指标 |
| `outputs/20261027/tables/timing_raw.csv`、`stage_timing_raw.csv` | 原始计时记录 |
| `outputs/20261027/figures/` | 已生成的42张结果图，包括5张三联混淆矩阵 |
| `report/` | 已有Word、PDF和Markdown课程报告初稿，以及对应图表 |
| `validation/` | 历史结构化核验记录、本次整理检查记录 |

可以直接查看CSV和图片，不需要重新训练。
报告保留上传时的原文，仍有待填写的作者信息；报告里的历史目录名称不是当前运行命令。
当前包没有`.ipynb`文件，Python主入口是实际可用的运行方式。

## 安装与完整运行

推荐 Python 3.11，在独立虚拟环境中安装依赖：

```bash
python -m pip install -r requirements.txt
python run_experiment.py --plan
```

`--plan`只显示搜索规模，不下载数据、不训练。
本配置含480个候选、2400次主交叉验证拟合，另有验证协议对照与15个最终模型。

将原始数据放回 `data/UCI HAR Dataset/`，或者指定官网下载的原始ZIP：

```bash
python run_experiment.py --data "UCI HAR Dataset.zip" --out outputs/my_run --workers 2
```

未指定`--data`时，程序先查找`data/`，找不到再尝试官方下载。
训练使用新的输出目录，不覆盖已发布的 `outputs/20261027/`。

中断后恢复新实验：

```bash
python run_experiment.py --resume outputs/my_run
```

恢复要求代码、数据、环境和配置一致。发布包已删除完成运行的旧断点，
不要对 `outputs/20261027/` 使用`--resume`。新实验的断点仍会正常保存。

## 使用已有模型核验、重画图或预测

另一个压缩包 `HAR_Local_Assets.zip` 只包含原始数据和15个二进制模型。
**将其内容解压到本项目根目录**，合并`data/`与`outputs/`后即可恢复本地完整功能。
也可以从你保留的原项目复制相同的两个位置：

```text
data/UCI HAR Dataset/
outputs/20261027/models/*.joblib
```

这些本地资源已由`.gitignore`排除。仓库内仍保留模型参数JSON与模型哈希，不能用任意重训模型替代旧模型后声称文件校验通过。

恢复资源之后运行：

```bash
python verify_results.py --run outputs/20261027
python redraw_figures.py --run outputs/20261027
```

第一条复算指标并检查原始数据、划分和模型哈希；第二条重画图，不训练。
**现有绘图入口也会检查模型文件哈希，所以仅有轻量仓库时不能直接执行全部重绘。**
模型未恢复时，直接查看仓库里已有的图片即可。

阶段耗时补测和新数据预测用法：

```bash
python measure_stages.py --run outputs/20261027
python predict.py --model "outputs/20261027/models/FisherLDA__full.joblib" --input "X_new.txt" --output "new_predictions.csv"
```

补测会更新计时记录，需要保留第一次计时的情况下先备份实验目录。
预测输入必须符合官方561维特征的定义和列序，不是手机原始传感器序列。
只加载自己训练或已确认可信的joblib文件。模型版本见对应JSON和实验清单。

## 实验设置

| 分类器 | 参数候选 |
| --- | --- |
| RBF-SVM | C=1、10、100、1000；gamma=scale、0.001 |
| RandomForest | max_depth=None、10、20；min_samples_leaf=1、3；固定200棵树 |
| LogisticRegression | C=0.01、0.1、1、10、100；L2正则化、lbfgs |
| KNN | n_neighbors=3、5、9、15；uniform、distance；欧氏距离 |
| FisherLDA | shrinkage=None、auto、0.05、0.2、0.5；lsqr |

PCA和ANOVA比较20、50、100、150、200、250、300维，全特征基准为561维。
每条压缩路线选择训练分组验证Macro-F1相对自身全特征基准下降不超过1个百分点的最小候选维数。
没有达标配置时保留最高验证分数方案作诊断，并记录失败，不能将其称为成功压缩。

预测计时采用单线程。阶段计时比较批量1、32、256、2947，分别记录标准化、压缩、分类和完整流程。
独立阶段的中位数不能严格相加为完整流程中位数；计时不含原始信号特征提取和手机功耗。
详细协议见[docs/protocol.md](docs/protocol.md)。

## 软件测试

```bash
python -m unittest discover -s tests -v
```

常规测试使用临时夹具，不需要下载HAR或加载旧模型。部分绘图测试需要支持中文的系统字体。
真实数据一致性测试 `tests/check_real_fisher.py` 则需要恢复数据，不能与离线单元测试混为一谈。
训练和绘图都不会自动生成课程报告。

## GitHub提交

解压后提交文件夹内容，不要把整个ZIP作为仓库主文件上传。
本地数据、模型、缓存、环境和新实验输出都已加入`.gitignore`；当前结果快照、图表和报告允许提交。
`.gitattributes`保留冻结记录的原始换行，避免Windows换行转换破坏哈希校验。
提交前可检查：

```bash
git status --short
git check-ignore "data/UCI HAR Dataset/train/X_train.txt"
git check-ignore "outputs/20261027/models/KNN__full.joblib"
```

若旧文件已经被Git跟踪，仅补充`.gitignore`不会自动移出版本管理。[GitHub说明](https://docs.github.com/en/get-started/git-basics/ignoring-files)
代码尚未添加开源许可证，需由仓库作者决定授权范围；没有擅自套用MIT等许可证。
整理范围见[docs/cleanup.md](docs/cleanup.md)。

## 数据与结果边界

数据来源：[UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)，DOI：10.24432/C54S4K。
使用的是官方已提取的561维特征，保留训练与测试人员划分。

官方测试集在课题探索中已查看过，因此这份结果属于扩展探索性实验。保留未达标和耗时增加的方案。
数据、指标、已保存模型和硬件计时之间的检查范围分别写在实验目录及`validation/`中；
本地哈希用于复核文件一致性，不构成不可篡改的研究记录，也不保证跨设备的相同耗时。
