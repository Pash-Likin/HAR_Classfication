# 实验协议：PCA 32 / 64 / 96 / 128

版本：`pca-only-highest-cv-2.1`。本文件是新运行协议，不代表本轮实验已经执行。

## 数据与路线

使用官方 UCI HAR 561 维已提取特征，保留官方 21 名训练人员和 9 名测试人员的划分。训练集 7352 个窗口，测试集 2947 个窗口。人员编号不进入 X，官方特征列顺序不改变。

新实验只包含两类路线：

1. `full`：StandardScaler → 分类器，保留 561 维。
2. `pca`：StandardScaler → PCA → 分类器，候选维数为 32、64、96、128。

删除 ANOVA / SelectKBest 筛选及其排名缓存，不生成新的 `selected_features.csv`。FisherLDA 保持为分类器，不是另一条降维路线。没有实现从原始信号重新提取特征，也没有测量该过程耗时。

## 训练内部评价

在官方训练人员中使用 5 折 StratifiedGroupKFold，seed=42。同一人员不得同时出现在某折训练和验证部分。所有模型、所有维数和参数候选使用同一组折。

每折只用训练部分拟合 StandardScaler 和 PCA，再变换该折验证部分。PCA 固定 `svd_solver="full"`、`whiten=False`，使用同一折完整 SVD 的前 k 个分量复用计算。不会先用全部训练集拟合 PCA 再分折，也不会用测试数据拟合变换。

每一维数遍历分类器的同一参数网格；全特征基准也使用该网格。主指标为六类 Macro-F1，Accuracy 和 balanced accuracy 为补充指标。

正式配置的主搜索为 160 个候选、800 次分类器逐折拟合。共享预处理不代表跳过候选分类器训练。原有随机窗口 / 人员分组协议对照使用固定参数，仅在训练数据内部再进行 50 次拟合，不参与选模。

## 维数与参数联合选择

对分类器 m，定义 S_m(k) 为维数 k 上遍历其参数网格所得的最高平均分组 CV Macro-F1。S_m(full) 是该分类器未降维调参后的同一指标。

选择规则：在 k∈{32,64,96,128} 中选择 S_m(k) 最大的维度及其参数，即 k*=argmax_k S_m(k)。直接比较未四舍五入的平均训练分组 CV Macro-F1。完全同分时优先较小 k，再按稳定 candidate_id 决定同维度同分参数。

配置字段 `pca_selection` 必须为 `highest_cv_macro_f1`。输出记录 `selection_rule="highest_cv_macro_f1"` 和 `selection_metric="cv_macro_f1"`。已移除 `allowed_macro_f1_drop`、`cv_threshold`、`meets_cv_tolerance` 等旧门槛字段。全特征仍独立调参；PCA 即使低于全特征，也选出其自身最高分候选，差值作为描述结果完整保留。

不同分类器可以选择不同 k。该比较评价调参后的完整管线；不能把全部变化都解释成 PCA 单一因素的因果效应。RF 的 `max_features="sqrt"` 会随输入维数改变实际分裂特征数；SVM 的 `gamma="scale"` 的有效值也会随表示变化。

本次最高分规则已在新的真实数据运行前确定。测试分数和模型耗时不参与选维数，也不会为偏好小维度而牺牲更高的验证分数。

## 固定参数与调参后的辅助比较

从同一次搜索记录中取出事先指定的固定参数候选，生成 `fixed_parameter_comparison.csv` 和 PCA 专用的 `pca_dimension_comparison.csv`，不额外重复拟合。

固定参数为：SVM C=10、gamma=0.001；RF max_depth=None、min_samples_leaf=1；LR C=1；KNN n_neighbors=5、weights=uniform；FisherLDA shrinkage=None。它们是本项目事先约定的对照，不能统称为所有 sklearn 分类器的库默认参数。

四维度表包含固定参数分数、每维调参最优分数及其标准差、最优参数、CV 排名、全特征参考分数、相对差值与最终选择标记。这里只回答不同维度在开发阶段的表现，不能把同折的调参增益当作无偏泛化增益。

若要正式比较“固定参数先选维度再调参”和“联合选维度与参数”两种完整策略，需要提前固定两套规则，并在共同外层人员验证折或独立留出数据上评价完整策略。本轮没有追加这项策略比较实验。

## 测试边界

生成所有训练 CV 表，选择维度与参数，写入 `selection_lock.json`，拟合最终 10 条管线后才读取测试数组、计算测试指标。测试分数不回流到选择逻辑。

历史中已接触官方测试集，本轮仍标记 `prior_test_exposure=True`，属于探索性续作。文件锁和哈希用于一致性检查，不等于不可篡改的第三方预注册。

## 图表与成本测量

每个分类器生成一张 full / PCA 双联混淆矩阵，共享计数色标。维数曲线只有 PCA，显示全特征参考分数并标出已锁定的最高 CV 分数维度。参数热图只使用所选维数的训练 CV 成绩，未评估格子为 NaN。

CV 曲线的误差线为折间标准差，不是置信区间；各折不当作独立重复实验。人员指标按人员单独计算，其平均值和全窗口指标含义不同。

预测计时包含标准化、PCA 变换（full 为结构性零）、分类与完整 Pipeline；不包含传感器采集、原始信号特征提取、模型加载、文件 IO 或手机功耗。分阶段中位数之和不必等于端到端中位数。模型文件体积不是峰值运行内存。

## 复现与历史隔离

本轮必须使用新的 `outputs/pca_best_cv_...` 目录。配置、代码、数据和环境指纹变化时禁止复用旧断点。旧三路线和上一版最小达标规则的结果不能恢复或核验为本版最高分结果。旧 `outputs/20261027/`、旧报告和既有验证记录仍保留原内容，不计为本次完成。

软件测试使用合成夹具，只验证流程正确性。真实实验成绩和所选 k 只能来自实际完成的官方数据运行。

## 一手参考

- [UCI HAR 数据及人员划分](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)，DOI 10.24432/C54S4K。
- [PCA 维数与分类器参数联合搜索](https://scikit-learn.org/1.8/auto_examples/compose/plot_digits_pipe.html)。
- [预处理、Pipeline 与数据泄漏](https://scikit-learn.org/1.8/common_pitfalls.html)。
- [StratifiedGroupKFold](https://scikit-learn.org/1.8/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)。
- [选参分数与外部评价的区别](https://scikit-learn.org/1.8/auto_examples/model_selection/plot_nested_cross_validation_iris.html)。
