# 数据位置

原始数据未纳入GitHub发布包。训练入口会查找本目录，缺失时尝试官方下载；也支持`--data`指定本地ZIP或目录。

官网下载：
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

也可将配套`HAR_Local_Assets.zip`解压到项目根目录，恢复原始数据和已保存模型。
恢复后应为：

```text
data/
├── README.md
└── UCI HAR Dataset/
    ├── features.txt
    ├── activity_labels.txt
    ├── train/
    │   ├── X_train.txt
    │   ├── y_train.txt
    │   └── subject_train.txt
    └── test/
        ├── X_test.txt
        ├── y_test.txt
        └── subject_test.txt
```

必须保留人员编号文件，不能换成缺少subject信息的合并CSV。
原实验八个必要数据文件的哈希保留在`outputs/20261027/protocol_frozen.json`。
参考：Reyes-Ortiz J, Anguita D, Ghio A, Oneto L, Parra X. Human Activity Recognition Using Smartphones. UCI, 2013. DOI: 10.24432/C54S4K。
数据文件中原有的作者与使用说明随本地资源保留，未改动或删除。
