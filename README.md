# 实验一：新闻文本分类代码说明

代码用于完成英文新闻/邮件文本 10 分类实验。统一入口为 `run_experiment.py`，核心实现位于 `textcls/` 包中。

## 1. 环境与依赖

Python 包依赖记录于 `requirements.txt`。运行代码前，应按该文件处理依赖。

Python 版本：3.10 或兼容版本。

## 2. 快速运行

在项目根目录运行：

```bash
python run_experiment.py
```

无参运行会复现实验完整流程：

```text
1. TF-IDF 下的扩展参数网格调参
2. BoW / TF-IDF / Char TF-IDF / Hashing / LSA 表示方法消融
3. Hashing + MLP 最优方案 loss/accuracy 曲线
4. Hashing + MLP 最终预测
```

主要输出文件：

```text
model_comparison.csv
representation_comparison.csv
representation_comparison_pivot.csv
pic/representation_comparison.png
pic/best_hashing_mlp_loss_accuracy.png
predictions.csv
```

格式要求：

```text
2457 行
1 列
无表头
无索引
```

## 3. 常用命令

### 3.1 生成最终预测

```bash
python run_experiment.py --mode final
```

该命令只执行最终预测流程，会读取：

```text
train_data.csv
test_data_unlabeled.csv
```

并生成：

```text
predictions.csv
```

### 3.2 指定特征和模型

示例：使用 TF-IDF + 逻辑回归。

```bash
python run_experiment.py --mode final --feature-method tfidf --model lr --C 100 --max-iter 2000
```

示例：使用 TF-IDF + 朴素贝叶斯。

```bash
python run_experiment.py --mode final --feature-method tfidf --model nb --alpha 0.5
```

### 3.3 模型调参实验

```bash
python run_experiment.py --mode tune --feature-method tfidf
```

扩展参数网格：

```text
NB: alpha={0.05,0.1,0.3,0.5,1.0,1.5,2.0}
SVM: kernel=linear, C={0.05,0.1,0.5,1,3,10,30,100}
LR: C={0.05,0.1,0.5,1,3,10,30,100}, max_iter={1000,2000,3000}
MLP: hidden={(30,),(50,),(100,),(50,25)}, alpha={0.00001,0.0001,0.001}, max_iter=200
```

输出：

```text
model_comparison.csv
```

如需同时生成每个候选模型的 epoch-accuracy 曲线：

```bash
python run_experiment.py --mode tune --feature-method tfidf --plot-curves
```

曲线保存到：

```text
pic/
```

### 3.4 文本表示方法消融

```bash
python run_experiment.py --mode representations
```

输出：

```text
representation_comparison.csv
representation_comparison_pivot.csv
pic/representation_comparison.png
```

### 3.5 一次运行全部流程

```bash
python run_experiment.py --mode all
```

该命令与无参运行等价。

该命令依次执行：

```text
模型调参
文本表示方法消融
最终预测
```

运行时间长于单独执行各流程。

### 3.6 查看全部参数

```bash
python run_experiment.py --help
```

## 4. 命令行参数

| 参数 | 可选值/默认值 | 作用 |
| --- | --- | --- |
| `--mode` | `final` / `tune` / `representations` / `all`；`run_experiment.py` 无参默认 `all` | 选择实验流程 |
| `--train-csv` | 默认 `train_data.csv` | 训练数据路径 |
| `--test-csv` | 默认 `test_data_unlabeled.csv` | 无标签测试数据路径 |
| `--output` | 默认 `predictions.csv` | 最终预测输出路径 |
| `--feature-method` | `bow` / `tfidf` / `tfidf_char` / `hashing` / `lsa`，默认 `hashing` | 文本表示方法 |
| `--model` | `nb` / `svm` / `lr` / `mlp`，默认 `mlp` | 分类模型 |
| `--alpha` | 浮点数 | NB 或 MLP 的正则/平滑参数 |
| `--C` | 浮点数 | SVM 或 LR 的正则参数 |
| `--kernel` | 默认按模型配置 | SVM 核函数 |
| `--max-iter` | 整数 | LR 或 MLP 最大迭代轮数 |
| `--hidden-layer-sizes` | 如 `50` 或 `100,50` | MLP 隐藏层结构 |
| `--no-validate` | 开关 | 最终训练前不打印验证集结果 |
| `--plot-curves` | 开关 | 调参时生成曲线图 |

## 5. 文件作用

### 5.1 根目录文件

| 文件 | 作用 |
| --- | --- |
| `run_experiment.py` | 统一入口。无参执行完整复现流程，也可按 `--mode` 执行单独流程 |
| `requirements.txt` | Python 包依赖清单 |
| `train_data.csv` | 有标签训练数据 |
| `test_data_unlabeled.csv` | 无标签测试数据 |

### 5.2 `textcls/` 模块

| 文件 | 作用 |
| --- | --- |
| `textcls/__init__.py` | 包初始化文件 |
| `textcls/cli.py` | 命令行参数定义与流程分发 |
| `textcls/config.py` | 随机种子、默认特征、默认模型、参数配置 |
| `textcls/data.py` | CSV 读取和 `predictions.csv` 写出 |
| `textcls/preprocessing.py` | 文本预处理：邮件头清理、空白归一化 |
| `textcls/features.py` | BoW、TF-IDF、Char TF-IDF、Hashing、LSA 特征提取 |
| `textcls/models.py` | NB、SVM、LR、MLP 模型构造、训练和预测 |
| `textcls/evaluation.py` | Accuracy、macro-F1 计算与结果表整理 |
| `textcls/plotting.py` | epoch-accuracy 曲线和表示方法对比图绘制 |
| `textcls/experiments.py` | 三类实验流程：最终预测、模型调参、表示方法消融 |

## 6. 复现实验结果

复现顺序：

```bash
python run_experiment.py
```

当前验证集最优配置：

```text
Hashing + MLPClassifier(hidden_layer_sizes=(50,), alpha=0.0001, max_iter=200)
Accuracy = 0.9233
macro-F1 = 0.9239
```

## 7. 运行约束

- CSV 需使用标准解析器读取；文本字段内部含逗号和换行。
- 调参只使用验证集；测试集只用于最终预测。
- 若运行 `--mode all` 或 MLP 相关实验，耗时会明显长于 NB/LR。
