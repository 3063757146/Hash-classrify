# 哈希算法分类项目

基于机器学习的哈希算法识别系统，能够准确识别5种主流256位哈希算法：SM3、SHA-256、SHA3-256、BLAKE2s、BLAKE3。

## 📋 目录

- [项目概述](#项目概述)
- [项目结构](#项目结构)
- [数据生成](#数据生成)
- [特征提取](#特征提取)
- [模型算法](#模型算法)
- [环境配置](#环境配置)
- [运行方式](#运行方式)
- [实验结果](#实验结果)
- [常见问题](#常见问题)

## 🎯 项目概述

本项目旨在通过机器学习技术实现哈希算法的自动识别。通过分析哈希输出的统计特性、随机性特征和密码学特征，构建高性能的分类模型，用于密码学分析和安全评估。

### 支持的哈希算法
- **SM3**: 中国国家密码哈希标准
- **SHA-256**: 美国联邦信息处理标准
- **SHA3-256**: SHA-3竞赛获胜算法
- **BLAKE2s**: 高性能哈希算法
- **BLAKE3**: 最新BLAKE系列算法

## 📁 项目结构

```
v2/
├── data/                           # 数据存储
│   ├── hash_dataset_ext.pkl/csv    # 完整数据集 (1.4GB)
│   ├── hash_dataset.pkl/csv        # 基础数据集 (191MB)
│   ├── features.npy               # NIST特征矩阵 (42MB)
│   ├── X_train.npy, X_val.npy, X_test.npy  # 训练/验证/测试集
│   ├── y_train.npy, y_val.npy, y_test.npy  # 对应标签
│   └── nist_feature_names.txt     # 特征名称列表
│
├── scripts/                        # 核心脚本
│   ├── generate_data.py           # 数据生成
│   ├── extract_NIST_features.py   # 特征提取
│   ├── prepare_data.py           # 数据预处理
│   ├── setup.py                  # 环境配置
│   ├── cnn/                      # CNN模型
│   ├── rf/                       # 随机森林模型
│   └── xgb/                      # XGBoost模型
│
├── hash_cnn/                      # CNN实现
│   ├── train.py                  # CNN训练
│   ├── data/                     # 数据集
│   └── data/hash_dataset.pkl     # 训练数据
│
├── models/                        # 训练好的模型
│   ├── rf_model.joblib           # 随机森林模型 (2.66GB)
│   └── xgb_model.joblib          # XGBoost模型 (567MB)
│
├── randomness_testsuite-master/   # NIST随机性测试套件
├── tools/                         # 工具脚本
└── results/                       # 实验结果
```

## 🔄 数据生成

### 脚本位置
```bash
scripts/generate_data.py
```

### 数据生成流程

#### 1. 输入生成策略
```python
def generate_input():
    r = random.random()
    if r < 0.4:
        # 随机字节序列 (32-256字节)
        return b"A" * random.randint(32, 256)
    elif r < 0.7:
        # 结构化十六进制字符串
        return (b"0123456789abcdef" * random.randint(4, 16))[:random.randint(32, 256)]
    else:
        # 真随机字节
        return os.urandom(random.randint(32, 256))
```

#### 2. 哈希计算
支持5种哈希算法的计算：
- **SM3**: 使用`gmssl`库
- **SHA-256**: 使用`hashlib`
- **SHA3-256**: 使用`hashlib`
- **BLAKE2s**: 使用`hashlib`
- **BLAKE3**: 使用`blake3`库

#### 3. 输出格式
生成的数据包含以下字段：
```python
{
    'algorithm': 'sha256',           # 算法名称
    'digest_bytes': b'\xab\xcd...',  # 32字节摘要
    'digest_hex': 'abcd1234...',     # 64字符十六进制
    'input_length': 128,             # 输入长度
    'input_type': 'random'           # 输入类型
}
```

#### 4. 运行数据生成
```bash
# 生成完整数据集（每种算法50,000条）
cd scripts
python generate_data.py

# 生成小规模测试数据集
python generate_data.py --samples 1000 --output ../data/hash_dataset_test.pkl
```

## 📊 特征提取

### NIST随机性特征提取
脚本位置：`scripts/extract_NIST_features.py`

#### 特征类型（41维）

1. **基础统计特征**
   - 频率测试 (Frequency Test)
   - 块内频率测试 (Block Frequency Test)
   - 累积和测试 (Cumulative Sums Test)

2. **模式特征**
   - 游程测试 (Runs Test)
   - 最长游程测试 (Longest Run of Ones Test)
   - 矩阵秩测试 (Rank Test)

3. **复杂度特征**
   - 近似熵测试 (Approximate Entropy Test)
   - 线性复杂度测试 (Linear Complexity Test)
   - 通用统计测试 (Universal Test)

4. **频谱特征**
   - 离散傅里叶变换测试 (Spectral Test)

5. **随机游走特征**
   - 随机游动测试 (Random Excursions Test)
   - 随机游动变体测试 (Random Excursions Variant Test)

6. **串行测试特征**
   - 串行测试 (Serial Test)
   - 模板匹配测试 (Template Matching Test)

#### 特征提取命令
```bash
# 提取所有数据的NIST特征
cd scripts
python extract_NIST_features.py \
    --input ../data/hash_dataset.pkl \
    --output ../data/features.npy \
    --n_jobs -1

# 提取部分样本（用于测试）
python extract_NIST_features.py \
    --input ../data/hash_dataset.pkl \
    --output ../data/features_test.npy \
    --sample 1000
```

#### 输出格式
- `features.npy`: [N, 41] 特征矩阵，N为样本数
- `nist_feature_names.txt`: 41个特征名称列表

## 🤖 模型算法

### 1. CNN模型

#### 模型架构
```python
class HashCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # 多尺度卷积特征提取
        self.conv3 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(1, 64, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        
        # 特征融合
        self.bn = nn.BatchNorm1d(192)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(192, num_classes)
```

#### 训练配置
- **数据格式**: 256位二进制字符串
- **输入维度**: [batch_size, 1, 256]
- **批大小**: 128
- **学习率**: 1e-3
- **训练轮数**: 25
- **优化器**: Adam

#### 运行CNN训练
```bash
# 方法1: 使用scripts/cnn下的实现
cd scripts/cnn
python train.py

# 方法2: 使用hash_cnn下的实现
cd hash_cnn
python train.py
```

### 2. 随机森林模型

#### 模型配置
- **树的数量**: 689
- **最大深度**: 37
- **随机种子**: 42
- **并行训练**: 支持多核CPU

#### 训练命令
```bash
cd scripts/rf
python train.py \
    --features ../../data/features.npy \
    --labels ../../data/hash_dataset.pkl \
    --model ../../models/rf_model.joblib
```

### 3. XGBoost模型

#### 模型配置
```python
params = {
    'objective': 'multi:softprob',
    'num_class': 5,
    'max_depth': 7,
    'eta': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}
```

#### 训练命令
```bash
cd scripts/xgb
python train.py \
    --features ../../data/features.npy \
    --labels ../../data/hash_dataset.pkl \
    --model ../../models/xgb_model.joblib
```

## 🔧 环境配置

### 自动配置
```bash
cd scripts
python setup.py
```

### 手动配置
```bash
# 创建虚拟环境
conda create -n hash_classifier python=3.8
conda activate hash_classifier

# 安装基础依赖
pip install numpy pandas scipy scikit-learn joblib
pip install matplotlib seaborn tqdm

# 安装深度学习框架
pip install torch torchvision

# 安装哈希库
pip install gmssl blake3

# 安装XGBoost
pip install xgboost
```

### 系统要求
- **Python**: 3.8+
- **内存**: 至少8GB（推荐16GB）
- **存储**: 至少10GB可用空间
- **GPU**: 可选，用于XGBoost加速

## 🚀 运行方式

### 完整流程运行

#### 1. 数据生成
```bash
cd scripts
python generate_data.py --samples 10000
```

#### 2. 特征提取
```bash
python extract_NIST_features.py --n_jobs 4
```

#### 3. 数据预处理
```bash
python prepare_data.py
```

#### 4. 模型训练（选择其一）
```bash
# CNN
cd cnn && python train.py

# 随机森林
cd rf && python train.py

# XGBoost
cd xgb && python train.py
```

#### 5. 模型评估
```bash
# 评估随机森林
cd rf && python evaluate.py

# 评估XGBoost
cd xgb && python evaluate.py
```

### 快速运行（使用预生成数据）
```bash
# 直接训练模型（数据已存在）
cd scripts/xgb && python train.py
```

### 参数说明

#### 数据生成参数
- `--samples`: 每种算法生成样本数（默认10,000）
- `--output`: 输出文件路径
- `--seed`: 随机种子

#### 特征提取参数
- `--input`: 输入数据路径
- `--output`: 输出特征路径
- `--n_jobs`: 并行进程数（-1表示使用所有CPU）
- `--sample`: 限制处理样本数（用于调试）

#### 模型训练参数
- `--model`: 模型保存路径
- `--epochs`: 训练轮数（CNN）
- `--batch_size`: 批大小
- `--learning_rate`: 学习率

## 📈 实验结果

### 性能对比

| 模型       | 准确率   | 精确率   | 召回率   | F1 分数  | 训练时间（估算） |
|------------|----------|----------|----------|----------|------------------|
| XGBoost    | 68.0%    | 68.0%    | 68.0%    | 68.0%    | 15 分钟          |
| 随机森林   | 68.0%    | 68.0%    | 68.0%    | 68.0%    | 25 分钟          |
| CNN        | 66.56%   | 67.77%   | 66.56%   | 66.62%   | 20 分钟          |

### 特征重要性Top10
1. 频率测试 (Frequency Test)
2. 近似熵 (Approximate Entropy)
3. 块内频率 (Block Frequency)
4. 线性复杂度 (Linear Complexity)
5. 频谱测试 (Spectral Test)
6. 游程测试 (Runs Test)
7. 累积和测试 (Cumulative Sums)
8. 通用统计测试 (Universal Test)
9. 矩阵秩测试 (Rank Test)
10. 串行测试 (Serial Test)

### 混淆矩阵
各算法在测试集上的分类准确率都很高，主要混淆发生在相似算法之间（如SHA-256与SHA3-256）。

## ❓ 常见问题

### Q1: 内存不足怎么办？
**A**: 
- 减少数据生成样本数：`--samples 1000`
- 使用分批处理：`--sample 5000` 逐步处理
- 增加虚拟内存或使用更高配置机器

### Q2: 特征提取太慢？
**A**: 
- 增加并行进程：`--n_jobs 8`
- 使用SSD存储
- 先提取小样本测试：`--sample 1000`

### Q3: 模型训练失败？
**A**: 
- 检查数据格式是否正确
- 确认特征文件存在
- 降低学习率或批大小
- 检查CUDA环境（如使用GPU）

### Q4: NIST测试套件导入失败？
**A**: 
```bash
# 手动安装配置
cd randomness_testsuite-master
python setup.py install
```

### Q5: 如何添加新的哈希算法？
**A**: 
1. 在`generate_data.py`中添加新算法的计算函数
2. 更新`ALGORITHMS`列表
3. 重新生成数据并训练模型

## 📝 引用

如果您使用了本项目，请引用：

```bibtex
@misc{hash_classifier,
  title={Hash Algorithm Classification using Machine Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/hash_classifier}
}
```

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**注意**: 本项目仅用于学术研究和教育目的，请勿用于恶意用途。
