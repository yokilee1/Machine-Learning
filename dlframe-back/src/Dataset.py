from dlframe import WebManager, Logger
import math
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

class IrisDataset:
    """
    创建一个鸢尾花分类数据集
    
    特征：
    - 4个数值特征：萼片长度、萼片宽度、花瓣长度、花瓣宽度
    - 所有特征值为浮点数
    
    分类：
    1. Setosa（山鸢尾）
    2. Versicolor（变色鸢尾）
    3. Virginica（维吉尼亚鸢尾）
    
    参数：
        无需参数，直接从sklearn加载
    """
    def __init__(self):
        from sklearn.datasets import load_iris
        iris = load_iris()
        self.data = iris.data
        self.target = iris.target
        self.logger = Logger.get_logger('IrisDataset')
        self.logger.print("Iris Dataset loaded")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    
class MNISTDataset:
    """
    创建一个手写数字识别数据集
    
    特征：
    - 784维向量（28x28像素图像展平）
    - 像素值范围0-1（归一化后）
    
    分类：
    - 10个类别（数字0-9）
    
    参数：
        无需参数，直接从openml加载
    """
    def __init__(self):
        from sklearn.datasets import fetch_openml
        from sklearn.model_selection import train_test_split
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
        X = X / 255.0  # 归一化
        self.data = list(zip(X, y))
        self.logger = Logger.get_logger('MNISTDataset')
        self.logger.print("MNIST Dataset loaded")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
class WineDataset:
    """
    创建一个葡萄酒分类数据集
    
    特征：
    - 13个化学成分指标
    - 包括酒精度、苹果酸、灰分等
    
    分类：
    - 3个不同品种的葡萄酒
    
    参数：
        无需参数，直接从sklearn加载
    """
    def __init__(self):
        from sklearn.datasets import load_wine
        import numpy as np
        
        # 加载红酒数据集
        wine = load_wine()
        self.X = wine.data
        self.y = wine.target
        self.feature_names = wine.feature_names
        self.target_names = wine.target_names
        
        # 将数据组织成所需的格式
        self.data = [(np.array(x, dtype=np.float64), int(y)) 
                     for x, y in zip(self.X, self.y)]
        
        # 打印数据集信息
        logger = Logger.get_logger('WineDataset')
        logger.print(f"Wine dataset loaded:")
        logger.print(f"- Number of samples: {len(self.data)}")
        logger.print(f"- Number of features: {len(self.feature_names)}")
        logger.print(f"- Number of classes: {len(self.target_names)}")
        logger.print(f"- Features: {self.feature_names}")
        logger.print(f"- Classes: {self.target_names}")
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)


class BreastCancerDataset:
    """
    创建一个乳腺癌诊断数据集
    
    特征：
    - 30个数值特征
    - 包括细胞核的各种特征测量值
    
    分类：
    1. 良性肿瘤
    2. 恶性肿瘤
    
    参数：
        无需参数，直接从sklearn加载
    """
    def __init__(self):
        from sklearn.datasets import load_breast_cancer
        import numpy as np
        
        # 加载乳腺癌数据集
        cancer = load_breast_cancer()
        self.X = cancer.data
        self.y = cancer.target
        self.feature_names = cancer.feature_names
        self.target_names = cancer.target_names
        
        # 将数据组织成所需的格式
        self.data = [(np.array(x, dtype=np.float64), int(y)) 
                     for x, y in zip(self.X, self.y)]
        
        # 打印数据集信息
        logger = Logger.get_logger('BreastCancerDataset')
        logger.print(f"Breast Cancer dataset loaded:")
        logger.print(f"- Number of samples: {len(self.data)}")
        logger.print(f"- Number of features: {len(self.feature_names)}")
        logger.print(f"- Number of classes: {len(self.target_names)}")
        logger.print(f"- Features: {self.feature_names}")
        logger.print(f"- Classes: {self.target_names}")
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    

class WeatherDataset:
    """
    创建一个天气-活动隐马尔可夫模型数据集
    
    特征：
    - 隐状态：晴天、雨天
    - 观测状态：散步、购物、清洁
    
    概率分布：
    1. 转移概率：天气状态转换概率
    2. 发射概率：不同天气下进行各项活动的概率
    
    参数：
        sequence_length: 生成序列的长度，默认2000
    """
    def __init__(self):
        self.states = ['Sunny', 'Rainy']  # 隐状态
        self.observations = ['Walk', 'Shop', 'Clean']  # 观测状态
        self.hidden_states, self.obs_sequence = self.generate_data()
        self.logger = Logger.get_logger('WeatherDataset')
        self.logger.print("天气-活动HMM数据集已加载")
        
    def __len__(self):
        return len(self.obs_sequence)
    
    def __getitem__(self, idx):
        """返回(观测值，隐状态)对"""
        return self.obs_sequence[idx], self.hidden_states[idx]
    
    def generate_data(self, sequence_length=2000):  # 增加序列长度
        """改进数据生成过程"""
        # 调整转移概率以提高模型性能
        transition_matrix = np.array([
            [0.8, 0.2],  # Sunny -> Sunny, Sunny -> Rainy
            [0.2, 0.8]   # Rainy -> Sunny, Rainy -> Rainy
        ])
        
        # 调整发射概率使状态更容易区分
        emission_matrix = np.array([
            [0.7, 0.2, 0.1],  # Sunny -> [Walk, Shop, Clean]
            [0.1, 0.2, 0.7]   # Rainy -> [Walk, Shop, Clean]
        ])
        
        # 均匀的初始分布
        initial_distribution = np.array([0.5, 0.5])
        
        # 生成隐状态序列
        hidden_states = []
        current_state = np.random.choice(len(self.states), p=initial_distribution)
        hidden_states.append(current_state)
        
        for _ in range(sequence_length - 1):
            current_state = np.random.choice(len(self.states), 
                                           p=transition_matrix[current_state])
            hidden_states.append(current_state)
            
        # 生成观测序列
        observations = []
        for state in hidden_states:
            observation = np.random.choice(len(self.observations), 
                                         p=emission_matrix[state])
            observations.append(observation)
            
        return np.array(hidden_states), np.array(observations)
    

class GMMDataset:
    """
    创建一个高斯混合模型聚类数据集
    
    特征：
    - 2维特征空间中的点
    - 来自多个高斯分布的混合
    
    聚类：
    - n_components个不同的高斯分布簇
    
    参数：
        n_samples: 样本数量，默认300
        n_components: 高斯分布的数量，默认3
        random_state: 随机种子，默认42
    """
    def __init__(self, n_samples=300, n_components=3, random_state=42):
        """
        创建一个用于EM算法的高斯混合模型数据集
        
        参数:
            n_samples: 样本数量
            n_components: 高斯分布的数量
            random_state: 随机种子
        """
        self.X, self.y = make_blobs(n_samples=n_samples, 
                                  centers=n_components,
                                  random_state=random_state)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] 

class GeneExpressionDataset:
    def __init__(self, n_samples=1000, random_state=42):
        """
        创建一个简化的癌症基因表达数据集（二维特征用于可视化）
        
        特征：
        - 选择2个重要基因的表达水平
        - 每个基因的表达值范围在0-15之间
        
        癌症亚型：
        1. A型：CCND1高表达，CD8A低表达
        2. B型：CCND1低表达，CD8A高表达
        3. C型：CCND1和CD8A都处于中等水平
        
        参数:
            n_samples: 样本数量
            random_state: 随机种子
        """
        np.random.seed(random_state)
        self.logger = Logger.get_logger('GeneExpressionDataset')
        
        # 每个亚型的样本量
        samples_per_type = n_samples // 3
        
        # 生成A型样本（CCND1高表达，CD8A低表达）
        type_a = np.random.multivariate_normal(
            mean=[12, 3],
            cov=[[1.0, 0.0], [0.0, 1.0]],
            size=samples_per_type
        )
        
        # 生成B型样本（CCND1低表达，CD8A高表达）
        type_b = np.random.multivariate_normal(
            mean=[3, 12],
            cov=[[1.0, 0.0], [0.0, 1.0]],
            size=samples_per_type
        )
        
        # 生成C型样本（两个基因都是中等表达）
        type_c = np.random.multivariate_normal(
            mean=[7.5, 7.5],
            cov=[[1.0, -0.5], [-0.5, 1.0]],
            size=samples_per_type
        )
        
        # 合并所有数据
        self.X = np.vstack([type_a, type_b, type_c])
        
        # 确保基因表达值在合理范围内（0-15）
        self.X = np.clip(self.X, 0, 15)
        
        # 生成标签
        self.y = np.concatenate([
            np.zeros(samples_per_type),      # A型
            np.ones(samples_per_type),       # B型
            2 * np.ones(samples_per_type)    # C型
        ])
        
        # 打乱数据顺序
        idx = np.random.permutation(len(self.X))
        self.X = self.X[idx]
        self.y = self.y[idx]
        
        # 基因名称
        self.feature_names = [
            'CCND1',  # 细胞增殖相关基因
            'CD8A'    # 免疫反应相关基因
        ]
        
        # 亚型名称
        self.target_names = [
            'A型(增殖活跃型)',
            'B型(免疫活性型)',
            'C型(平衡型)'
        ]
        
        self.logger.print("癌症基因表达数据集已加载：")
        self.logger.print(f"- 总样本数: {len(self.X)}")
        self.logger.print(f"- 基因数量: {len(self.feature_names)}")
        self.logger.print(f"- 癌症亚型数: {len(self.target_names)}")
        self.logger.print(f"- 选定基因: {', '.join(self.feature_names)}")
        self.logger.print("- 数据分布特点:")
        self.logger.print("  * A型: CCND1高表达，CD8A低表达")
        self.logger.print("  * B型: CCND1低表达，CD8A高表达")
        self.logger.print("  * C型: 两个基因表达水平均衡")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

