from dlframe import WebManager, Logger
import math
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

class IrisDataset:
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
    

class HMMDataset:
    def __init__(self):
        self.states = ['Sunny', 'Rainy']  # 隐状态
        self.observations = ['Walk', 'Shop', 'Clean']  # 观测状态
        self.hidden_states, self.obs_sequence = self.generate_data()
        self.logger = Logger.get_logger('HMMDataset')
        self.logger.print("天气-活动HMM数据集已加载")
        
    def __len__(self):
        return len(self.obs_sequence)
    
    def __getitem__(self, idx):
        """返回(观测值，隐状态)对"""
        return self.obs_sequence[idx], self.hidden_states[idx]
    
    def generate_data(self, sequence_length=2000):  # 增加序列长度
        """改进数据生成过程"""
        # 更现实的转移概率
        transition_matrix = np.array([
            [0.7, 0.3],  # Sunny -> Sunny, Sunny -> Rainy
            [0.3, 0.7]   # Rainy -> Sunny, Rainy -> Rainy
        ])
        
        # 更明显的发射概率差异
        emission_matrix = np.array([
            [0.5, 0.3, 0.2],  # Sunny -> [Walk, Shop, Clean]
            [0.2, 0.3, 0.5]   # Rainy -> [Walk, Shop, Clean]
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
    

