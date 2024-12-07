from dlframe import WebManager, Logger
import math
import numpy as np

class KFoldSplitter:
    def __init__(self, k=5, shuffle=True, random_state=None):
        self.k = k
        self.shuffle = shuffle
        self.random_state = random_state
        self.logger = Logger.get_logger('KFoldSplitter')
        self.logger.print(f"KFold Splitter initialized with k={k}, shuffle={shuffle}")
        
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer greater than 1")

    def split(self, dataset):
        """
        将数据集分割为k个折叠
        
        Args:
            dataset: 输入数据集
            
        Returns:
            list: k个折叠的列表
        """
        import random
        n = len(dataset)
        if n < self.k:
            raise ValueError(f"Dataset size {n} is smaller than k={self.k}")
            
        # 生成索引
        indices = list(range(n))
        if self.shuffle:
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(indices)
            
        # 计算每个折叠的大小
        fold_size = n // self.k
        folds = []
        
        for i in range(self.k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.k - 1 else n
            fold = [dataset[indices[j]] for j in range(start, end)]
            folds.append(fold)
            
        return folds
    
    def get_train_valid_split(self, dataset, fold_idx):
        """
        获取指定折叠索引的训练集和验证集
        
        Args:
            dataset: 输入数据集
            fold_idx: 用作验证集的折叠索引
            
        Returns:
            tuple: (训练集, 验证集)
        """
        if not 0 <= fold_idx < self.k:
            raise ValueError(f"fold_idx must be between 0 and {self.k-1}")
            
        folds = self.split(dataset)
        valid_set = folds[fold_idx]
        train_set = []
        for i, fold in enumerate(folds):
            if i != fold_idx:
                train_set.extend(fold)
                
        return train_set, valid_set

class RandomSplitter:
    def __init__(self, ratio):
        self.ratio = ratio
        self.logger = Logger.get_logger('RandomSplitter')
        self.logger.print("Random Splitter initialized with ratio={}".format(ratio))

    def split(self, dataset):
        import random
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        split_index = int(len(dataset) * self.ratio)
        train_indices = indices[:split_index]
        test_indices = indices[split_index:]
        train_set = [dataset[i] for i in train_indices]
        test_set = [dataset[i] for i in test_indices]
        return train_set, test_set

class HMMDataSplitter:
    def __init__(self, test_size=0.2):
        self.test_size = test_size
        self.logger = Logger.get_logger('HMMDataSplitter')
        
    def split(self, dataset):
        """将数据集分割为训练集和测试集"""
        data_size = len(dataset)
        train_size = int(data_size * (1 - self.test_size))
        
        # 分割数据
        train_data = [dataset[i] for i in range(train_size)]
        test_data = [dataset[i] for i in range(train_size, data_size)]

        return train_data, test_data
    
    def prepare_sequences(self, observations, sequence_length=100):
        """将长序列分割成多个短序列"""
        sequences = []
        for i in range(0, len(observations) - sequence_length + 1, sequence_length):
            sequence = observations[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)