from dlframe import WebManager, Logger
import math
import numpy as np

class KFoldSplitter:
    def __init__(self, k=5):
        self.k = k
        self.logger = Logger.get_logger('KFoldSplitter')
        self.logger.print("KFold Splitter initialized with k={}".format(k))

    def split(self, dataset):
        n = len(dataset)
        fold_size = n // self.k
        folds = []
        for i in range(self.k):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.k - 1 else n
            fold = [dataset[j] for j in range(start, end)]
            folds.append(fold)
        return folds
    
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
        
        self.logger.print(f"数据集分割完成 - 训练集: {len(train_data)}样本, 测试集: {len(test_data)}样本")
        return train_data, test_data
    
    def prepare_sequences(self, observations, sequence_length=100):
        """将长序列分割成多个短序列"""
        sequences = []
        for i in range(0, len(observations) - sequence_length + 1, sequence_length):
            sequence = observations[i:i + sequence_length]
            sequences.append(sequence)
        return np.array(sequences)

