import sys
sys.path.append('dlframe-back')  # 导入父目录模块
from dlframe import WebManager, Logger
from src.Model import *
from src.Dataset import *
from src.Splitter import *
from src.Judger import *
import math
import numpy as np
import os
os.environ['OMP_NUM_THREADS'] = '1'  # 设置为单线程运行，避免 KMeans 的内存泄漏

# 数据集
class TestDataset:
    def __init__(self, num) -> None:
        super().__init__()
        self.num = range(num)  # 创建一个从0到num-1的数字序列
        self.logger = Logger.get_logger('TestDataset')  # 获取日志记录器
        self.logger.print("I'm in range 0, {}".format(num))  # 打印数据集范围信息

    def __len__(self) -> int:
        return len(self.num)  # 返回数据集长度

    def __getitem__(self, idx: int):
        return self.num[idx]  # 返回指定索引的数据

class TrainTestDataset:
    def __init__(self, item) -> None:
        super().__init__()
        self.item = item  # 存储训练或测试数据

    def __len__(self) -> int:
        return len(self.item)  # 返回数据集长度

    def __getitem__(self, idx: int):
        return self.item[idx]  # 返回指定索引的数据

# 数据集切分器
class TestSplitter:
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio  # 训练集和测试集的分割比例
        self.logger = Logger.get_logger('TestSplitter')
        self.logger.print("I'm ratio:{}".format(self.ratio))

    def split(self, dataset):
        # 根据比例分割数据集为训练集
        trainingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio))]
        trainingSet = TrainTestDataset(trainingSet)

        # 剩余部分作为测试集
        testingSet = [dataset[i] for i in range(math.floor(len(dataset) * self.ratio), len(dataset))]
        testingSet = TrainTestDataset(testingSet)

        self.logger.print("split!")
        self.logger.print("training_len = {}".format([trainingSet[i] for i in range(len(trainingSet))]))
        return trainingSet, testingSet

# 模型
class TestModel:
    def __init__(self, learning_rate) -> None:
        super().__init__()
        self.learning_rate = learning_rate  # 学习率
        self.logger = Logger.get_logger('TestModel')

    def train(self, trainDataset) -> None:
        # 训练模型并打印信息
        self.logger.print("trainging, lr = {}, trainDataset = {}".format(self.learning_rate, trainDataset))

    def test(self, testDataset):
        # 测试模型并显示随机生成的图像
        self.logger.print("testing")
        self.logger.imshow(np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8))
        return testDataset

# 结果判别器
class TestJudger:
    def __init__(self) -> None:
        super().__init__()
        self.logger = Logger.get_logger('TestJudger')

    def judge(self, y_hat, test_dataset) -> None:
        # 比较预测结果和真实值
        self.logger.print("y_hat = {}".format([y_hat[i] for i in range(len(y_hat))]))
        self.logger.print("gt = {}".format([test_dataset[i] for i in range(len(test_dataset))]))

if __name__ == '__main__':
    with WebManager(host='0.0.0.0', port=8765) as manager:
        # 注册数据集
        dataset = manager.register('数据集', {
            'iris(鸢尾花)': IrisDataset(),
            'wine(红酒)': WineDataset(),
            'breast-cancer(乳腺癌)': BreastCancerDataset(),
            'weather(天气)': HMMDataset(),  # 天气-活动HMM数据集
        })
        
        splitter = manager.register_element('数据分割', {
            'ratio:0.8': RandomSplitter(0.8), 
            'ratio:0.5': RandomSplitter(0.5),
            'HMMDataSplitter': HMMDataSplitter()
        })
    
        train_data_test_data = splitter.split(dataset)  # 分割数据集
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]  # 获取训练集和测试集
    
        # 注册模型，为HMM模型提供正确的参数
        Model = manager.register('模型', {
            'knn': KNNModel(),
            'decision-tree': DecisionTreeModel(), 
            'svm': SVMModel(),
            'naive-bayes': NaiveBayesModel(),
            'boosting': BoostingModel(),
            'Logistic Regression': LogisticRegressionModel(),
            'Maximum Entropy': MaximumEntropyModel(),
            'em': EMModel(),
            'k-means': KMeansModel(),
            'HMM': HMMModel(n_states=2, n_observations=3),  # 2个状态(晴天/雨天)，3个观测(散步/购物/清洁)
        })
        
        Model.train(train_data)  # 训练模型
        y_hat = Model.test(test_data)  # 测试模型
        
        judger = manager.register('评价指标', {
            '准确率accuracy': AccuracyJudger(), 
            '精确率precision': PrecisionJudger(),
            '召回率recall': RecallJudger(),
            'f1': F1ScoreJudger(),
            'EM评价器': EMJudger(),
            'Clustering(使用于k-means)': ClusteringJudger(),
            '天气预报评价器': HMMJudger(),
            'ROC-AUC曲线': ROCAUCJudger()
        })

        # 执行完整的训练和测试流程
        judger.judge(y_hat, test_data)  # 评估结果