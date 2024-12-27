import sys
import asyncio
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

import threading

def run_in_new_thread(manager):
    """在新线程中运行模型训练和评估"""
    try:
        # 注册数据集
        dataset = manager.register('数据集', {
            'iris(鸢尾花)': IrisDataset(),
            'wine(红酒)': WineDataset(),
            'breast-cancer(乳腺癌)': BreastCancerDataset(),
            'Gaussian(高斯分布)': GMMDataset(),  
            'Gene(癌症基因表达)': GeneExpressionDataset(),
            'weather(天气)': WeatherDataset(),  
        })
        
        splitter = manager.register_element('数据分割', {
            'Random:0.8': RandomSplitter(0.8), 
            'Random:0.5': RandomSplitter(0.5),
            'KFold:5': KFoldSplitter(k=5, shuffle=True, random_state=42),
            'KFold:10': KFoldSplitter(k=10, shuffle=True, random_state=42),
        })
    
        train_data_test_data = splitter.split(dataset)  # 分割数据集
        train_data, test_data = train_data_test_data[0], train_data_test_data[1]  # 获取训练集和测试集
    
        # 注册模型，为HMM模型提供正确的参数
        Model = manager.register('模型', {
            'KNN': KNNModel(),
            'Decision-Tree': DecisionTreeModel(), 
            'SVM': SVMModel(),
            'Naive-Bayes': NaiveBayesModel(),
            'Boosting': BoostingModel(),
            'Logistic Regression': LogisticRegressionModel(),
            'Maximum Entropy': MaximumEntropyModel(),
            'EM': EMModel(),
            'K-MEANS': KMeansModel(),
            'HMM': HMMModel(n_states=2, n_observations=3),  # 2个状态(晴天/雨天)，3个观测(散步/购物/清洁)
        })
        
        Model.train(train_data)  # 训练模型
        y_hat = Model.test(test_data)  # 测试模型
        
        judger = manager.register('评价指标', {
            '准确率accuracy': AccuracyJudger(), 
            '精确率precision': PrecisionJudger(),
            '召回率recall': RecallJudger(),
            'f1系数': F1ScoreJudger(),
            'ROC-AUC曲线': ROCAUCJudger(),
            '轮廓系数Clustering': ClusteringJudger(),
            '天气预报指标': HMMJudger()
        })

        # 执行完整的训练和测试流程
        judger.judge(y_hat, test_data)  # 评估结果
        

    except Exception as e:
        print(f"运行时错误: {e}")

if __name__ == '__main__':
    # 创建新的事件循环
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        with WebManager(host='0.0.0.0', port=8765) as manager:
            # 在新线程中运行模型训练和评估
            training_thread = threading.Thread(target=run_in_new_thread, args=(manager,))
            training_thread.start()
            
            # 等待训练线程完成
            training_thread.join()
    finally:
        loop.close()