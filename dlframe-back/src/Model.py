from dlframe import WebManager, Logger

import math
import numpy as np
from hmmlearn.hmm import GaussianHMM
from hmmlearn import hmm
from dlframe import Logger
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sns
from PIL import Image
from sklearn.decomposition import PCA
from copy import deepcopy
import json

class KNNModel:
    def __init__(self, k=3):
        self.k = k
        self.logger = Logger.get_logger('KNNModel')
        self.logger.print("KNN Model initialized with k={}".format(k))
    def train(self, trainDataset):
        self.trainData = [x[0] for x in trainDataset]
        self.trainLabels = [x[1] for x in trainDataset]
        self.logger.print("Training completed")
    
    def visualize(self , testDataset):
        predictions = []
        distances_list = []

        for data in testDataset:
            distances = [np.linalg.norm(data[0] - x) for x in self.trainData]
            distances_list.append(distances)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.trainLabels[i] for i in nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)

        # 计算预测标签的分布
        unique_labels, counts = np.unique(predictions, return_counts=True)
        # 可视化
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. 距离分布直方图
        distances_flat = np.array(distances_list).flatten()
        ax1.hist(distances_flat, bins=50)
        ax1.set_title('Distance Distribution')
        ax1.set_xlabel('Distance')
        ax1.set_ylabel('Count')
        
        # 2. 预测标签分布
        ax2.bar(unique_labels, counts)
        ax2.set_title('Prediction Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        
        # 3. K近邻示例(第一个测试样本)
        first_distances = distances_list[0]
        sorted_indices = np.argsort(first_distances)[:self.k]
        ax3.scatter(range(self.k), [first_distances[i] for i in sorted_indices])
        ax3.set_title(f'K={self.k} Nearest Neighbors\nfor First Test Sample')
        ax3.set_xlabel('Neighbor Rank')
        ax3.set_ylabel('Distance')
        
        # 4. 准确率随K值变化
        k_range = range(1, min(20, len(self.trainData)))
        accuracies = []
        for k in k_range:
            correct = 0
            for i, data in enumerate(testDataset):
                distances = distances_list[i]
                k_nearest = np.argsort(distances)[:k]
                pred = Counter([self.trainLabels[j] for j in k_nearest]).most_common(1)[0][0]
                if pred == data[1]:
                    correct += 1
            accuracies.append(correct/len(testDataset))
        
        ax4.plot(k_range, accuracies)
        ax4.set_title('Accuracy vs K')
        ax4.set_xlabel('K')
        ax4.set_ylabel('Accuracy')
        
        plt.tight_layout()
        
        # 转换图像为数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)
        
    def test(self, testDataset):
        predictions = []
        distances_list = []

        for data in testDataset:
            distances = [np.linalg.norm(data[0] - x) for x in self.trainData]
            distances_list.append(distances)
            nearest_indices = np.argsort(distances)[:self.k]
            nearest_labels = [self.trainLabels[i] for i in nearest_indices]
            most_common_label = Counter(nearest_labels).most_common(1)[0][0]
            predictions.append(most_common_label)
        self.visualize(testDataset)
        return predictions
    
class DecisionTreeModel:
    def __init__(self):
        from sklearn.tree import DecisionTreeClassifier
        self.model = DecisionTreeClassifier()
        self.logger = Logger.get_logger('DecisionTreeModel')
        self.logger.print("决策树模型初始化完成")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("训练完成")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        from sklearn.tree import plot_tree
        import numpy as np
        from io import BytesIO
        from PIL import Image
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 决策树结构图
        plot_tree(self.model, ax=ax1, filled=True, feature_names=None)
        ax1.set_title('decision tree')
        
        # 2. 特征重要性
        importances = self.model.feature_importances_
        ax2.bar(range(len(importances)), importances)
        ax2.set_title('feature importance')
        ax2.set_xlabel('feature')
        ax2.set_ylabel('importance')
        
        # 3. 预测结果分布
        unique_labels, counts = np.unique(y_pred, return_counts=True)
        ax3.pie(counts, labels=[f'sample {i}' for i in unique_labels], autopct='%1.1f%%')
        ax3.set_title('prediction distribution')
        
        # 4. 准确率统计
        correct = np.sum(y_pred == y_test)
        accuracy = correct / len(y_test)
        ax4.bar(['accuracy'], [accuracy])
        ax4.set_ylim(0, 1)
        ax4.set_title(f'model accuracy: {accuracy:.2%}')
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)

    def test(self, testDataset):
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        self.visualize(testDataset)
        return y_pred

class SVMModel:
    def __init__(self):
        from sklearn.svm import SVC
        # 初始化时启用概率预测
        self.model = SVC(probability=True)
        self.logger = Logger.get_logger('SVMModel')
        self.logger.print("SVM模型初始化完成")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("训练完成")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        from sklearn.metrics import confusion_matrix
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 特征空间散点图(使用前两个特征)
        scatter = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.8)
        if hasattr(self.model, 'support_vectors_'):
            ax1.scatter(self.model.support_vectors_[:, 0], 
                       self.model.support_vectors_[:, 1], 
                       c='r', marker='x', label='Support Vectors')
        ax1.set_title('feature space scatter plot')
        ax1.set_xlabel('feature1')
        ax1.set_ylabel('feature2')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1)
        
        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('confusion matrix')
        plt.colorbar(im, ax=ax2)
        
        # 在混淆矩阵中添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 3. 预测概率分布
        for i in range(y_prob.shape[1]):
            ax3.hist(y_prob[:, i], bins=20, alpha=0.5, label=f'label {i}')
        ax3.set_title('prediction probability distribution')
        ax3.set_xlabel('probability')
        ax3.set_ylabel('count')
        ax3.legend()
        
        # 4. 准确率统计
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        ax4.bar(['accuracy'], [accuracy])
        ax4.set_ylim(0, 1)
        ax4.set_title(f'model accuracy: {accuracy:.2%}')
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)

    def test(self, testDataset):
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        self.visualize(testDataset)
        
        return y_pred
    
class NaiveBayesModel:
    def __init__(self):
        from sklearn.naive_bayes import GaussianNB
        self.model = GaussianNB()
        self.logger = Logger.get_logger('NaiveBayesModel')
        self.logger.print("朴素贝叶斯模型初始化完成")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("训练完成")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        from sklearn.metrics import confusion_matrix
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 类条件概率密度
        feature_idx = 0  # 使用第一个特征作为示例
        for i in range(len(self.model.classes_)):
            class_samples = X_test[y_test == i][:, feature_idx]
            if len(class_samples) > 0:
                ax1.hist(class_samples, bins=20, alpha=0.5, 
                        label=f'label {i}', density=True)
        ax1.set_title('class conditional probability density')
        ax1.set_xlabel('feature value')
        ax1.set_ylabel('density')
        ax1.legend()
        
        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('confusion matrix')
        plt.colorbar(im, ax=ax2)
        
        # 在混淆矩阵中添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 3. 预测概率分布
        for i in range(y_prob.shape[1]):
            ax3.hist(y_prob[:, i], bins=20, alpha=0.5, 
                    label=f'label {i}')
        ax3.set_title('prediction probability distribution')
        ax3.set_xlabel('probability')
        ax3.set_ylabel('count')
        ax3.legend()
        
        # 4. 准确率和先验概率
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        priors = self.model.class_prior_
        x = np.arange(len(priors))
        width = 0.35
        
        ax4.bar(x - width/2, priors, width, label='prior')
        ax4.bar(x + width/2, [np.mean(y_pred == i) for i in range(len(priors))], 
                width, label='prediction')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'label {i}' for i in range(len(priors))])
        ax4.set_title(f'prior and prediction accuracy: {accuracy:.2%}')
        ax4.legend()
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)
    def test(self, testDataset):
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 行预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        self.visualize(testDataset)  
        return y_pred
    
class KMeansModel:
    def __init__(self, k=3):
        import os
        from sklearn.cluster import KMeans
        
        # 设置OMP_NUM_THREADS环境变量来避免内存泄漏
        os.environ['OMP_NUM_THREADS'] = '5'
        
        self.model = KMeans(n_clusters=k)
        self.logger = Logger.get_logger('KMeansModel')
        self.logger.print("K-means模型初始化完成")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        self.model.fit(X_train)
        self.logger.print("训练完成")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        from sklearn.metrics import silhouette_score, silhouette_samples
        from sklearn.decomposition import PCA
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        
        # 进行测
        labels = self.model.predict(X_test)
        
        # 创建图
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 二维聚类结果图(使用PCA降维)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_test)
        centroids_pca = pca.transform(self.model.cluster_centers_)
        
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        ax1.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='centroids')
        ax1.set_title('result of clustering(PCA)')
        ax1.set_xlabel('main component1')
        ax1.set_ylabel('main component2')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1)
        
        # 2. 各簇样本数量分布
        unique_labels, counts = np.unique(labels, return_counts=True)
        ax2.bar([f'cluster {i}' for i in unique_labels], counts)
        ax2.set_title('cluster sample count distribution')
        ax2.set_xlabel('cluster')
        ax2.set_ylabel('count')
        
        # 3. 轮廓系数分布
        silhouette_avg = silhouette_score(X_test, labels)
        ax3.hist(silhouette_samples(X_test, labels), bins=20)
        ax3.axvline(silhouette_avg, color='red', linestyle='--', 
                   label=f'avg: {silhouette_avg:.3f}')
        ax3.set_title('cluster silhouette coefficient distribution')
        ax3.set_xlabel('silhouette coefficient')
        ax3.set_ylabel('count')
        ax3.legend()
        
        # 4. 簇内距离分布
        intra_dists = []
        for i in range(self.model.n_clusters):
            cluster_points = X_test[labels == i]
            centroid = self.model.cluster_centers_[i]
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            intra_dists.append(dists)
        
        ax4.boxplot(intra_dists, labels=[f'cluster {i}' for i in range(self.model.n_clusters)])
        ax4.set_title('cluster intra-distance distribution')
        ax4.set_xlabel('cluster')
        ax4.set_ylabel('distance')
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)   

    def test(self, testDataset):        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        
        # 进行预测
        labels = self.model.predict(X_test)
        self.visualize(testDataset)  

        return labels
    
class BoostingModel:
    def __init__(self):
        from sklearn.ensemble import AdaBoostClassifier
        self.model = AdaBoostClassifier()
        self.logger = Logger.get_logger('BoostingModel')
        self.logger.print("Boosting Model initialized")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("Training completed")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 获取特征重要性
        importances = self.model.feature_importances_
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 特征重要性
        feature_indices = range(len(importances))
        ax1.bar(feature_indices, importances)
        ax1.set_title('feature importance')
        ax1.set_xlabel('feature')
        ax1.set_ylabel('importance')
        ax1.grid(True)
        
        # 2. 预测概率分布
        for i in range(y_pred_proba.shape[1]):
            ax2.hist(y_pred_proba[:, i], bins=20, alpha=0.5, label=f'label {i}')
        ax2.set_title('prediction probability distribution')
        ax2.set_xlabel('probability')
        ax2.set_ylabel('count')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 预测结果分布
        unique_labels, counts = np.unique(y_pred, return_counts=True)
        ax3.pie(counts, labels=[f'label {i}' for i in unique_labels], autopct='%1.1f%%')
        ax3.set_title('prediction result distribution')
        
        # 4. 准确率统计
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        ax4.bar(['accuracy'], [accuracy])
        ax4.set_ylim(0, 1)
        ax4.set_title(f'model accuracy: {accuracy:.2%}')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)
        
    def test(self, testDataset):
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        self.visualize(testDataset)  
        return y_pred
    
class EMModel:
    def __init__(self, n_components=3, max_iter=100):
        from sklearn.mixture import GaussianMixture
        self.n_components = n_components
        self.max_iter = max_iter
        self.model = GaussianMixture(
            n_components=n_components,
            max_iter=max_iter,
            random_state=42
        )
        self.logger = Logger.get_logger('EMModel')
        self.logger.print(f"EM Model initialized with {n_components} components")

    def train(self, trainDataset):
        import numpy as np
        
        try:
            # 提取特征
            X = np.array([x[0] for x in trainDataset])
            self.logger.print(f"Training data shape: {X.shape}")
            
            # 训练模型
            self.model.fit(X)
            self.logger.print("Model fitting completed")
            
            # 返回预测概率
            probabilities = self.model.predict_proba(X)
            self.logger.print(f"Training completed in {self.model.n_iter_} iterations")
            self.logger.print(f"Convergence status: {self.model.converged_}")
            
            return probabilities
            
        except Exception as e:
            self.logger.print(f"Error during EM training: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None

    def test(self, testDataset):
        import numpy as np
        
        try:
            # 提取特征
            X = np.array([x[0] for x in testDataset])
            self.logger.print(f"Test data shape: {X.shape}")
            
            # 返回预测概率
            return self.model.predict_proba(X)
            
        except Exception as e:
            self.logger.print(f"Error during EM testing: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None


class LogisticRegressionModel:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression()
        self.logger = Logger.get_logger('LogisticRegressionModel')
        self.logger.print("Logistic Regression Model initialized")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("Training completed")

    def visualize(self , testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        from sklearn.metrics import confusion_matrix
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 模型系数权重
        coef = self.model.coef_[0]
        feature_indices = range(len(coef))
        ax1.bar(feature_indices, coef)
        ax1.set_title('coefficient weights')
        ax1.set_xlabel('feature')
        ax1.set_ylabel('impact on probability')
        ax1.grid(True)
        
        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('confusion matrix')
        plt.colorbar(im, ax=ax2)
        
        # 在混淆矩阵中添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 3. 预测概分布
        for i in range(y_pred_proba.shape[1]):
            ax3.hist(y_pred_proba[:, i], bins=20, alpha=0.5, label=f'label {i}')
        ax3.set_title('prediction probability distribution')
        ax3.set_xlabel('probability')
        ax3.set_ylabel('count')
        ax3.legend()
        ax3.grid(True)
        
        # 4. 准确率统计
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        ax4.bar(['accuracy'], [accuracy])
        ax4.set_ylim(0, 1)
        ax4.set_title(f'model accuracy: {accuracy:.2%}')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)

    def test(self, testDataset):
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        self.visualize(testDataset)  
        return y_pred
    
class MaximumEntropyModel:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
        self.logger = Logger.get_logger('MaximumEntropyModel')
        self.logger.print("Maximum Entropy Model initialized")

    def train(self, trainDataset):
        X_train = [x[0] for x in trainDataset]
        y_train = [x[1] for x in trainDataset]
        self.model.fit(X_train, y_train)
        self.logger.print("Training completed")

    def visualize(self, testDataset):
        import matplotlib.pyplot as plt
        import numpy as np
        from io import BytesIO
        from PIL import Image
        from sklearn.metrics import confusion_matrix
        
        # 获取测试数据
        X_test = np.array([x[0] for x in testDataset])
        y_test = np.array([x[1] for x in testDataset])
        
        # 进行预测
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)
        
        # 创建可视化图表
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 特征空间散点图(使用前两个特征)
        scatter = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, alpha=0.8)
        if hasattr(self.model, 'support_vectors_'):
            ax1.scatter(self.model.support_vectors_[:, 0], 
                       self.model.support_vectors_[:, 1], 
                       c='r', marker='x', label='Support Vectors')
        ax1.set_title('feature space scatter plot')
        ax1.set_xlabel('feature1')
        ax1.set_ylabel('feature2')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1)
        
        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('confusion matrix')
        plt.colorbar(im, ax=ax2)
        
        # 在混淆矩阵中添加数值标注
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        # 3. 预测概率分布
        for i in range(y_prob.shape[1]):
            ax3.hist(y_prob[:, i], bins=20, alpha=0.5, label=f'label {i}')
        ax3.set_title('prediction probability distribution')
        ax3.set_xlabel('probability')
        ax3.set_ylabel('count')
        ax3.legend()
        
        # 4. 准确率统计
        accuracy = np.sum(y_pred == y_test) / len(y_test)
        ax4.bar(['accuracy'], [accuracy])
        ax4.set_ylim(0, 1)
        ax4.set_title(f'model accuracy: {accuracy:.2%}')
        
        plt.tight_layout()
        
        # 转换为图像数组
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 显示图像
        self.logger.imshow(img_array)

    def test(self, testDataset):
        X_test = [x[0] for x in testDataset]
        self.visualize(testDataset)
        return self.model.predict(X_test)
    
class HMMModel:
    def __init__(self, n_states, n_observations):
        """
        初始化HMM模型
        
        Args:
            n_states: 隐状态数量
            n_observations: 观测状态数量
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.logger = Logger.get_logger('HMMModel')
        
        # 初始化模型参数
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """初始化模型参数"""
        # 使用更好的初始化策略
        self.A = np.random.dirichlet(np.ones(self.n_states) * 10, size=self.n_states)
        self.B = np.random.dirichlet(np.ones(self.n_observations) * 10, size=self.n_states)
        self.pi = np.random.dirichlet(np.ones(self.n_states) * 10)
        
    def _normalize(self, matrix):
        """归一化矩阵，避免数值下溢"""
        return matrix / (np.sum(matrix, axis=1, keepdims=True) + 1e-12)
    
    def forward(self, observations):
        """前向算法的向量化实现"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        scaling_factors = np.zeros(T)
        
        # 初始化
        alpha[0] = self.pi * self.B[:, observations[0]]
        scaling_factors[0] = np.sum(alpha[0])
        alpha[0] /= scaling_factors[0]
        
        # 递归计算
        for t in range(1, T):
            alpha[t] = np.dot(alpha[t-1], self.A) * self.B[:, observations[t]]
            scaling_factors[t] = np.sum(alpha[t])
            alpha[t] /= scaling_factors[t]
            
        return alpha, scaling_factors
    
    def backward(self, observations, scaling_factors):
        """后向算法的向量化实现"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))
        
        # 初始化
        beta[-1] = 1 / scaling_factors[-1]
        
        # 递归计算
        for t in range(T-2, -1, -1):
            beta[t] = np.dot(self.A, (self.B[:, observations[t+1]] * beta[t+1]))
            beta[t] /= scaling_factors[t]
            
        return beta
    
    def train(self, trainDataset):
        """训练模型的包装方法"""
        try:
            # 提取观测序列
            observations = np.array([x[0] for x in trainDataset])
            
            # 训练模型
            return self.train_internal(observations)
            
        except Exception as e:
            self.logger.print(f"训练过程中出现错误: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            
    def train_internal(self, observations, n_iterations=200, tolerance=1e-7):
        """内部训练方法"""
        prev_log_likelihood = float('-inf')
        final_log_likelihood = float('-inf')  # 初始化为负无穷
        
        try:
            for iteration in range(n_iterations):
                # E步
                alpha, scaling_factors = self.forward(observations)
                beta = self.backward(observations, scaling_factors)
                
                # 计算log likelihood
                current_log_likelihood = np.sum(np.log(scaling_factors + 1e-12))  # 添加小量避免log(0)
                final_log_likelihood = current_log_likelihood  # 更新最终的似然值
                
                # 检查收敛
                if abs(current_log_likelihood - prev_log_likelihood) < tolerance:
                    self.logger.print(f"模型在第{iteration}次迭代后收敛")
                    break
                    
                prev_log_likelihood = current_log_likelihood
                
                # 计算gamma和xi
                gamma = alpha * beta
                xi = self._compute_xi(observations, alpha, beta)
                
                # M步：更新参数
                self._update_parameters(observations, gamma, xi)
                
            return final_log_likelihood  # 返回最终的似然值
            
        except Exception as e:
            self.logger.print(f"训练迭代过程中出现错误: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return float('-inf')  # 出错时返回负无穷
    
    def _compute_xi(self, observations, alpha, beta):
        """计算xi矩阵"""
        T = len(observations)
        xi = np.zeros((T-1, self.n_states, self.n_states))
        
        for t in range(T-1):
            numerator = (alpha[t].reshape(-1, 1) * self.A * 
                        self.B[:, observations[t+1]].reshape(1, -1) * 
                        beta[t+1].reshape(1, -1))
            xi[t] = numerator / np.sum(numerator)
            
        return xi
        
    def _update_parameters(self, observations, gamma, xi):
        """更新模型参数"""
        # 更新初始概率
        self.pi = gamma[0]
        
        # 更新转移概率
        self.A = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0).reshape(-1, 1)
        
        # 更新发射概率
        for j in range(self.n_states):
            for k in range(self.n_observations):
                self.B[j,k] = np.sum(gamma[observations == k, j]) / np.sum(gamma[:, j])
                
        # 归一化参数
        self.A = self._normalize(self.A)
        self.B = self._normalize(self.B)
        
    def predict(self, observations):
        """使用维特比算法进行预测"""
        T = len(observations)
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # 初始化
        delta[0] = np.log(self.pi) + np.log(self.B[:, observations[0]])
        
        # 递归
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] + np.log(self.A[:, j])
                delta[t,j] = np.max(temp) + np.log(self.B[j, observations[t]])
                psi[t,j] = np.argmax(temp)
        
        # 回溯
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
            
        return states
    
    def visualize(self, save_path=None):
        """可视化模型参数"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
        # 转移概率矩阵
        sns.heatmap(self.A, annot=True, cmap='YlOrRd', ax=ax1)
        ax1.set_title('Transition Matrix')  # 使用英文替代中文
    
        # 发射概率矩阵
        sns.heatmap(self.B, annot=True, cmap='YlOrRd', ax=ax2)
        ax2.set_title('Emission Matrix')  # 使用英文替代中文
    
        # 初始概率分布
        ax3.bar(range(self.n_states), self.pi)
        ax3.set_title('Initial State Distribution')  # 使用英文替代中文
        ax3.set_xlabel('State')
        ax3.set_ylabel('Probability')
    
        # 状态转移图
        ax4.imshow(np.log(self.A), cmap='coolwarm')
        ax4.set_title('Log Transition Probabilities')  # 使用英文替代中文
    
        plt.tight_layout()
    
        if save_path:
           plt.savefig(save_path)
        
        # 转换为图像数组并显示
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
    
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
    
        plt.close()
        buf.close()
    
        self.logger.imshow(img_array)
    
    def test(self, testDataset):
        """测试模型"""
        try:
            # 提取观测序列
            observations = np.array([x[0] for x in testDataset])
            
            # 使用维特比算法预测隐状态
            predicted_states = self.predict(observations)
            
            # 可视化模型参数
            self.visualize()
            
            return predicted_states
            
        except Exception as e:
            self.logger.print(f"测试过程中出现错误: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None