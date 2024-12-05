import numpy as np
from dlframe import Logger
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score, mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from io import BytesIO
from PIL import Image




class AccuracyJudger:
    def __init__(self):
        self.logger = Logger.get_logger('AccuracyJudger')

    def judge(self, y_hat, test_dataset):
        # 从测试数据集中提取真实标签
        y_true = np.array([x[1] for x in test_dataset])
        y_hat = np.array(y_hat)
        
        # 计算准确率
        accuracy = np.sum(y_hat == y_true) / len(y_true)
        
        self.logger.print(f"Accuracy: {accuracy:.4f}")
        return accuracy

class PrecisionJudger:
    def __init__(self):
        self.logger = Logger.get_logger('PrecisionJudger')

    def judge(self, y_hat, test_dataset):
        y_true = np.array([x[1] for x in test_dataset])
        y_hat = np.array(y_hat)
        
        # 计算精确率
        true_positive = np.sum((y_hat == 1) & (y_true == 1))
        predicted_positive = np.sum(y_hat == 1)
        
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        
        self.logger.print(f"Precision: {precision:.4f}")
        return precision

class RecallJudger:
    def __init__(self):
        self.logger = Logger.get_logger('RecallJudger')

    def judge(self, y_hat, test_dataset):
        y_true = np.array([x[1] for x in test_dataset])
        y_hat = np.array(y_hat)
        
        # 计算召回率
        true_positive = np.sum((y_hat == 1) & (y_true == 1))
        actual_positive = np.sum(y_true == 1)
        
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        
        self.logger.print(f"Recall: {recall:.4f}")
        return recall

class F1ScoreJudger:
    def __init__(self):
        self.logger = Logger.get_logger('F1ScoreJudger')

    def judge(self, y_hat, test_dataset):
        y_true = np.array([x[1] for x in test_dataset])
        y_hat = np.array(y_hat)
        
        # 计算精确率和召回率
        true_positive = np.sum((y_hat == 1) & (y_true == 1))
        predicted_positive = np.sum(y_hat == 1)
        actual_positive = np.sum(y_true == 1)
        
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        
        # 计算F1分数
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        self.logger.print(f"F1 Score: {f1:.4f}")
        return f1

class ClusteringJudger:
    
    def __init__(self):
        self.logger = Logger.get_logger('ClusteringJudger')
        
    def judge(self, cluster_labels, test_dataset):
        """
        评估聚类结果
        cluster_labels: 聚类预测的标签
        test_dataset: 原始数据集
        """
        import numpy as np
        from sklearn.metrics import (
            silhouette_score,
            calinski_harabasz_score,
            davies_bouldin_score
        )
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        try:
            # 提取特征数据
            X = np.array([x[0] for x in test_dataset])
            
            # 计算各种评估指标
            scores = {}
            
            # 1. 轮廓系数 (范围：[-1, 1]，越大越好)
            silhouette = silhouette_score(X, cluster_labels)
            scores['silhouette'] = silhouette
            
            # 2. Calinski-Harabasz指数 (越大越好)
            calinski = calinski_harabasz_score(X, cluster_labels)
            scores['calinski_harabasz'] = calinski
            
            # 3. Davies-Bouldin指数 (越小越好)
            davies = davies_bouldin_score(X, cluster_labels)
            scores['davies_bouldin'] = davies
            
            # 4. 计算簇内距离和簇间距离
            unique_labels = np.unique(cluster_labels)
            n_clusters = len(unique_labels)
            
            # 计算簇中心
            centroids = []
            intra_distances = []  # 簇内距离
            for label in unique_labels:
                mask = cluster_labels == label
                cluster_points = X[mask]
                centroid = np.mean(cluster_points, axis=0)
                centroids.append(centroid)
                
                # 计算簇内平均距离
                if len(cluster_points) > 1:
                    distances = np.linalg.norm(cluster_points - centroid, axis=1)
                    intra_distances.append(np.mean(distances))
                else:
                    intra_distances.append(0)
            
            centroids = np.array(centroids)
            
            # 计算簇间距离
            inter_distances = []
            for i in range(n_clusters):
                for j in range(i + 1, n_clusters):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    inter_distances.append(dist)
            
            # 打印评估结果
            self.logger.print("\nClustering Evaluation Results:")
            self.logger.print(f"Number of clusters: {n_clusters}")
            self.logger.print(f"Silhouette Score: {silhouette:.3f}")
            self.logger.print(f"Calinski-Harabasz Score: {calinski:.3f}")
            self.logger.print(f"Davies-Bouldin Score: {davies:.3f}")
            self.logger.print("\nCluster Statistics:")
            for i, label in enumerate(unique_labels):
                cluster_size = np.sum(cluster_labels == label)
                self.logger.print(f"Cluster {label}: {cluster_size} samples, "
                                f"Average intra-cluster distance: {intra_distances[i]:.3f}")
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski,
                'davies_bouldin_score': davies,
                'n_clusters': n_clusters,
                'cluster_sizes': [np.sum(cluster_labels == label) for label in unique_labels],
                'intra_cluster_distances': intra_distances,
                'inter_cluster_distances': inter_distances
            }
            
        except Exception as e:
            self.logger.print(f"Error during clustering evaluation: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None
        
class EMJudger:
    def __init__(self):
        self.logger = Logger.get_logger('EMJudger')

        
    # def judge(self, y_pred, test_dataset):
    #      # 提取特征数据
    #      X = np.array([x[0] for x in test_dataset])
    #      y_true = np.array([x[1] for x in test_dataset])
         
    #      # 计算似然函数值（假设使用高斯分布）
    #      mean = np.mean(X, axis=0)
    #      cov = np.cov(X, rowvar=False)
    #      likelihood = np.sum(-0.5 * (np.log(np.linalg.det(cov)) + 
    #                                    np.sum((X - mean) @ np.linalg.inv(cov) * (X - mean), axis=1)))
    #      self.logger.print(f"似然函数值Likelihood (calculated): {likelihood:.4f}")
         
    #      # 收敛速度（这里我们假设为固定值，实际情况需要根据具体算法实现）
    #      convergence_speed = 1  # 这里可以根据实际情况进行调整
    #      self.logger.print(f"收敛速度Convergence Speed (assumed): {convergence_speed}")
         
    #      # 模型复杂度（假设为特征数量）
    #      model_complexity = X.shape[1]  # 特征数量
    #      self.logger.print(f"模型复杂度Model Complexity (features): {model_complexity}")
         
    #      # 拟合优度（使用AIC）
    #      aic = 2 * model_complexity - 2 * likelihood
    #      self.logger.print(f"拟合优度AIC (Goodness of Fit): {aic:.4f}")
         
    #      from sklearn.metrics import precision_score, recall_score, f1_score
         
    #      # 从y_pred中提取预测的类别标签
    #      predicted_labels = np.argmax(y_pred, axis=1)  # 获取每个样本的预测类别
    #      accuracy = np.sum(predicted_labels == y_true) / len(y_true)
    #      self.logger.print(f"Classification Accuracy: {accuracy:.4f}")
         
    #      # 计算其他评估指标
    #      precision = precision_score(y_true, predicted_labels, average='weighted')
    #      recall = recall_score(y_true, predicted_labels, average='weighted')
    #      f1 = f1_score(y_true, predicted_labels, average='weighted')
         
    #      self.logger.print(f"Precision: {precision:.4f}")
    #      self.logger.print(f"Recall: {recall:.4f}")
    #      self.logger.print(f"F1 Score: {f1:.4f}")
         
    #      return {
    #          'likelihood': likelihood,
    #          'convergence_speed': convergence_speed,
    #          'model_complexity': model_complexity,
    #          'aic': aic,
    #          'accuracy': accuracy,
    #          'precision': precision,
    #          'recall': recall,
    #          'f1_score': f1
    #      }
    def judge(self, y_pred, test_dataset):
        import numpy as np
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        
        try:
            # 提取特征数据
            X = np.array([x[0] for x in test_dataset])
            
            # 获取预测标签
            labels = np.argmax(y_pred, axis=1)
            
            # 计算基本聚类评估指标
            scores = {
                'silhouette': silhouette_score(X, labels),
                'calinski_harabasz': calinski_harabasz_score(X, labels),
                'davies_bouldin': davies_bouldin_score(X, labels)
            }
            
            # 计算不确定性
            uncertainty = -np.sum(y_pred * np.log(y_pred + 1e-10), axis=1)
            avg_uncertainty = np.mean(uncertainty)
            max_uncertainty = np.max(uncertainty)
            
            # 创建可视化
            plt.switch_backend('Agg')
            fig = plt.figure(figsize=(15, 10))
            
            # 1. 预测概率分布
            ax1 = plt.subplot(221)
            n_components = y_pred.shape[1]
            ax1.boxplot([y_pred[:, i] for i in range(n_components)],
                       labels=[f'Component {i+1}' for i in range(n_components)])
            ax1.set_title('Component Probability Distribution')
            ax1.set_ylabel('Probability')
            ax1.grid(True)
            
            # 2. 不确定性分布
            ax2 = plt.subplot(222)
            ax2.hist(uncertainty, bins=30)
            ax2.axvline(avg_uncertainty, color='r', linestyle='--', 
                       label=f'Mean={avg_uncertainty:.3f}')
            ax2.set_title('Uncertainty Distribution')
            ax2.set_xlabel('Uncertainty')
            ax2.set_ylabel('Count')
            ax2.legend()
            ax2.grid(True)
            
            # 3. 聚类评估指标
            ax3 = plt.subplot(223)
            metric_names = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
            metric_values = [scores['silhouette'], 
                           scores['calinski_harabasz']/1000,
                           scores['davies_bouldin']]
            ax3.bar(metric_names, metric_values)
            ax3.set_title('Clustering Metrics')
            ax3.set_xticklabels(metric_names, rotation=45)
            ax3.grid(True)
            
            # 4. 类别分布
            ax4 = plt.subplot(224)
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            ax4.bar([f'Cluster {i+1}' for i in range(len(unique_labels))], 
                   label_counts)
            ax4.set_title('Cluster Size Distribution')
            ax4.set_ylabel('Number of Samples')
            ax4.grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            
            img = Image.open(buf)
            img = img.convert('RGB')
            img_array = np.array(img)
            
            plt.close()
            buf.close()
            
            # 显示图像
            self.logger.imshow(img_array)
            
            # 打印评估结果
            self.logger.print("\nEM Algorithm Evaluation Results:")
            self.logger.print(f"\nClustering Metrics:")
            self.logger.print(f"- 轮廓系数Silhouette Score: {scores['silhouette']:.3f}")
            self.logger.print(f"- 方差比Calinski-Harabasz Score: {scores['calinski_harabasz']:.3f}")
            self.logger.print(f"- 戴维森堡丁指数Davies-Bouldin Score: {scores['davies_bouldin']:.3f}")
            self.logger.print(f"\nUncertainty Analysis:")
            self.logger.print(f"- 平均不确定性Average uncertainty: {avg_uncertainty:.3f}")
            self.logger.print(f"- 最大不确定性Maximum uncertainty: {max_uncertainty:.3f}")
            self.logger.print(f"\nCluster Distribution:")
            for i, count in enumerate(label_counts):
                self.logger.print(f"- Cluster {i+1}: {count} samples")
            
            return {
                'clustering_scores': scores,
                'uncertainty': {
                    'mean': float(avg_uncertainty),
                    'max': float(max_uncertainty)
                },
                'cluster_distribution': {
                    'labels': labels.tolist(),
                    'counts': label_counts.tolist()
                }
            }
            
        except Exception as e:
            self.logger.print(f"Error during EM evaluation: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None

class HMMJudger:
    def __init__(self):
        self.logger = Logger.get_logger('HMMJudger')
        
    def judge(self, predicted_states, test_dataset):
        """
        评估HMM模型的性能
        
        Args:
            predicted_states: 模型预测的隐状态序列
            test_dataset: 测试数据集，包含真实的隐状态
        """
        try:
            # 获取真实隐状态
            true_states = np.array([x[1] for x in test_dataset])
            
            # 计算评估指标
            results = self.evaluate(true_states, predicted_states)
            
            # 打印评估结果
            self.print_results(results)
            
            # 可视化评估结果
            self._visualize_results(results, true_states, predicted_states)
            
            return results
            
        except Exception as e:
            self.logger.print(f"评估过程中出现错误: {str(e)}")
            import traceback
            self.logger.print(traceback.format_exc())
            return None
            
    def evaluate(self, true_states, predicted_states):
        """评估HMM模型的性能"""
        results = {}
        
        # 计算准确率
        results['accuracy'] = accuracy_score(true_states, predicted_states)
        
        # 计算混淆矩阵
        results['confusion_matrix'] = confusion_matrix(true_states, predicted_states)
        
        # 计算每个状态的精确率和召回率
        cm = results['confusion_matrix']
        n_states = cm.shape[0]
        
        precision = np.zeros(n_states)
        recall = np.zeros(n_states)
        
        for i in range(n_states):
            precision[i] = cm[i,i] / np.sum(cm[:,i]) if np.sum(cm[:,i]) != 0 else 0
            recall[i] = cm[i,i] / np.sum(cm[i,:]) if np.sum(cm[i,:]) != 0 else 0
        
        results['precision'] = precision
        results['recall'] = recall
        
        # 计算F1分数
        results['f1'] = 2 * (precision * recall) / (precision + recall)
        results['f1'] = np.nan_to_num(results['f1'])  # 处理除零情况
        
        return results
    
    def print_results(self, results):
        """打印评估结果"""
        self.logger.print(f"准确率: {results['accuracy']:.4f}")
        self.logger.print("\n混淆矩阵:")
        self.logger.print(results['confusion_matrix'])
        self.logger.print("\n每个状态的评估指标:")
        for i in range(len(results['precision'])):
            self.logger.print(f"状态 {i}:")
            self.logger.print(f"精确率: {results['precision'][i]:.4f}")
            self.logger.print(f"召回率: {results['recall'][i]:.4f}")
            self.logger.print(f"F1分数: {results['f1'][i]:.4f}")
            
    def _visualize_results(self, results, true_states, predicted_states):
        """可视化评估结果"""
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        plt.switch_backend('Agg')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 混淆矩阵热力图
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', ax=ax1)
        ax1.set_title('Confusion Matrix')  # 使用英文替代中文
        ax1.set_xlabel('Predicted State')
        ax1.set_ylabel('True State')
        
        # 2. 状态序列比较
        sequence_length = min(100, len(true_states))  # 只显示前100个时间步
        ax2.plot(range(sequence_length), true_states[:sequence_length], 
                label='True State', marker='o')
        ax2.plot(range(sequence_length), predicted_states[:sequence_length], 
                label='Predicted State', marker='x')
        ax2.set_title('State Sequence Comparison (First 100 Steps)')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('State')
        ax2.legend()
        ax2.grid(True)
        
        # 3. 评估指标柱状图
        metrics = ['Precision', 'Recall', 'F1']
        x = np.arange(len(metrics))
        width = 0.35
        
        for i in range(len(results['precision'])):
            values = [results[m.lower()][i] for m in metrics]
            ax3.bar(x + i*width, values, width, label=f'State {i}')
            
        ax3.set_ylabel('Score')
        ax3.set_title('Evaluation Metrics by State')
        ax3.set_xticks(x + width/2)
        ax3.set_xticklabels(metrics)
        ax3.legend()
        
        # 4. 准确率随时间变化
        window_size = 50
        accuracies = []
        for i in range(0, len(true_states) - window_size + 1):
            window_accuracy = accuracy_score(
                true_states[i:i+window_size],
                predicted_states[i:i+window_size]
            )
            accuracies.append(window_accuracy)
            
        ax4.plot(range(len(accuracies)), accuracies)
        ax4.set_title(f'Accuracy over Time (Window Size={window_size})')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Accuracy')
        ax4.grid(True)
        
        plt.tight_layout()
        
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

class ROCAUCJudger:
    def __init__(self) -> None:
        self.logger = Logger.get_logger('ROCAUCJudger')

    def judge(self, y_pred, test_dataset) -> None:
        # 获取真实标签
        y_true = np.array([x[1] for x in test_dataset])
        y_pred = np.array(y_pred)

        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 切换到Agg后端
        plt.switch_backend('Agg')
        
        fig = plt.figure(figsize=(10, 8))
        
        # 检查是否为多分类问题
        unique_classes = np.unique(y_true)
        n_classes = len(unique_classes)

        if n_classes > 2:
            # 多分类问题
            if len(y_pred.shape) == 1:
                # 如果预测值不是概率形式，转换为one-hot编码
                y_pred_prob = np.zeros((len(y_pred), n_classes))
                for i, pred in enumerate(y_pred):
                    y_pred_prob[i, pred] = 1
                y_pred = y_pred_prob
            
            # 计算每个类别的ROC曲线
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_classes))
            
            for i, class_idx in enumerate(unique_classes):
                # 将当前类别视为正类，其他类别视为负类
                y_true_binary = (y_true == class_idx).astype(int)
                y_score = y_pred[:, i] if len(y_pred.shape) > 1 else (y_pred == class_idx).astype(int)
                
                try:
                    fpr[i], tpr[i], _ = roc_curve(y_true_binary, y_score)
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    
                    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                            label=f'ROC 曲线 (类别 {class_idx}) (AUC = {roc_auc[i]:.2f})')
                except Exception as e:
                    self.logger.print(f"类别 {class_idx} 计算ROC曲线时出错: {str(e)}")
                    continue
                
        else:
            # 二分类问题
            if len(y_pred.shape) > 1:
                # 如果是概率预测值，取正类的概率
                y_pred = y_pred[:, 1]
            
            try:
                fpr, tpr, _ = roc_curve(y_true, y_pred)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
            except Exception as e:
                self.logger.print(f"计算ROC曲线时出错: {str(e)}")
                return

        # 绘制对角线
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (False Positive Rate)')
        plt.ylabel('真正率 (True Positive Rate)')
        plt.title('接收者操作特征(ROC)曲线')
        plt.legend(loc="lower right")
        
        # 将图像转换为数组格式
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        img = Image.open(buf)
        img = img.convert('RGB')
        img_array = np.array(img)
        
        plt.close()
        buf.close()
        
        # 使用logger显示图像数组
        self.logger.imshow(img_array)

        # 打印AUC值
        if isinstance(roc_auc, dict):
            mean_auc = 0
            for i in roc_auc:
                self.logger.print(f"类别 {unique_classes[i]} 的AUC值: {roc_auc[i]:.4f}")
                mean_auc += roc_auc[i]
            self.logger.print(f"平均AUC值: {mean_auc/len(roc_auc):.4f}")
        else:
            self.logger.print(f"AUC值: {roc_auc:.4f}")


