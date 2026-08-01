# ============================================
# LSTM多变量时间序列预测
# 使用TensorFlow/Keras进行深度学习预测
# ============================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
tf.random.set_seed(13)
np.random.seed(13)

class LSTMForecast:
    """
    LSTM多变量时间序列预测
    """
    def __init__(self, data_path, feature_cols, target_col):
        """
        初始化
        :param data_path: 数据路径
        :param feature_cols: 特征列名列表
        :param target_col: 目标列名
        """
        self.data_path = data_path
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.df = None
        self.dataset = None
        self.data_mean = None
        self.data_std = None
        self.model = None
        
    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.set_index('year')
        features = self.df[self.feature_cols]
        self.dataset = features.values
        print(f"数据加载完成，形状: {self.dataset.shape}")
        return self.dataset
    
    def normalize_data(self, train_split):
        """标准化数据"""
        self.data_mean = self.dataset[:train_split].mean(axis=0)
        self.data_std = self.dataset[:train_split].std(axis=0)
        self.dataset = (self.dataset - self.data_mean) / (self.data_std + 1e-8)
        return self.dataset
    
    def create_sequences(self, dataset, target, start_index, end_index, 
                         history_size, target_size, step=1):
        """
        创建时间序列样本
        """
        data = []
        labels = []
        
        start_index = start_index + history_size
        
        if end_index is None:
            end_index = len(dataset) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i - history_size, i, step)
            data.append(dataset[indices])
            labels.append(target[i:i + target_size])
        
        return np.array(data), np.array(labels)
    
    def build_model(self, input_shape):
        """
        构建LSTM模型
        """
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=input_shape),
            tf.keras.layers.LSTM(16, activation='relu'),
            tf.keras.layers.Dense(72)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), 
            loss='mae'
        )
        self.model = model
        print("模型构建完成")
        return model
    
    def train(self, train_data, val_data, epochs=10, steps_per_epoch=10, validation_steps=50):
        """
        训练模型
        """
        history = self.model.fit(
            train_data,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_data,
            validation_steps=validation_steps
        )
        return history
    
    def plot_training_history(self, history, save_path=None):
        """绘制训练过程"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='训练损失')
        plt.plot(history.history['val_loss'], label='验证损失')
        plt.xlabel('迭代次数')
        plt.ylabel('损失 (MAE)')
        plt.title('LSTM模型训练过程')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # ============================================
    # 使用示例
    # ============================================
    
    # 1. 创建LSTM预测器
    lstm = LSTMForecast(
        data_path="data/processed/数字经济体系数据.csv",
        feature_cols=['光缆长度', '互联网上网人数', '专利申请数', 'GDP'],  # 根据实际列名修改
        target_col='城乡收入比'
    )
    
    # 2. 加载数据
    lstm.load_data()
    
    # 3. 标准化
    train_split = 48
    lstm.normalize_data(train_split)
    
    # 4. 创建训练数据
    past_history = 60
    future_target = 72
    step = 6
    
    X_train, y_train = lstm.create_sequences(
        lstm.dataset, lstm.dataset[:, 0], 0, train_split,
        past_history, future_target, step
    )
    
    # 5. 构建模型
    lstm.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # 6. 训练
    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_data = train_data.batch(8).repeat()
    
    history = lstm.train(train_data, None)
    
    # 7. 绘制训练过程
    lstm.plot_training_history(history, save_path="results/figures/lstm_training.png")
    
    print("LSTM模型训练完成！")