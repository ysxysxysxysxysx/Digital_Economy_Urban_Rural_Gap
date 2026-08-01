# ============================================
# ARIMA时间序列预测
# 数字经济发展对城乡收入差距的影响研究
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class ARIMAForecast:
    """
    ARIMA模型预测类
    """
    def __init__(self, data_path, target_col):
        """
        初始化
        :param data_path: 数据文件路径
        :param target_col: 目标列名（城乡收入比）
        """
        self.data_path = data_path
        self.target_col = target_col
        self.df = None
        self.model = None
        self.fitted = None
        
    def load_data(self):
        """加载数据"""
        self.df = pd.read_csv(self.data_path, parse_dates=['year'])
        self.df = self.df.set_index('year')
        print(f"数据加载完成，形状: {self.df.shape}")
        return self.df
    
    def adf_test(self, series):
        """ADF单位根检验"""
        result = adfuller(series.dropna())
        print(f'\nADF检验结果:')
        print(f'  ADF统计量: {result[0]:.4f}')
        print(f'  p值: {result[1]:.4f}')
        print(f'  临界值: 1%={result[4]["1%"]:.4f}, 5%={result[4]["5%"]:.4f}')
        
        if result[1] < 0.05:
            print('  ✅ 序列是平稳的 (p < 0.05)')
        else:
            print('  ❌ 序列是非平稳的 (p >= 0.05)')
        return result
    
    def find_best_order(self, series, max_ar=10, max_ma=5):
        """自动寻找最优ARIMA阶数"""
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        for p in range(max_ar + 1):
            for q in range(max_ma + 1):
                try:
                    model = sm.tsa.arima.ARIMA(series, order=(p, 1, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, 1, q)
                except:
                    continue
        
        print(f'\n最优ARIMA阶数: {best_order}, AIC: {best_aic:.4f}')
        return best_order
    
    def fit_model(self, order):
        """拟合ARIMA模型"""
        self.model = sm.tsa.arima.ARIMA(self.df[self.target_col], order=order)
        self.fitted = self.model.fit()
        print(f'\nARIMA{order} 模型拟合完成')
        print(self.fitted.summary())
        return self.fitted
    
    def plot_acf_pacf(self, series, save_path=None):
        """绘制自相关和偏自相关图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        plot_acf(series, ax=axes[0], lags=20)
        axes[0].set_title('自相关函数 (ACF)')
        
        plot_pacf(series, ax=axes[1], lags=20)
        axes[1].set_title('偏自相关函数 (PACF)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def forecast(self, steps=50, save_path=None):
        """进行预测"""
        forecast = self.fitted.forecast(steps=steps)
        forecast_index = pd.date_range(
            self.df.index[-1], 
            periods=steps + 1, 
            freq='Y'
        )[1:]
        
        # 绘制预测图
        plt.figure(figsize=(14, 7))
        
        # 历史数据
        plt.plot(self.df.index, self.df[self.target_col], 
                 label='历史数据', color='blue', linewidth=2)
        
        # 拟合值
        plt.plot(self.df.index, self.fitted.fittedvalues, 
                 label='拟合值', color='green', linestyle='--', alpha=0.7)
        
        # 预测值
        plt.plot(forecast_index, forecast, 
                 label='预测值', color='red', linewidth=2)
        
        # 置信区间
        se_forecast = self.fitted.se_forecast(steps=steps)
        plt.fill_between(forecast_index,
                         forecast - 1.96 * se_forecast,
                         forecast + 1.96 * se_forecast,
                         color='red', alpha=0.15, label='95%置信区间')
        
        plt.xlabel('年份')
        plt.ylabel('城乡收入比')
        plt.title('ARIMA模型预测效果图')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # 返回预测结果
        results = pd.DataFrame({
            '年份': forecast_index.strftime('%Y'),
            '预测值': forecast.values
        })
        return results
    
    def evaluate(self):
        """模型评估"""
        residuals = self.fitted.resid
        
        # 白噪声检验
        lb_test = acorr_ljungbox(residuals, lags=[6, 12], boxpierce=True)
        print(f'\n白噪声检验 (Ljung-Box):')
        print(lb_test)
        
        # 残差分布
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        plt.title('残差分布')
        plt.xlabel('残差')
        plt.ylabel('频数')
        
        plt.subplot(1, 2, 2)
        plt.plot(residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('残差序列')
        plt.xlabel('时间')
        plt.ylabel('残差')
        
        plt.tight_layout()
        plt.show()
        
        return residuals

if __name__ == "__main__":
    # ============================================
    # 使用示例
    # ============================================
    
    # 1. 创建预测器
    forecaster = ARIMAForecast(
        data_path="data/processed/数据预处理后——城乡收入差距指标选取年度数据总.csv",
        target_col="城乡收入比"  # 根据实际列名修改
    )
    
    # 2. 加载数据
    forecaster.load_data()
    
    # 3. 平稳性检验
    forecaster.adf_test(forecaster.df['城乡收入比'])
    
    # 4. 绘制ACF/PACF
    forecaster.plot_acf_pacf(forecaster.df['城乡收入比'], save_path="results/figures/acf_pacf.png")
    
    # 5. 寻找最优阶数
    best_order = forecaster.find_best_order(forecaster.df['城乡收入比'])
    
    # 6. 拟合模型
    forecaster.fit_model(best_order)
    
    # 7. 模型评估
    forecaster.evaluate()
    
    # 8. 预测
    forecast_results = forecaster.forecast(steps=50, save_path="results/figures/forecast.png")
    print("\n预测结果:")
    print(forecast_results.head(10))