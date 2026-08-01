# ============================================
# 数据预处理模块
# 将原始年度数据按月平均转换为年度数据
# 第九届全国大学生统计建模大赛
# ============================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

def process_yearly_data(input_file, output_file):
    """
    将原始数据转换为年度平均值
    对应Java代码中的Data_deal类
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 提取第一列作为年份，其他列作为数据
    # 假设数据格式：每12行为一年的数据
    years = []
    yearly_avg = []
    
    for i in range(0, len(df), 12):
        year_data = df.iloc[i:i+12]
        if len(year_data) >= 12:
            # 计算年度平均值（跳过第一列年份列）
            avg_values = year_data.iloc[:, 1:].mean()
            years.append(int(year_data.iloc[0, 0]))
            yearly_avg.append(avg_values.values)
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(yearly_avg, columns=df.columns[1:])
    result_df.insert(0, 'year', years)
    
    # 保留3位小数
    result_df = result_df.round(3)
    
    # 保存结果
    result_df.to_csv(output_file, index=False)
    print(f"数据预处理完成！输出文件: {output_file}")
    print(f"数据形状: {result_df.shape}")
    
    return result_df

def clean_data(df):
    """
    数据清洗：处理缺失值、异常值
    """
    # 缺失值处理：用众数填充
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
    
    # 异常值处理：IQR法
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    return df

def normalize_data(df, columns):
    """
    Min-Max标准化
    """
    for col in columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df

if __name__ == "__main__":
    # 示例用法
    input_file = "data/raw/第一问数据预处理后——数字经济体系的构建.csv"
    output_file = "data/processed/数据预处理后——城乡收入差距指标选取年度数据总.csv"
    
    # 确保输出目录存在
    os.makedirs("data/processed", exist_ok=True)
    
    # 执行预处理
    result = process_yearly_data(input_file, output_file)
    print("\n前5行数据预览:")
    print(result.head())