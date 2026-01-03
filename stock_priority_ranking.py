#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票优先级排序脚本
基于涨停次数和多因子选股进行综合评分排序
"""

import pandas as pd
import numpy as np
import os

def calculate_priority_score(df):
    """
    计算股票优先级评分
    
    评分因子：
    1. 涨停次数 (limit_up_days_1y) - 权重40%
    2. 振幅百分比 (range_pct) - 权重30% 
    3. 成交量异动 (vol_spike_5d) - 权重20%
    4. 价格位置 (基于high/low比值) - 权重10%
    
    Args:
        df: 包含股票数据的DataFrame
        
    Returns:
        df: 添加了评分和排名的DataFrame
    """
    
    # 创建副本避免修改原数据
    df_score = df.copy()
    
    # 1. 涨停次数评分 (0-100分)
    # 涨停次数越多分数越高
    max_limit_up = df_score['limit_up_days_1y'].max()
    min_limit_up = df_score['limit_up_days_1y'].min()
    
    if max_limit_up > min_limit_up:
        df_score['limit_up_score'] = ((df_score['limit_up_days_1y'] - min_limit_up) / 
                                     (max_limit_up - min_limit_up)) * 100
    else:
        df_score['limit_up_score'] = 100
    
    # 2. 振幅评分 (0-100分)
    # 振幅适中更好，过高风险大，过低活跃度不够
    # 使用倒U型函数，15%左右为最优
    optimal_range = 15.0
    df_score['range_score'] = 100 - np.abs(df_score['range_pct'] - optimal_range) * 3
    df_score['range_score'] = np.clip(df_score['range_score'], 0, 100)
    
    # 3. 成交量异动评分 (0-100分)
    df_score['vol_score'] = df_score['vol_spike_5d'].apply(lambda x: 100 if x else 50)
    
    # 4. 价格位置评分 (0-100分)
    # 当前价格接近区间高点更好
    df_score['price_position'] = ((df_score['last_close'] - df_score['low']) / 
                                 (df_score['high'] - df_score['low'])) * 100
    df_score['price_score'] = df_score['price_position']
    
    # 综合评分计算
    weights = {
        'limit_up': 0.40,  # 涨停次数权重40%
        'range': 0.30,     # 振幅权重30%
        'volume': 0.20,    # 成交量权重20%
        'price': 0.10      # 价格位置权重10%
    }
    
    df_score['total_score'] = (
        df_score['limit_up_score'] * weights['limit_up'] +
        df_score['range_score'] * weights['range'] +
        df_score['vol_score'] * weights['volume'] +
        df_score['price_score'] * weights['price']
    )
    
    # 按总分排序
    df_score = df_score.sort_values('total_score', ascending=False)
    df_score['rank'] = range(1, len(df_score) + 1)
    
    return df_score

def format_output(df_scored):
    """
    格式化输出结果
    """
    # 选择要显示的列
    output_cols = [
        'rank', 'symbol', 'code', 'market',
        'last_close', 'range_pct', 'limit_up_days_1y', 'vol_spike_5d',
        'total_score', 'limit_up_score', 'range_score', 'vol_score', 'price_score'
    ]
    
    df_output = df_scored[output_cols].copy()
    
    # 格式化数值
    df_output['total_score'] = df_output['total_score'].round(2)
    df_output['limit_up_score'] = df_output['limit_up_score'].round(1)
    df_output['range_score'] = df_output['range_score'].round(1)
    df_output['vol_score'] = df_output['vol_score'].round(1)
    df_output['price_score'] = df_output['price_score'].round(1)
    
    return df_output

def main():
    """
    主函数
    """
    # 文件路径
    input_file = 'output/selection_local.csv'
    output_file = 'output/stock_priority_ranking.csv'
    
    try:
        # 读取数据
        print("正在读取股票数据...")
        df = pd.read_csv(input_file)
        print(f"共读取到 {len(df)} 只股票")
        
        # 计算优先级评分
        print("正在计算优先级评分...")
        df_scored = calculate_priority_score(df)
        
        # 格式化输出
        df_output = format_output(df_scored)
        
        # 保存结果
        df_output.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"优先级排序结果已保存到: {output_file}")
        
        # 显示排序结果
        print("\n=== 股票优先级排序结果 ===")
        print("评分规则：")
        print("- 涨停次数：权重40%，次数越多分数越高")
        print("- 振幅百分比：权重30%，15%左右最优")
        print("- 成交量异动：权重20%，有异动得100分，无异动得50分")
        print("- 价格位置：权重10%，越接近区间高点分数越高")
        print("-" * 80)
        
        # 显示详细结果
        for idx, row in df_output.iterrows():
            print(f"排名 {row['rank']:2d}: {row['symbol']} ({row['code']})")
            print(f"         总分: {row['total_score']:6.2f} | "
                  f"涨停: {row['limit_up_days_1y']:2d}次({row['limit_up_score']:5.1f}) | "
                  f"振幅: {row['range_pct']:6.2f}%({row['range_score']:5.1f}) | "
                  f"量能: {'异动' if row['vol_spike_5d'] else '正常'}({row['vol_score']:5.1f}) | "
                  f"位置: {row['price_score']:5.1f}")
            print()
        
        # 投资建议
        print("=== 投资建议 ===")
        top_stock = df_output.iloc[0]
        print(f"最优选择: {top_stock['symbol']} (总分: {top_stock['total_score']:.2f})")
        
        if top_stock['limit_up_days_1y'] >= 10:
            print("- 该股票年内涨停次数较多，趋势强劲")
        
        if top_stock['vol_spike_5d']:
            print("- 近5日成交量异动，关注度较高")
            
        if top_stock['range_score'] > 80:
            print("- 振幅适中，风险可控")
        elif top_stock['range_score'] < 50:
            print("- 注意：振幅较大，风险较高")
            
        print("\n注意：本分析仅供参考，投资有风险，决策需谨慎！")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        print("请确保文件路径正确")
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")

if __name__ == "__main__":
    main()
