#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
止损止盈管理脚本
功能：
1. 根据selection_local.csv买入股票
2. 计算止损点（买入价-3%）
3. 动态计算止盈点（最高价-3%）
4. 每日维护持仓记录
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, date
import json
import requests
import re
import time

class StopLossGainManager:
    def __init__(self, data_dir='data/pytdx/daily_raw', output_dir='output'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.portfolio_file = os.path.join(output_dir, 'portfolio_management.csv')
        self.log_file = os.path.join(output_dir, 'trading_log.txt')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 交易参数
        self.stop_loss_pct = 0.03  # 止损3%
        self.stop_gain_pct = 0.03  # 止盈3%（从最高点回撤）
        
    def log_message(self, message):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    
    def get_latest_price(self, symbol):
        """获取股票最新价格 - 从在线API获取"""
        try:
            # 转换股票代码格式
            # 600609.SH -> sh600609
            # 002995.SZ -> sz002995
            if '.SH' in symbol:
                code = 'sh' + symbol.replace('.SH', '')
            elif '.SZ' in symbol:
                code = 'sz' + symbol.replace('.SZ', '')
            else:
                self.log_message(f"警告：无法识别股票代码格式 {symbol}")
                return None
            
            # 使用东方财富API
            url = f"http://push2.eastmoney.com/api/qt/stock/get"
            
            # 转换为东方财富的代码格式
            if '.SH' in symbol:
                em_code = f"1.{symbol.replace('.SH', '')}"
            elif '.SZ' in symbol:
                em_code = f"0.{symbol.replace('.SZ', '')}"
            else:
                return None
            
            params = {
                'secid': em_code,
                'fields': 'f43,f44,f45,f46,f47,f48,f57,f58,f162,f169,f170,f171',
                'cb': 'jsonp'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'http://quote.eastmoney.com/',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                self.log_message(f"东方财富API获取 {symbol} 数据失败，HTTP状态码: {response.status_code}")
                return self.get_latest_price_sina(symbol)  # fallback to sina
            
            # 解析JSON数据
            content = response.text
            if 'jsonp(' in content:
                json_str = content[content.find('(')+1:content.rfind(')')]
                data = json.loads(json_str)
                
                if 'data' in data and data['data']:
                    stock_data = data['data']
                    
                    # 解析字段
                    current_price = stock_data.get('f43', 0) / 100  # 当前价
                    open_price = stock_data.get('f46', 0) / 100   # 开盘价
                    high_price = stock_data.get('f44', 0) / 100   # 最高价
                    low_price = stock_data.get('f45', 0) / 100    # 最低价
                    prev_close = stock_data.get('f60', 0) / 100   # 昨收价
                    volume = stock_data.get('f47', 0)             # 成交量
                    name = stock_data.get('f58', '')              # 股票名称
                    
                    if current_price <= 0:
                        self.log_message(f"{symbol} 当前价格无效: {current_price}")
                        return self.get_latest_price_sina(symbol)
                    
                    today = datetime.now().strftime('%Y-%m-%d')
                    
                    result = {
                        'date': today,
                        'name': name,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': current_price,
                        'prev_close': prev_close,
                        'volume': volume
                    }
                    
                    self.log_message(f"成功获取 {symbol}({name}) 实时数据: 当前价 {current_price:.2f}")
                    return result
                else:
                    self.log_message(f"东方财富API返回数据为空: {symbol}")
                    return self.get_latest_price_sina(symbol)
            else:
                self.log_message(f"东方财富API返回格式错误: {symbol}")
                return self.get_latest_price_sina(symbol)
                
        except requests.RequestException as e:
            self.log_message(f"东方财富API网络请求 {symbol} 失败: {str(e)}")
            return self.get_latest_price_sina(symbol)
        except (ValueError, json.JSONDecodeError) as e:
            self.log_message(f"东方财富API解析 {symbol} 数据失败: {str(e)}")
            return self.get_latest_price_sina(symbol)
        except Exception as e:
            self.log_message(f"东方财富API获取 {symbol} 价格时出错: {str(e)}")
            return self.get_latest_price_sina(symbol)
    
    def get_latest_price_sina(self, symbol):
        """备用方法1：从新浪财经获取股价（更好的请求头）"""
        try:
            # 转换股票代码格式
            if '.SH' in symbol:
                code = 'sh' + symbol.replace('.SH', '')
            elif '.SZ' in symbol:
                code = 'sz' + symbol.replace('.SZ', '')
            else:
                return None
            
            # 新浪财经API
            url = f"http://hq.sinajs.cn/list={code}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://finance.sina.com.cn/',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Cache-Control': 'no-cache'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'gbk'
            
            if response.status_code != 200:
                self.log_message(f"新浪API获取 {symbol} 数据失败，HTTP状态码: {response.status_code}")
                return self.get_latest_price_backup(symbol)
            
            # 解析数据
            content = response.text
            if not content or 'var hq_str_' not in content:
                self.log_message(f"新浪API获取 {symbol} 数据为空或格式错误")
                return self.get_latest_price_backup(symbol)
            
            # 提取数据
            match = re.search(r'"([^"]*)"', content)
            if not match:
                self.log_message(f"新浪API解析 {symbol} 数据失败")
                return self.get_latest_price_backup(symbol)
            
            data_str = match.group(1)
            data_parts = data_str.split(',')
            
            if len(data_parts) < 9:
                self.log_message(f"新浪API {symbol} 数据格式不完整")
                return self.get_latest_price_backup(symbol)
            
            # 解析各字段
            name = data_parts[0]
            open_price = float(data_parts[1]) if data_parts[1] else 0
            prev_close = float(data_parts[2]) if data_parts[2] else 0
            current_price = float(data_parts[3]) if data_parts[3] else 0
            high_price = float(data_parts[4]) if data_parts[4] else 0
            low_price = float(data_parts[5]) if data_parts[5] else 0
            volume = float(data_parts[8]) if data_parts[8] else 0
            
            if current_price <= 0:
                self.log_message(f"新浪API {symbol} 当前价格无效: {current_price}")
                return self.get_latest_price_backup(symbol)
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            result = {
                'date': today,
                'name': name,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': current_price,
                'prev_close': prev_close,
                'volume': volume
            }
            
            self.log_message(f"新浪API成功获取 {symbol}({name}) 实时数据: 当前价 {current_price:.2f}")
            return result
            
        except Exception as e:
            self.log_message(f"新浪API获取 {symbol} 失败: {str(e)}")
            return self.get_latest_price_backup(symbol)
    
    def get_latest_price_backup(self, symbol):
        """备用方法2：从腾讯财经获取股价"""
        try:
            # 腾讯财经API
            if '.SH' in symbol:
                code = 'sh' + symbol.replace('.SH', '')
            elif '.SZ' in symbol:
                code = 'sz' + symbol.replace('.SZ', '')
            else:
                return None
            
            url = f"http://qt.gtimg.cn/q={code}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Referer': 'https://stockapp.finance.qq.com/',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Connection': 'keep-alive'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.encoding = 'gbk'
            
            if response.status_code != 200:
                self.log_message(f"腾讯API获取 {symbol} 数据失败，HTTP状态码: {response.status_code}")
                return self.get_latest_price_manual(symbol)
            
            content = response.text
            if not content:
                self.log_message(f"腾讯API获取 {symbol} 数据为空")
                return self.get_latest_price_manual(symbol)
            
            # 解析腾讯数据格式
            # v_sh600609="1~股票名称~股票代码~当前价格~涨跌~涨跌%~成交量~成交额~..."
            match = re.search(r'"([^"]*)"', content)
            if not match:
                self.log_message(f"腾讯API解析 {symbol} 数据失败")
                return self.get_latest_price_manual(symbol)
            
            data_str = match.group(1)
            data_parts = data_str.split('~')
            
            if len(data_parts) < 8:
                self.log_message(f"腾讯API {symbol} 数据格式不完整")
                return self.get_latest_price_manual(symbol)
            
            name = data_parts[1] if len(data_parts) > 1 else ''
            current_price = float(data_parts[3]) if data_parts[3] else 0
            
            if current_price <= 0:
                self.log_message(f"腾讯API {symbol} 当前价格无效: {current_price}")
                return self.get_latest_price_manual(symbol)
            
            today = datetime.now().strftime('%Y-%m-%d')
            
            result = {
                'date': today,
                'name': name,
                'open': current_price,  # 简化处理
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 0
            }
            
            self.log_message(f"腾讯API成功获取 {symbol}({name}) 实时数据: 当前价 {current_price:.2f}")
            return result
            
        except Exception as e:
            self.log_message(f"腾讯API获取 {symbol} 失败: {str(e)}")
            return self.get_latest_price_manual(symbol)
    
    def get_latest_price_manual(self, symbol):
        """手动输入价格的备用方法"""
        try:
            self.log_message(f"所有API都失败，请手动输入 {symbol} 的当前价格")
            print(f"\n⚠️  无法自动获取 {symbol} 的价格")
            print("请手动输入当前价格（直接回车跳过该股票）:")
            
            price_input = input(f"{symbol} 当前价格: ").strip()
            
            if not price_input:
                self.log_message(f"跳过 {symbol} 的价格更新")
                return None
            
            try:
                current_price = float(price_input)
                if current_price <= 0:
                    self.log_message(f"输入的价格无效: {current_price}")
                    return None
                
                today = datetime.now().strftime('%Y-%m-%d')
                
                result = {
                    'date': today,
                    'name': '手动输入',
                    'open': current_price,
                    'high': current_price,
                    'low': current_price,
                    'close': current_price,
                    'volume': 0
                }
                
                self.log_message(f"手动输入 {symbol} 价格: {current_price:.2f}")
                return result
                
            except ValueError:
                self.log_message(f"输入的价格格式错误: {price_input}")
                return None
                
        except Exception as e:
            self.log_message(f"手动输入价格失败: {str(e)}")
            return None
    
    def load_portfolio(self):
        """加载持仓组合"""
        if os.path.exists(self.portfolio_file):
            try:
                df = pd.read_csv(self.portfolio_file)
                return df
            except Exception as e:
                self.log_message(f"加载持仓文件出错: {str(e)}")
                return pd.DataFrame()
        else:
            # 创建空的持仓文件
            columns = [
                '股票代码', '买入日期', '买入价格', '持仓数量', '总成本',
                '止损价格', '当前价格', '历史最高价', '止盈价格',
                '当前市值', '未实现盈亏', '盈亏百分比',
                '状态', '最后更新'
            ]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.portfolio_file, index=False, encoding='utf-8-sig')
            return df
    
    def save_portfolio(self, df):
        """保存持仓组合"""
        try:
            df.to_csv(self.portfolio_file, index=False, encoding='utf-8-sig')
            self.log_message(f"持仓文件已更新: {self.portfolio_file}")
        except Exception as e:
            self.log_message(f"保存持仓文件出错: {str(e)}")
    
    def buy_stocks_from_selection(self, selection_file='output/selection_local.csv', 
                                 total_capital=100000, equal_weight=True):
        """根据选股结果买入股票"""
        try:
            # 读取选股结果
            selection_df = pd.read_csv(selection_file)
            self.log_message(f"读取选股结果，共 {len(selection_df)} 只股票")
            
            # 加载当前持仓
            portfolio_df = self.load_portfolio()
            
            # 检查是否已经有持仓
            if len(portfolio_df) > 0:
                active_positions = portfolio_df[portfolio_df['状态'] == 'active']
                if len(active_positions) > 0:
                    self.log_message("当前已有持仓，请先处理现有持仓或选择添加模式")
                    return False
            
            # 计算每只股票的投资金额
            num_stocks = len(selection_df)
            if equal_weight:
                capital_per_stock = total_capital / num_stocks
            
            new_positions = []
            today = date.today().strftime('%Y-%m-%d')
            
            for idx, row in selection_df.iterrows():
                symbol = row['symbol']
                
                # 获取当前价格（使用last_close作为买入价）
                buy_price = float(row['last_close'])
                
                # 计算买入数量（取整百股）
                if equal_weight:
                    quantity = int(capital_per_stock / buy_price / 100) * 100
                else:
                    # 可以根据其他逻辑分配
                    quantity = int(capital_per_stock / buy_price / 100) * 100
                
                if quantity < 100:  # 最少买100股
                    quantity = 100
                
                total_cost = buy_price * quantity
                stop_loss_price = buy_price * (1 - self.stop_loss_pct)
                
                # 创建新持仓记录
                position = {
                    '股票代码': symbol,
                    '买入日期': today,
                    '买入价格': buy_price,
                    '持仓数量': quantity,
                    '总成本': total_cost,
                    '止损价格': stop_loss_price,
                    '当前价格': buy_price,
                    '历史最高价': buy_price,
                    '止盈价格': buy_price,  # 初始等于买入价
                    '当前市值': total_cost,
                    '未实现盈亏': 0,
                    '盈亏百分比': 0,
                    '状态': 'active',
                    '最后更新': today
                }
                
                new_positions.append(position)
                self.log_message(f"买入 {symbol}: {quantity}股 @ {buy_price:.2f}, 总成本: {total_cost:.2f}")
            
            # 保存新持仓
            new_portfolio_df = pd.DataFrame(new_positions)
            self.save_portfolio(new_portfolio_df)
            
            self.log_message(f"成功买入 {len(new_positions)} 只股票，总投入: {sum(p['total_cost'] for p in new_positions):.2f}")
            return True
            
        except Exception as e:
            self.log_message(f"买入股票时出错: {str(e)}")
            return False
    
    def update_portfolio(self):
        """更新持仓组合的价格和止损止盈点"""
        portfolio_df = self.load_portfolio()
        
        if len(portfolio_df) == 0:
            self.log_message("当前无持仓")
            return
        
        active_positions = portfolio_df[portfolio_df['状态'] == 'active'].copy()
        
        if len(active_positions) == 0:
            self.log_message("当前无活跃持仓")
            return
        
        today = date.today().strftime('%Y-%m-%d')
        updated_count = 0
        
        for idx, position in active_positions.iterrows():
            symbol = position['股票代码']
            
            # 获取最新价格（尝试多个API源）
            latest_data = self.get_latest_price_sina(symbol)  # 先尝试新浪
            
            if latest_data is None:
                # 尝试东方财富API
                self.log_message(f"尝试东方财富API获取 {symbol} 数据...")
                latest_data = self.get_latest_price(symbol)
                
            if latest_data is None:
                # 尝试腾讯API
                self.log_message(f"尝试腾讯API获取 {symbol} 数据...")
                latest_data = self.get_latest_price_backup(symbol)
                
            if latest_data is None:
                self.log_message(f"无法获取 {symbol} 最新价格，跳过更新")
                continue
            
            current_price = latest_data['close']
            daily_high = latest_data.get('high', current_price)
            
            # 更新最高价（如果今日最高价更高）
            highest_price = max(position['历史最高价'], daily_high)
            
            # 计算止盈价格（从最高价回撤3%）
            stop_gain_price = highest_price * (1 - self.stop_gain_pct)
            
            # 计算当前价值和盈亏
            current_value = current_price * position['持仓数量']
            unrealized_pnl = current_value - position['总成本']
            unrealized_pnl_pct = (unrealized_pnl / position['总成本']) * 100
            
            # 检查是否触发止损或止盈
            status = 'active'
            if current_price <= position['止损价格']:
                status = 'stopped_loss'
                self.log_message(f"🔴 {symbol} 触发止损！当前价: {current_price:.2f}, 止损价: {position['止损价格']:.2f}")
            elif current_price <= stop_gain_price and highest_price > position['买入价格']:
                status = 'stopped_gain'
                self.log_message(f"🟢 {symbol} 触发止盈！当前价: {current_price:.2f}, 止盈价: {stop_gain_price:.2f}")
            
            # 更新数据
            portfolio_df.loc[idx, '当前价格'] = current_price
            portfolio_df.loc[idx, '历史最高价'] = highest_price
            portfolio_df.loc[idx, '止盈价格'] = stop_gain_price
            portfolio_df.loc[idx, '当前市值'] = current_value
            portfolio_df.loc[idx, '未实现盈亏'] = unrealized_pnl
            portfolio_df.loc[idx, '盈亏百分比'] = unrealized_pnl_pct
            portfolio_df.loc[idx, '状态'] = status
            portfolio_df.loc[idx, '最后更新'] = today
            
            updated_count += 1
            
            self.log_message(f"更新 {symbol}: 当前价 {current_price:.2f}, "
                           f"最高价 {highest_price:.2f}, 止盈价 {stop_gain_price:.2f}, "
                           f"盈亏 {unrealized_pnl_pct:.2f}%")
            
            # 添加短暂延迟，避免请求过于频繁
            time.sleep(0.5)
        
        # 保存更新后的持仓
        self.save_portfolio(portfolio_df)
        self.log_message(f"已更新 {updated_count} 个持仓")
        
        # 显示持仓摘要
        self.show_portfolio_summary()
    
    def show_portfolio_summary(self):
        """显示持仓摘要"""
        portfolio_df = self.load_portfolio()
        
        if len(portfolio_df) == 0:
            return
        
        active_df = portfolio_df[portfolio_df['状态'] == 'active']
        
        print("\n" + "="*80)
        print("📊 持仓摘要")
        print("="*80)
        
        if len(active_df) > 0:
            total_cost = active_df['总成本'].sum()
            total_value = active_df['当前市值'].sum()
            total_pnl = active_df['未实现盈亏'].sum()
            total_pnl_pct = (total_pnl / total_cost) * 100
            
            print(f"活跃持仓: {len(active_df)} 只")
            print(f"总成本: {total_cost:,.2f}")
            print(f"总市值: {total_value:,.2f}")
            print(f"总盈亏: {total_pnl:,.2f} ({total_pnl_pct:+.2f}%)")
            print("-" * 80)
            
            # 显示个股详情
            for idx, row in active_df.iterrows():
                print(f"{row['股票代码']:12} | "
                      f"买入: {row['买入价格']:7.2f} | "
                      f"当前: {row['当前价格']:7.2f} | "
                      f"最高: {row['历史最高价']:7.2f} | "
                      f"止损: {row['止损价格']:7.2f} | "
                      f"止盈: {row['止盈价格']:7.2f} | "
                      f"盈亏: {row['盈亏百分比']:+6.2f}%")
        
        # 显示已平仓统计
        closed_df = portfolio_df[portfolio_df['状态'].isin(['stopped_loss', 'stopped_gain'])]
        if len(closed_df) > 0:
            print(f"\n已平仓: {len(closed_df)} 只")
            stop_loss_count = len(closed_df[closed_df['状态'] == 'stopped_loss'])
            stop_gain_count = len(closed_df[closed_df['状态'] == 'stopped_gain'])
            print(f"止损: {stop_loss_count} 只 | 止盈: {stop_gain_count} 只")
        
        print("="*80)
    
    def export_daily_report(self):
        """导出每日报告"""
        portfolio_df = self.load_portfolio()
        
        if len(portfolio_df) == 0:
            return
        
        today = date.today().strftime('%Y-%m-%d')
        report_file = os.path.join(self.output_dir, f'daily_report_{today}.csv')
        
        # 添加一些计算列
        report_df = portfolio_df.copy()
        report_df['持仓天数'] = pd.to_datetime(today) - pd.to_datetime(report_df['买入日期'])
        report_df['持仓天数'] = report_df['持仓天数'].dt.days
        
        # 保存报告
        report_df.to_csv(report_file, index=False, encoding='utf-8-sig')
        self.log_message(f"每日报告已导出: {report_file}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='止损止盈管理脚本')
    parser.add_argument('--action', choices=['buy', 'update', 'summary'], 
                       default='update', help='执行的操作')
    parser.add_argument('--capital', type=float, default=100000, 
                       help='总投资金额（仅买入时使用）')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = StopLossGainManager()
    
    if args.action == 'buy':
        print("🛒 开始买入股票...")
        success = manager.buy_stocks_from_selection(total_capital=args.capital)
        if success:
            print("✅ 买入完成")
        else:
            print("❌ 买入失败")
    
    elif args.action == 'update':
        print("🔄 更新持仓信息...")
        manager.update_portfolio()
        manager.export_daily_report()
        print("✅ 更新完成")
    
    elif args.action == 'summary':
        print("📊 显示持仓摘要...")
        manager.show_portfolio_summary()

if __name__ == "__main__":
    main()
