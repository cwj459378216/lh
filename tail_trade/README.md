# 尾盘买入策略系统 (tail_trade)

## 📖 策略概述

短线尾盘买入策略：在收盘前 10 分钟（14:50-15:00）检测符合条件的股票并发送企业微信通知。

### 买入条件（全部满足）

| 条件 | 默认阈值 | 说明 |
|------|----------|------|
| 当日涨幅 | < 5% | 排除已大涨的票 |
| 尾盘涨幅 | > 0.5% | 14:50-15:00 价格上涨 |
| 尾盘量比 | > 1.5 | 尾盘每分钟均量 / 全日每分钟均量 |

## 📁 项目结构

```
tail_trade/
├── __init__.py              # 包初始化
├── config.py                # 配置（Webhook、路径、参数）
├── strategy.py              # 策略逻辑
├── notifier.py              # 企业微信通知
├── logger.py                # 日志模块
├── backtest.py              # 历史回测（可直接运行）
├── monitor.py               # 实时监控（可直接运行）
├── fetch_minute_data.py     # 分钟线数据下载
├── stock_pool.csv           # 股票池
├── data/
│   ├── __init__.py
│   ├── loader.py            # 数据加载（日线/分钟线/AkShare）
│   └── minute/              # 分钟线 CSV（需下载）
├── logs/                    # 日志文件（自动生成）
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install pandas akshare requests chinese-calendar pytdx
```

### 2. 配置企业微信

编辑 `config.py`，将 `WECHAT_WEBHOOK` 设置为你的群机器人 Webhook URL。

### 3. 配置股票池

编辑 `stock_pool.csv`：

```csv
code,name
000001.SZ,平安银行
600519.SH,贵州茅台
```

### 4. 下载分钟线数据（回测用）

```bash
# 默认使用 pytdx（推荐，无限频限制，可获取较长历史）
python -m tail_trade.fetch_minute_data --pool
python -m tail_trade.fetch_minute_data --pool --days 30   # 下载最近 30 个交易日

# 下载指定股票
python -m tail_trade.fetch_minute_data --codes 000001.SZ,600519.SH

# 强制使用 AkShare（仅近 5 个交易日，有限频风险）
python -m tail_trade.fetch_minute_data --pool --source akshare --delay 3
```

### 5. 运行回测

```bash
# 回测单只股票
python -m tail_trade.backtest --code 000001.SZ --name 平安银行 --start 2024-01-01

# 回测整个股票池
python -m tail_trade.backtest --start 2024-01-01 --end 2024-12-31

# 自定义参数
python -m tail_trade.backtest --code 000001.SZ --gain-max 0.03 --tail-rise 0.003 --vol-ratio 2.0

# 导出信号记录到 CSV
python -m tail_trade.backtest --output signals.csv
```

### 6. 启动实时监控

```bash
# 使用股票池监控
python -m tail_trade.monitor

# 监控指定股票
python -m tail_trade.monitor --codes 000001.SZ,600519.SH

# 调整轮询间隔（秒）
python -m tail_trade.monitor --interval 45

# 测试模式（立即执行一次，不等待时间窗口）
python -m tail_trade.monitor --test
```

## ⚙️ 参数调整

### 在 config.py 中修改

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DAILY_GAIN_MAX` | 0.05 | 当日涨幅上限 |
| `TAIL_START_TIME` | "14:50" | 尾盘起始 |
| `TAIL_END_TIME` | "15:00" | 尾盘结束 |
| `TAIL_MINUTES` | 10 | 尾盘分钟数 |
| `TAIL_PRICE_RISE_MIN` | 0.005 | 尾盘涨幅下限 |
| `TAIL_VOLUME_RATIO_MIN` | 1.5 | 尾盘量比下限 |
| `MONITOR_POLL_INTERVAL` | 60 | 轮询间隔（秒） |
| `SNAPSHOT_MAX_RETRIES` | 3 | 快照请求最大重试次数 |
| `SNAPSHOT_RETRY_BASE_DELAY` | 5 | 重试基础延时（指数退避 5→10→20s） |
| `SNAPSHOT_CACHE_TTL` | 15 | 快照内存缓存 TTL（秒） |
| `MONITOR_MAX_CONSECUTIVE_FAILURES` | 5 | 连续失败上限（超过后发送告警） |
| `MINUTE_DATA_SOURCE` | pytdx | 分钟线数据源（pytdx / akshare） |

### 通过环境变量覆盖

```bash
DAILY_GAIN_MAX=0.03 TAIL_PRICE_RISE_MIN=0.003 python -m tail_trade.monitor
```

### 通过命令行参数

```bash
python -m tail_trade.backtest --gain-max 0.03 --tail-rise 0.003 --vol-ratio 2.0
```

## 📊 回测报告示例

```
══════════════════════════════════════════════════
          📊 尾盘策略回测报告
══════════════════════════════════════════════════
  总触发次数 :  42
  有效统计   :  40（有次日收益数据）
  胜率       :  62.5%（25/40）
  平均次日收益:  +0.38%
  最大单次盈利:  +3.21%
  最大单次亏损:  -2.15%
  累计收益   :  +15.20%（简单累加）
══════════════════════════════════════════════════
```

## 📝 日志

日志自动写入 `tail_trade/logs/` 目录，按模块和日期分文件：

- `backtest_20240115.log`
- `monitor_20240115.log`
- `notifier_20240115.log`

## ⚠️ 注意事项

1. **AkShare 限频保护**：
   - 实时快照使用全市场批量接口（`stock_zh_a_spot_em`）+ 内存缓存
   - 请求失败自动 **指数退避重试**（5s → 10s → 20s，最多 3 次）
   - 连续失败超过 5 轮自动发送 **企业微信告警**
   - 连续失败时自动 **扩大轮询间隔**，减少无效请求
2. **分钟线数据**：
   - **推荐使用 pytdx**（通达信协议，无限频限制，可获取 30+ 个交易日）
   - AkShare 仅保留近 5 个交易日，批量下载有限频风险
   - pytdx 失败时自动回退到 AkShare
3. **手动交易**：本系统仅发送信号通知，不自动下单
4. **数据路径**：日线数据默认读取 `通达信/data/pytdx/daily_raw/`，可在 config.py 中修改
