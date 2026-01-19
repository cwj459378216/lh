import os
# import argparse  # 不再使用命令行参数
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
from tqdm import tqdm
import requests
import json
from dataclasses import dataclass
from dataclasses import field
from typing import List

# python select_stocks_local.py --end-date 20260110
# python select_stocks_local.py --end-date 20251013 --debug-symbol 600868.SH

# 统一参数配置（集中管理默认值）
@dataclass
class Config:
    # 路径与输出（硬编码）：数据源目录与结果文件路径
    data_dir: str = os.path.join(os.path.dirname(__file__), '通达信', 'data', 'pytdx', 'daily_raw')
    # 默认文件名改为带时间戳，避免多次单独运行覆盖
    default_out: str = os.path.join(
        os.path.dirname(__file__),
        'output',
        f"selection_local_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )
    date_subdir: bool = True  # 是否按当天日期(YYYYMMDD)创建子文件夹保存输出

    # 评分过滤（按原始评分 raw_score）：None 表示不启用
    min_score: float | None = None  # 例如 70.0

    # 市值过滤（单位：元）：总市值区间与是否跳过过滤
    mktcap_min: float = 8e9   # 市值下限，默认80亿
    mktcap_max: float = 15e9  # 市值上限，默认150亿
    skip_mktcap: bool = True # 跳过市值/估值过滤开关
    # 选股核心参数（全部硬编码）：窗口、振幅、接近低点、涨停统计与阈值、放量参数、A股10%前缀过滤
    months: int = 3                   # 近几个月作为振幅窗口
    range_lower: float = 0.10         # 振幅下限（比例）
    range_upper: float = 0.20         # 振幅上限（比例）
    near_low_tol: float = 0.10        # 接近近低容差（比例）
    limitup_months: int = 12          # 涨停统计窗口（月）
    limitup_threshold: float = 0.098  # 涨停阈值（收盘涨幅近似≥9.8%）
    min_limitup_count: int = 3        # 正价涨停次数最少要求（近一年）
    vol_days: int = 5                 # 放量考察天数（当前逻辑已不用最近n天，而是当日 vs 前一日）
    vol_factor: float = 2           # 放量倍数阈值（当日成交量≥前一日X倍）
    only_10pct_a: bool = True         # 是否仅保留10%涨停上限A股常见前缀

    # 新增：价格区间过滤（排除某些买入价区间）
    # exclude_price_ranges: list[tuple[float, float]] = ()
    # 例如 [(15,20),(25,30)] 表示排除 [15,20) 与 [25,30)
    exclude_price_ranges: list[tuple[float, float]] = ((25.0, 30.0),)
    # exclude_price_ranges: list[tuple[float, float]] = ()

    # 新增：一年内位置过滤（排除某些位置区间，单位：%）
    # 例如 [(60, 1e9)] 表示排除 [60, +inf)
    # 例如 [(60, 70), (80, 90)] 表示排除 [60,70) 与 [80,90)
    exclude_pos_1y_ranges_pct: list[tuple[float, float]] = ((0.0, 10.0), (80.0, 90.0))
    # exclude_pos_1y_ranges_pct: list[tuple[float, float]] = ()

    # 估值过滤（PE-TTM）：估值上下限，None 表示不限制
    pe_min: float | None = None       # PE(TTM)下限，例如10.0
    pe_max: float | None = None       # PE(TTM)上限，例如40.0
    # 形态附加条件：最近N天全部收红，或最近N天内存在单日涨幅≥阈值
    use_extra_cond: bool = True       # 是否启用附加形态条件过滤
    last_n_days_red_n: int = 3        # 最近收红天数要求的窗口（默认3天），也作为“连续上涨天数”
    up_pct_days_n: int = 3            # 最近涨幅检查天数（当前逻辑由“任意一日”改为“当日”）
    up_pct_threshold: float = 0.03     # 单日涨幅阈值（比例，默认3%）
    # 当日最高价与收盘价允许的最大偏离比例（默认 10%）
    high_close_tol: float = 0.10
    # 启动质量开关：是否启用启动质量过滤
    enable_start_quality: bool = True
    # 1年区间位置过滤（单位：%）。None 表示不启用。
    # pos_1y_min_pct: float | None = 30.0
    # pos_1y_max_pct: float | None = 65.0
    pos_1y_min_pct: float | None = None
    pos_1y_max_pct: float | None = None
    # 单股测试：仅扫描该股票文件名（如 "603598.SH"），None 表示不限制
    test_single_symbol: str | None = None

    # 新增：信号日涨幅过滤（排除某些涨幅区间，单位：%）
    # 例如 [(5.0, 6.0)] 表示排除信号日涨幅位于 [5.0, 6.0)
    # exclude_sig_up_pct_ranges: list[tuple[float, float]] = field(default_factory=lambda: [(5.0, 6.0)])  # type: ignore[assignment]
    exclude_sig_up_pct_ranges: list[tuple[float, float]] = field(default_factory=lambda: [()])  # type: ignore[assignment]

    # 新增：信号日放量倍数过滤（排除某些放量倍数区间）
    # 例如 [(2.0, 3.0)] 表示排除信号日放量倍数位于 [2.0, 3.0)
    # exclude_sig_vol_ratio_ranges: list[tuple[float, float]] = field(default_factory=lambda: [(7.0, 9.0)])  # type: ignore[assignment]
    exclude_sig_vol_ratio_ranges: list[tuple[float, float]] = field(default_factory=lambda: [()])  # type: ignore[assignment]

    # --- 新增：评分权重配置（原为模块常量 W_*）
    # 三项权重：股价段 / 一年位置段 / 三连涨信号
    w_price: float = 0.23
    w_pos: float = 0.23
    w_sig: float = 0.08

    # 新增两项权重：信号日涨幅档位 / 信号日放量倍数档位
    w_sig_up_pct: float = 0.23
    w_sig_vol_ratio: float = 0.23

    # 类别内：胜率/均盈亏权重
    w_win_in_cat: float = 0.7
    w_pnl_in_cat: float = 0.3

    def __post_init__(self):
        # 保持兼容：若外部显式传入 None，则转为空列表
        if self.exclude_sig_up_pct_ranges is None:
            self.exclude_sig_up_pct_ranges = []
        if self.exclude_sig_vol_ratio_ranges is None:
            self.exclude_sig_vol_ratio_ranges = []

CFG = Config()

# 数据目录格式：

# CSV 文件名形如 000001.SZ.csv
# 列为 trade_date,open,high,low,close,volume,amount
# 默认筛选条件与 Tushare 版一致：

# 近3个月振幅在 10%~20%
# 收盘接近近低 3% 内
# 最近一年涨停日数 > 5（按收盘涨幅≥9.8%近似）
# 最近5日存在放量（当日成交量≥前一日2倍）
# 仅 A 股 10% 涨停上限常见前缀（沪:600/601/603/605，深:000/001/002/003）
# 使用示例（你的数据目录）： powershell: python select_stocks_local.py --data-dir "E:\work\SynologyDrive\量化交易\通达信\通达信\data\pytdx\daily_raw" --out ".\output\selection_local.csv"

# 如需调整阈值，可加参数：

# --months 3
# --range-lower 0.10 --range-upper 0.20
# --near-low-tol 0.03
# --limitup-months 12 --limitup-threshold 0.098
# --vol-days 5 --vol-factor 2.0
# 关闭仅10% A股过滤：--no-only-10pct-a

# 仅保留 A 股中 10% 涨停上限常见前缀（不含 B 股）
ALLOW_PREFIXES_10P_SH = ("600", "601", "603", "605")  # 沪A
ALLOW_PREFIXES_10P_SZ = ("000", "001", "002", "003")  # 深A

# --- 打分配置（来自回测汇总输出）
# 说明：以下胜率/平均盈亏来自你贴的统计表。若以后你更新统计，只需改这里即可。
PRICE_SEG_STATS = {
    "0-5": {"win_rate": 0.547619, "avg_pnl": 199.949000},
    "5-10": {"win_rate": 0.420000, "avg_pnl": 42.254467},
    "10-15": {"win_rate": 0.543478, "avg_pnl": 113.287826},
    "15-20": {"win_rate": 0.440476, "avg_pnl": -23.591905},
    "20-25": {"win_rate": 0.475000, "avg_pnl": -0.002250},
    "25-30": {"win_rate": 0.352941, "avg_pnl": -75.239118},
    "30+": {"win_rate": 0.444444, "avg_pnl": 44.042407},
}

POS_SEG_STATS = {
    "<0": {"win_rate": None, "avg_pnl": None},
    "0-10": {"win_rate": 0.388889, "avg_pnl": -11.345333},
    "10-20": {"win_rate": 0.464481, "avg_pnl": 68.579399},
    "20-30": {"win_rate": 0.478873, "avg_pnl": 96.020634},
    "30-40": {"win_rate": 0.500000, "avg_pnl": 69.060844},
    "40-50": {"win_rate": 0.555556, "avg_pnl": 129.442302},
    "50-60": {"win_rate": 0.483871, "avg_pnl": 45.827204},
    "60-70": {"win_rate": 0.404762, "avg_pnl": 121.017143},
    "70-80": {"win_rate": 0.428571, "avg_pnl": 378.411905},
    "80-90": {"win_rate": 0.222222, "avg_pnl": -140.286667},
    "90-100": {"win_rate": None, "avg_pnl": None},
    "100+": {"win_rate": None, "avg_pnl": None},
}

SIG_STATS = {
    "否": {"win_rate": 0.480055, "avg_pnl": 90.855681},
    "是": {"win_rate": 0.443609, "avg_pnl": 15.681353},
}

# --- 新增：信号日涨幅(%) 分段统计
# 档位：<=3.99, 4, 5, 6, 7, 8, 9<（保留 10+ 以防未来出现）
SIG_UP_PCT_SEG_STATS = {
    "<=3.99": {"win_rate": 0.489669, "avg_pnl": 86.430723},
    "4": {"win_rate": 0.458716, "avg_pnl": 97.439266},
    "5": {"win_rate": 0.413462, "avg_pnl": -30.966346},
    "6": {"win_rate": 0.459459, "avg_pnl": 277.994324},
    "7": {"win_rate": 0.625000, "avg_pnl": 78.337500},
    "8": {"win_rate": 1.000000, "avg_pnl": 503.795000},
    "9<": {"win_rate": None, "avg_pnl": None},
    "10+": {"win_rate": None, "avg_pnl": None},
}

# --- 新增：信号日放量倍数 分段统计
# 档位：>=2, 3, 4, 5, 6, 7, 8, 9, 10+（保留 <=2 以防未来出现）
SIG_VOL_RATIO_SEG_STATS = {
    "<=2": {"win_rate": None, "avg_pnl": None},
    ">=2": {"win_rate": 0.449438, "avg_pnl": 101.329978},
    "3": {"win_rate": 0.464912, "avg_pnl": 29.550307},
    "4": {"win_rate": 0.566667, "avg_pnl": 123.392222},
    "5": {"win_rate": 0.621622, "avg_pnl": 248.105405},
    "6": {"win_rate": 0.571429, "avg_pnl": 25.779643},
    "7": {"win_rate": 0.375000, "avg_pnl": -7.408125},
    "8": {"win_rate": 0.000000, "avg_pnl": -246.790000},
    "9": {"win_rate": 0.333333, "avg_pnl": 38.700000},
    "10+": {"win_rate": 0.600000, "avg_pnl": 55.376000},
}

# 评分归一化：满分 100 分
# - 方法：为每一类（股价段/一年位置段/三连涨信号）先算一个 0~100 的“类别分”，再按权重累加成总分（0~100）
# - 总分是固定标尺（跨批次可比），不再对“本次选股结果”做 min-max
SCORE_MAX = 100.0
SCORE_MIN = 0.0
SCORE_FLAT_DEFAULT = 50.0

# 三项权重：股价段 / 一年位置段 / 三连涨信号
W_PRICE = 0.23
W_POS = 0.23
W_SIG = 0.08

# 新增两项权重：信号日涨幅档位 / 信号日放量倍数档位
W_SIG_UP_PCT = 0.23
W_SIG_VOL_RATIO = 0.23

# 类别内：胜率/均盈亏权重
W_WIN_IN_CAT = 0.7
W_PNL_IN_CAT = 0.3


def _min_max_normalize(series: pd.Series, vmin: float = SCORE_MIN, vmax: float = SCORE_MAX, flat_default: float = SCORE_FLAT_DEFAULT) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.dropna().empty:
        return pd.Series([flat_default] * len(series), index=series.index, dtype=float)

    mn = float(s.min())
    mx = float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([flat_default] * len(series), index=series.index, dtype=float)

    out = (s - mn) / (mx - mn) * (vmax - vmin) + vmin
    return out.astype(float)


def _min_max_norm_value(x: float | None, vmin_x: float | None, vmax_x: float | None, out_min: float = SCORE_MIN, out_max: float = SCORE_MAX, flat_default: float = SCORE_FLAT_DEFAULT) -> float:
    """把单个标量 x 做 min-max 映射到 [out_min, out_max]。

    - x 或 min/max 缺失 -> flat_default
    - vmax==vmin -> flat_default
    - 结果会 clip 到区间内
    """
    if x is None or pd.isna(x) or vmin_x is None or vmax_x is None:
        return float(flat_default)
    try:
        mn = float(vmin_x)
        mx = float(vmax_x)
        if mx - mn <= 1e-12:
            return float(flat_default)
        v = (float(x) - mn) / (mx - mn) * (out_max - out_min) + out_min
        v = max(float(out_min), min(float(out_max), float(v)))
        return float(v)
    except Exception:
        return float(flat_default)


def _stats_min_max(stats: dict) -> dict[str, tuple[float | None, float | None]]:
    """从某一类 stats（PRICE_SEG_STATS/POS_SEG_STATS/SIG_STATS）里抽取 win_rate / avg_pnl 的 min/max。"""
    wrs: list[float] = []
    pnls: list[float] = []
    for _, v in (stats or {}).items():
        if not isinstance(v, dict):
            continue
        wr = v.get('win_rate')
        pnl = v.get('avg_pnl')
        if wr is not None and not pd.isna(wr):
            wrs.append(float(wr))
        if pnl is not None and not pd.isna(pnl):
            pnls.append(float(pnl))

    wr_min = min(wrs) if wrs else None
    wr_max = max(wrs) if wrs else None
    pnl_min = min(pnls) if pnls else None
    pnl_max = max(pnls) if pnls else None
    return {
        'win_rate': (wr_min, wr_max),
        'avg_pnl': (pnl_min, pnl_max),
    }


# 预先计算各类别内部的 min/max，用于固定标尺 0~100
_PRICE_MM = _stats_min_max(PRICE_SEG_STATS)
_POS_MM = _stats_min_max(POS_SEG_STATS)
_SIG_MM = _stats_min_max(SIG_STATS)
# 新增：两类信号日统计 min/max
_SIG_UP_PCT_MM = _stats_min_max(SIG_UP_PCT_SEG_STATS)
_SIG_VOL_RATIO_MM = _stats_min_max(SIG_VOL_RATIO_SEG_STATS)


def _score_category(win_rate: float | None, avg_pnl: float | None, mm: dict[str, tuple[float | None, float | None]], w_win: float = W_WIN_IN_CAT, w_pnl: float = W_PNL_IN_CAT) -> tuple[float, float, float]:
    """按某一类的统计表，把(胜率,均盈亏) -> (类别分, 胜率子分, 均盈亏子分)，全部 0~100。"""
    wr_min, wr_max = mm.get('win_rate', (None, None))
    pnl_min, pnl_max = mm.get('avg_pnl', (None, None))

    win_score = _min_max_norm_value(win_rate, wr_min, wr_max, out_min=SCORE_MIN, out_max=SCORE_MAX, flat_default=SCORE_FLAT_DEFAULT)
    pnl_score = _min_max_norm_value(avg_pnl, pnl_min, pnl_max, out_min=SCORE_MIN, out_max=SCORE_MAX, flat_default=SCORE_FLAT_DEFAULT)

    cat_score = float(w_win) * float(win_score) + float(w_pnl) * float(pnl_score)
    cat_score = max(SCORE_MIN, min(SCORE_MAX, cat_score))
    return float(cat_score), float(win_score), float(pnl_score)


def _score_category(
    win_rate: float | None,
    avg_pnl: float | None,
    mm: dict[str, tuple[float | None, float | None]],
    w_win: float | None = None,
    w_pnl: float | None = None,
) -> tuple[float, float, float]:
    """按某一类的统计表，把(胜率,均盈亏) -> (类别分, 胜率子分, 均盈亏子分)，全部 0~100。

    权重默认从 CFG 读取，确保批量/回测可复现。
    """
    if w_win is None:
        w_win = float(getattr(CFG, 'w_win_in_cat', W_WIN_IN_CAT))
    if w_pnl is None:
        w_pnl = float(getattr(CFG, 'w_pnl_in_cat', W_PNL_IN_CAT))

    wr_min, wr_max = mm.get('win_rate', (None, None))
    pnl_min, pnl_max = mm.get('avg_pnl', (None, None))

    win_score = _min_max_norm_value(win_rate, wr_min, wr_max, out_min=SCORE_MIN, out_max=SCORE_MAX, flat_default=SCORE_FLAT_DEFAULT)
    pnl_score = _min_max_norm_value(avg_pnl, pnl_min, pnl_max, out_min=SCORE_MIN, out_max=SCORE_MAX, flat_default=SCORE_FLAT_DEFAULT)

    cat_score = float(w_win) * float(win_score) + float(w_pnl) * float(pnl_score)
    cat_score = max(SCORE_MIN, min(SCORE_MAX, cat_score))
    return float(cat_score), float(win_score), float(pnl_score)


def _format_seg_value(seg: str | None) -> str:
    return "N/A" if seg is None else str(seg)


def _price_segment(price: float | None) -> str | None:
    if price is None or pd.isna(price):
        return None
    p = float(price)
    if p < 5:
        return "0-5"
    if p < 10:
        return "5-10"
    if p < 15:
        return "10-15"
    if p < 20:
        return "15-20"
    if p < 25:
        return "20-25"
    if p < 30:
        return "25-30"
    return "30+"


def _pos_segment_pct(pos_pct: float | None) -> str | None:
    if pos_pct is None or pd.isna(pos_pct):
        return None
    v = float(pos_pct)
    if v < 0:
        return "<0"
    if v < 10:
        return "0-10"
    if v < 20:
        return "10-20"
    if v < 30:
        return "20-30"
    if v < 40:
        return "30-40"
    if v < 50:
        return "40-50"
    if v < 60:
        return "50-60"
    if v < 70:
        return "60-70"
    if v < 80:
        return "70-80"
    if v < 90:
        return "80-90"
    if v < 100:
        return "90-100"
    return "100+"


def _sig_up_pct_segment(up_pct: float | None) -> str | None:
    """信号日涨幅分段（单位：%）。"""
    if up_pct is None or pd.isna(up_pct):
        return None
    v = float(up_pct)
    # 注意：你的统计表档位是 <=3.99, 4, 5, 6, 7, 8, 9<
    if v <= 3.99:
        return "<=3.99"
    if v < 4.999999:
        return "4"
    if v < 5.999999:
        return "5"
    if v < 6.999999:
        return "6"
    if v < 7.999999:
        return "7"
    if v < 8.999999:
        return "8"
    if v < 10.0:
        return "9<"
    return "10+"


def _sig_vol_ratio_segment(vol_ratio: float | None) -> str | None:
    """信号日放量倍数分段（volume_today / volume_yesterday）。"""
    if vol_ratio is None or pd.isna(vol_ratio):
        return None
    v = float(vol_ratio)
    # 统计表以 >=2 为入门；其后按整数档位
    if v < 2.0:
        return "<=2"
    if v < 3.0:
        return ">=2"
    if v < 4.0:
        return "3"
    if v < 5.0:
        return "4"
    if v < 6.0:
        return "5"
    if v < 7.0:
        return "6"
    if v < 8.0:
        return "7"
    if v < 9.0:
        return "8"
    if v < 10.0:
        return "9"
    return "10+"


def _today_up_pct_value(df: pd.DataFrame) -> float | None:
    """返回当日涨幅（单位：%），计算口径：close_today vs close_yesterday。"""
    if df is None or len(df) < 2 or 'close' not in df.columns:
        return None
    prev_close = float(df['close'].iloc[-2])
    today_close = float(df['close'].iloc[-1])
    if prev_close <= 0:
        return None
    return float((today_close - prev_close) / prev_close * 100.0)


def _today_vol_ratio_value(df: pd.DataFrame) -> float | None:
    """返回当日放量倍数：volume_today / volume_yesterday。"""
    if df is None or len(df) < 2 or 'volume' not in df.columns:
        return None
    prev_vol = float(df['volume'].iloc[-2])
    today_vol = float(df['volume'].iloc[-1])
    if prev_vol <= 0:
        return None
    return float(today_vol / prev_vol)


def score_row_by_backtest_stats(
    last_close: float | None,
    pos_in_1y_pct: float | None,
    last3_up: bool | None,
    sig_today_up_pct: float | None = None,
    sig_today_vol_ratio: float | None = None,
) -> tuple[float, str]:
    """根据回测统计表为单条选股结果打分。

    返回：
    - raw_score: 固定标尺总分（0~100，跨批次可比）
    - reason: 可读原因（包含各项类别分及其子分）
    """
    # 1) 股价区间
    price_seg = _price_segment(last_close)
    pstat = PRICE_SEG_STATS.get(price_seg or "", {})
    p_wr = pstat.get("win_rate")
    p_pnl = pstat.get("avg_pnl")

    # 2) 一年内位置区间
    pos_seg = _pos_segment_pct(pos_in_1y_pct)
    ostat = POS_SEG_STATS.get(pos_seg or "", {})
    o_wr = ostat.get("win_rate")
    o_pnl = ostat.get("avg_pnl")

    # 3) 三连涨信号
    sig_key = "是" if bool(last3_up) else "否"
    sstat = SIG_STATS.get(sig_key, {})
    s_wr = sstat.get("win_rate")
    s_pnl = sstat.get("avg_pnl")

    # 4) 新增：信号日涨幅档位
    sig_up_seg = _sig_up_pct_segment(sig_today_up_pct)
    upstat = SIG_UP_PCT_SEG_STATS.get(sig_up_seg or "", {})
    u_wr = upstat.get("win_rate")
    u_pnl = upstat.get("avg_pnl")

    # 5) 新增：信号日放量倍数档位
    sig_vol_seg = _sig_vol_ratio_segment(sig_today_vol_ratio)
    vstat = SIG_VOL_RATIO_SEG_STATS.get(sig_vol_seg or "", {})
    v_wr = vstat.get("win_rate")
    v_pnl = vstat.get("avg_pnl")

    # --- 分项打分（每类内部先归一化到 0~100，再按权重累加成总分 0~100）
    # 权重从 CFG 读取（总和应为 1.0），确保配置可落盘/可复现。
    w_price = float(getattr(CFG, 'w_price', W_PRICE))
    w_pos = float(getattr(CFG, 'w_pos', W_POS))
    w_sig = float(getattr(CFG, 'w_sig', W_SIG))
    w_up = float(getattr(CFG, 'w_sig_up_pct', W_SIG_UP_PCT))
    w_vol = float(getattr(CFG, 'w_sig_vol_ratio', W_SIG_VOL_RATIO))

    score_price, score_price_win, score_price_pnl = _score_category(p_wr, p_pnl, _PRICE_MM)
    score_pos, score_pos_win, score_pos_pnl = _score_category(o_wr, o_pnl, _POS_MM)
    score_sig, score_sig_win, score_sig_pnl = _score_category(s_wr, s_pnl, _SIG_MM)
    score_up, score_up_win, score_up_pnl = _score_category(u_wr, u_pnl, _SIG_UP_PCT_MM)
    score_vol, score_vol_win, score_vol_pnl = _score_category(v_wr, v_pnl, _SIG_VOL_RATIO_MM)

    raw_score = (
        w_price * float(score_price)
        + w_pos * float(score_pos)
        + w_sig * float(score_sig)
        + w_up * float(score_up)
        + w_vol * float(score_vol)
    )
    raw_score = max(SCORE_MIN, min(SCORE_MAX, float(raw_score)))

    # --- 原因文本（带上分项分数，方便校验）
    reasons: list[str] = []
    reasons.append(
        f"股价区间={_format_seg_value(price_seg)}(胜率={('N/A' if p_wr is None else round(float(p_wr)*100,2))}%,均盈亏={('N/A' if p_pnl is None else round(float(p_pnl),2))})"
        f"=> 类别分={score_price:.2f}(胜率分={score_price_win:.2f},盈亏分={score_price_pnl:.2f},权重={w_price:.3f})"
    )
    reasons.append(
        f"一年位置={_format_seg_value(pos_seg)}(胜率={('N/A' if o_wr is None else round(float(o_wr)*100,2))}%,均盈亏={('N/A' if o_pnl is None else round(float(o_pnl),2))})"
        f"=> 类别分={score_pos:.2f}(胜率分={score_pos_win:.2f},盈亏分={score_pos_pnl:.2f},权重={w_pos:.3f})"
    )
    reasons.append(
        f"三连涨信号={sig_key}(胜率={('N/A' if s_wr is None else round(float(s_wr)*100,2))}%,均盈亏={('N/A' if s_pnl is None else round(float(s_pnl),2))})"
        f"=> 类别分={score_sig:.2f}(胜率分={score_sig_win:.2f},盈亏分={score_sig_pnl:.2f},权重={w_sig:.3f})"
    )

    up_val_str = 'N/A' if sig_today_up_pct is None or pd.isna(sig_today_up_pct) else f"{float(sig_today_up_pct):.2f}%"
    reasons.append(
        f"信号日涨幅={up_val_str},档位={_format_seg_value(sig_up_seg)}(胜率={('N/A' if u_wr is None else round(float(u_wr)*100,2))}%,均盈亏={('N/A' if u_pnl is None else round(float(u_pnl),2))})"
        f"=> 类别分={score_up:.2f}(胜率分={score_up_win:.2f},盈亏分={score_up_pnl:.2f},权重={w_up:.2f})"
    )

    vol_val_str = 'N/A' if sig_today_vol_ratio is None or pd.isna(sig_today_vol_ratio) else f"{float(sig_today_vol_ratio):.2f}x"
    reasons.append(
        f"信号日放量倍数={vol_val_str},档位={_format_seg_value(sig_vol_seg)}(胜率={('N/A' if v_wr is None else round(float(v_wr)*100,2))}%,均盈亏={('N/A' if v_pnl is None else round(float(v_pnl),2))})"
        f"=> 类别分={score_vol:.2f}(胜率分={score_vol_win:.2f},盈亏分={score_vol_pnl:.2f},权重={w_vol:.2f})"
    )

    return raw_score, reasons


def load_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    # 规范列
    exp = ['trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    missing = [c for c in exp if c not in df.columns]
    if missing:
        return pd.DataFrame()

    # 解析日期并排序（兼容 trade_date 为 int/float 导致被误解析成 1970 时间戳的问题）
    # 预期格式：YYYYMMDD（如 20260112）
    td_raw = df['trade_date']
    td_str = td_raw.astype(str).str.strip()
    td_str = td_str.str.replace(r'\.0$', '', regex=True)

    td = pd.to_datetime(td_str, format='%Y%m%d', errors='coerce')
    # 若不是 YYYYMMDD（例如带 '-' 或 'YYYY/MM/DD'），则回退通用解析
    mask = td.isna() & td_str.ne('')
    if bool(mask.any()):
        td.loc[mask] = pd.to_datetime(td_str.loc[mask], errors='coerce')

    df['trade_date'] = td
    df = df.dropna(subset=['trade_date']).sort_values('trade_date').reset_index(drop=True)

    # 数值列
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def is_10pct_a_share(code: str, market: str) -> bool:
    if market == 'SH':
        return code.startswith(ALLOW_PREFIXES_10P_SH)
    else:
        return code.startswith(ALLOW_PREFIXES_10P_SZ)


def calc_metrics(df: pd.DataFrame) -> tuple[float, float, float, float]:
    if df.empty:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    high = float(df['high'].max())
    low = float(df['low'].min())
    last_close = float(df['close'].iloc[-1])
    if low <= 0:
        return (high, low, last_close, float('nan'))
    rng = (high - low) / low
    return (high, low, last_close, rng)


def near_low_enough(last_close: float, low: float, tol: float) -> bool:
    import math
    if any(math.isnan(x) for x in [last_close, low]) or low <= 0:
        return False
    return abs(last_close - low) / low <= tol


def count_limit_up_days(df: pd.DataFrame, threshold: float = 0.098) -> int:
    if df is None or df.empty:
        return 0
    pre_close = df['close'].shift(1)
    pct = (df['close'] - pre_close) / pre_close
    cond = (pre_close > 0) & (pct >= (threshold - 1e-4))
    return int(cond.sum())


def has_volume_spike_last_n_days(df: pd.DataFrame, n: int = 5, factor: float = 2.0) -> bool:
    if df is None or df.empty or 'volume' not in df.columns:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    ratios = sub['volume'] / sub['volume'].shift(1)
    return bool((ratios.iloc[1:] >= factor).any())


def _last_n_days_red(df: pd.DataFrame, n: int = CFG.last_n_days_red_n) -> bool:
    """最近 n 天全部收红（收盘价 > 开盘价）。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n)
    if len(sub) < n:
        return False
    return bool((sub['close'] > sub['open']).all())


def _has_single_day_up_pct(df: pd.DataFrame, n: int = CFG.up_pct_days_n, threshold: float = CFG.up_pct_threshold) -> bool:
    """[旧逻辑] 最近 n 天内存在单日涨幅 ≥ threshold（按收盘相对前一日收盘）。保留以兼容历史，如不需要可删除。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1).copy()
    if len(sub) < 2:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 排除第一天无前收
    return bool((pct >= threshold - 1e-4).any())


def _is_today_volume_spike(df: pd.DataFrame, factor: float = CFG.vol_factor) -> bool:
    """当日成交量是否相对前一日放量：volume_today >= factor * volume_yesterday。"""
    if df is None or len(df) < 2 or 'volume' not in df.columns:
        return False
    today_vol = df['volume'].iloc[-1]
    prev_vol = df['volume'].iloc[-2]
    if prev_vol <= 0:
        return False
    return bool(today_vol >= factor * prev_vol)


def _is_today_up_pct(df: pd.DataFrame, threshold: float = CFG.up_pct_threshold) -> bool:
    """当日收盘相对前一日收盘涨幅是否 ≥ threshold。"""
    if df is None or len(df) < 2 or 'close' not in df.columns:
        return False
    prev_close = df['close'].iloc[-2]
    today_close = df['close'].iloc[-1]
    if prev_close <= 0:
        return False
    pct = (today_close - prev_close) / prev_close
    return bool(pct >= threshold - 1e-4)


def _is_last_n_days_all_up(df: pd.DataFrame, n: int = CFG.last_n_days_red_n) -> bool:
    """最近 n 天连续上涨：每一天的收盘价都高于前一日收盘价。"""
    if df is None or df.empty:
        return False
    sub = df.tail(n + 1)
    if len(sub) < n + 1:
        return False
    pre_close = sub['close'].shift(1)
    pct = (sub['close'] - pre_close) / pre_close
    pct = pct.iloc[1:]  # 最近 n 天
    return bool((pct > 0).all())


def _in_any_range(value: float, ranges: list[tuple[float, float]]) -> bool:
    """是否落在任意一个区间 [lo, hi)。"""
    if value is None or pd.isna(value):
        return False
    for lo, hi in ranges or []:
        try:
            if float(lo) <= float(value) < float(hi):
                return True
        except Exception:
            continue
    return False


def _in_any_price_range(price: float, ranges: list[tuple[float, float]]) -> bool:
    """是否落在任意一个区间 [lo, hi)。"""
    return _in_any_range(price, ranges)


def scan_dir(data_dir: str,
             months_lookback: int,
             range_lower: float,
             range_upper: float,
             near_low_tol: float,
             limitup_lookback_months: int,
             limitup_threshold: float,
             volume_spike_days: int,
             volume_spike_factor: float,
             only_10pct_a: bool,
             end_date: pd.Timestamp | None = None,
             preloaded: dict[str, pd.DataFrame] | None = None,
             quiet: bool = False,
             pos_1y_min_pct: float | None = None,
             pos_1y_max_pct: float | None = None,
             min_limitup_count: int | None = None,
             debug_symbol: str | None = None) -> pd.DataFrame:
    # 允许传入截断日期（用于回测），默认使用今天
    end_dt = (pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(datetime.today().strftime('%Y-%m-%d')))
    end = end_dt.date()
    start_3m = end - relativedelta(months=months_lookback)
    start_1y = end - relativedelta(months=limitup_lookback_months)
    s3 = pd.to_datetime(start_3m.strftime('%Y-%m-%d'))
    s1 = pd.to_datetime(start_1y.strftime('%Y-%m-%d'))
    e = pd.to_datetime(end.strftime('%Y-%m-%d'))

    # 准备迭代的源：文件或预加载数据
    if preloaded is None:
        files = [fn for fn in os.listdir(data_dir) if fn.lower().endswith('.csv')]
        files.sort()
        iterable = [(os.path.splitext(fn)[0], None, os.path.join(data_dir, fn)) for fn in files]
        pbar = None if quiet else tqdm(total=len(files), desc='筛选(本地CSV)', dynamic_ncols=True)
    else:
        keys = sorted(preloaded.keys())
        iterable = [(stem, preloaded.get(stem), None) for stem in keys]
        pbar = None if quiet else tqdm(total=len(keys), desc='筛选(预加载)', dynamic_ncols=True)

    results = []
    found = 0

    for stem, df_pre, fp in iterable:
        # 如果设置了单股测试，只保留指定 symbol（例如 "603598.SH"）
        if CFG.test_single_symbol is not None and stem != CFG.test_single_symbol:
            if pbar: pbar.update(1)
            continue

        # 新增：命令行调试单票
        if debug_symbol is not None and stem != debug_symbol:
            if pbar: pbar.update(1)
            continue

        # 解析 symbol/code/market（兼容 preloaded 模式，没有文件名时也能得到 code/market）
        parts = stem.split('.')
        if len(parts) != 2:
            if pbar: pbar.update(1)
            continue
        code, market = parts[0], parts[1].upper()
        if only_10pct_a and not is_10pct_a_share(code, market):
            if pbar: pbar.update(1)
            continue

        # 加载数据
        if df_pre is not None:
            df = df_pre
        else:
            df = load_csv(fp) if fp is not None else pd.DataFrame()
        if df.empty or len(df) < 10:
            if pbar: pbar.update(1)
            continue

        # 窗口切片（按传入的 e 截断）
        df_cut = df[df['trade_date'] <= e]
        if df_cut.empty or len(df_cut) < 10:
            if pbar: pbar.update(1)
            continue
        bars_all = df_cut[(df_cut['trade_date'] >= s1) & (df_cut['trade_date'] <= e)]
        bars_3m = df_cut[(df_cut['trade_date'] >= s3) & (df_cut['trade_date'] <= e)]
        if bars_3m.empty or len(bars_3m) < 10:
            if pbar: pbar.update(1)
            continue

        high, low, last_close, rng = calc_metrics(bars_3m)
        if pd.isna(rng):
            if pbar: pbar.update(1)
            continue

        # 新增：股价区间过滤（排除指定价格段）
        price_ok = True
        if CFG.exclude_price_ranges:
            price_ok = not _in_any_price_range(last_close, list(CFG.exclude_price_ranges))

        # 计算当日 K 线实体位置约束：最高价与收盘价偏离不超过 CFG.high_close_tol
        today_high = float(bars_3m['high'].iloc[-1])
        today_low = float(bars_3m['low'].iloc[-1])
        today_close = float(bars_3m['close'].iloc[-1])
        high_close_ok = True
        if today_close > 0:
            high_close_ok = (today_high - today_close) / today_close <= CFG.high_close_tol + 1e-4

        # --- 新增：信号日涨幅/放量倍数过滤（必须在 passed 里使用前先定义）
        # 说明：
        # - end_date=e 时，“信号日”定义为 bars_3m 的最后一根K线（即 e 对应/之前的最后交易日）。
        # - 若无法计算指标（数据不足、缺列、pre_close<=0 等），则不过滤（保持 True），避免误杀。
        sig_today_up_pct = None
        sig_today_vol_ratio = None
        sig_up_ok = True
        sig_vol_ok = True

        try:
            # 信号日涨幅(%)：用当日 close 与前一日 close 计算（与其它逻辑保持一致）
            if bars_3m is not None and len(bars_3m) >= 2:
                _pc = float(bars_3m['close'].iloc[-2])
                _c = float(bars_3m['close'].iloc[-1])
                if _pc > 0:
                    sig_today_up_pct = (_c - _pc) / _pc * 100.0
        except Exception:
            sig_today_up_pct = None

        try:
            # 信号日放量倍数：今日 volume / 昨日 volume
            if bars_3m is not None and len(bars_3m) >= 2 and 'volume' in bars_3m.columns:
                _v0 = float(bars_3m['volume'].iloc[-2])
                _v1 = float(bars_3m['volume'].iloc[-1])
                if _v0 > 0 and _v1 > 0:
                    sig_today_vol_ratio = _v1 / _v0
        except Exception:
            sig_today_vol_ratio = None

        # 应用排除区间：只有当指标可计算时才生效
        try:
            _ranges = list(getattr(CFG, 'exclude_sig_up_pct_ranges', []) or [])
            if sig_today_up_pct is not None and (not pd.isna(sig_today_up_pct)) and _ranges:
                sig_up_ok = not _in_any_range(float(sig_today_up_pct), _ranges)
        except Exception:
            sig_up_ok = True

        try:
            _ranges = list(getattr(CFG, 'exclude_sig_vol_ratio_ranges', []) or [])
            if sig_today_vol_ratio is not None and (not pd.isna(sig_today_vol_ratio)) and _ranges:
                sig_vol_ok = not _in_any_range(float(sig_today_vol_ratio), _ranges)
        except Exception:
            sig_vol_ok = True

        # 计算今年区间内位置（按近一年窗口 bars_all）
        pos_pct = None
        if not bars_all.empty:
            high_1y = float(bars_all['high'].max())
            low_1y = float(bars_all['low'].min())
            if high_1y > low_1y and low_1y > 0:
                pos_pct = (last_close - low_1y) / (high_1y - low_1y)
                # 夹在 0~1 范围
                pos_pct = max(0.0, min(1.0, pos_pct))

        # 新增：一年内位置区间过滤（可配置多个排除段）
        # 过滤逻辑：若 pos_pct 无法计算 -> 不通过；若落在任意排除段 -> 不通过
        pos_exclude_ok = True
        if CFG.exclude_pos_1y_ranges_pct:
            if not isinstance(pos_pct, float):
                pos_exclude_ok = False
            else:
                pos_pct100 = pos_pct * 100.0
                pos_exclude_ok = not _in_any_range(pos_pct100, list(CFG.exclude_pos_1y_ranges_pct))

        # 1年区间位置过滤（pos_in_1y，单位%）：默认要求 30~65
        pos_ok = True
        if pos_1y_min_pct is not None or pos_1y_max_pct is not None:
            if not isinstance(pos_pct, float):
                pos_ok = False
            else:
                pos_pct100 = pos_pct * 100.0
                if pos_1y_min_pct is not None:
                    pos_ok = pos_ok and (pos_pct100 >= pos_1y_min_pct - 1e-9)
                if pos_1y_max_pct is not None:
                    pos_ok = pos_ok and (pos_pct100 <= pos_1y_max_pct + 1e-9)

        limitup_cnt = count_limit_up_days(bars_all, limitup_threshold)

        # 统一涨停次数门槛：优先使用传入参数，否则退化为全局配置 CFG.min_limitup_count
        _min_lu = CFG.min_limitup_count if min_limitup_count is None else int(min_limitup_count)

        vol_spike = _is_today_volume_spike(bars_all, volume_spike_factor)
        today_up = _is_today_up_pct(bars_all, CFG.up_pct_threshold)
        cont_up_n = _is_last_n_days_all_up(bars_all, CFG.last_n_days_red_n)
        extra_cond_val = (vol_spike and today_up) or cont_up_n
        extra_cond = (not CFG.use_extra_cond) or extra_cond_val

        # 启动质量：当日上涨 + 放量 + 收盘价不低于当日中轴
        start_quality = True
        if CFG.enable_start_quality:
            mid_price = (today_high + today_low) / 2.0 if today_high > 0 and today_low > 0 else float('inf')
            start_quality = bool(today_up and vol_spike and today_close >= mid_price - 1e-9)

        # 使用配置的 min_limitup_count 作为统一门槛
        passed = (
            range_lower - 1e-9 <= rng <= range_upper + 1e-9
            and near_low_enough(last_close, low, near_low_tol)
            and limitup_cnt >= _min_lu
            and vol_spike
            and extra_cond
            and high_close_ok
            and start_quality
            and pos_ok
            and price_ok
            and pos_exclude_ok
            and sig_up_ok
            and sig_vol_ok
        )

        # 新增：指定 debug_symbol 时，即使未通过也输出原因（配合 --end-date 用来看某天为什么没触发）
        if debug_symbol is not None and stem == debug_symbol and not quiet:
            try:
                # 必选条件（passed 的直接组成）
                cond_rng = (range_lower - 1e-9 <= rng <= range_upper + 1e-9)
                cond_near_low = near_low_enough(last_close, low, near_low_tol)
                cond_limitup = (limitup_cnt >= _min_lu)
                cond_vol = bool(vol_spike)
                cond_extra = bool(extra_cond)
                cond_high_close = bool(high_close_ok)
                cond_start_q = bool(start_quality)
                cond_pos = bool(pos_ok)

                # 派生条件（用于理解 extra_cond / start_quality 等内部细节）
                pct_to_low = None
                if isinstance(low, (int, float)) and low and low > 0:
                    pct_to_low = (last_close - low) / low

                # 当日涨幅（用于 today_up）
                today_pct = None
                if bars_all is not None and len(bars_all) >= 2:
                    prev_close = float(bars_all['close'].iloc[-2])
                    today_close_ = float(bars_all['close'].iloc[-1])
                    if prev_close > 0:
                        today_pct = (today_close_ - prev_close) / prev_close

                # 当日量比（用于 vol_spike）
                vol_ratio = None
                if bars_all is not None and len(bars_all) >= 2:
                    prev_vol = float(bars_all['volume'].iloc[-2])
                    today_vol = float(bars_all['volume'].iloc[-1])
                    if prev_vol > 0:
                        vol_ratio = today_vol / prev_vol

                # 高收偏离（用于 high_close_ok）
                high_close_gap = None
                if today_close and today_close > 0:
                    high_close_gap = (today_high - today_close) / today_close

                cutoff_dt = e.strftime('%Y-%m-%d') if hasattr(e, 'strftime') else str(e)

                def _flag(ok: bool) -> str:
                    return '✅' if ok else '❌'

                print(f"[DEBUG] 截止日期={cutoff_dt} 标的={stem} 入选={passed}")
                print("[DEBUG] 必选条件（全部满足才会入选）:")

                print(
                    f"  [{_flag(cond_rng)} 必选] 振幅区间: 近{months_lookback}个月最高/最低 -> rng={(rng*100):.2f}% "
                    f"要求[{range_lower*100:.2f}%,{range_upper*100:.2f}%]"
                )

                if pct_to_low is None:
                    print(
                        f"  [{_flag(cond_near_low)} 必选] 接近近低: 缺少有效 low 数据（low={low}）"
                    )
                else:
                    print(
                        f"  [{_flag(cond_near_low)} 必选] 接近近低: (收盘-近{months_lookback}个月最低)/最低 = "
                        f"({last_close:.3f}-{low:.3f})/{low:.3f} = {pct_to_low*100:.2f}% "
                        f"要求≤{near_low_tol*100:.2f}%"
                    )

                print(
                    f"  [{_flag(cond_limitup)} 必选] 涨停次数: 近{limitup_lookback_months}个月 cnt={limitup_cnt} "
                    f"门槛≥{_min_lu}（阈值≈{limitup_threshold*100:.2f}%）"
                )

                if vol_ratio is None:
                    print(
                        f"  [{_flag(cond_vol)} 必选] 当日放量: 缺少有效成交量/前一日数据（factor={volume_spike_factor}）"
                    )
                else:
                    print(
                        f"  [{_flag(cond_vol)} 必选] 当日放量: 量比=今日量/昨量 = {vol_ratio:.2f}x "
                        f"要求≥{volume_spike_factor:.2f}x"
                    )

                # extra_cond 内部解释（派生，但 extra_cond 本身是必选）
                if today_pct is None:
                    today_pct_str = 'N/A'
                else:
                    today_pct_str = f"{today_pct*100:.2f}%（阈值≥{CFG.up_pct_threshold*100:.2f}%）"

                print(
                    f"  [{_flag(cond_extra)} 必选] 附加条件: (当日放量 且 当日涨幅达标) 或 (近{CFG.last_n_days_red_n}天连续上涨)"
                )
                print(
                    f"      [派生] 当日涨幅: {today_pct_str} -> {'达标' if bool(today_up) else '未达标'}"
                )
                print(
                    f"      [派生] 近{CFG.last_n_days_red_n}天连续上涨: {('是' if bool(cont_up_n) else '否')}"
                )

                if high_close_gap is None:
                    print(
                        f"  [{_flag(cond_high_close)} 必选] 高收偏离: 缺少有效收盘价数据（tol={CFG.high_close_tol*100:.2f}%）"
                    )
                else:
                    print(
                        f"  [{_flag(cond_high_close)} 必选] 高收偏离: (最高-收盘)/收盘 = "
                        f"({today_high:.3f}-{today_close:.3f})/{today_close:.3f} = {high_close_gap*100:.2f}% "
                        f"要求≤{CFG.high_close_tol*100:.2f}%"
                    )

                # start_quality 内部解释（派生，但 start_quality 本身是必选）
                mid_price = (today_high + today_low) / 2.0 if today_high > 0 and today_low > 0 else None
                if mid_price is None:
                    mid_str = 'N/A'
                else:
                    mid_str = f"中轴={(mid_price):.3f}（要求收盘≥中轴）"

                print(
                    f"  [{_flag(cond_start_q)} 必选] 启动质量: (当日上涨 且 放量 且 收盘≥当日中轴)"
                )
                print(
                    f"      [派生] {mid_str} | 收盘={today_close:.3f}"
                )

                # pos_ok 当前默认不启用（min/max None），但 pos_ok 仍在 passed 里，因此仍标为必选
                pos_val_str = 'N/A' if not isinstance(pos_pct, float) else f"{pos_pct*100:.2f}%"
                print(
                    f"  [{_flag(cond_pos)} 必选] 一年位置过滤: min={pos_1y_min_pct} max={pos_1y_max_pct} pos={pos_val_str}"
                )

                # 非条件说明（信息，不影响入选）
                print("[DEBUG] 说明:")
                print("  - 上面标记【必选】的条件，是 passed 的组成部分；任何一个为 ❌ 都不会入选")
                print("  - 标记【派生】的是用于解释必选条件内部计算的中间量，本身不单独决定入选")

            except Exception:
                pass

        # 单股测试时，如果没通过，打印各条件方便排查
        if CFG.test_single_symbol is not None and stem == CFG.test_single_symbol and not passed and not quiet:
            try:
                print(
                    f"[DEBUG] {stem} 未入选原因: "
                    f"振幅={rng:.4f} 是否在区间[{range_lower},{range_upper}]内={range_lower - 1e-9 <= rng <= range_upper + 1e-9}; "
                    f"接近近低={near_low_enough(last_close, low, near_low_tol)}; "
                    f"近一年涨停天数={limitup_cnt} 是否≥配置门槛{_min_lu}={limitup_cnt >= _min_lu}; "
                    f"当日放量={vol_spike}; 当日涨幅达标={today_up}; 近{CFG.last_n_days_red_n}天连续上涨={cont_up_n}; "
                    f"当日高收价偏离≤{CFG.high_close_tol*100:.1f}%={high_close_ok}; "
                    f"启动质量(涨+量+收盘≥中轴)={start_quality}; "
                    f"附加条件整体={extra_cond}; "
                    f"价格过滤(排除{list(CFG.exclude_price_ranges)})={price_ok}; "
                    f"一年内位置过滤(排除{list(CFG.exclude_pos_1y_ranges_pct)})={pos_exclude_ok}; "
                    f"信号日涨幅过滤(排除{list(CFG.exclude_sig_up_pct_ranges)})={sig_up_ok}; "
                    f"信号日放量倍数过滤(排除{list(CFG.exclude_sig_vol_ratio_ranges)})={sig_vol_ok}"
                )
            except Exception:
                pass

        if passed:
            _raw_score, _reason = score_row_by_backtest_stats(
                last_close=last_close,
                pos_in_1y_pct=(pos_pct * 100.0 if isinstance(pos_pct, float) else None),
                last3_up=_is_last_n_days_all_up(bars_all, CFG.last_n_days_red_n),
                sig_today_up_pct=sig_today_up_pct,
                sig_today_vol_ratio=sig_today_vol_ratio,
            )

            results.append({
                'symbol': stem,
                'code': code,
                'market': market,
                'high': round(high, 3),
                'low': round(low, 3),
                'last_close': round(last_close, 3),
                'range_pct': round(rng * 100, 2),
                'limit_up_days_1y': '近一年涨停天数',
                'vol_spike_5d': bool(vol_spike),
                'pos_in_1y': (round(pos_pct * 100, 2) if isinstance(pos_pct, float) else None),
                'last3_up': _is_last_n_days_all_up(bars_all, CFG.last_n_days_red_n),
                'today_up_ge_3pct': _is_today_up_pct(bars_all, CFG.up_pct_threshold),
                # 新增：信号日涨幅与放量倍数（用于评分与输出）
                'sig_today_up_pct': (round(float(sig_today_up_pct), 4) if sig_today_up_pct is not None and not pd.isna(sig_today_up_pct) else None),
                'sig_today_vol_ratio': (round(float(sig_today_vol_ratio), 4) if sig_today_vol_ratio is not None and not pd.isna(sig_today_vol_ratio) else None),
                # 可选：输出启动质量标志
                'start_quality': bool(start_quality),
                # 评分
                'raw_score': _raw_score,
                'score_reason': _reason,
            })
            found += 1
            if not quiet:
                try:
                    tqdm.write(
                        f"入选: {stem} | 高:{round(high,3)} 低:{round(low,3)} 现:{round(last_close,3)} 振幅:{round(rng*100,2)}% "
                        f"涨停:{limitup_cnt} 放量:{'是' if vol_spike else '否'} 位置(1年):{(round(pos_pct*100,2) if isinstance(pos_pct,float) else 'N/A')}% "
                        f"近{CFG.last_n_days_red_n}天连续上涨:{'是' if _is_last_n_days_all_up(bars_all,CFG.last_n_days_red_n) else '否'} "
                        f"当日涨幅≥{int(CFG.up_pct_threshold*100)}%:{'是' if _is_today_up_pct(bars_all,CFG.up_pct_threshold) else '否'} "
                        f"附加条件:{'未启用' if not CFG.use_extra_cond else '启用'}"
                    )
                except Exception:
                    pass
        if pbar:
            pbar.set_postfix(found=found, code=stem)
            pbar.update(1)

    if pbar:
        pbar.close()
    return pd.DataFrame(results)


def _fetch_market_cap_eastmoney(code: str, market: str, timeout: float = 5.0) -> float | None:
    """通过东方财富接口获取总市值(元)。返回 float 或 None。
    市场编码：SH->1，SZ->0
    字段 f20: 总市值(元)
    """
    try:
        secid = ('1' if market.upper() == 'SH' else '0') + f'.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f20'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get('data') or {}
        val = data.get('f20')
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _fetch_pe_ttm_eastmoney(code: str, market: str, timeout: float = 5.0) -> float | None:
    """通过东方财富接口获取 PE(TTM)。字段 f9。返回 float 或 None。"""
    try:
        secid = ('1' if market.upper() == 'SH' else '0') + f'.{code}'
        url = f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}&fields=f9'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None
        j = r.json()
        data = j.get('data') or {}
        val = data.get('f9')
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _filter_by_market_cap_and_valuation(rows: list[dict], mc_min: float, mc_max: float, pe_min: float | None, pe_max: float | None) -> list[dict]:
    """按总市值与估值过滤。"""
    out = []
    pbar = tqdm(total=len(rows), desc='市值/估值过滤', dynamic_ncols=True)
    for r in rows:
        code = r.get('code', '')
        market = r.get('market', '')
        mc = _fetch_market_cap_eastmoney(code, market)
        pe = _fetch_pe_ttm_eastmoney(code, market)
        r['market_cap'] = mc
        r['pe_ttm'] = pe
        ok_mc = (mc is not None and mc_min - 1e-6 <= mc <= mc_max + 1e-6)
        ok_pe = True
        if pe_min is not None:
            ok_pe = ok_pe and (pe is not None and pe >= pe_min - 1e-9)
        if pe_max is not None:
            ok_pe = ok_pe and (pe is not None and pe <= pe_max + 1e-9)
        if ok_mc and ok_pe:
            out.append(r)
        pbar.set_postfix(code=f"{code}.{market}", mc=mc, pe=pe)
        pbar.update(1)
    pbar.close()
    return out


def main():
    # 使用硬编码的配置
    data_dir = CFG.data_dir
    out_path = CFG.default_out

    # 评分过滤阈值（按原始评分 raw_score 过滤；None 表示不启用）
    min_score_arg = CFG.min_score

    # 可选命令行参数：指定截止日期与静默模式
    end_date_arg = None
    quiet_arg = False
    debug_symbol_arg = None
    try:
        import argparse
        parser = argparse.ArgumentParser(description='本地CSV选股')
        parser.add_argument('--end-date', type=str, help='指定筛选截止日期(YYYYMMDD)，默认今天')
        parser.add_argument('--quiet', action='store_true', help='静默模式，减少日志输出')
        parser.add_argument('--debug-symbol', type=str, help='调试单票未触发原因，例如 603598.SH（配合 --end-date）')
        args, _ = parser.parse_known_args()
        if args.end_date:
            try:
                end_date_arg = pd.to_datetime(args.end_date)
            except Exception:
                print('end-date 参数格式错误，应为 YYYYMMDD，例如 20250901')
        quiet_arg = bool(args.quiet)
        if getattr(args, 'debug_symbol', None):
            debug_symbol_arg = str(args.debug_symbol).strip()
    except Exception:
        pass

    date_subdir = CFG.date_subdir
    mktcap_min = CFG.mktcap_min
    mktcap_max = CFG.mktcap_max
    skip_mktcap = CFG.skip_mktcap
    months = CFG.months
    range_lower = CFG.range_lower
    range_upper = CFG.range_upper
    near_low_tol = CFG.near_low_tol
    limitup_months = CFG.limitup_months
    limitup_threshold = CFG.limitup_threshold
    min_limitup_count = CFG.min_limitup_count
    vol_days = CFG.vol_days
    vol_factor = CFG.vol_factor
    only_10pct_a = CFG.only_10pct_a
    pe_min = CFG.pe_min
    pe_max = CFG.pe_max

    # 处理按日期子目录
    if date_subdir:
        date_str = datetime.today().strftime('%Y%m%d') if end_date_arg is None else pd.to_datetime(end_date_arg).strftime('%Y%m%d')
        out_dir = os.path.dirname(out_path) if out_path.lower().endswith('.csv') else out_path
        out_dir = os.path.join(out_dir, date_str)
        base_name = os.path.basename(out_path) if out_path.lower().endswith('.csv') else os.path.basename(CFG.default_out)
        out_path = os.path.join(out_dir, base_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 扫描并筛选
    df = scan_dir(
        data_dir=data_dir,
        months_lookback=months,
        range_lower=range_lower,
        range_upper=range_upper,
        near_low_tol=near_low_tol,
        limitup_lookback_months=limitup_months,
        limitup_threshold=limitup_threshold,
        volume_spike_days=vol_days,
        volume_spike_factor=vol_factor,
        only_10pct_a=only_10pct_a,
        end_date=end_date_arg,
        quiet=quiet_arg,
        pos_1y_min_pct=CFG.pos_1y_min_pct,
        pos_1y_max_pct=CFG.pos_1y_max_pct,
        min_limitup_count=min_limitup_count,
        debug_symbol=debug_symbol_arg,
    )

    if df.empty:
        print('无符合条件的标的')
        return

    # ---- 评分：固定标尺（0~100），raw_score 本身就是总分
    # 为保持与 backtest_select_stocks_local 一致：这里也做一次数值化 + 归一化(0~100)
    if 'raw_score' in df.columns:
        df['_raw_score_num'] = pd.to_numeric(df['raw_score'], errors='coerce')
        df['score'] = _min_max_normalize(df['_raw_score_num'], vmin=0.0, vmax=100.0, flat_default=50.0).round(2)
        df = df.drop(columns=['_raw_score_num'])

    # 评分过滤（按原始评分 raw_score）
    if min_score_arg is not None:
        if 'raw_score' in df.columns:
            df = df[pd.to_numeric(df['raw_score'], errors='coerce') >= float(min_score_arg)].copy()
        else:
            # 若无 raw_score 列则无法按评分过滤
            if not quiet_arg:
                print('未生成 raw_score 列，跳过评分过滤（min_score_arg 已设置但未产生 raw_score）')
        if df.empty:
            print(f'评分过滤后无标的（min_score={min_score_arg}）')
            return

    # 市值/估值过滤（使用硬编码参数）
    if not skip_mktcap:
        rows = df.to_dict(orient='records')
        rows = _filter_by_market_cap_and_valuation(rows, mktcap_min, mktcap_max, pe_min, pe_max)
        df = pd.DataFrame(rows)
        if df.empty:
            print('市值/估值过滤后无标的（检查区间是否合理）')
            return

    # 输出前将列名改为中文
   
    col_map = {
        'symbol': '标的',
        'code': '代码',
       
        'market': '市场',
        'high': '最高价',
        'low': '最低价',
        'last_close': '最新收盘价',
        'range_pct': '振幅(%)',
        'limit_up_days_1y': '近一年涨停天数',
        'vol_spike_5d': '当日放量',
        'pos_in_1y': '一年区间位置(%)',
        'last3_up': '近3天连续上涨',
        'today_up_ge_3pct': '当日涨幅≥3%',
        'raw_score': '原始评分',
        'score': '评分',
        'score_reason': '打分原因',
        # 新增输出字段
        'sig_today_up_pct': '信号日涨幅(%)',
        'sig_today_vol_ratio': '信号日放量倍数',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # 排序：优先评分（高分靠前），再按振幅（低振幅靠前）
    if '评分' in df.columns:
        df = df.sort_values(['评分', '振幅(%)'], ascending=[False, True]).reset_index(drop=True)
    else:
        df = df.sort_values('振幅(%)').reset_index(drop=True)

    if not quiet_arg:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', 120)
        print(df)

    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已保存到: {out_path} | 入选数量: {len(df)} | 截止日期: {(pd.to_datetime(end_date_arg).strftime('%Y-%m-%d') if end_date_arg is not None else datetime.today().strftime('%Y-%m-%d'))}")


if __name__ == '__main__':
    main()
