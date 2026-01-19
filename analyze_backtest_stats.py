import pandas as pd

path = r"e:\work\SynologyDrive\量化交易\通达信\output\backtest\20260117_2016\买入明细.csv"

# 新增：如果存在“权益曲线.csv”，则按年指标优先从权益曲线计算（更准）
EQUITY_CURVE_CANDIDATES = [
    r"e:\work\SynologyDrive\量化交易\通达信\output\backtest\20260117_2016\权益曲线.csv",
    r"e:\work\SynologyDrive\量化交易\通达信\output\backtest\20260117_2016\equity_curve.csv",
]

df = pd.read_csv(path, encoding="utf-8-sig")

# 统一字段/容错
df["买入价格"] = pd.to_numeric(df["买入价格"], errors="coerce")
df["前一日一年内位置(%)"] = pd.to_numeric(df["前一日一年内位置(%)"], errors="coerce")
# 卖出盈亏：正数=盈利，负数=亏损
df["卖出盈亏"] = pd.to_numeric(df["卖出盈亏"], errors="coerce")

# 新增：评分字段统计（每10分一组）
# 兼容没有该列的情况
if "原始评分" in df.columns:
    df["原始评分"] = pd.to_numeric(df["原始评分"], errors="coerce")
if "评分" in df.columns:
    df["评分"] = pd.to_numeric(df["评分"], errors="coerce")

# 新增：信号日涨幅(%) 分段统计
# 兼容没有该列的情况
if "信号日涨幅(%)" in df.columns:
    df["信号日涨幅(%)"] = pd.to_numeric(df["信号日涨幅(%)"], errors="coerce")

# 新增：信号日放量倍数 分段统计
# 兼容没有该列的情况
if "信号日放量倍数" in df.columns:
    df["信号日放量倍数"] = pd.to_numeric(df["信号日放量倍数"], errors="coerce")


def win_rate(x):
    x = x.dropna()
    if len(x) == 0:
        return float("nan")
    return (x == "盈利").mean()


def profit_mean(x):
    # 平均盈利金额（单位：与卖出盈亏一致）
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return x.mean()


def profit_sum(x):
    # 累计盈亏金额
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return float("nan")
    return x.sum()


def _safe_days_between(start: pd.Series, end: pd.Series) -> pd.Series:
    s = pd.to_datetime(start, errors="coerce")
    e = pd.to_datetime(end, errors="coerce")
    d = (e - s).dt.days
    return d


def _load_equity_curve() -> pd.DataFrame | None:
    """读取权益曲线.csv（若存在）。要求至少包含：日期, 总权益, 现金。"""
    import os

    for p in EQUITY_CURVE_CANDIDATES:
        if os.path.exists(p):
            eq = pd.read_csv(p, encoding="utf-8-sig")
            # 标准化列名
            eq.columns = [str(c).strip() for c in eq.columns]
            if "日期" not in eq.columns:
                continue
            if "总权益" not in eq.columns:
                continue
            # 现金缺失也能算回撤/收益，但占用率需要现金
            eq = eq.copy()
            eq["日期"] = pd.to_datetime(eq["日期"], errors="coerce")
            eq["总权益"] = pd.to_numeric(eq["总权益"], errors="coerce")
            if "现金" in eq.columns:
                eq["现金"] = pd.to_numeric(eq["现金"], errors="coerce")
            return eq.dropna(subset=["日期", "总权益"])
    return None


def _compute_yearly_metrics_from_equity_curve(eq: pd.DataFrame) -> pd.DataFrame:
    """基于每日权益曲线按年计算：资金平均占用率、总收益率、最大回撤。

    - 总收益率：当年最后一个交易日总权益 / 当年第一个交易日总权益 - 1
    - 最大回撤：当年内基于总权益的 peak-to-trough 最大回撤
    - 资金平均占用率：当年平均(1 - 现金/总权益)
      （若缺少现金列则为 NaN）
    """

    w = eq.copy().sort_values("日期")
    w["__year__"] = w["日期"].dt.year

    def _year_return(g: pd.DataFrame) -> float:
        g = g.dropna(subset=["总权益"])
        if len(g) == 0:
            return float("nan")
        first = g.iloc[0]["总权益"]
        last = g.iloc[-1]["总权益"]
        if pd.isna(first) or pd.isna(last) or first == 0:
            return float("nan")
        return (last / first) - 1

    def _year_max_dd(g: pd.DataFrame) -> float:
        s = g["总权益"].dropna()
        if len(s) == 0:
            return float("nan")
        peak = s.cummax()
        dd = (peak - s) / peak
        return float(dd.max())

    def _year_occ(g: pd.DataFrame) -> float:
        if "现金" not in g.columns:
            return float("nan")
        s_eq = g["总权益"].astype("float64")
        s_cash = g["现金"].astype("float64")
        occ = 1 - (s_cash / s_eq.replace({0: pd.NA}))
        return float(occ.mean() * 100)

    out = (
        w.groupby("__year__", dropna=False, observed=False)
         .apply(lambda g: pd.Series({
             "资金平均占用率": _year_occ(g),
             "总收益率": _year_return(g),
             "最大回撤": _year_max_dd(g),
         }))
         .reset_index(names="__year__")
    )

    return out


def _compute_yearly_metrics_from_trades(df_: pd.DataFrame, year_col: str) -> pd.DataFrame:
    """基于买入/卖出明细，按年估算：资金平均占用率、总收益率、最大回撤。

    说明（都是“近似”）：
    - 资金平均占用率：用“买入金额 * 持仓天数”的年内资金占用总和 / (年初资金 * 年天数)
      其中年初资金=该年所有交易的最大买入金额总和的近似（用全年累计买入金额作为保守上界）。
    - 总收益率：年内(卖出盈亏合计) / 年内(买入金额合计)
    - 最大回撤：用按年累计权益（逐年叠加年盈亏）构造年度权益曲线，计算峰值回撤（跨年尺度）。
    """

    col_buy_amt = "买入金额" if "买入金额" in df_.columns else None
    col_profit = "卖出盈亏" if "卖出盈亏" in df_.columns else None
    col_buy_date = "买入日期" if "买入日期" in df_.columns else None
    col_sell_date = "卖出日期" if "卖出日期" in df_.columns else None

    if col_buy_amt is None or col_profit is None:
        return pd.DataFrame(columns=["__year__", "资金平均占用率", "总收益率", "最大回撤"])

    work = df_.copy()
    work[col_buy_amt] = pd.to_numeric(work[col_buy_amt], errors="coerce")
    work[col_profit] = pd.to_numeric(work[col_profit], errors="coerce")

    # 以“买入日期”为准估算资金占用天数；若缺失则回退到“持仓天数”列
    if col_buy_date is not None and col_sell_date is not None:
        hold_days = _safe_days_between(work[col_buy_date], work[col_sell_date])
    else:
        hold_days = pd.Series([pd.NA] * len(work), index=work.index, dtype="float64")

    if "持仓天数" in work.columns:
        hold_days2 = pd.to_numeric(work["持仓天数"], errors="coerce")
        hold_days = hold_days.fillna(hold_days2)

    # 极端/异常修正：非正天数按 0 处理
    hold_days = hold_days.fillna(0)
    hold_days = hold_days.clip(lower=0)

    work["__hold_days__"] = hold_days

    # 年内资金占用总额（金额*天）
    work["__amt_days__"] = work[col_buy_amt].fillna(0) * work["__hold_days__"].fillna(0)

    # 按年汇总
    g = work.groupby(year_col, dropna=False, observed=False)
    yearly = g.agg(
        年买入金额=(col_buy_amt, "sum"),
        年盈亏=(col_profit, "sum"),
        年资金占用金额天=("__amt_days__", "sum"),
    ).reset_index(names="__year__")

    # 年天数（闰年处理）
    y = pd.to_numeric(yearly["__year__"], errors="coerce")
    is_leap = (y % 4 == 0) & ((y % 100 != 0) | (y % 400 == 0))
    days_in_year = is_leap.map({True: 366, False: 365}).astype("float64")

    # 资金平均占用率（%）：占用金额天 / (年买入金额 * 年天数)
    # 若年买入金额为 0，则为 NaN
    denom = yearly["年买入金额"].replace({0: pd.NA}).astype("float64") * days_in_year
    occ_pct = (yearly["年资金占用金额天"].astype("float64") / denom) * 100
    yearly["资金平均占用率"] = occ_pct

    # 总收益率（ratio）：年盈亏 / 年买入金额
    yearly["总收益率"] = yearly["年盈亏"].astype("float64") / yearly["年买入金额"].replace({0: pd.NA}).astype("float64")

    # 最大回撤（ratio）：用“逐年累计权益曲线”计算（跨年维度）
    yearly_sorted = yearly.sort_values("__year__").reset_index(drop=True)
    equity = (1 + yearly_sorted["总收益率"].fillna(0)).cumprod()
    peak = equity.cummax()
    dd = (peak - equity) / peak
    yearly_sorted["最大回撤"] = dd

    return yearly_sorted[["__year__", "资金平均占用率", "总收益率", "最大回撤"]]

def _to_year(s: pd.Series) -> pd.Series:
    """从日期列推导年份；若解析失败返回 NaN。"""
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.year


def _pick_year_col(df_: pd.DataFrame) -> str | None:
    """猜测用于“按年统计”的日期列名。"""
    candidates = [
        "买入日期",
        "信号日期",
        "日期",
        "交易日期",
        "买入时间",
        "信号时间",
        "时间",
    ]
    for c in candidates:
        if c in df_.columns:
            return c
    # 兜底：找包含“日期/时间”的列
    for c in df_.columns:
        if ("日期" in str(c)) or ("时间" in str(c)):
            return c
    return None


def _pick_year_summary_col(df_: pd.DataFrame) -> str | None:
    """猜测年度汇总表中的年份列名。"""
    for c in ["年份", "年", "year", "Year"]:
        if c in df_.columns:
            return c
    for c in df_.columns:
        if "年" in str(c).lower() or "year" in str(c).lower():
            return c
    return None


def _try_parse_percent(s: pd.Series) -> pd.Series:
    """把 '65.82%' / 65.82 / '0.6582' 等尽量转为“百分数数值”(0-100)。"""
    if s is None:
        return pd.Series(dtype="float64")
    ss = s.astype(str).str.strip()
    has_pct = ss.str.contains("%", na=False)
    v = pd.to_numeric(ss.str.replace("%", "", regex=False), errors="coerce")
    # 对非百分号且看起来是比例(<=1)的值，转成百分数
    non_pct = ~has_pct
    v.loc[non_pct & (v <= 1)] = v.loc[non_pct & (v <= 1)] * 100
    return v


def _try_parse_ratio(s: pd.Series) -> pd.Series:
    """把 '3.84%' / 0.0384 / 3.84 等尽量转为“比例数值”(0-1)。"""
    if s is None:
        return pd.Series(dtype="float64")
    ss = s.astype(str).str.strip()
    has_pct = ss.str.contains("%", na=False)
    v = pd.to_numeric(ss.str.replace("%", "", regex=False), errors="coerce")
    # 有百分号：3.84% -> 0.0384
    v.loc[has_pct] = v.loc[has_pct] / 100
    # 无百分号但看起来像“百分数”(>1)：3.84 -> 0.0384
    v.loc[~has_pct & (v > 1)] = v.loc[~has_pct & (v > 1)] / 100
    return v


# 1) 股价分段胜率（细分）
# 0-5, 5-10, 10-15, 15-20, 20-25, 25-30, 30+
# 说明：使用 right=False -> 左闭右开 [a,b)
price_bins = pd.cut(
    df["买入价格"],
    bins=[-float("inf"), 5, 10, 15, 20, 25, 30, float("inf")],
    right=False,
    labels=["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30+"],
)
price_stat = (
    df.groupby(price_bins, dropna=False, observed=False)
      .agg(
          样本数=("是否盈利", "count"),
          胜率=("是否盈利", win_rate),
          平均盈亏=("卖出盈亏", profit_mean),
          累计盈亏=("卖出盈亏", profit_sum),
      )
      .reset_index(names="股价区间")
)

# 2) 一年内位置分段胜率（每 10% 一档）
# 0-10,10-20,...,90-100,100%+
# 说明：同样用 [a,b)
pos_bins = pd.cut(
    df["前一日一年内位置(%)"],
    bins=[-float("inf"), 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, float("inf")],
    right=False,
    labels=[
        "<0",
        "0-10",
        "10-20",
        "20-30",
        "30-40",
        "40-50",
        "50-60",
        "60-70",
        "70-80",
        "80-90",
        "90-100",
        "100+",
    ],
)
pos_stat = (
    df.groupby(pos_bins, dropna=False, observed=False)
      .agg(
          样本数=("是否盈利", "count"),
          胜率=("是否盈利", win_rate),
          平均盈亏=("卖出盈亏", profit_mean),
          累计盈亏=("卖出盈亏", profit_sum),
      )
      .reset_index(names="一年内位置区间(%)")
)

# 3) 三连涨信号胜率
# 兼容空字符串
df["是否三连涨信号"] = df["是否三连涨信号"].fillna("缺失").astype(str).str.strip().replace({"": "缺失"})
sig_stat = (
    df.groupby("是否三连涨信号", dropna=False)
      .agg(
          样本数=("是否盈利", "count"),
          胜率=("是否盈利", win_rate),
          平均盈亏=("卖出盈亏", profit_mean),
          累计盈亏=("卖出盈亏", profit_sum),
      )
      .reset_index()
)

# 4) 卖出原因统计（频次 + 胜率 + 盈亏）
df["卖出原因"] = df["卖出原因"].fillna("缺失").astype(str).str.strip().replace({"": "缺失"})
sell_reason_stat = (
    df.groupby("卖出原因", dropna=False)
      .agg(
          样本数=("是否盈利", "count"),
          胜率=("是否盈利", win_rate),
          平均盈亏=("卖出盈亏", profit_mean),
          累计盈亏=("卖出盈亏", profit_sum),
      )
      .sort_values("样本数", ascending=False)
      .reset_index()
)

# 5) 评分区间统计（原始评分 / 评分，每5分一组）
# 说明：用 [a,b) 左闭右开；并额外增加 <0 与 100+ 两档
# - edges: [-inf, 0,5,...,100, +inf]
# - labels: ['<0', '0-5', ..., '95-100', '100+']
SCORE_STEP = 5
SCORE_EDGES = [-float("inf"), *list(range(0, 101, SCORE_STEP)), float("inf")]
SCORE_LABELS = ["<0"] + [f"{i}-{i+SCORE_STEP}" for i in range(0, 100, SCORE_STEP)] + ["100+"]

raw_score_bin_stat = None
if "原始评分" in df.columns:
    raw_score_bins = pd.cut(
        df["原始评分"],
        bins=SCORE_EDGES,
        right=False,
        labels=SCORE_LABELS,
    )
    raw_score_bin_stat = (
        df.groupby(raw_score_bins, dropna=False, observed=False)
          .agg(
              样本数=("是否盈利", "count"),
              胜率=("是否盈利", win_rate),
              平均盈亏=("卖出盈亏", profit_mean),
              累计盈亏=("卖出盈亏", profit_sum),
          )
          .reset_index(names="原始评分区间")
    )

score_bin_stat = None
if "评分" in df.columns:
    score_bins = pd.cut(
        df["评分"],
        bins=SCORE_EDGES,
        right=False,
        labels=SCORE_LABELS,
    )
    score_bin_stat = (
        df.groupby(score_bins, dropna=False, observed=False)
          .agg(
              样本数=("是否盈利", "count"),
              胜率=("是否盈利", win_rate),
              平均盈亏=("卖出盈亏", profit_mean),
              累计盈亏=("卖出盈亏", profit_sum),
          )
          .reset_index(names="评分区间")
    )

# 6) 信号日涨幅(%) 分段统计（按档位：<=3.99 ,4 ,5 ,6 ,7 ,8 ,9<）
# 说明：用 [a,b) 左闭右开
# - "<=3.99"：(-inf,4)（即 <4，包含 3.99 档位）
# - "4"： [4,5)
# - ...
# - "9<"： [9,10)
sig_ret_stat = None
if "信号日涨幅(%)" in df.columns:
    sig_ret_bins = pd.cut(
        df["信号日涨幅(%)"],
        bins=[-float("inf"), 4, 5, 6, 7, 8, 9, 10, float("inf")],
        right=False,
        labels=["<=3.99", "4", "5", "6", "7", "8", "9<", "10+"],
    )
    sig_ret_stat = (
        df.groupby(sig_ret_bins, dropna=False, observed=False)
          .agg(
              样本数=("是否盈利", "count"),
              胜率=("是否盈利", win_rate),
              平均盈亏=("卖出盈亏", profit_mean),
              累计盈亏=("卖出盈亏", profit_sum),
          )
          .reset_index(names="信号日涨幅档位(%)")
    )

# 7) 信号日放量倍数 分段统计（按档位：>=2,3,4,5,6,7,8,9,10<）
# 说明：用 [a,b) 左闭右开
# - "<=2"：(-inf,2)
# - ">=2"：[2,3)
# - ...
# - "10<"：[9,10)
# - "10+"：[10, +inf)
vol_mult_stat = None
if "信号日放量倍数" in df.columns:
    vol_mult_bins = pd.cut(
        df["信号日放量倍数"],
        bins=[-float("inf"), 2, 3, 4, 5, 6, 7, 8, 9, 10, float("inf")],
        right=False,
        labels=["<=2", ">=2", "3", "4", "5", "6", "7", "8", "9", "10+"],
    )
    # 注：这里的标签含义为：
    # - ">=2" 代表 [2,3)
    # - "3"   代表 [3,4)
    # - ...
    # - "9"   代表 [9,10)（即你说的 10<）
    vol_mult_stat = (
        df.groupby(vol_mult_bins, dropna=False, observed=False)
          .agg(
              样本数=("是否盈利", "count"),
              胜率=("是否盈利", win_rate),
              平均盈亏=("卖出盈亏", profit_mean),
              累计盈亏=("卖出盈亏", profit_sum),
          )
          .reset_index(names="信号日放量倍数档位")
    )

# 总体汇总（便于看整体赚多少）
overall = pd.Series({
    "样本数": df["是否盈利"].notna().sum(),
    "胜率": win_rate(df["是否盈利"]),
    "平均盈亏": profit_mean(df["卖出盈亏"]),
    "累计盈亏": profit_sum(df["卖出盈亏"]),
})

# 8) 按年统计：资金平均占用率 / 总收益率 / 最大回撤 + 平仓统计/胜率
# 说明：
# - 年份优先从“买入日期/信号日期/日期...”推导；找不到则跳过该统计。
# - 字段名做容错：优先匹配常见列名，找不到则不输出该项。

year_stat = None
_year_col = _pick_year_col(df)
if _year_col is not None:
    df["__year__"] = _to_year(df[_year_col])

    # 优先：若有权益曲线，用权益曲线计算三项指标（更准）
    _eq = _load_equity_curve()
    _computed_from_eq = _compute_yearly_metrics_from_equity_curve(_eq) if _eq is not None else None

    # 其次：若没有权益曲线，则基于交易明细估算
    _computed_year_metrics = _compute_yearly_metrics_from_trades(df, "__year__")

    # 指标列名（容错匹配）
    occ_col = None
    for c in ["资金平均占用率", "平均占用率", "资金占用率"]:
        if c in df.columns:
            occ_col = c
            break

    ret_col = None
    for c in ["总收益率", "收益率", "策略收益率"]:
        if c in df.columns:
            ret_col = c
            break

    dd_col = None
    for c in ["最大回撤", "回撤", "最大回撤率"]:
        if c in df.columns:
            dd_col = c
            break

    # 解析（明细本身有列才解析）
    if occ_col is not None:
        df["__occ_pct__"] = _try_parse_percent(df[occ_col])
    if ret_col is not None:
        df["__ret_ratio__"] = _try_parse_ratio(df[ret_col])
    if dd_col is not None:
        df["__dd_ratio__"] = _try_parse_ratio(df[dd_col])

    def _close_counts(x: pd.Series) -> pd.Series:
        s = x.fillna("缺失").astype(str).str.strip().replace({"": "缺失"})
        win_cnt = (s == "盈利").sum()
        loss_cnt = (s == "亏损").sum()
        flat_cnt = (s == "持平").sum()
        total = len(s)
        denom = win_cnt + loss_cnt
        wr = (win_cnt / denom) if denom > 0 else float("nan")
        return pd.Series({
            "盈利": win_cnt,
            "亏损": loss_cnt,
            "持平": flat_cnt,
            "平仓总笔数": total,
            "胜率(按平仓,不含持平)": wr,
        })

    aggs: dict[str, tuple[str, str]] = {}
    if occ_col is not None:
        aggs["资金平均占用率"] = ("__occ_pct__", "mean")
    if ret_col is not None:
        aggs["总收益率"] = ("__ret_ratio__", "sum")
    if dd_col is not None:
        aggs["最大回撤"] = ("__dd_ratio__", "max")

    if len(aggs) > 0:
        base = df.groupby("__year__", dropna=False, observed=False).agg(**aggs)
    else:
        base = pd.DataFrame(index=df.groupby("__year__", dropna=False, observed=False).size().index)

    # 覆盖/补齐三项指标：权益曲线 > 交易明细估算
    if _computed_from_eq is not None and len(_computed_from_eq) > 0:
        ym = _computed_from_eq.set_index("__year__")
        for c in ["资金平均占用率", "总收益率", "最大回撤"]:
            if c in ym.columns:
                base[c] = ym[c]
    elif _computed_year_metrics is not None and len(_computed_year_metrics) > 0:
        ym = _computed_year_metrics.set_index("__year__")
        for c in ["资金平均占用率", "总收益率", "最大回撤"]:
            if c in ym.columns:
                base[c] = ym[c]

    close_part = (
        df.groupby("__year__", dropna=False, observed=False)["是否盈利"]
          .apply(_close_counts)
          .unstack()
    )

    year_stat = base.join(close_part, how="outer").reset_index(names="年份")

pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 220)

print("\n=== 总体汇总 ===")
print(overall)

print("\n=== 股价分段胜率/盈亏 ===")
print(price_stat)

print("\n=== 前一日一年内位置分段胜率/盈亏 ===")
print(pos_stat)

print("\n=== 是否三连涨信号胜率/盈亏 ===")
print(sig_stat)

print("\n=== 卖出原因统计（样本数/胜率/盈亏） ===")
print(sell_reason_stat)

if raw_score_bin_stat is not None:
    print("\n=== 原始评分分段统计（每5分一组） ===")
    print(raw_score_bin_stat)

if score_bin_stat is not None:
    print("\n=== 评分分段统计（每5分一组） ===")
    print(score_bin_stat)

if sig_ret_stat is not None:
    print("\n=== 信号日涨幅(%) 分段统计（档位：<=3.99 ,4 ,5 ,6 ,7 ,8 ,9<） ===")
    print(sig_ret_stat)

if vol_mult_stat is not None:
    print("\n=== 信号日放量倍数 分段统计（档位：>=2,3,4,5,6,7,8,9,10<） ===")
    print(vol_mult_stat)

if year_stat is not None:
    # 更贴近你给的输出格式：每年一行“核心三指标 + 平仓统计 + 胜率”
    print("\n=== 按年统计 ===")

    def _fmt_pct(v):
        return ("nan" if pd.isna(v) else f"{v:.2f}%")

    def _fmt_ratio_pct(v):
        return ("nan" if pd.isna(v) else f"{v * 100:.2f}%")

    for _, r in year_stat.iterrows():
        y = r.get("年份")
        occ = r.get("资金平均占用率")
        ret = r.get("总收益率")
        dd = r.get("最大回撤")

        line1_parts = []
        if "资金平均占用率" in year_stat.columns:
            line1_parts.append(f"资金平均占用率: {_fmt_pct(occ)}")
        if "总收益率" in year_stat.columns:
            line1_parts.append(f"总收益率: {_fmt_ratio_pct(ret)}")
        if "最大回撤" in year_stat.columns:
            line1_parts.append(f"最大回撤: {_fmt_ratio_pct(dd)}")

        win_cnt = int(r.get("盈利", 0) if pd.notna(r.get("盈利", 0)) else 0)
        loss_cnt = int(r.get("亏损", 0) if pd.notna(r.get("亏损", 0)) else 0)
        flat_cnt = int(r.get("持平", 0) if pd.notna(r.get("持平", 0)) else 0)
        total_cnt = int(r.get("平仓总笔数", 0) if pd.notna(r.get("平仓总笔数", 0)) else 0)
        wr = r.get("胜率(按平仓,不含持平)")

        print(f"\n[{y}] " + " | ".join(line1_parts))
        print(f"平仓统计: 盈利 {win_cnt} | 亏损 {loss_cnt} | 持平 {flat_cnt} | 平仓总笔数 {total_cnt}")
        print(f"胜率(按平仓, 不含持平): {('nan' if pd.isna(wr) else f'{wr * 100:.2f}%')}")