import pandas as pd

path = r"e:\work\SynologyDrive\量化交易\通达信\output\backtest\20260115_1419\买入明细.csv"

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

# 5) 评分区间统计（原始评分 / 评分，每10分一组）
# 说明：用 [a,b) 左闭右开；并额外增加 <0 与 100+ 两档
# - edges: [-inf, 0,10,...,100, +inf]
# - labels: ['<0', '0-10', ..., '90-100', '100+']
SCORE_EDGES = [-float("inf"), *list(range(0, 101, 10)), float("inf")]
SCORE_LABELS = ["<0"] + [f"{i}-{i+10}" for i in range(0, 100, 10)] + ["100+"]

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

# 总体汇总（便于看整体赚多少）
overall = pd.Series({
    "样本数": df["是否盈利"].notna().sum(),
    "胜率": win_rate(df["是否盈利"]),
    "平均盈亏": profit_mean(df["卖出盈亏"]),
    "累计盈亏": profit_sum(df["卖出盈亏"]),
})

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
    print("\n=== 原始评分分段统计（每10分一组） ===")
    print(raw_score_bin_stat)

if score_bin_stat is not None:
    print("\n=== 评分分段统计（每10分一组） ===")
    print(score_bin_stat)