import json
import random
import time
from typing import Any, Dict
from urllib.parse import quote

import requests


def _build_headers() -> Dict[str, str]:
    # 尽量模拟常见浏览器请求头（不保证能绕过风控，仅用于降低“过于像脚本”的特征）
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": "https://backtest.10jqka.com.cn/",
        "Origin": "https://backtest.10jqka.com.cn",
    }


def _human_sleep(min_s: float = 0.8, max_s: float = 2.2) -> None:
    # 加一点“人工抖动”
    time.sleep(random.uniform(min_s, max_s))


def fetch_json(session: requests.Session, url: str, *, retries: int = 3) -> Any:
    # 简单重试 + 退避（遇到限流/临时网络抖动时更稳）
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        _human_sleep(0.6, 1.8)
        try:
            resp = session.get(url, timeout=30)

            # 常见限流：429 Too Many Requests；以及 5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = min(10.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.6))
                time.sleep(backoff)
                continue

            resp.raise_for_status()

            try:
                return resp.json()
            except ValueError:
                return json.loads(resp.text)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            backoff = min(10.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.6))
            time.sleep(backoff)

    raise last_exc if last_exc else RuntimeError("Request failed")


def fetch_list(session: requests.Session, page: int = 1, page_num: int = 10, sort_type: str = "hot", keyword: str = "") -> Any:
    url = (
        "https://backtest.10jqka.com.cn/strategysquare/list"
        f"?order=desc&page={page}&pageNum={page_num}&sortType={sort_type}&keyword={keyword}"
    )
    return fetch_json(session, url)


def fetch_detail(session: requests.Session, strategy_id: int) -> Any:
    url = f"https://backtest.10jqka.com.cn/strategysquare/detail?strategyId={strategy_id}"
    return fetch_json(session, url)


def fetch_historypick(session: requests.Session, query: str, hold_num: str | int, trade_date: str) -> Any:
    q = quote(str(query), safe="")
    url = (
        "https://backtest.10jqka.com.cn/tradebacktest/historypick"
        f"?query={q}&hold_num={hold_num}&trade_date={trade_date}"
    )
    return fetch_json(session, url)


def fetch_backtestresult(session: requests.Session, strategy_id: int) -> Any:
    url = f"https://backtest.10jqka.com.cn/strategysquare/backtestresult?strategyId={strategy_id}"
    return fetch_json(session, url)


def _get_report_data(backtestresult: Any) -> Dict[str, Any]:
    if not isinstance(backtestresult, dict):
        return {}
    return (
        backtestresult.get("result", {})
        .get("statistics", {})
        .get("reportData", {})
    )


def _get_max_annual_yield(backtestresult: Any) -> Any:
    return _get_report_data(backtestresult).get("maxAnnualYield")


def _get_max_win_rate(backtestresult: Any) -> Any:
    return _get_report_data(backtestresult).get("maxWinRate")


def _extract_mainboard_stock_map(stocks: Any) -> dict[str, str]:
    stock_map: dict[str, str] = {}
    if not isinstance(stocks, list):
        return stock_map

    for s in stocks:
        if not isinstance(s, dict):
            continue
        code = s.get("stock_code") or s.get("stockCode") or s.get("code")
        name = s.get("stock_name") or s.get("stockName") or s.get("name")
        if not (code and name):
            continue

        code_str = str(code)
        # 沪深主板：60xxxx（沪A）/00xxxx（深A）
        if code_str.startswith("60") or code_str.startswith("00"):
            stock_map[code_str] = str(name)

    return stock_map


def _parse_win_rate_ratio(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        num = float(v)
        # 兼容百分数（80 表示 80%）
        if num > 1.0:
            return num / 100.0
        return num
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        if s.endswith("%"):
            try:
                return float(s[:-1].strip()) / 100.0
            except Exception:
                return None
        try:
            num = float(s)
        except Exception:
            return None
        if num > 1.0:
            return num / 100.0
        return num
    return None


def _collect_rows_for_sort(
    session: requests.Session,
    *,
    sort_type: str,
    keyword: str,
    trade_date: str,
    min_output_count: int,
    max_pages: int,
    page_num: int,
    mwr0_min: float,
    mwr1_max: float,
) -> list[dict[str, Any]]:
    scanned_strategies = 0
    matched_filters = 0
    filtered_out_mainboard = 0
    filtered_out_mwr0 = 0

    final_rows: list[dict[str, Any]] = []

    page = 1
    while page <= max_pages and len(final_rows) < min_output_count:
        lst = fetch_list(session, page=page, page_num=page_num, sort_type=sort_type, keyword=keyword)
        items = lst.get("result", {}).get("list", []) if isinstance(lst, dict) else []
        if not items:
            break

        for it in items:
            if len(final_rows) >= min_output_count:
                break

            prop = it.get("property", {}) if isinstance(it, dict) else {}
            sid = prop.get("id")
            if sid is None:
                continue

            scanned_strategies += 1

            if scanned_strategies % 5 == 0:
                print(
                    f"进度[{sort_type}]: page={page}/{max_pages}, 已扫描策略={scanned_strategies}, "
                    f"mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
                    f"已收集输出={len(final_rows)}/{min_output_count}"
                )

            detail = fetch_detail(session, int(sid))
            result = detail.get("result", {}) if isinstance(detail, dict) else {}
            qs = result.get("queryString", {}) if isinstance(result, dict) else {}

            query = qs.get("query")
            day_buy_stock_num = qs.get("dayBuyStockNum")
            if not query or not day_buy_stock_num:
                continue

            historypick = fetch_historypick(session, query=query, hold_num=day_buy_stock_num, trade_date=trade_date)

            stocks = None
            if isinstance(historypick, dict):
                hp_result = historypick.get("result")
                if isinstance(hp_result, dict):
                    stocks = hp_result.get("stocks")
                else:
                    preview = str(hp_result)
                    if len(preview) > 120:
                        preview = preview[:120] + "..."
                    print(
                        f"跳过[{sort_type}]：historypick.result 非 dict，sid={sid}，type={type(hp_result).__name__}，value={preview}"
                    )

            if not stocks:
                continue

            stock_map = _extract_mainboard_stock_map(stocks)
            if not stock_map:
                filtered_out_mainboard += 1
                continue

            backtestresult = fetch_backtestresult(session, int(sid))
            max_annual_yield = _get_max_annual_yield(backtestresult)
            max_win_rate = _get_max_win_rate(backtestresult)

            mwr0 = max_win_rate[0] if isinstance(max_win_rate, list) and len(max_win_rate) > 0 else None
            mwr0_num = _parse_win_rate_ratio(mwr0)
            if mwr0_num is None or mwr0_num < mwr0_min:
                filtered_out_mwr0 += 1
                continue

            mwr1 = max_win_rate[1] if isinstance(max_win_rate, list) and len(max_win_rate) > 1 else None
            try:
                mwr1_num = float(mwr1) if mwr1 is not None else None
            except Exception:
                mwr1_num = None
            if mwr1_num is None or mwr1_num > mwr1_max:
                continue

            matched_filters += 1

            merged = dict(historypick)
            merged["sort_type"] = sort_type
            merged["query"] = query
            merged["property_id"] = sid
            merged["trade_date"] = trade_date
            merged["dayBuyStockNum"] = day_buy_stock_num
            merged["fallIncome"] = result.get("fallIncome")
            merged["upperIncome"] = result.get("upperIncome")
            merged["lowerIncome"] = result.get("lowerIncome")
            merged["maxAnnualYield"] = max_annual_yield
            merged["maxWinRate"] = max_win_rate
            merged["mainboardStockMap"] = stock_map

            final_rows.append(merged)

        print(
            f"进度[{sort_type}]: page={page}/{max_pages} 完成, 本页策略={len(items)}, 已扫描策略={scanned_strategies}, "
            f"mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
            f"已收集输出={len(final_rows)}/{min_output_count}"
        )

        page += 1

    print(
        f"汇总[{sort_type}]: 已扫描策略={scanned_strategies}, mwr0>={mwr0_min:g} 且 mwr1<={mwr1_max:g} 命中={matched_filters}, "
        f"胜率过滤剔除={filtered_out_mwr0}, 主板过滤剔除={filtered_out_mainboard}, "
        f"最终输出={len(final_rows)} (目标 min_count={min_output_count})"
    )

    if len(final_rows) < min_output_count:
        print(f"提示[{sort_type}]: 未达到 min_count，已收集 {len(final_rows)}/{min_output_count}")

    return final_rows


def main(
    *,
    # trade_date: str | None = None,
    trade_date: str | None = "2026-01-23",
    min_count: int = 5,
    max_pages: int = 100,
    page_num: int = 10,
    # gain(本月), winRate(成功率), profit(年化收益), hot(热度)
    # sort_types: list[str] | None = None,
    sort_types: list[str] | None = ["winRate"],
    keyword: str = "",
) -> None:
    # =============================
    # 可配置参数（集中放这里，方便修改）
    # =============================
    # 交易日（None 表示当天）
    if trade_date is None:
        trade_date = time.strftime("%Y-%m-%d")

    # 需要最少输出多少条“符合要求”的策略
    MIN_OUTPUT_COUNT = int(min_count)

    # 最大翻页数
    MAX_PAGES = int(max_pages)

    # 每页拉取策略数量
    PAGE_NUM = int(page_num)

    # 筛选阈值：只保留 maxWinRate 的持股周期（mwr1）<= 该值
    MWR1_MAX = 5.0

    # 筛选阈值：只保留最大胜率（mwr0）>= 该值
    MWR0_MIN = 0.8

    # 输出 CSV：按当前日期时间命名
    OUTPUT_CSV_FILENAME = time.strftime("%Y%m%d_%H%M%S") + ".csv"

    # 是否将“入选股票”展开为单独列（否则为一个 JSON 字符串列）
    EXPAND_STOCKS = False

    # =============================

    with requests.Session() as session:
        session.headers.update(_build_headers())

        # 访问网页首页一次，让 Cookie/会话更像“正常用户”
        try:
            session.get("https://backtest.10jqka.com.cn/", timeout=15)
        except Exception:
            pass
        _human_sleep(1.0, 2.8)

        effective_sort_types = sort_types or ["gain", "winRate", "profit", "hot"]

        final_rows: list[dict[str, Any]] = []
        for st in effective_sort_types:
            rows = _collect_rows_for_sort(
                session,
                sort_type=st,
                keyword=keyword,
                trade_date=trade_date,
                min_output_count=MIN_OUTPUT_COUNT,
                max_pages=MAX_PAGES,
                page_num=PAGE_NUM,
                mwr0_min=MWR0_MIN,
                mwr1_max=MWR1_MAX,
            )
            final_rows.extend(rows)

        # =============================
        # 输出：控制台 + CSV
        # =============================
        import csv
        import os

        # 整理为可写入 CSV 的结构（final_rows 已是最终口径）
        csv_rows: list[dict[str, Any]] = []

        for r in final_rows:
            st = r.get("sort_type")
            q = r.get("query")
            pid = r.get("property_id")
            day_buy = r.get("dayBuyStockNum")
            upper = r.get("upperIncome")
            fall = r.get("fallIncome")
            lower = r.get("lowerIncome")
            may = r.get("maxAnnualYield")
            mwr = r.get("maxWinRate")

            stock_map = r.get("mainboardStockMap") if isinstance(r, dict) else None
            if not isinstance(stock_map, dict) or not stock_map:
                # 理论上不会发生（已前置过滤），兜底避免 KeyError
                continue

            may0 = may[0] if isinstance(may, list) and len(may) > 0 else None
            may1 = may[1] if isinstance(may, list) and len(may) > 1 else None
            mwr0 = mwr[0] if isinstance(mwr, list) and len(mwr) > 0 else None
            mwr1 = mwr[1] if isinstance(mwr, list) and len(mwr) > 1 else None

            # 控制台输出
            print(f"策略：{q}")
            print(f"策略ID：{pid}")
            print(f"策略地址：https://backtest.10jqka.com.cn/backtest/app.html#/strategysquare/{pid}")
            print(f"排序方式: {st}")
            print(f"交易日: {trade_date}")
            print(f"单日买入数: {day_buy}")
            print(f"止盈: 收益率 ≥ {upper} % 时坚定持有;直到最高收益回落 ≤ {fall} %")
            print(f"止损: 收益率 ≤ -{lower} %")
            print(f"最大预期: 年化收益率{may0}, 回测持股周期 {may1} 天")
            print(f"最大胜率: 胜率 {mwr0}, 回测持股周期 {mwr1} 天")
            print(f"入选股票：{stock_map}")
            print("-" * 60)

            base_row: dict[str, Any] = {
                "sort_type": st,
                "trade_date": trade_date,
                "query": q,
                "property_id": pid,
                "url": f"https://backtest.10jqka.com.cn/backtest/app.html#/strategysquare/{pid}",
                "dayBuyStockNum": day_buy,
                "upperIncome": upper,
                "fallIncome": fall,
                "lowerIncome": lower,
                "maxAnnualYield0": may0,
                "maxAnnualYield1": may1,
                "maxWinRate0": mwr0,
                "maxWinRate1": mwr1,
            }

            if EXPAND_STOCKS:
                # 展开股票为多列（stock_1_code, stock_1_name ...）
                for i, (code, name) in enumerate(stock_map.items(), start=1):
                    base_row[f"stock_{i}_code"] = code
                    base_row[f"stock_{i}_name"] = name
            else:
                base_row["stocks"] = json.dumps(stock_map, ensure_ascii=False)

            csv_rows.append(base_row)

        # 写 CSV
        if csv_rows:
            out_path = os.path.join(os.getcwd(), OUTPUT_CSV_FILENAME)
            fieldnames = list(csv_rows[0].keys())
            with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(csv_rows)
            print(f"CSV 已输出: {out_path}")

            # 统计：每只股票出现次数（按 sort_type 去重：同一 sort_type 内只算一次）
            from collections import defaultdict

            stock_to_sort_types: dict[tuple[str, str], set[str]] = defaultdict(set)
            for row in csv_rows:
                st = str(row.get("sort_type") or "")
                stocks_json = row.get("stocks")
                if isinstance(stocks_json, str) and stocks_json.strip():
                    try:
                        sm = json.loads(stocks_json)
                    except Exception:
                        sm = None
                    if isinstance(sm, dict):
                        for code, name in sm.items():
                            if not code:
                                continue
                            stock_to_sort_types[(str(code), str(name))].add(st)
                    continue

                # 兼容 EXPAND_STOCKS=True 的情况
                for k, v in row.items():
                    if not (isinstance(k, str) and k.endswith("_code")):
                        continue
                    code = str(v) if v is not None else ""
                    if not code:
                        continue
                    name = str(row.get(k.replace("_code", "_name")) or "")
                    stock_to_sort_types[(code, name)].add(st)

            stock_count_rows: list[dict[str, Any]] = []
            for (code, name), sts in stock_to_sort_types.items():
                sts_sorted = sorted(s for s in sts if s)
                stock_count_rows.append(
                    {
                        "stock_code": code,
                        "stock_name": name,
                        "count": len(sts_sorted),
                        "sort_types": ",".join(sts_sorted),
                    }
                )

            stock_count_rows.sort(key=lambda r: (-int(r.get("count") or 0), str(r.get("stock_code") or "")))

            counts_filename = OUTPUT_CSV_FILENAME
            if counts_filename.lower().endswith(".csv"):
                counts_filename = counts_filename[:-4] + "_stock_counts.csv"
            else:
                counts_filename = counts_filename + "_stock_counts.csv"
            counts_path = os.path.join(os.getcwd(), counts_filename)

            if stock_count_rows:
                with open(counts_path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=["stock_code", "stock_name", "count", "sort_types"])
                    w.writeheader()
                    w.writerows(stock_count_rows)
                print(f"股票出现次数统计已输出: {counts_path}")
            else:
                print("股票出现次数统计为空：没有解析到 stocks")
        else:
            print("没有可输出的结果（主板过滤后为空），未生成 CSV")


if __name__ == "__main__":
    main()
