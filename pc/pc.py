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


def main(
    *,
    trade_date: str | None = None,
    min_count: int = 5,
    max_pages: int = 100,
    page_num: int = 10,
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

        # 进度统计
        scanned_strategies = 0  # 已扫描策略数
        matched_mwr1 = 0  # 满足 mwr1<=阈值 的策略数（不含主板过滤）

        final_rows = []

        # 分页拉取：直到满足最少输出数量，或达到最大翻页
        page = 1
        while page <= MAX_PAGES and len(final_rows) < MIN_OUTPUT_COUNT:
            lst = fetch_list(session, page=page, page_num=PAGE_NUM, sort_type="hot", keyword="")
            items = lst.get("result", {}).get("list", []) if isinstance(lst, dict) else []
            if not items:
                break

            for it in items:
                if len(final_rows) >= MIN_OUTPUT_COUNT:
                    break

                prop = it.get("property", {}) if isinstance(it, dict) else {}
                sid = prop.get("id")
                if sid is None:
                    continue

                scanned_strategies += 1

                # 简单进度：每扫描 5 个 / 每页结束都会看到一次
                if scanned_strategies % 5 == 0:
                    print(
                        f"进度: page={page}/{MAX_PAGES}, 已扫描策略={scanned_strategies}, "
                        f"mwr1<={MWR1_MAX:g} 命中={matched_mwr1}, 已收集输出={len(final_rows)}/{MIN_OUTPUT_COUNT}"
                    )

                detail = fetch_detail(session, int(sid))
                result = detail.get("result", {}) if isinstance(detail, dict) else {}
                qs = result.get("queryString", {}) if isinstance(result, dict) else {}

                query = qs.get("query")
                day_buy_stock_num = qs.get("dayBuyStockNum")

                if not query or not day_buy_stock_num:
                    continue

                historypick = fetch_historypick(session, query=query, hold_num=day_buy_stock_num, trade_date=trade_date)

                stocks = historypick.get("result", {}).get("stocks") if isinstance(historypick, dict) else None

                # 以 result.stocks 为准：有内容才输出
                if not stocks:
                    continue

                # 回测汇总结果
                backtestresult = fetch_backtestresult(session, int(sid))
                max_annual_yield = _get_max_annual_yield(backtestresult)
                max_win_rate = _get_max_win_rate(backtestresult)

                # 只显示 mwr1(回测持股周期) <= 阈值 的数据
                mwr1 = max_win_rate[1] if isinstance(max_win_rate, list) and len(max_win_rate) > 1 else None
                try:
                    mwr1_num = float(mwr1) if mwr1 is not None else None
                except Exception:
                    mwr1_num = None
                if mwr1_num is None or mwr1_num > MWR1_MAX:
                    continue

                matched_mwr1 += 1

                merged = dict(historypick)
                merged["query"] = query
                merged["property_id"] = sid
                merged["trade_date"] = trade_date
                # 单日买入数
                merged["dayBuyStockNum"] = day_buy_stock_num
                # 收益率 ≥ {upperIncome} % 时坚定持有;直到最高收益回落 ≤ {fallIncome} %
                merged["fallIncome"] = result.get("fallIncome")
                merged["upperIncome"] = result.get("upperIncome")
                # 收益率 ≤ -{lowerIncome} %
                merged["lowerIncome"] = result.get("lowerIncome")
                # 最大预期年化收益率maxAnnualYield[0], 回测持股周期 maxAnnualYield[1] 天
                merged["maxAnnualYield"] = max_annual_yield
                # 最大胜率 maxWinRate[0], 回测持股周期 maxWinRate[1] 天
                merged["maxWinRate"] = max_win_rate

                final_rows.append(merged)

            # 每页结束打印一次更直观
            print(
                f"进度: page={page}/{MAX_PAGES} 完成, 本页策略={len(items)}, 已扫描策略={scanned_strategies}, "
                f"mwr1<={MWR1_MAX:g} 命中={matched_mwr1}, 已收集输出={len(final_rows)}/{MIN_OUTPUT_COUNT}"
            )

            page += 1

        # 结束汇总
        print(
            f"汇总: 已扫描策略={scanned_strategies}, mwr1<={MWR1_MAX:g} 命中={matched_mwr1}, "
            f"最终输出={len(final_rows)} (目标 min_count={MIN_OUTPUT_COUNT})"
        )

        # =============================
        # 输出：控制台 + CSV
        # =============================
        import csv
        import os

        # 先做最终输出过滤（沪深主板），并整理为可写入 CSV 的结构
        csv_rows: list[dict[str, Any]] = []

        for r in final_rows:
            q = r.get("query")
            pid = r.get("property_id")
            day_buy = r.get("dayBuyStockNum")
            upper = r.get("upperIncome")
            fall = r.get("fallIncome")
            lower = r.get("lowerIncome")
            may = r.get("maxAnnualYield")
            mwr = r.get("maxWinRate")

            stocks = r.get("result", {}).get("stocks") if isinstance(r, dict) else None
            stock_map: dict[str, str] = {}
            if isinstance(stocks, list):
                for s in stocks:
                    if not isinstance(s, dict):
                        continue
                    code = s.get("stock_code") or s.get("stockCode") or s.get("code")
                    name = s.get("stock_name") or s.get("stockName") or s.get("name")
                    if not (code and name):
                        continue

                    code_str = str(code)
                    if code_str.startswith("60") or code_str.startswith("00"):
                        stock_map[code_str] = str(name)

            if not stock_map:
                continue

            may0 = may[0] if isinstance(may, list) and len(may) > 0 else None
            may1 = may[1] if isinstance(may, list) and len(may) > 1 else None
            mwr0 = mwr[0] if isinstance(mwr, list) and len(mwr) > 0 else None
            mwr1 = mwr[1] if isinstance(mwr, list) and len(mwr) > 1 else None

            # 控制台输出
            print(f"策略：{q}")
            print(f"策略ID：{pid}")
            print(f"策略地址：https://backtest.10jqka.com.cn/backtest/app.html#/strategysquare/{pid}")
            print(f"交易日: {trade_date}")
            print(f"单日买入数: {day_buy}")
            print(f"止盈: 收益率 ≥ {upper} % 时坚定持有;直到最高收益回落 ≤ {fall} %")
            print(f"止损: 收益率 ≤ -{lower} %")
            print(f"最大预期: 年化收益率{may0}, 回测持股周期 {may1} 天")
            print(f"最大胜率: 胜率 {mwr0}, 回测持股周期 {mwr1} 天")
            print(f"入选股票：{stock_map}")
            print("-" * 60)

            base_row: dict[str, Any] = {
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
        else:
            print("没有可输出的结果（主板过滤后为空），未生成 CSV")


if __name__ == "__main__":
    main()
