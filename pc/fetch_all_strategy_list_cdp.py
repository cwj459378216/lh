#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""同花顺回测广场（浏览器模式）：批量 yieldbacktest，只输出命中记录 CSV。

为什么需要浏览器模式：
- /tradebacktest/yieldbacktest 经常在网关层返回 "Nginx forbidden" 403。
- 站点可能要求动态请求头/指纹（例如 hexin-v）且会随时间轮换。

工作方式：
1) 用 Playwright 启动 Chromium（非 headless），你手动微信登录。
2) 你在页面里随便触发一次回测，让浏览器发出真实的 yieldbacktest XHR。
   脚本会捕获该请求的 headers（含动态字段），并用同一浏览器会话的 cookies
   通过 APIRequestContext 批量调用 list/detail/yieldbacktest。
3) 若跑批过程中再次 403，脚本会暂停并提示你再触发一次回测以刷新 headers，然后继续。

输出：仅写命中记录 CSV（每个策略只保留 winRate 最高且 >55% 的一条）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

from playwright.sync_api import Playwright, Request, Response, sync_playwright


DEFAULT_PAGE_NUM = 10000
DEFAULT_ORDER = "desc"
MIN_WIN_RATE = 0.55
HIT_WIN_RATE_PCT = 55.0
DEFAULT_RETRIES = 3

DEFAULT_SLEEP_MIN_S = 0.2
DEFAULT_SLEEP_MAX_S = 0.5

LOGIN_COOKIE_NAMES = ("u_ukey", "sess_tk", "u_did")

SECRETS_TOML = Path(__file__).with_name("ths_secrets.toml")
SECRETS_JSON = Path(__file__).with_name("ths_secrets.json")
PROFILE_DIR = Path(__file__).with_name(".pw-profile")


def _truthy(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "on")
    return False


def _load_local_secrets() -> dict[str, str]:
    secrets: dict[str, str] = {}
    allowed_keys = {
        "user_agent",
        "browser_channel",
        "browser_devtools",
        "browser_slow_mo_ms",
        "login_cookie_timeout_s",
        "test_one",
        "print_payload",
        "yieldbacktest_sleep_min_s",
        "yieldbacktest_sleep_max_s",
        "max_consecutive_403",
        "debug_cookie_snapshot",
        "debug_cookie_snapshot_full",
    }

    if SECRETS_TOML.exists():
        try:
            import tomllib

            with open(SECRETS_TOML, "rb") as f:
                obj = tomllib.load(f)
            if isinstance(obj, dict):
                for k in allowed_keys:
                    if k not in obj:
                        continue
                    v = obj.get(k)
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        s = str(v).strip()
                        if s:
                            secrets[k] = s
        except Exception as e:
            print(f"提示：读取 {SECRETS_TOML.name} 失败 ({type(e).__name__}: {e})")

    if not secrets and SECRETS_JSON.exists():
        try:
            with open(SECRETS_JSON, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                for k in allowed_keys:
                    if k not in obj:
                        continue
                    v = obj.get(k)
                    if v is None:
                        continue
                    if isinstance(v, (str, int, float, bool)):
                        s = str(v).strip()
                        if s:
                            secrets[k] = s
        except Exception as e:
            print(f"提示：读取 {SECRETS_JSON.name} 失败 ({type(e).__name__}: {e})")

    return secrets


def _mask_secret(s: str, *, keep: int = 6) -> str:
    s = s or ""
    if len(s) <= keep * 2:
        return "***"
    return f"{s[:keep]}***{s[-keep:]}"


def _as_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(str(v).strip())
    except Exception:
        return None


def _as_int(v: Any) -> int | None:
    try:
        if v is None:
            return None
        return int(float(str(v).strip()))
    except Exception:
        return None


def _human_sleep(min_s: float = DEFAULT_SLEEP_MIN_S, max_s: float = DEFAULT_SLEEP_MAX_S) -> None:
    time.sleep(random.uniform(min_s, max_s))


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_out_dir(out_dir: str | None) -> Path:
    base = Path(out_dir) if out_dir else (Path.cwd() / "csv")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _parse_win_rate_ratio(v: Any) -> float | None:
    if v is None:
        return None

    if isinstance(v, (int, float)):
        num = float(v)
        return num / 100.0 if num > 1.0 else num

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
        return num / 100.0 if num > 1.0 else num

    return None


def _compute_end_date_yesterday() -> date:
    return date.today().fromordinal(date.today().toordinal() - 1)


def _compute_start_date_5y_ago(today_: date) -> date:
    y = today_.year - 5
    m = today_.month
    d = today_.day
    while True:
        try:
            return date(y, m, d)
        except ValueError:
            d -= 1
            if d <= 0:
                return date(y, m, 1)


def _as_str(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = max(1, int(total))
        self.last_print_ts = 0.0
        self.enabled = sys.stdout.isatty()

    def update(self, current: int, *, hits: int) -> None:
        if not self.enabled:
            return
        now = time.time()
        if current != self.total and (now - self.last_print_ts) < 0.15:
            return
        pct = min(100.0, max(0.0, current * 100.0 / self.total))
        msg = f"\r进度 {current}/{self.total} ({pct:5.1f}%) 命中={hits}"
        sys.stdout.write(msg)
        sys.stdout.flush()
        self.last_print_ts = now

    def done(self) -> None:
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _write_hits_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["strategy_id", "createTime", "daySaleStrategy", "winRate"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _sanitize_headers(h: dict[str, str]) -> dict[str, str]:
    # Playwright/APIRequest 会自己处理 Host/Content-Length/Encoding 等。
    drop = {
        "host",
        "content-length",
        "accept-encoding",
        "connection",
    }
    out: dict[str, str] = {}
    for k, v in (h or {}).items():
        lk = k.lower()
        if lk in drop:
            continue
        out[k] = v
    return out


def _cookie_fingerprint(cookies: list[dict[str, Any]]) -> str:
    items = []
    for c in cookies:
        name = str(c.get("name") or "")
        domain = str(c.get("domain") or "")
        path = str(c.get("path") or "")
        value = str(c.get("value") or "")
        if not name:
            continue
        items.append((name, domain, path, value))
    items.sort(key=lambda x: (x[0], x[1], x[2]))
    src = "\n".join([f"{n}\t{d}\t{p}\t{v}" for (n, d, p, v) in items])
    return hashlib.sha256(src.encode("utf-8", errors="ignore")).hexdigest()[:12]


def _dump_cookie_snapshot(cookies: list[dict[str, Any]], *, stage: str, show_full: bool, limit: int = 10) -> None:
    fp = _cookie_fingerprint(cookies)
    v_values = [str(c.get("value") or "").strip() for c in cookies if c.get("name") == "v" and str(c.get("value") or "").strip()]
    v_show = "(empty)"
    if v_values:
        v0 = v_values[0]
        v_show = v0 if show_full else _mask_secret(v0)
        if len(v_values) > 1:
            v_show += f" (+{len(v_values) - 1})"

    print(f"cookie_snapshot[{stage}]: count={len(cookies)} v={v_show} fp={fp}")
    shown = cookies[: max(0, int(limit))]
    parts: list[str] = []
    for c in shown:
        name = str(c.get("name") or "")
        domain = str(c.get("domain") or "")
        path = str(c.get("path") or "")
        value = str(c.get("value") or "").strip()
        if not name:
            continue
        parts.append(f"{name}@{domain}{path}={value if show_full else _mask_secret(value)}")
    if parts:
        extra = len(cookies) - len(shown)
        msg = "cookie_snapshot_entries=" + "; ".join(parts)
        if extra > 0:
            msg += f" ; ...(+{extra})"
        print(msg)


def _has_login_cookies(cookies: list[dict[str, Any]]) -> bool:
    names = {str(c.get("name") or "") for c in cookies}
    return any(n in names for n in LOGIN_COOKIE_NAMES)


def _wait_for_login_cookies(context, *, timeout_s: float = 180.0) -> list[dict[str, Any]] | None:
    print("等待登录 cookie 生效（u_ukey/sess_tk/u_did 任一即可）…")
    start = time.time()
    while True:
        cookies = context.cookies(["https://backtest.10jqka.com.cn", "https://upass.10jqka.com.cn"])
        if _has_login_cookies(cookies):
            print("已检测到登录 cookie。")
            return cookies
        if (time.time() - start) > timeout_s:
            return None
        time.sleep(1.0)


class _HeaderCapture:
    def __init__(self) -> None:
        self.latest: dict[str, str] | None = None
        self.latest_ts = 0.0

    def handler(self, req: Request) -> None:
        # 允许捕获 yieldbacktest (POST) 或 strategysquare/list (GET) 的请求头
        url = req.url
        is_yield = "/tradebacktest/yieldbacktest" in url and req.method.upper() == "POST"
        is_list = "/strategysquare/list" in url
        
        if not (is_yield or is_list):
            return
            
        print(f"[调试] 捕获到目标请求: {url}")
        h = dict(req.headers)
        self.latest = _sanitize_headers(h)
        
        # 如果是从 list 接口 (GET) 捕获的，可能缺 Content-Type，手动补齐以防复用于 POST 时出错
        if is_list:
            self.latest["content-type"] = "application/json"
            self.latest["Content-Type"] = "application/json"

        self.latest_ts = time.time()
        hxv = (self.latest.get("hexin-v") or self.latest.get("Hexin-V") or "").strip()
        if hxv:
            print(f"已捕获请求头 (source={'yield' if is_yield else 'list'})：hexin-v={_mask_secret(hxv)}")
        else:
            print(f"已捕获请求头 (source={'yield' if is_yield else 'list'})，但未看到 hexin-v 字段")


def _wait_for_manual_login(page, *, timeout_s: int = 600) -> None:
    print("浏览器已打开：请完成微信登录。登录完成后回到终端按回车继续…")
    try:
        _ = input()
        return
    except Exception:
        # 某些环境不允许 input；退化为超时等待
        start = time.time()
        while (time.time() - start) < timeout_s:
            time.sleep(0.25)
        return


def _wait_for_header_capture(capture: _HeaderCapture, *, timeout_s: int = 300) -> dict[str, str]:
    print("请在浏览器页面里随便触发一次回测（让页面发送 yieldbacktest 请求），脚本将自动捕获动态请求头…")
    start = time.time()
    while True:
        if capture.latest:
            return capture.latest
        if (time.time() - start) > timeout_s:
            raise TimeoutError("等待捕获 yieldbacktest 请求头超时。请确认已在页面触发回测。")
        time.sleep(0.25)


def _api_get_json(ctx, url: str, *, headers: dict[str, str] | None = None, retries: int = 3, timeout_s: float = 60000.0) -> Any:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        _human_sleep(0.4, 1.2)
        try:
            # timeout 默认是毫秒，这里我们参数名虽然叫 timeout_s 默认给了 30.0 (秒)，但传递给 ctx.get 时
            # playwright expects millis if explicitly passed, or 0 for no timeout.
            # 之前的 timeout_s=30.0 被当作 30ms 导致了超时错误 "Timeout 30ms exceeded"
            
            # 修复：确保传入的是毫秒
            # 实际上 APIRequestContext.get 的 timeout 参数如果传入 float，会被认为是毫秒
            # 30.0 ms 显然太短。改为传入 30000 ms (30s)
            
            resp = ctx.get(url, headers=headers, timeout=timeout_s * 1000 if timeout_s < 1000 else timeout_s)
            
            if resp.status in (429, 500, 502, 503, 504):
                time.sleep(min(12.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.8)))
                continue
            if not resp.ok:
                raise RuntimeError(f"GET {url} failed: status={resp.status} body={resp.text()[:200]}")
            return resp.json()
        except Exception as e:
            last = e
            time.sleep(min(12.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.8)))
    raise last if last else RuntimeError("Request failed")


def _api_post_json(ctx, url: str, *, payload: dict[str, Any], headers: dict[str, str] | None, retries: int = 3, timeout_s: float = 60000.0) -> Any:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        _human_sleep(0.4, 1.2)
        try:
            # 同样修复 timeout 单位问题
            resp = ctx.post(url, data=json.dumps(payload), headers=headers, timeout=timeout_s * 1000 if timeout_s < 1000 else timeout_s)
            
            if resp.status in (429, 500, 502, 503, 504):
                time.sleep(min(12.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.8)))
                continue
            if resp.status == 403:
                snippet = (resp.text() or "").strip().replace("\n", " ")[:500]
                raise PermissionError(f"403 Forbidden for url: {url} ; body_snippet={snippet}")
            if not resp.ok:
                raise RuntimeError(f"POST {url} failed: status={resp.status} body={resp.text()[:200]}")
            return resp.json()
        except Exception as e:
            last = e
            time.sleep(min(12.0, 0.8 * (2 ** (attempt - 1)) + random.uniform(0, 0.8)))
    raise last if last else RuntimeError("Request failed")


def _extract_items(list_json: Any) -> list[dict[str, Any]]:
    if not isinstance(list_json, dict):
        return []
    result = list_json.get("result")
    if not isinstance(result, dict):
        return []
    items = result.get("list")
    return items if isinstance(items, list) else []


def _get_property_id(item: Any) -> int | None:
    if not isinstance(item, dict):
        return None
    prop = item.get("property")
    if not isinstance(prop, dict):
        return None
    sid = prop.get("id")
    try:
        return int(sid)
    except Exception:
        return None


def _pick_first(d: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _extract_strategy_params_for_yieldbacktest(detail_json: Any) -> dict[str, Any] | None:
    if not isinstance(detail_json, dict):
        return None
    result = detail_json.get("result")
    if not isinstance(result, dict):
        return None

    qs = result.get("queryString")
    if not isinstance(qs, dict):
        qs = {}

    query = _as_str(qs.get("query"))
    if not query:
        return None

    day_buy_stock_num = _pick_first(qs, ["dayBuyStockNum", "day_buy_stock_num"])
    day_buy_stock_num_s = _as_str(day_buy_stock_num)

    period = _pick_first(qs, ["daysForSaleStrategy", "period", "periodStr", "periods"]) if isinstance(qs, dict) else None
    if period is None:
        period = _pick_first(result, ["daysForSaleStrategy", "period", "periodStr", "periods"]) if isinstance(result, dict) else None
    period_s = _as_str(period)

    stock_hold_count = _pick_first(qs, ["stockHoldCount", "stock_hold_count", "stockHold", "stock_hold"]) if isinstance(qs, dict) else None
    if stock_hold_count is None:
        stock_hold_count = _pick_first(result, ["stockHoldCount", "stock_hold_count", "stockHold", "stock_hold"]) if isinstance(result, dict) else None
    stock_hold_s = _as_str(stock_hold_count)

    upper_income_s = _as_str(_pick_first(result, ["upperIncome", "upper_income"]))
    lower_income_s = _as_str(_pick_first(result, ["lowerIncome", "lower_income"]))
    fall_income_s = _as_str(_pick_first(result, ["fallIncome", "fall_income"]))

    return {
        "query": query,
        "day_buy_stock_num": day_buy_stock_num_s,
        "period": period_s,
        "stock_hold": stock_hold_s,
        "upper_income": upper_income_s,
        "lower_income": lower_income_s,
        "fall_income": fall_income_s,
    }


def _fetch_list(ctx, *, page: int, page_num: int, sort_type: str, keyword: str, order: str) -> Any:
    url = (
        "https://backtest.10jqka.com.cn/strategysquare/list"
        f"?order={order}&page={page}&pageNum={page_num}&sortType={sort_type}&keyword={keyword}"
    )
    # 注意：这里我们不再传递错误的超时值，使用函数默认修正后的值
    return _api_get_json(ctx, url, retries=DEFAULT_RETRIES)


def _fetch_detail(ctx, strategy_id: int) -> Any:
    url = f"https://backtest.10jqka.com.cn/strategysquare/detail?strategyId={int(strategy_id)}"
    return _api_get_json(ctx, url, retries=DEFAULT_RETRIES)


def _yield_backtest(ctx, payload: dict[str, Any], *, headers: dict[str, str], retries: int = 3) -> Any:
    url = "https://backtest.10jqka.com.cn/tradebacktest/yieldbacktest"
    return _api_post_json(ctx, url, payload=payload, headers=headers, retries=retries)


def _open_browser(
    playwright: Playwright,
    *,
    user_agent: str | None,
    channel: str | None,
    devtools: bool,
    slow_mo_ms: int | None,
) -> tuple[Any, Any, Any]:
    print("正在尝试连接已打开的 Chrome (CDP port 9222)...")
    print("如果连接失败，请在终端运行：")
    print('/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222 --user-data-dir="/tmp/chrome_dev_test"')
    
    try:
        # 尝试连接本地 9222 端口
        browser = playwright.chromium.connect_over_cdp("http://localhost:9222")
        context = browser.contexts[0]
        # 获取当前活跃页面或新建
        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()
        
        # 获取浏览器当前的 UA
        ua = page.evaluate("navigator.userAgent")
        print("已成功连接到 Chrome！")
        return context, page, ua
    except Exception as e:
        print(f"连接 Chrome 失败: {e}")
        print("请确保已完全关闭 Chrome 并通过命令行带参数启动。")
        raise e


def _attach_login_diagnostics(page) -> None:
    def on_request_failed(req: Request) -> None:
        try:
            f = req.failure
            err = f.error_text if f else "(unknown)"
            url = req.url
            if any(x in url for x in ("login", "weixin", "wx", "upass", "passport")):
                print(f"requestfailed: {req.method} {url} err={err}")
        except Exception:
            return

    def on_response(resp: Response) -> None:
        try:
            url = resp.url
            if any(x in url for x in ("login", "weixin", "wx", "upass", "passport")):
                print(f"response: {resp.status} {url}")
        except Exception:
            return

    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)


def main() -> int:
    ap = argparse.ArgumentParser(description="同花顺回测广场（CDP 接管模式）")
    ap.add_argument("--sort-type", default="createTime", help="排序: createTime/gain/winRate/profit 等")
    ap.add_argument("--keyword", default="", help="关键词")
    ap.add_argument("--page-num", type=int, default=DEFAULT_PAGE_NUM, help="每页数量 pageNum（默认 10000）")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认当前目录下 csv/）")

    args = ap.parse_args()

    local = _load_local_secrets()
    test_one = _truthy(local.get("test_one"))
    print_payload = _truthy(local.get("print_payload"))
    debug_cookie_snapshot = _truthy(local.get("debug_cookie_snapshot"))
    debug_cookie_snapshot_full = _truthy(local.get("debug_cookie_snapshot_full"))

    browser_channel = (local.get("browser_channel") or "").strip() or "chrome"
    browser_devtools = _truthy(local.get("browser_devtools"))
    slow_mo_ms = _as_int(local.get("browser_slow_mo_ms"))
    login_cookie_timeout = _as_float(local.get("login_cookie_timeout_s"))
    if login_cookie_timeout is None:
        login_cookie_timeout = 180.0

    yb_sleep_min = _as_float(local.get("yieldbacktest_sleep_min_s"))
    yb_sleep_max = _as_float(local.get("yieldbacktest_sleep_max_s"))
    if yb_sleep_min is None:
        yb_sleep_min = 10.0
    if yb_sleep_max is None:
        yb_sleep_max = 20.0
    if yb_sleep_max < yb_sleep_min:
        yb_sleep_max = yb_sleep_min

    max_403 = _as_int(local.get("max_consecutive_403"))
    if max_403 is None:
        max_403 = 3

    out_dir = _ensure_out_dir(args.out_dir)
    prefix = f"yield_hits_browser_{args.sort_type}_wrgt{int(HIT_WIN_RATE_PCT)}_{_timestamp()}"
    hits_path_csv = out_dir / f"{prefix}.csv"

    seen_ids: set[int] = set()
    hit_rows: list[dict[str, Any]] = []

    with sync_playwright() as p:
        # 在这里直接调用连接逻辑，不再传入 launch 参数（因为是 connect）
        context, page, ua = _open_browser(p, user_agent=None, channel=None, devtools=False, slow_mo_ms=None)

        # 监听 Headers
        capture = _HeaderCapture()
        page.on("request", capture.handler)
        
        # 显式打开回测页面（避免用户当前在其他页面）
        if "app.html" not in page.url:
            print("正在导航到回测页面...")
            page.goto("https://backtest.10jqka.com.cn/backtest/app.html", wait_until="domcontentloaded")
        
        print("请在浏览器页面里随便触发一次回测（让页面发送 yieldbacktest 请求），脚本将自动捕获动态请求头…")
        
        # 使用循环检测，避免 wait_for_header_capture 阻塞太死
        while not capture.latest:
            try:
                page.wait_for_timeout(500)
            except KeyboardInterrupt:
                print("\n等待中断。")
                return 1

        print("抓取动态请求头成功，继续…")
        
        yb_headers = capture.latest

        # API 请求上下文：共享同一浏览器会话 cookie
        req_ctx = context.request

        if debug_cookie_snapshot:
            ck = context.cookies("https://backtest.10jqka.com.cn")
            _dump_cookie_snapshot(ck, stage="after_login", show_full=debug_cookie_snapshot_full)

        # 拉取 list
        page_num = int(args.page_num)
        lst = _fetch_list(
            req_ctx,
            page=1,
            page_num=page_num,
            sort_type=str(args.sort_type),
            keyword=str(args.keyword),
            order=DEFAULT_ORDER,
        )
        items = _extract_items(lst)
        raw_total = len(items)
        progress = _Progress(raw_total)

        end_date = _compute_end_date_yesterday()
        start_date = _compute_start_date_5y_ago(date.today())

        consecutive_403 = 0

        for idx, it in enumerate(items, start=1):
            progress.update(idx, hits=len(hit_rows))
            sid = _get_property_id(it)
            if sid is None or sid in seen_ids:
                continue

            prop = it.get("property") if isinstance(it, dict) else None
            win_rate_raw = prop.get("winRate") if isinstance(prop, dict) else None
            win_rate = _parse_win_rate_ratio(win_rate_raw)
            if win_rate is None or win_rate <= MIN_WIN_RATE:
                continue

            create_time_s = None
            if isinstance(prop, dict):
                create_time_s = _as_str(prop.get("ctime"))

            seen_ids.add(sid)

            try:
                detail = _fetch_detail(req_ctx, sid)
            except Exception as e:
                print(f"跳过：detail 获取失败 sid={sid} ({type(e).__name__}: {e})")
                continue

            params = _extract_strategy_params_for_yieldbacktest(detail)
            if not params:
                print(f"跳过：无法从 detail 提取参数 sid={sid}")
                continue

            # 浏览器模式：仍沿用你本地兜底配置（避免 detail 缺字段）
            payload = {
                "query": params["query"],
                "period": params.get("period") or local.get("default_yieldbacktest_period") or "2,3",
                "start_date": start_date.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
                "stock_hold": params.get("stock_hold") or local.get("fallback_stock_hold") or "5",
                "upper_income": params.get("upper_income") or local.get("fallback_upper_income") or "10",
                "lower_income": params.get("lower_income") or local.get("fallback_lower_income") or "10",
                "fall_income": params.get("fall_income") or local.get("fallback_fall_income") or "9",
                "day_buy_stock_num": params.get("day_buy_stock_num"),
                "engine": local.get("yieldbacktest_engine") or "online",
                "capital": local.get("yieldbacktest_capital") or "50000",
            }

            if print_payload:
                print(f"yieldbacktest payload sid={sid}: " + json.dumps(payload, ensure_ascii=False, sort_keys=True))

            _human_sleep(yb_sleep_min, yb_sleep_max)

            # 403 时自动进入“重新捕获头”流程
            try:
                y = _yield_backtest(req_ctx, payload, headers=yb_headers, retries=DEFAULT_RETRIES)
            except PermissionError as e:
                consecutive_403 += 1
                # 只要出现 403，立即执行刷新并重试（不再等待 max_403 阈值）
                # 注意：为了让当前失败的 sid 能重试，我们需要一个小的内部循环或者在这里用 goto 逻辑
                # 但 Python 没有 goto。最简单的是：把上面的逻辑包裹在一个 while True 里，或者仅仅是刷新后重新发起一次请求
                
                # 这里修改为：立即刷新，然后在本轮循环内“重试当前 sid”
                # 由于原 for 循环无法直接回退，我们可以用一个 while 重试当前 sid 的逻辑包裹 yieldbacktest
                
                # 但要修改整个结构比较大。
                # 方案 B：直接在这里刷新，然后 raise 一个自定义 Retry 当前项的信号？不，太复杂。
                # 方案 C（推荐）：把上面的 yieldbacktest 调用改为带重试的循环。
                pass
            except Exception as e:
                print(f"跳过：yieldbacktest 失败 sid={sid} ({type(e).__name__}: {e})")
                continue
            
            # 如果刚才触发了 PermissionError（被上面的 pass 捕获了），说明需要“刷新+重试”
            # 我们在这里处理重试逻辑
            if consecutive_403 > 0:
                 print(f"\n[提示] 遇到 403 Forbidden，立即刷新 Token 并重试当前策略 (sid={sid})...")
                 
                 # === 执行刷新逻辑 (复用之前的自动操作代码) ===
                 capture.latest = None
                 try:
                    # 使用 reload 强制刷新页面，确保触发网络请求
                    print("正在执行页面刷新 (page.reload)...")
                    page.reload()
                    
                    wait_start = time.time()
                    while not capture.latest:
                        if (time.time() - wait_start) > 20:
                            print("[警告] 刷新后等待新请求头超时 (20s)。")
                            break
                        page.wait_for_timeout(500)
                 except Exception as reload_err:
                    print(f"[错误] 刷新逻辑异常: {reload_err}")
                
                 if capture.latest:
                    yb_headers = capture.latest
                    hxv = yb_headers.get('hexin-v') or yb_headers.get('Hexin-V')
                    print(f"已自动获取新 headers (hexin-v={_mask_secret(hxv)})，正在重试 sid={sid} ...")
                    
                    # === 立即重试一次当前 sid ===
                    try:
                        y = _yield_backtest(req_ctx, payload, headers=yb_headers, retries=DEFAULT_RETRIES)
                        print(f"重试成功 sid={sid}，继续后续解析。")
                    except Exception as e_retry:
                        print(f"重试失败 sid={sid}: {e_retry}，跳过该策略。")
                        continue
                 else:
                    print("未能自动捕获及恢复，跳过当前策略。")
                    continue

            consecutive_403 = 0

            result = y.get("result", {}) if isinstance(y, dict) else {}
            backtest_data = result.get("backtestData") if isinstance(result, dict) else None
            if not isinstance(backtest_data, list):
                backtest_data = []

            best_day_sale: str | None = None
            best_wr_ratio: float | None = None

            for row in backtest_data:
                if not isinstance(row, dict):
                    continue
                day_sale_s = _as_str(row.get("daySaleStrategy"))
                if not day_sale_s:
                    continue
                wr_ratio = _parse_win_rate_ratio(row.get("winRate"))
                if wr_ratio is None:
                    continue
                if wr_ratio * 100.0 <= HIT_WIN_RATE_PCT:
                    continue
                if best_wr_ratio is None or wr_ratio > best_wr_ratio:
                    best_wr_ratio = wr_ratio
                    best_day_sale = day_sale_s

            if best_day_sale is not None and best_wr_ratio is not None:
                hit_rows.append(
                    {
                        "strategy_id": sid,
                        "createTime": create_time_s,
                        "daySaleStrategy": best_day_sale,
                        "winRate": round(best_wr_ratio * 100.0, 4),
                    }
                )

            if test_one:
                print("提示：test_one=1，仅测试 1 个策略，提前结束")
                break

        progress.done()
        print(f"完成：raw_items={raw_total} list_winRate>{MIN_WIN_RATE:g} 后={len(seen_ids)} 命中记录={len(hit_rows)}")
        _write_hits_csv(hits_path_csv, hit_rows)
        print(f"命中记录 CSV 已输出: {hits_path_csv}")

        # 不自动关闭浏览器，方便你查看；脚本结束时会自动退出 context
        try:
            context.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
