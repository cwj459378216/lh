import sys
import os
import pandas as pd
import difflib
import requests
import time
import random
import json

# -----------------------------------------------------------------------------
# 网络请求相关工具函数 (复用自 pc.py 或简化版)
# -----------------------------------------------------------------------------

def _build_headers():
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Connection": "keep-alive",
        "Referer": "https://backtest.10jqka.com.cn/",
    }

def _human_sleep(min_s=0.5, max_s=1.5):
    time.sleep(random.uniform(min_s, max_s))

def fetch_strategy_query(session, strategy_id):
    """
    在线获取策略的 query 字符串
    """
    url = f"https://backtest.10jqka.com.cn/strategysquare/detail?strategyId={strategy_id}"
    try:
        _human_sleep() # 避免请求过快
        resp = session.get(url, timeout=10)
        if resp.status_code == 429:
            print("  [429 Too Many Requests] Sleeping 5s...")
            time.sleep(5)
            # Retry once
            resp = session.get(url, timeout=10)
            
        if resp.status_code != 200:
            print(f"  [Error] HTTP {resp.status_code} for ID {strategy_id}")
            return None

        data = resp.json()
        result = data.get("result", {})
        if not result:
            return None
            
        qs = result.get("queryString", {})
        return qs.get("query")
        
    except Exception as e:
        print(f"  [Exception] Fetching ID {strategy_id}: {e}")
        return None

# -----------------------------------------------------------------------------
# 相似度计算核心逻辑
# -----------------------------------------------------------------------------

def _normalize_query_text(q):
    if q is None:
        return ""
    s = str(q).strip().lower()
    if not s:
        return ""
    
    trans = str.maketrans({
        " ": "", "\t": "", "\r": "", "\n": "",
        ",": "", "，": "", ";": "", "；": "",
        "|": "", "/": "", "\\": "", "-": "", "_": "",
        ":": "", "：": "", "（": "", "）": "", "(": "", ")": "",
        "[": "", "]": "", "{": "", "}": "", "\"": "", "'": ""
    })
    return s.translate(trans)

def _query_similarity_ratio(a, b):
    # a, b assumed normalized
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------

def process_file(csv_file, threshold=0.7):
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    print(f"Reading {csv_file} ...")
    try:
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(csv_file, encoding='utf-8')
    
    # Check for strategy_id
    id_col = None
    for col in ['strategy_id', 'property_id', 'id']:
        if col in df.columns:
            id_col = col
            break
            
    if not id_col:
        print("Error: Could not find a strategy ID column (strategy_id, property_id, id).")
        return

    # Check query column
    has_query_col = 'query' in df.columns
    if not has_query_col:
        print("Column 'query' not found. Will fetch from online API.")
    else:
        print("Column 'query' found. Using local data.")

    # Init Session
    session = requests.Session()
    session.headers.update(_build_headers())
    # Visit home to warm up
    try:
        session.get("https://backtest.10jqka.com.cn/", timeout=5)
    except:
        pass

    kept_rows = []
    seen_queries_norm = []
    
    dropped_count = 0
    total_rows = len(df)
    
    print(f"Total strategies: {total_rows}. Sim Threshold: {threshold}")
    
    rows = df.to_dict('records')
    
    for i, row in enumerate(rows):
        sid = row.get(id_col)
        
        # 1. Get Query
        query_text = ""
        if has_query_col:
            query_text = row.get('query')
        else:
            if i % 10 == 0:
                print(f"Processing {i+1}/{total_rows}... fetching info for {sid}")
            
            online_query = fetch_strategy_query(session, sid)
            if online_query:
                query_text = online_query
                row['fetched_query'] = online_query # Save fetched query to output
            else:
                print(f"  [Warn] Could not fetch query for {sid}, keeping it safely.")
                kept_rows.append(row)
                continue
        
        # Check for '未来' keyword
        if query_text and "未来" in str(query_text):
            dropped_count += 1
            print(f"  [Drop] ID {sid} (Contains '未来')")
            continue

        # 2. Check Similarity
        q_norm = _normalize_query_text(query_text)
        
        if not q_norm:
            kept_rows.append(row)
            continue
            
        is_duplicate = False
        best_ratio = 0.0
        
        for seen_q in seen_queries_norm:
            ratio = _query_similarity_ratio(q_norm, seen_q)
            if ratio > threshold:
                is_duplicate = True
                best_ratio = ratio
                break
        
        if is_duplicate:
            dropped_count += 1
            print(f"  [Drop] ID {sid} (Sim: {best_ratio:.2f})")
        else:
            kept_rows.append(row)
            seen_queries_norm.append(q_norm)

    # Output
    output_df = pd.DataFrame(kept_rows)
    
    dir_name, file_name = os.path.split(csv_file)
    name, ext = os.path.splitext(file_name)
    output_path = os.path.join(dir_name, f"{name}_dedup{int(threshold*100)}{ext}")
    
    output_df.to_csv(output_path, index=False, encoding='utf-8-sig') # with BOM for Excel

    print("-" * 40)
    print(f"Detailed Clean-up Finished.")
    print(f"Original: {total_rows}")
    print(f"Kept:     {len(output_df)}")
    print(f"Dropped:  {dropped_count}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # 默认路径
    default_csv = '/Users/frank/work/code/lh/lh/pc/csv/1.31.csv'
    
    target_file = default_csv
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    threshold = 0.7
    if len(sys.argv) > 2:
        try:
            val = float(sys.argv[2])
            if val > 1.0: val /= 100.0
            threshold = val
        except:
            pass
            
    process_file(target_file, threshold)
