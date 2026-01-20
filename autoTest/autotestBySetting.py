import os
import re
import sys
import ast
import shutil
import hashlib
from datetime import datetime

import pandas as pd

# 允许从 autoTest 子目录直接运行
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import backtest_select_stocks_local as bt
import select_stocks_local as sel
from stop_loss_rules import StopLossConfig


# --- 批量隔离执行（强隔离：每个配置单独进程）

def _run_in_subprocess(setting_csv_path: str, base_out_dir: str | None = None) -> str:
    """在子进程中运行单个配置，返回输出目录。

    说明：
    - 通过重新启动 Python 进程，实现 backtest_select_stocks_local / select_stocks_local / stop_loss_rules
      以及所有模块级全局变量的完全隔离。
    - 子进程会把最终输出目录写到 run_out_dir.txt；父进程读取该文件作为返回值。
    """
    setting_csv_path = os.path.abspath(setting_csv_path)

    cfg_name = _safe_name(os.path.splitext(os.path.basename(setting_csv_path))[0])
    cfg_hash = _hash_file(setting_csv_path)

    if base_out_dir is None:
        base_out_dir = os.path.join(os.path.dirname(__file__), 'output')
    base_out_dir = os.path.abspath(base_out_dir)
    os.makedirs(base_out_dir, exist_ok=True)

    cfg_root = os.path.join(base_out_dir, f'{cfg_name}_{cfg_hash}')
    os.makedirs(cfg_root, exist_ok=True)

    # 子进程输出落在 cfg_root 下（回测脚本内部再加时间戳子目录）
    run_out_dir_flag = os.path.join(cfg_root, 'run_out_dir.txt')

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        '--child',
        setting_csv_path,
        '--base-out-dir',
        cfg_root,
        '--flag',
        run_out_dir_flag,
    ]

    import subprocess

    # 父进程不抛异常：让上层决定是否跳过；这里返回空串即可
    try:
        p = subprocess.run(cmd, cwd=ROOT_DIR)
    except Exception:
        return ''

    if p.returncode != 0:
        return ''

    if os.path.isfile(run_out_dir_flag):
        try:
            out_dir = open(run_out_dir_flag, 'r', encoding='utf-8').read().strip()
            if out_dir:
                return out_dir
        except Exception:
            pass

    # 兜底：至少返回 cfg_root，方便留痕
    return cfg_root


def _safe_name(s: str, max_len: int = 80) -> str:
    s = str(s or '')
    s = re.sub(r'[^0-9A-Za-z\u4e00-\u9fff._-]+', '_', s)
    s = s.strip('._-')
    if not s:
        s = 'cfg'
    return s[:max_len]


def _parse_value(v: str):
    """把 CSV 里的字符串转回 Python 值（尽量温和，失败则保留原字符串）。"""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None

    s = str(v).strip()
    if s == '' or s.lower() in ('nan', 'none', 'null'):
        return None

    # bool
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False

    # int/float
    try:
        if re.fullmatch(r'[-+]?\d+', s):
            return int(s)
        if re.fullmatch(r'[-+]?\d*\.\d+(e[-+]?\d+)?', s.lower()) or re.fullmatch(r'[-+]?\d+e[-+]?\d+', s.lower()):
            return float(s)
    except Exception:
        pass

    # list/dict/tuple
    try:
        obj = ast.literal_eval(s)

        # 关键修复：用户常写 "((25.0, 30.0))" 这种，Python 会把它解析成 (25.0, 30.0) 而不是 ((25.0,30.0),)
        # 但业务上我们需要的是“区间列表”，所以将 (lo,hi) 自动提升为 [(lo,hi)]。
        if isinstance(obj, tuple) and len(obj) == 2:
            a, b = obj
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return [(float(a), float(b))]

        # 也兼容用户写成单个区间的 list: [25.0, 30.0]
        if isinstance(obj, list) and len(obj) == 2:
            a, b = obj
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return [(float(a), float(b))]

        return obj
    except Exception:
        return s


def load_setting_csv(setting_csv_path: str) -> dict[str, dict]:
    """读取一个 回测配置.csv，按模块拆成 dict。"""
    df = pd.read_csv(setting_csv_path, encoding='utf-8-sig')
    # 兼容用户手动保存导致的列名空格
    df.columns = [str(c).strip() for c in df.columns]

    required = {'模块', '参数', '值'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f'配置文件缺少列: 需要 {required}, 实际 {set(df.columns)}')

    out: dict[str, dict] = {}
    for _, r in df.iterrows():
        module = str(r.get('模块') or '').strip()
        key = str(r.get('参数') or '').strip()
        val = _parse_value(r.get('值'))
        if not module or not key:
            continue
        out.setdefault(module, {})[key] = val
    return out


def apply_configs(configs: dict[str, dict]):
    """把配置写回到 bt.CFG / sel.CFG，并构造 sl_cfg。"""
    # 1) 回测配置
    bt_cfg = bt.CFG
    for k, v in (configs.get('backtest_select_stocks_local') or {}).items():
        if hasattr(bt_cfg, k):
            setattr(bt_cfg, k, v)

    # 2) 选股配置
    sel_cfg = getattr(sel, 'CFG', None)
    if sel_cfg is not None:
        for k, v in (configs.get('select_stocks_local') or {}).items():
            if hasattr(sel_cfg, k):
                setattr(sel_cfg, k, v)

    # 3) 止损配置：用 stop_loss_rules 的模块段（如果缺字段，StopLossConfig 会用默认值）
    sl_kwargs = (configs.get('stop_loss_rules') or {}).copy()
    sl_cfg = StopLossConfig(**{k: v for k, v in sl_kwargs.items() if v is not None})

    return bt_cfg, sel_cfg, sl_cfg


def _hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def run_one_setting(setting_csv_path: str, base_out_dir: str | None = None) -> str:
    """跑单个配置；返回该次回测输出目录。

    批量模式一致性要求：
    - 默认使用“子进程隔离”以避免模块级全局状态污染。

    约定：
    - 若运行失败，返回空字符串（由上层决定跳过）。
    """
    # 默认强隔离：每个配置单独进程
    if os.environ.get('AUTOTEST_DISABLE_SUBPROCESS', '').strip().lower() in ('1', 'true', 'yes'):
        pass
    else:
        return _run_in_subprocess(setting_csv_path, base_out_dir=base_out_dir)

    # --- 兼容：仍保留原“同进程运行”逻辑（仅用于调试）
    try:
        setting_csv_path = os.path.abspath(setting_csv_path)
        configs = load_setting_csv(setting_csv_path)

        bt_cfg, _, _ = apply_configs(configs)

        # 让这个配置跑出来的结果也保留：覆盖 bt_cfg.out_dir 到 autoTest 的批量输出目录
        # 目录结构：autoTest/output/<配置文件名>_<hash>/<timestamp>/...
        if base_out_dir is None:
            base_out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(base_out_dir, exist_ok=True)

        cfg_name = _safe_name(os.path.splitext(os.path.basename(setting_csv_path))[0])
        cfg_hash = _hash_file(setting_csv_path)
        cfg_root = os.path.join(base_out_dir, f'{cfg_name}_{cfg_hash}')
        os.makedirs(cfg_root, exist_ok=True)

        # 告诉回测：它内部还会再加时间戳子目录
        bt_cfg.out_dir = cfg_root

        run_out_dir = bt.main()

        # 兼容：回测脚本可能因为“无交易日”等原因提前 return None
        if not run_out_dir:
            run_out_dir = cfg_root

        # 把本次使用的配置文件原件也拷贝到 run_out_dir，方便追溯
        try:
            shutil.copy2(setting_csv_path, os.path.join(run_out_dir, '回测配置_输入.csv'))
        except Exception:
            pass

        # 写一份批量运行留痕
        try:
            pd.DataFrame([
                {'项': '配置文件', '值': setting_csv_path},
                {'项': '输出目录', '值': run_out_dir},
                {'项': '完成时间', '值': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
            ]).to_csv(os.path.join(run_out_dir, '批量运行信息.csv'), index=False, encoding='utf-8-sig')
        except Exception:
            pass

        return run_out_dir
    except Exception:
        return ''


def run_all_settings(settings_dir: str, pattern: str = '*.csv') -> list[str]:
    """批量跑一个目录下的所有配置文件（扫描 *.csv，并按列名过滤出“回测配置表”）。"""
    settings_dir = os.path.abspath(settings_dir)
    if not os.path.isdir(settings_dir):
        raise FileNotFoundError(settings_dir)

    import glob
    files = sorted(glob.glob(os.path.join(settings_dir, pattern)))

    if not files:
        raise FileNotFoundError(f'未找到CSV文件: {settings_dir}\\{pattern}')

    def _is_valid_setting_csv(path: str) -> bool:
        try:
            df0 = pd.read_csv(path, encoding='utf-8-sig', nrows=0)
            cols = [str(c).strip() for c in df0.columns]
            return {'模块', '参数', '值'}.issubset(set(cols))
        except Exception:
            return False

    setting_files = [fp for fp in files if _is_valid_setting_csv(fp)]
    if not setting_files:
        raise FileNotFoundError(f'未找到“回测配置.csv”格式的文件（需要列: 模块/参数/值）：{settings_dir}')

    results = []
    for fp in setting_files:
        print('=' * 80)
        print('开始运行配置：', fp)
        try:
            out_dir = run_one_setting(fp)
            if out_dir:
                results.append(out_dir)
                print('完成，输出目录：', out_dir)
            else:
                print('运行失败，已跳过：', fp)
        except Exception as e:
            # 兜底：任何异常都不阻断批量
            print('运行失败，已跳过：', fp)
            print('异常：', repr(e))

    return results


def main():
    """用法：
    - python autoTest\\autotestBySetting.py  (默认跑 autoTest 目录下的所有 *.csv，并按列名筛选回测配置表)
    - python autoTest\\autotestBySetting.py <配置目录>
    - python autoTest\\autotestBySetting.py <单个配置文件.csv>

    子进程模式内部调用：
    - python autoTest\\autotestBySetting.py --child <配置文件.csv> --base-out-dir <输出根> --flag <run_out_dir.txt>
    """
    base = os.path.dirname(__file__)

    # --- 子进程入口（只跑一个配置）
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        if len(sys.argv) < 3:
            raise ValueError('child 模式缺少配置文件参数')
        setting_csv = os.path.abspath(sys.argv[2])

        # 解析可选参数
        base_out_dir = None
        flag_path = None
        args = sys.argv[3:]
        i = 0
        while i < len(args):
            a = args[i]
            if a == '--base-out-dir' and i + 1 < len(args):
                base_out_dir = args[i + 1]
                i += 2
                continue
            if a == '--flag' and i + 1 < len(args):
                flag_path = args[i + 1]
                i += 2
                continue
            i += 1

        # 在子进程内：同进程运行（但每次都是全新进程，等价于隔离）
        configs = load_setting_csv(setting_csv)
        bt_cfg, _, _ = apply_configs(configs)

        # base_out_dir 这里传的是 cfg_root（父进程已确保唯一）
        if base_out_dir is None:
            base_out_dir = os.path.join(os.path.dirname(__file__), 'output')
        os.makedirs(base_out_dir, exist_ok=True)

        bt_cfg.out_dir = os.path.abspath(base_out_dir)
        run_out_dir = bt.main()
        if not run_out_dir:
            run_out_dir = bt_cfg.out_dir

        # 拷贝配置原件 + 留痕（子进程写到真实输出目录）
        try:
            shutil.copy2(setting_csv, os.path.join(run_out_dir, '回测配置_输入.csv'))
        except Exception:
            pass

        try:
            pd.DataFrame([
                {'项': '配置文件', '值': setting_csv},
                {'项': '输出目录', '值': run_out_dir},
                {'项': '完成时间', '值': datetime.now().strftime('%Y-%m-%d %H:%M:%S')},
                {'项': '运行模式', '值': 'child-subprocess'},
            ]).to_csv(os.path.join(run_out_dir, '批量运行信息.csv'), index=False, encoding='utf-8-sig')
        except Exception:
            pass

        if flag_path:
            try:
                os.makedirs(os.path.dirname(flag_path), exist_ok=True)
                with open(flag_path, 'w', encoding='utf-8') as f:
                    f.write(str(run_out_dir))
            except Exception:
                pass

        return

    # --- 原入口
    if len(sys.argv) <= 1:
        # 默认：跑当前 autoTest 目录下所有“*.csv”（并按列名筛选）
        target = base
    else:
        target = sys.argv[1]

    target = os.path.abspath(target)
    if os.path.isdir(target):
        outs = run_all_settings(target, pattern='*.csv')
        print('\n全部完成：')
        for o in outs:
            print(' -', o)
        return

    if os.path.isfile(target):
        out_dir = run_one_setting(target)
        print('完成，输出目录：', out_dir)
        return

    raise FileNotFoundError(target)


if __name__ == '__main__':
    main()
