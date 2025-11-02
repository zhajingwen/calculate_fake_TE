# spurious_te_signal.py
import ccxt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# ================== 1. 数据下载（CCXT / KuCoin） ==================
def _period_to_bars(period: str, timeframe: str) -> int:
    """
    将周期转换为 bars
    period: 周期，如 "60d", 天为单位
    timeframe: 时间周期，如 "5m", 分钟为单位
    return: bars
    """
    # assert timeframe == "5m"
    days = int(period.rstrip('d'))
    timeframe = int(timeframe.rstrip('m'))
    bars_per_day = int(24 * 60 / timeframe)
    return days * bars_per_day

def download_ccxt_data(symbol: str, period: str, timeframe: str) -> pd.DataFrame:
    """
    下载数据
    symbol: 交易对，如 "BTC/USDT"
    period: 周期，如 "60d"
    timeframe: 时间周期，如 "5m"
    return: DataFrame
    """
    # 更长超时 + 限速；若 .com 超时，将切换到 .cc 镜像
    exchange = ccxt.kucoin({"timeout": 30000})
    exchange.load_markets()

    target_bars = _period_to_bars(period, timeframe)
    # timeframe为分钟级
    ms_per_bar = int(timeframe.rstrip('m')) * 60 * 1000
    now_ms = exchange.milliseconds()
    since = now_ms - target_bars * ms_per_bar

    all_rows = []
    fetched = 0
    while True:
        # 带重试抓取 OHLCV
        last_exc = None
        for attempt in range(5):
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                import time
                time.sleep(1.5 * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        if not ohlcv:
            break
        all_rows.extend(ohlcv)
        fetched += len(ohlcv)
        since = ohlcv[-1][0] + ms_per_bar
        if len(ohlcv) < 1500 or fetched >= target_bars:
            break

    if not all_rows:
        return pd.DataFrame(columns=["Open","High","Low","Close","Volume"])  

    df = pd.DataFrame(all_rows, columns=["Timestamp","Open","High","Low","Close","Volume"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    df = df.set_index("Timestamp").sort_index()
    df['return'] = df['Close'].pct_change().fillna(0)
    df['volume_usd'] = df['Volume'] * df['Close']
    return df

# ================== 2. 找最优延迟 τ* ==================
def find_optimal_delay(btc_ret, alt_ret, max_lag=48):
    corrs = []
    lags = list(range(0, max_lag + 1))
    for lag in lags:
        if lag > 0:
            # ALT 滞后 BTC：验证 BTC[t] 是否影响 ALT[t+lag]
            x = btc_ret[:-lag]  # BTC 的前 n-lag 个
            y = alt_ret[lag:]   # ALT 的后 n-lag 个（跳过前 lag 个）
        elif lag == 0:
            # lag == 0，用全样本
            x = alt_ret
            y = btc_ret
        m = min(len(x), len(y))
        if m < 10:
            corrs.append(-1)
            continue
        # print(x[:m], y[:m])
        related_matrix = np.corrcoef(x[:m], y[:m])[0, 1]
        print(f'lag: {lag}, related_matrix: {related_matrix}')

        # # 转成 DataFrame
        # df = pd.DataFrame({'BTC': x[:m], 'KCS': y[:m]})
        # print(df)
        # print(df.corr())    
        # # 画热力图
        # plt.figure(figsize=(6, 5))
        # sns.heatmap(
        #     df.corr(),
        #     annot=True,          # 显示数字
        #     cmap='coolwarm',     # 红正蓝负
        #     center=0,            # 0 为中心（可选）
        #     square=True,         # 正方形格子
        #     fmt='.2f',           # 保留2位小数
        #     cbar_kws={'label': '相关系数'}  # 颜色条标签
        # )
        # plt.title('BTC vs ETH 收益率相关性热力图')
        # plt.show()

        # time.sleep(1000)
                
        
        
        corrs.append(related_matrix)
    # print(corrs)
    tau_star = lags[np.argmax(corrs)]
    print(f'tau_star: {tau_star}')
    return tau_star, corrs

# ================== 3. 计算虚假 TE: T_{ALT → BTC}(τ) ==================
def compute_spurious_te(btc_ret, alt_ret, delay, k=3):
    # 采用离散近似的 TE 估计：TE ≈ I(X_t; Y_{t-τ} | X_{t-1})，固定 k=1
    if len(btc_ret) < 200 or len(alt_ret) < 200 or delay < 0:
        return 0.0
    try:
        num_bins = 3
        # 分位数分箱（避免极端值影响）
        def discretize(x):
            qs = np.nanquantile(x, [1/num_bins, 2/num_bins])
            return np.digitize(x, qs)

        xb = discretize(btc_ret)
        yb = discretize(alt_ret)

        # 对齐：X_t, X_{t-1}, Y_{t-τ}
        X_t = xb[1+delay:]
        X_p = xb[delay:-1]
        Y_p = yb[:-1-delay]

        n = min(len(X_t), len(X_p), len(Y_p))
        if n <= 100:
            return 0.0
        X_t, X_p, Y_p = X_t[:n], X_p[:n], Y_p[:n]

        # 计算条件互信息 I(X_t; Y_p | X_p)
        # 统计联合频率
        max_x = int(max(X_t.max(), X_p.max())) + 1
        max_y = int(Y_p.max()) + 1
        p_xyz = np.zeros((max_x, max_x, max_y), dtype=np.float64)
        p_xz = np.zeros((max_x, max_y), dtype=np.float64)
        p_yz = np.zeros((max_x, max_y), dtype=np.float64)
        p_z = np.zeros((max_x,), dtype=np.float64)

        for a, b, c in zip(X_t, X_p, Y_p):
            p_xyz[a, b, c] += 1
            p_xz[a, c] += 1
            p_yz[b, c] += 1
            p_z[b] += 1

        p_xyz /= n
        p_xz /= n
        p_yz /= n
        p_z /= n

        eps = 1e-12
        te = 0.0
        it = np.nditer(p_xyz, flags=['multi_index'])
        while not it.finished:
            pabc = float(it[0])
            if pabc > 0:
                a, b, c = it.multi_index
                num = pabc * p_z[b]
                den = (p_yz[b, c] * p_xz[a, c])
                if num > 0 and den > 0:
                    te += pabc * np.log2((num + eps) / (den + eps))
            it.iternext()

        return max(float(te), 0.0)
    except Exception:
        return 0.0

# ================== 4. 信号生成 ==================
def generate_signal(te_value, threshold=0.05):
    if te_value > threshold:
        return "ENTER: 延迟套利信号触发！"
    else:
        return "HOLD: 虚假 TE 不足"

# ================== 5. 主函数：KCS vs BTC（以 USDT 计价，5m/60d） ==================
def main():
    timeframes = ["1m","5m"]
    periods = ["1d", "7d", "30d", "60d"]
    for timeframe in timeframes:
        for period in periods:
            print(f"正在从 KuCoin 下载 BTC/USDT 和 KCS/USDT 的 {timeframe}、{period} 数据...")
            btc_df = download_ccxt_data("BTC/USDT", period=period, timeframe=timeframe)
            alt_df = download_ccxt_data("KCS/USDT", period=period, timeframe=timeframe)
            
            common_idx = btc_df.index.intersection(alt_df.index)
            btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
            
            btc_ret = btc_df['return'].values
            alt_ret = alt_df['return'].values
            
            # 1. 找最优延迟（单位：分钟级 bars）
            tau_star, corr_curve = find_optimal_delay(btc_ret, alt_ret)
            print(f'timeframe: {timeframe}, period: {period}, tau_star: {tau_star}')
            # return tau_star, corr_curve


# ================== 运行 ==================
if __name__ == "__main__":
    main()
