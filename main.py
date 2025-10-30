# spurious_te_signal.py
import ccxt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================== 1. 数据下载（CCXT / KuCoin） ==================
def _period_to_bars(period: str, timeframe: str) -> int:
    assert timeframe == "5m"
    days = int(period.rstrip('d'))
    bars_per_day = int(24 * 60 / 5)
    return days * bars_per_day

def download_ccxt_data(symbol: str, period: str = "60d", timeframe: str = "5m") -> pd.DataFrame:
    exchange = ccxt.kucoin({"enableRateLimit": True})
    target_bars = _period_to_bars(period, timeframe)
    ms_per_bar = 5 * 60 * 1000
    now_ms = exchange.milliseconds()
    since = now_ms - target_bars * ms_per_bar

    all_rows = []
    fetched = 0
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
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
    lags = list(range(-max_lag, max_lag + 1))
    for lag in lags:
        if lag >= 0:
            # ALT 滞后 BTC
            corr = np.corrcoef(alt_ret[:-lag], btc_ret[lag:])[0,1] if lag < len(alt_ret)-10 else -1
        else:
            # BTC 滞后 ALT（一般不会）
            corr = np.corrcoef(alt_ret[-lag:], btc_ret[:lag])[0,1] if -lag < len(btc_ret)-10 else -1
        corrs.append(corr)
    tau_star = lags[np.argmax(corrs)]
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
    print("正在从 KuCoin 下载 BTC/USDT 和 KCS/USDT 的 5m、60d 数据...")
    btc_df = download_ccxt_data("BTC/USDT", period="60d", timeframe="5m")
    alt_df = download_ccxt_data("KCS/USDT", period="60d", timeframe="5m")
    
    common_idx = btc_df.index.intersection(alt_df.index)
    btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
    
    btc_ret = btc_df['return'].values
    alt_ret = alt_df['return'].values
    
    # 1. 找最优延迟（单位：5m bars）
    tau_star, corr_curve = find_optimal_delay(btc_ret, alt_ret)
    print(f"最优延迟 τ* = {tau_star:+} 个 5m bars（约 {tau_star*5} 分钟）")
    
    # 2. 计算虚假 TE
    te_false = compute_spurious_te(btc_ret, alt_ret, delay=tau_star)
    print(f"虚假转移熵 T_{{KCS→BTC}}(τ*) = {te_false:.4f} bits")
    
    # 3. 信号
    signal = generate_signal(te_false, threshold=0.05)
    print(f"信号: {signal}")
    
    # 4. 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    lags = list(range(-48, 49))
    plt.plot(lags, corr_curve, 'b-', linewidth=2)
    plt.axvline(tau_star, color='r', linestyle='--', label=f'τ* = {tau_star} bars')
    plt.title('交叉相关 vs 延迟')
    plt.xlabel('延迟 τ (5m bars)')
    plt.ylabel('相关系数')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    te_curve = [compute_spurious_te(btc_ret, alt_ret, d) for d in range(0, 25)]
    plt.plot(range(0, 25), te_curve, 'r-', linewidth=2)
    plt.axvline(abs(tau_star), color='r', linestyle='--', label=f'τ* = {abs(tau_star)} bars')
    plt.axhline(0.05, color='k', linestyle=':', label='阈值 0.05')
    plt.title('虚假 TE vs 正延迟')
    plt.xlabel('延迟 τ (5m bars)')
    plt.ylabel('T_{KCS→BTC} (bits)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("spurious_te_kcs_btc.png", dpi=300)
    plt.show()
    
    return te_false, tau_star

# ================== 运行 ==================
if __name__ == "__main__":
    te, tau = main()
