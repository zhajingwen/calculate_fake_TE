# spurious_te_signal.py
import yfinance as yf
import numpy as np
import pandas as pd
from pyinform import transfer_entropy
import matplotlib.pyplot as plt

# ================== 1. 数据下载 ==================
def download_data(symbol, period="60d", interval="1h"):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True)
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
    if len(btc_ret) < 100 or len(alt_ret) < 100:
        return 0.0
    try:
        # 构造嵌入向量
        X = np.column_stack([btc_ret[k+i-1:-i] for i in range(k)])        # BTC 作为目标
        Y = np.column_stack([alt_ret[k+i-1+delay:-i-delay] for i in range(k)])  # ALT 滞后 delay
        min_len = min(len(X), len(Y))
        X, Y = X[:min_len], Y[:min_len]
        te = transfer_entropy(Y, X, k=k, local=False)  # T_{ALT → BTC}
        return max(te, 0.0)
    except:
        return 0.0

# ================== 4. 信号生成 ==================
def generate_signal(te_value, threshold=0.05):
    if te_value > threshold:
        return "ENTER: 延迟套利信号触发！"
    else:
        return "HOLD: 虚假 TE 不足"

# ================== 5. 主函数：SHIB vs BTC 实时计算 ==================
def main():
    print("正在下载 BTC 和 SHIB 数据...")
    btc_df = download_data("BTC-USD")
    alt_df = download_data("SHIB-USD")
    
    common_idx = btc_df.index.intersection(alt_df.index)
    btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
    
    btc_ret = btc_df['return'].values
    alt_ret = alt_df['return'].values
    
    # 1. 找最优延迟
    tau_star, corr_curve = find_optimal_delay(btc_ret, alt_ret)
    print(f"最优延迟 τ* = {tau_star:+} 小时")
    
    # 2. 计算虚假 TE
    te_false = compute_spurious_te(btc_ret, alt_ret, delay=tau_star)
    print(f"虚假转移熵 T_{{SHIB→BTC}}(τ*) = {te_false:.4f} bits")
    
    # 3. 信号
    signal = generate_signal(te_false, threshold=0.05)
    print(f"信号: {signal}")
    
    # 4. 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    lags = list(range(-48, 49))
    plt.plot(lags, corr_curve, 'b-', linewidth=2)
    plt.axvline(tau_star, color='r', linestyle='--', label=f'τ* = {tau_star}h')
    plt.title('交叉相关 vs 延迟')
    plt.xlabel('延迟 τ (小时)')
    plt.ylabel('相关系数')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    te_curve = [compute_spurious_te(btc_ret, alt_ret, d) for d in range(0, 25)]
    plt.plot(range(0, 25), te_curve, 'r-', linewidth=2)
    plt.axvline(abs(tau_star), color='r', linestyle='--', label=f'τ* = {abs(tau_star)}h')
    plt.axhline(0.05, color='k', linestyle=':', label='阈值 0.05')
    plt.title('虚假 TE vs 正延迟')
    plt.xlabel('延迟 τ (小时)')
    plt.ylabel('T_{SHIB→BTC} (bits)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("spurious_te_shib_btc.png", dpi=300)
    plt.show()
    
    return te_false, tau_star

# ================== 运行 ==================
if __name__ == "__main__":
    te, tau = main()
