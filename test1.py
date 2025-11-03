# spurious_te_signal.py
import ccxt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from retry import retry


class SpuriousTEAnalyzer:
    """
    虚假转移熵分析器类
    用于分析加密货币之间的虚假转移熵和延迟相关性
    """
    
    def __init__(self, exchange_name="kucoin", timeout=30000, default_timeframes=None, default_periods=None):
        """
        初始化分析器
        
        Args:
            exchange_name: 交易所名称，默认为 "kucoin"
            timeout: 请求超时时间（毫秒）
            default_timeframes: 默认时间周期列表
            default_periods: 默认数据周期列表
        """
        self.exchange = getattr(ccxt, exchange_name)({"timeout": timeout})
        self.timeframes = default_timeframes or ["1m", "5m"]
        self.periods = default_periods or ["1d", "7d", "30d", "60d"]
        self.btc_symbol = "BTC/USDT"
        self.btc_df_cache = {}  # 缓存字典，key为 (timeframe, period)
    
    @staticmethod
    def _period_to_bars(period: str, timeframe: str) -> int:
        """
        将周期转换为 bars
        period: 周期，如 "60d", 天为单位
        timeframe: 时间周期，如 "5m", 分钟为单位
        return: bars
        """
        days = int(period.rstrip('d'))
        timeframe_minutes = int(timeframe.rstrip('m'))
        bars_per_day = int(24 * 60 / timeframe_minutes)
        return days * bars_per_day
    
    @retry(tries=10, delay=5, backoff=2)
    def download_ccxt_data(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """
        下载数据
        symbol: 交易对，如 "BTC/USDT"
        period: 周期，如 "60d"
        timeframe: 时间周期，如 "5m"
        return: DataFrame
        """
        target_bars = self._period_to_bars(period, timeframe)
        # timeframe为分钟级
        ms_per_bar = int(timeframe.rstrip('m')) * 60 * 1000
        now_ms = self.exchange.milliseconds()
        since = now_ms - target_bars * ms_per_bar

        all_rows = []
        fetched = 0
        while True:
            # 带重试抓取 OHLCV
            last_exc = None
            for attempt in range(5):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
                    last_exc = None
                    break
                except Exception as e:
                    last_exc = e
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
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        df = df.set_index("Timestamp").sort_index()
        df['return'] = df['Close'].pct_change().fillna(0)
        df['volume_usd'] = df['Volume'] * df['Close']
        return df
    
    @staticmethod
    def find_optimal_delay(btc_ret, alt_ret, max_lag=48):
        """
        找最优延迟 τ*
        
        Args:
            btc_ret: BTC收益率数组
            alt_ret: 山寨币收益率数组
            max_lag: 最大延迟值
        
        Returns:
            tau_star: 最优延迟
            corrs: 相关系数列表
            max_related_matrix: 最大相关系数
        """
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
            related_matrix = np.corrcoef(x[:m], y[:m])[0, 1]
            corrs.append(related_matrix)
        tau_star = lags[np.argmax(corrs)]
        max_related_matrix = max(corrs)
        return tau_star, corrs, max_related_matrix
    
    @staticmethod
    def compute_spurious_te(btc_ret, alt_ret, delay, k=3):
        """
        计算虚假 TE: T_{ALT → BTC}(τ)
        
        Args:
            btc_ret: BTC收益率数组
            alt_ret: 山寨币收益率数组
            delay: 延迟值
            k: 参数（默认3）
        
        Returns:
            TE值
        """
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
    
    @staticmethod
    def generate_signal(te_value, threshold=0.05):
        """
        信号生成
        
        Args:
            te_value: TE值
            threshold: 阈值
        
        Returns:
            信号字符串
        """
        if te_value > threshold:
            return "ENTER: 延迟套利信号触发！"
        else:
            return "HOLD: 虚假 TE 不足"
    
    def one_coin_analysis(self, coin: str):
        """
        分析单个币种，返回最大相关系数
        coin: 币种，如 "KCS/USDT"
        """
        max_related_matrix_list = {}
        # 下载数据
        for timeframe in self.timeframes:
            for period in self.periods:
                print(f"正在从 KuCoin 下载 BTC/USDT 和 {coin} 的 {timeframe}、{period} 数据...")
                # 缓存 BTC 数据，避免重复下载（因为 BTC 数据对所有币种都相同）
                cache_key = (timeframe, period)
                if cache_key not in self.btc_df_cache:
                    self.btc_df_cache[cache_key] = self.download_ccxt_data(
                        self.btc_symbol, period=period, timeframe=timeframe
                    )
                # 必须使用 .copy()，避免后续操作（如 loc 索引）修改缓存的数据
                btc_df = self.btc_df_cache[cache_key].copy()
                alt_df = self.download_ccxt_data(coin, period=period, timeframe=timeframe)
                
                common_idx = btc_df.index.intersection(alt_df.index)
                btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
                
                btc_ret = btc_df['return'].values
                alt_ret = alt_df['return'].values
                
                # 1. 找最优延迟（单位：分钟级 bars）
                tau_star, corr_curve, max_related_matrix = self.find_optimal_delay(btc_ret, alt_ret)
                print(f'timeframe: {timeframe}, period: {period}, tau_star: {tau_star}, max_related_matrix: {max_related_matrix}')

                max_related_matrix_list[max_related_matrix] = (timeframe, period, tau_star)

        max_related_matrix_list = sorted(max_related_matrix_list.items(), key=lambda x: x[0], reverse=True)
        
        # 转换为 pandas DataFrame
        df_results = pd.DataFrame([
            {
                '最大相关系数': max_corr,
                '时间周期': timeframe,
                '数据周期': period,
                '最优延迟': tau_star
            }
            for max_corr, (timeframe, period, tau_star) in max_related_matrix_list
        ])
        
        # 格式化输出
        print("\n" + "="*60)
        print(f"{coin}相关系数分析结果")
        print("="*60)
        print(df_results.to_string(index=False))
        print("="*60)
    
    def run(self, quote_currency="USDT"):
        """
        主运行方法，分析所有币种
        
        Args:
            quote_currency: 计价货币，默认为 "USDT"
        """
        all_coins = self.exchange.load_markets()
        for coin in all_coins:
            coin_item = all_coins[coin]
            if coin_item['quote'] != quote_currency:
                continue
            self.one_coin_analysis(coin)
            time.sleep(1)


# ================== 运行 ==================
if __name__ == "__main__":
    analyzer = SpuriousTEAnalyzer()
    analyzer.run()
