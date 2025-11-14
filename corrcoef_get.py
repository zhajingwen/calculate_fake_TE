# spurious_te_signal.py
"""
虚假传递熵（Spurious Transfer Entropy）分析器

本模块用于分析山寨币与BTC之间的相关性，通过计算皮尔逊相关系数和传递熵（TE）
来识别可能存在的虚假因果关系。主要功能包括：
1. 从交易所下载历史K线数据
2. 计算不同时间周期和延迟下的相关系数
3. 找到最优延迟（tau_star）
4. 计算虚假传递熵值
5. 生成交易信号

核心思想：如果山寨币与BTC存在高相关性，可能存在套利机会或虚假因果关系
"""
import ccxt
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from retry import retry


class SpuriousTEAnalyzer:
    """
    虚假传递熵分析器类
    
    用于分析各个山寨币与BTC的皮尔逊相关系数，并计算虚假传递熵。
    通过分析不同时间周期（timeframe）和数据周期（period）下的相关性，
    找出最优延迟（tau_star），从而识别潜在的套利机会或虚假因果关系。
    """
    
    def __init__(self, exchange_name="kucoin", timeout=30000, default_timeframes=None, default_periods=None):
        """
        初始化分析器
        
        Args:
            exchange_name (str): 交易所名称，默认为 "kucoin"
                                支持的交易所包括：binance, okx, kucoin等（需ccxt支持）
            timeout (int): 请求超时时间（毫秒），默认30000ms（30秒）
            default_timeframes (list): 默认时间周期列表，如 ["1m", "5m", "15m"]
                                      时间周期表示K线的颗粒度（每根K线的时间长度）
            default_periods (list): 默认数据周期列表，如 ["1d", "7d", "30d", "60d"]
                                    数据周期表示要分析的历史数据的时间范围
        """
        # 初始化交易所连接对象，使用ccxt库动态获取交易所类
        self.exchange = getattr(ccxt, exchange_name)({"timeout": timeout})
        
        # 设置默认时间周期：1分钟和5分钟K线
        # 时间周期越小，数据越细粒度，但计算量越大
        self.timeframes = default_timeframes or ["1m", "5m"]
        
        # 设置默认数据周期：1天、7天、30天、60天的历史数据
        # 数据周期越长，分析越全面，但需要下载更多数据
        self.periods = default_periods or ["1d", "7d", "30d", "60d"]
        
        # BTC的交易对名称（基准货币）
        # 所有山寨币将与BTC进行相关性分析
        self.btc_symbol = "BTC/USDT"
        
        # 缓存BTC各个颗粒度、各个周期级别的数据
        # 由于BTC数据对所有山寨币都相同，缓存可以避免重复下载，提高效率
        # key格式: (timeframe, period)，如 ("5m", "30d")
        # value: 对应的DataFrame数据
        self.btc_df_cache = {}
    
    @staticmethod
    def _period_to_bars(period: str, timeframe: str) -> int:
        """
        将周期转换为K线数量（bars）
        
        计算在指定时间周期（timeframe）下，覆盖指定数据周期（period）需要多少根K线。
        例如：60天的数据，5分钟K线，需要 60 * 24 * 60 / 5 = 17280 根K线
        
        Args:
            period (str): 数据周期（K线的覆盖范围），如 "60d"（60天）
                         格式：数字 + 'd'（天为单位）
            timeframe (str): 时间周期（K线的颗粒度），如 "5m"（5分钟）
                            格式：数字 + 'm'（分钟为单位）
        
        Returns:
            int: 统计周期内K线总条数（bars）
        
        Example:
            >>> _period_to_bars("30d", "5m")
            8640  # 30天 * 24小时 * 60分钟 / 5分钟 = 8640根K线
        """
        # 提取天数：去掉末尾的'd'字符，转换为整数
        days = int(period.rstrip('d'))
        
        # 提取时间周期（分钟数）：去掉末尾的'm'字符，转换为整数
        timeframe_minutes = int(timeframe.rstrip('m'))
        
        # 计算每天有多少根K线：24小时 * 60分钟 / 每根K线的分钟数
        bars_per_day = int(24 * 60 / timeframe_minutes)
        
        # 返回总K线数：天数 * 每天K线数
        return days * bars_per_day
    
    @retry(tries=10, delay=5, backoff=2)
    def download_ccxt_data(self, symbol: str, period: str, timeframe: str) -> pd.DataFrame:
        """
        从交易所下载指定交易对的历史K线数据（OHLCV）
        
        使用ccxt库从交易所API获取历史数据。由于交易所API通常有单次请求数量限制（如1500条），
        需要分批次请求并合并数据。包含重试机制以应对网络波动。
        
        Args:
            symbol (str): 交易对符号，如 "BTC/USDT"、"ETH/USDT"
            period (str): 数据周期，如 "60d"（60天历史数据）
            timeframe (str): 时间周期，如 "5m"（5分钟K线）
        
        Returns:
            pd.DataFrame: 包含以下列的DataFrame：
                - Timestamp: 时间戳（作为索引）
                - Open: 开盘价
                - High: 最高价
                - Low: 最低价
                - Close: 收盘价
                - Volume: 成交量（基础货币单位）
                - return: 收益率（价格变化百分比）
                - volume_usd: 成交量（USD计价）
        
        Note:
            - 使用@retry装饰器，最多重试10次，延迟5秒，指数退避
            - 单次最多获取1500条K线（交易所API限制）
            - 如果数据为空，返回空DataFrame但保留列结构
        """
        # 计算需要获取的K线总数
        target_bars = self._period_to_bars(period, timeframe)
        
        # 计算每根K线的时间长度（毫秒）
        # 例如：5分钟K线 = 5 * 60 * 1000 = 300000毫秒
        ms_per_bar = int(timeframe.rstrip('m')) * 60 * 1000
        
        # 获取当前时间戳（毫秒）
        now_ms = self.exchange.milliseconds()
        
        # 计算起始时间戳：当前时间 - 需要的K线数 * 每根K线时长
        since = now_ms - target_bars * ms_per_bar

        # 存储所有获取到的K线数据
        all_rows = []
        # 已获取的K线数量
        fetched = 0
        
        # 循环获取数据，直到获取足够的数据或没有更多数据
        while True:
            # 带重试机制抓取OHLCV数据
            # 最多尝试5次，每次失败后等待时间递增（指数退避）
            last_exc = None
            for attempt in range(5):
                try:
                    # 从交易所获取OHLCV数据
                    # limit=1500: 单次最多获取1500条（交易所API限制）
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1500)
                    last_exc = None
                    break
                except Exception as e:
                    # 记录异常，等待后重试
                    last_exc = e
                    # 指数退避：第1次等待1.5秒，第2次等待3秒，第3次等待4.5秒...
                    time.sleep(1.5 * (attempt + 1))
            
            # 如果5次尝试都失败，抛出最后一个异常
            if last_exc is not None:
                raise last_exc
            
            # 如果没有获取到数据，退出循环
            if not ohlcv:
                break
            
            # 将本次获取的数据添加到总列表
            all_rows.extend(ohlcv)
            fetched += len(ohlcv)
            
            # 更新起始时间戳：从最后一条数据的下一根K线开始
            # ohlcv[-1][0] 是最后一条数据的时间戳
            since = ohlcv[-1][0] + ms_per_bar
            
            # 如果获取的数据少于1500条（说明已经到最新数据），或已获取足够数据，退出循环
            if len(ohlcv) < 1500 or fetched >= target_bars:
                break

        # 如果没有获取到任何数据，返回空DataFrame但保留列结构
        if not all_rows:
            return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "return", "volume_usd"])

        # 将原始数据转换为DataFrame
        # OHLCV数据格式：[timestamp, open, high, low, close, volume]
        df = pd.DataFrame(all_rows, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"])
        
        # 将时间戳转换为datetime对象
        # unit="ms": 时间戳单位是毫秒
        # utc=True: 先转换为UTC时间
        # dt.tz_convert(None): 再转换为本地时区（去掉时区信息）
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        
        # 将时间戳设置为索引，并按时间排序
        df = df.set_index("Timestamp").sort_index()
        
        # 计算收益率：当前收盘价相对于前一个收盘价的变化百分比
        # pct_change(): 计算百分比变化
        # fillna(0): 第一行没有前一行数据，填充为0
        df['return'] = df['Close'].pct_change().fillna(0)
        
        # 计算USD计价的成交量：成交量（基础货币） * 收盘价（USD）
        df['volume_usd'] = df['Volume'] * df['Close']
        
        return df
    
    @staticmethod
    def find_optimal_delay(btc_ret, alt_ret, max_lag=48):
        """
        寻找最优延迟 τ*（tau star）
        
        通过计算不同延迟（lag）下BTC和山寨币收益率的皮尔逊相关系数，
        找出使相关系数最大的延迟值。这个延迟值表示山寨币价格变化滞后于BTC的程度。
        
        算法原理：
        - 对于每个延迟lag（0到max_lag），计算BTC[t]与ALT[t+lag]的相关系数
        - lag=0: 计算同时刻的相关系数
        - lag>0: 计算BTC领先ALT的相关系数（验证BTC是否影响ALT的未来价格）
        - 选择相关系数最大的lag作为最优延迟τ*
        
        Args:
            btc_ret (np.ndarray): BTC收益率数组，形状为(n,)
            alt_ret (np.ndarray): 山寨币收益率数组，形状为(n,)
            max_lag (int): 最大延迟值（单位：K线数量），默认48
                          例如：5分钟K线，max_lag=48表示最多延迟240分钟（4小时）
        
        Returns:
            tuple: (tau_star, corrs, max_related_matrix)
                - tau_star (int): 最优延迟值，使相关系数最大的lag
                - corrs (list): 所有延迟下的相关系数列表，长度为max_lag+1
                - max_related_matrix (float): 最大相关系数值，范围[-1, 1]
        
        Note:
            - 相关系数接近1表示强正相关，接近-1表示强负相关，接近0表示无相关
            - 如果数据点少于10个，相关系数设为-1（无效值）
            - tau_star越大，说明山寨币价格变化滞后BTC越久
        """
        # 存储所有延迟下的相关系数
        corrs = []
        
        # 生成延迟列表：[0, 1, 2, ..., max_lag]
        lags = list(range(0, max_lag + 1))
        
        # 遍历每个延迟值，计算对应的相关系数
        for lag in lags:
            if lag > 0:
                # ALT滞后BTC的情况：验证BTC[t]是否影响ALT[t+lag]
                # 例如：lag=5表示BTC在t时刻的价格变化，是否影响ALT在t+5时刻的价格
                # x: BTC的前n-lag个数据点（去掉最后lag个）
                x = btc_ret[:-lag]
                # y: ALT的后n-lag个数据点（跳过前lag个，与x对齐）
                y = alt_ret[lag:]
            elif lag == 0:
                # lag == 0：计算同时刻的相关系数
                # 注意：这里x和y的顺序可能看起来反了，但这是为了保持一致性
                x = alt_ret
                y = btc_ret
            
            # 确保x和y长度一致（取最小值）
            m = min(len(x), len(y))
            
            # 如果数据点太少（少于10个），无法计算可靠的相关系数
            if m < 10:
                corrs.append(-1)  # 使用-1作为无效标记
                continue
            
            # 计算皮尔逊相关系数
            # np.corrcoef返回2x2的相关系数矩阵：
            # [[corr(x,x), corr(x,y)],
            #  [corr(y,x), corr(y,y)]]
            # 我们只需要corr(x,y)，即[0, 1]位置的元素
            related_matrix = np.corrcoef(x[:m], y[:m])[0, 1]
            corrs.append(related_matrix)
        
        # 找到使相关系数最大的延迟值（最优延迟τ*）
        tau_star = lags[np.argmax(corrs)]
        
        # 获取最大相关系数值
        max_related_matrix = max(corrs)
        
        return tau_star, corrs, max_related_matrix
    
    @staticmethod
    def compute_spurious_te(btc_ret, alt_ret, delay, k=3):
        """
        计算虚假传递熵（Spurious Transfer Entropy）: T_{ALT → BTC}(τ)
        
        传递熵（TE）是一种信息论度量，用于量化一个时间序列对另一个时间序列的因果影响。
        虚假传递熵特指在已知BTC自身历史的情况下，山寨币对BTC的额外信息贡献。
        
        数学定义：
        TE(Y→X|X_past) = I(X_t; Y_{t-τ} | X_{t-1})
        其中：
        - X_t: BTC在t时刻的状态
        - X_{t-1}: BTC在t-1时刻的状态（历史信息）
        - Y_{t-τ}: 山寨币在t-τ时刻的状态（延迟τ）
        - I(·;·|·): 条件互信息
        
        算法步骤：
        1. 将连续收益率离散化为有限状态（分箱）
        2. 对齐时间序列：X_t, X_{t-1}, Y_{t-τ}
        3. 统计联合概率分布和边缘概率分布
        4. 使用条件互信息公式计算TE值
        
        Args:
            btc_ret (np.ndarray): BTC收益率数组
            alt_ret (np.ndarray): 山寨币收益率数组
            delay (int): 延迟值τ（单位：K线数量）
            k (int): 参数，当前未使用（保留用于未来扩展），默认3
        
        Returns:
            float: 传递熵值（单位：比特），值越大表示山寨币对BTC的信息贡献越大
                  如果计算失败或数据不足，返回0.0
        
        Note:
            - 需要至少200个数据点才能进行可靠计算
            - 使用3分位数分箱将连续值离散化
            - 如果对齐后的数据点少于100个，返回0.0
            - TE值可能为负，但实际意义中我们只关心正值（负值表示无信息传递）
        """
        # 采用离散近似的TE估计：TE ≈ I(X_t; Y_{t-τ} | X_{t-1})
        # 其中X_t是BTC当前状态，X_{t-1}是BTC历史状态，Y_{t-τ}是山寨币延迟状态
        
        # 数据量检查：需要足够的数据点才能进行可靠计算
        if len(btc_ret) < 200 or len(alt_ret) < 200 or delay < 0:
            return 0.0
        
        try:
            # 分箱数量：将连续收益率离散化为3个状态
            # 状态0: 低收益率（底部33%）
            # 状态1: 中收益率（中间33%）
            # 状态2: 高收益率（顶部33%）
            num_bins = 3
            
            # 分位数分箱函数（避免极端值影响）
            # 使用分位数而不是等间距分箱，可以更好地处理数据分布的不均匀性
            def discretize(x):
                """
                将连续收益率离散化为有限状态
                
                使用分位数将数据分为num_bins个区间，每个数据点被分配到对应的区间编号
                """
                # 计算分位数阈值：1/3和2/3分位数
                qs = np.nanquantile(x, [1/num_bins, 2/num_bins])
                # np.digitize返回每个值所属的区间编号（0, 1, 2）
                return np.digitize(x, qs)

            # 将BTC和山寨币的收益率离散化
            xb = discretize(btc_ret)  # BTC的离散状态序列
            yb = discretize(alt_ret)  # 山寨币的离散状态序列

            # 对齐时间序列：X_t, X_{t-1}, Y_{t-τ}
            # X_t: BTC在t时刻的状态（从1+delay开始，确保有历史数据）
            X_t = xb[1+delay:]
            # X_p (X_past): BTC在t-1时刻的状态（从delay开始，与X_t对齐）
            X_p = xb[delay:-1]
            # Y_p (Y_past): 山寨币在t-τ时刻的状态（去掉最后1+delay个，与X_t对齐）
            Y_p = yb[:-1-delay]

            # 确保三个序列长度一致
            n = min(len(X_t), len(X_p), len(Y_p))
            
            # 如果对齐后的数据点太少，无法进行可靠计算
            if n <= 100:
                return 0.0
            
            # 截取相同长度的序列
            X_t, X_p, Y_p = X_t[:n], X_p[:n], Y_p[:n]

            # 计算条件互信息 I(X_t; Y_p | X_p)
            # 需要统计以下概率分布：
            # - p(X_t, X_p, Y_p): 三维联合概率
            # - p(X_t, Y_p): 二维联合概率
            # - p(X_p, Y_p): 二维联合概率
            # - p(X_p): 一维边缘概率
            
            # 确定状态空间大小（离散状态的最大值+1）
            max_x = int(max(X_t.max(), X_p.max())) + 1
            max_y = int(Y_p.max()) + 1
            
            # 初始化概率分布数组
            p_xyz = np.zeros((max_x, max_x, max_y), dtype=np.float64)  # p(X_t, X_p, Y_p)
            p_xz = np.zeros((max_x, max_y), dtype=np.float64)          # p(X_t, Y_p)
            p_yz = np.zeros((max_x, max_y), dtype=np.float64)          # p(X_p, Y_p)
            p_z = np.zeros((max_x,), dtype=np.float64)                 # p(X_p)

            # 统计频率：遍历所有数据点，统计各种组合的出现次数
            for a, b, c in zip(X_t, X_p, Y_p):
                # a: X_t的状态值
                # b: X_p的状态值
                # c: Y_p的状态值
                p_xyz[a, b, c] += 1  # 三维联合频率
                p_xz[a, c] += 1      # (X_t, Y_p)联合频率
                p_yz[b, c] += 1      # (X_p, Y_p)联合频率
                p_z[b] += 1          # X_p边缘频率

            # 将频率转换为概率（归一化）
            p_xyz /= n
            p_xz /= n
            p_yz /= n
            p_z /= n

            # 计算传递熵：TE = Σ p(x_t, x_p, y_p) * log2(p(x_t|x_p) * p(y_p|x_p) / p(x_t, y_p|x_p))
            # 简化形式：TE = Σ p(x_t, x_p, y_p) * log2(p(x_t, x_p, y_p) * p(x_p) / (p(x_p, y_p) * p(x_t, y_p)))
            eps = 1e-12  # 小常数，避免log(0)
            te = 0.0
            
            # 遍历所有可能的状态组合
            it = np.nditer(p_xyz, flags=['multi_index'])
            while not it.finished:
                pabc = float(it[0])  # p(X_t=a, X_p=b, Y_p=c)
                
                if pabc > 0:  # 只计算非零概率
                    a, b, c = it.multi_index
                    
                    # 计算条件互信息的每一项
                    # 分子：p(x_t, x_p, y_p) * p(x_p)
                    num = pabc * p_z[b]
                    # 分母：p(x_p, y_p) * p(x_t, y_p)
                    den = (p_yz[b, c] * p_xz[a, c])
                    
                    # 只有当分子和分母都大于0时才计算（避免log(0)）
                    if num > 0 and den > 0:
                        # 累加条件互信息：p * log2(p * q / (r * s))
                        te += pabc * np.log2((num + eps) / (den + eps))
                
                it.iternext()

            # 返回非负的TE值（负值在信息论中表示无信息传递，实际中设为0）
            return max(float(te), 0.0)
        
        except Exception:
            # 如果计算过程中出现任何异常，返回0.0
            return 0.0
    
    @staticmethod
    def generate_signal(te_value, threshold=0.05):
        """
        根据传递熵值生成交易信号
        
        如果传递熵值超过阈值，说明山寨币对BTC存在显著的信息传递，
        可能存在延迟套利机会（山寨币价格滞后于BTC）。
        
        Args:
            te_value (float): 传递熵值（单位：比特）
            threshold (float): 信号触发阈值，默认0.05比特
                              当TE值超过此阈值时，认为存在显著的信息传递
        
        Returns:
            str: 交易信号字符串
                - "ENTER: 延迟套利信号触发！": TE值超过阈值，可能存在套利机会
                - "HOLD: 虚假 TE 不足": TE值低于阈值，无显著信息传递
        
        Note:
            - 阈值0.05是一个经验值，可根据实际回测结果调整
            - 更高的阈值意味着更严格的信号筛选，但可能错过一些机会
        """
        if te_value > threshold:
            return "ENTER: 延迟套利信号触发！"
        else:
            return "HOLD: 虚假 TE 不足"
    
    def one_coin_analysis(self, coin: str):
        """
        分析单个币种与BTC的相关性
        
        对指定的山寨币，在不同时间周期（timeframe）和数据周期（period）下，
        计算与BTC的最大相关系数和最优延迟。结果按相关系数从高到低排序。
        
        分析流程：
        1. 遍历所有时间周期和数据周期的组合
        2. 下载BTC和山寨币的历史数据
        3. 对齐时间序列（取交集）
        4. 计算最优延迟和最大相关系数
        5. 汇总所有组合的结果并排序输出
        
        Args:
            coin (str): 币种交易对符号，如 "KCS/USDT"、"ETH/USDT"
        
        Returns:
            None: 结果直接打印到控制台，不返回值
        
        Note:
            - BTC数据会被缓存，避免重复下载（因为BTC数据对所有币种都相同）
            - 使用.copy()避免修改缓存数据
            - 如果数据为空或缺少必要列，会跳过该组合并打印警告
            - 结果按最大相关系数降序排列，方便识别最相关的组合
        """
        # 存储所有组合的最大相关系数和对应的参数
        # key: 最大相关系数, value: (timeframe, period, tau_star)
        max_related_matrix_list = {}
        
        # 遍历所有时间周期和数据周期的组合
        for timeframe in self.timeframes:
            for period in self.periods:
                print(f"正在从 KuCoin 下载 {coin} 的 {timeframe}、{period} 数据...")
                
                # 缓存BTC数据，避免重复下载（因为BTC数据对所有币种都相同）
                cache_key = (timeframe, period)
                if cache_key not in self.btc_df_cache:
                    # 如果缓存中没有，下载并缓存BTC数据
                    self.btc_df_cache[cache_key] = self.download_ccxt_data(
                        self.btc_symbol, period=period, timeframe=timeframe
                    )
                
                # 必须使用.copy()，避免后续操作（如loc索引）修改缓存的数据
                # 如果直接赋值，修改btc_df会影响缓存，导致后续分析出错
                btc_df = self.btc_df_cache[cache_key].copy()
                
                # 下载山寨币数据
                alt_df = self.download_ccxt_data(coin, period=period, timeframe=timeframe)
                
                # 对齐时间序列：取BTC和山寨币时间索引的交集
                # 这样可以确保两个序列的时间点完全一致
                common_idx = btc_df.index.intersection(alt_df.index)
                btc_df, alt_df = btc_df.loc[common_idx], alt_df.loc[common_idx]
                
                # 检查数据是否为空或缺少必要的列
                if len(btc_df) == 0 or len(alt_df) == 0:
                    print(f"  警告: {coin} 的 {timeframe}/{period} 数据为空，跳过...")
                    continue
                if 'return' not in btc_df.columns or 'return' not in alt_df.columns:
                    print(f"  警告: {coin} 的 {timeframe}/{period} 数据缺少 'return' 列，跳过...")
                    continue
                
                # 提取收益率数组
                btc_ret = btc_df['return'].values
                alt_ret = alt_df['return'].values
                
                # 找最优延迟（单位：K线数量，即bars）
                # tau_star: 最优延迟值
                # corr_curve: 所有延迟下的相关系数曲线（当前未使用）
                # max_related_matrix: 最大相关系数
                tau_star, corr_curve, max_related_matrix = self.find_optimal_delay(btc_ret, alt_ret)
                print(f'timeframe: {timeframe}, period: {period}, tau_star: {tau_star}, max_related_matrix: {max_related_matrix}')

                # 存储结果：使用最大相关系数作为key（如果重复，后面的会覆盖前面的）
                max_related_matrix_list[max_related_matrix] = (timeframe, period, tau_star)

        # 按最大相关系数降序排序（从高到低）
        # 这样可以快速识别哪些组合下相关性最强
        max_related_matrix_list = sorted(max_related_matrix_list.items(), key=lambda x: x[0], reverse=True)
        
        # 转换为pandas DataFrame，方便格式化输出
        df_results = pd.DataFrame([
            {
                '最大相关系数': max_corr,
                '时间周期': timeframe,
                '数据周期': period,
                '最优延迟': tau_star
            }
            for max_corr, (timeframe, period, tau_star) in max_related_matrix_list
        ])
        
        # 格式化输出结果
        print("\n" + "="*60)
        print(f"{coin}相关系数分析结果")
        print("="*60)
        print(df_results.to_string(index=False))
        print("="*60)
    
    def run(self, quote_currency="USDT"):
        """
        主运行方法，分析所有币种与BTC的相关性
        
        遍历交易所中所有以指定计价货币（如USDT）计价的交易对，
        对每个交易对执行相关性分析。
        
        流程：
        1. 加载交易所的所有交易对信息
        2. 筛选出以指定计价货币计价的交易对
        3. 对每个交易对调用one_coin_analysis进行分析
        4. 每次分析后等待1秒，避免API请求过于频繁
        
        Args:
            quote_currency (str): 计价货币，默认为 "USDT"
                                 只分析以该货币计价的交易对，如 "BTC/USDT"、"ETH/USDT"
        
        Returns:
            None: 结果直接打印到控制台
        
        Note:
            - 会分析交易所中所有符合条件的交易对，可能需要较长时间
            - 每次分析后等待1秒，避免触发交易所API限流
            - 可以通过修改quote_currency参数分析其他计价货币的交易对
        """
        # 加载交易所的所有交易对信息
        # 返回一个字典，key是交易对符号（如"BTC/USDT"），value是交易对详细信息
        all_coins = self.exchange.load_markets()
        
        # 遍历所有交易对
        for coin in all_coins:
            coin_item = all_coins[coin]
            
            # 只分析以指定计价货币计价的交易对
            # 例如：如果quote_currency="USDT"，只分析"BTC/USDT"、"ETH/USDT"等
            if coin_item['quote'] != quote_currency:
                continue
            
            # 对当前交易对进行相关性分析
            self.one_coin_analysis(coin)
            
            # 等待1秒，避免API请求过于频繁（防止触发限流）
            time.sleep(1)


# ================== 主程序入口 ==================
if __name__ == "__main__":
    """
    主程序入口
    
    当直接运行此脚本时，会执行以下操作：
    1. 创建SpuriousTEAnalyzer分析器实例（使用默认参数）
    2. 调用run()方法，分析所有USDT计价的交易对与BTC的相关性
    
    使用示例：
        # 直接运行（使用默认参数）
        python corrcoef_get.py
        
        # 或在代码中自定义参数
        analyzer = SpuriousTEAnalyzer(
            exchange_name="binance",  # 使用币安交易所
            default_timeframes=["1m", "5m", "15m"],  # 自定义时间周期
            default_periods=["7d", "30d"]  # 自定义数据周期
        )
        analyzer.run(quote_currency="USDT")  # 分析USDT交易对
    """
    # 创建分析器实例（使用默认参数：KuCoin交易所，1m和5m时间周期，1d/7d/30d/60d数据周期）
    analyzer = SpuriousTEAnalyzer()
    
    # 运行分析：分析所有USDT计价的交易对
    analyzer.run()
