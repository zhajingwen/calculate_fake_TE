# Calculate Fake Transfer Entropy

一个用于计算和分析虚假转移熵（Spurious Transfer Entropy）的 Python 工具，用于检测加密货币市场中的虚假因果关系和延迟套利机会。

## 📌 项目简介

本项目通过分析山寨币和 BTC 之间的虚假转移熵和相关系数，识别由共同驱动因素引起的虚假因果关系，并生成交易信号。支持单个币种分析和批量币种分析。

## 🎯 主要功能

1. **数据下载**: 使用 CCXT 从 KuCoin 获取加密货币历史 K 线数据，支持多种时间周期（1m、5m）和数据周期（1d、7d、30d、60d）
2. **批量分析**: 支持分析所有 USDT 交易对与 BTC 的相关系数
3. **数据缓存**: 智能缓存 BTC 数据，避免重复下载，提高分析效率
4. **最优延迟计算**: 通过交叉相关分析找到最优延迟 τ*
5. **虚假转移熵计算**: 计算 T_{ALT → BTC}(τ)
6. **异常检测**: 识别相关系数异常的币种（高相关性但极低相关性并存的情况）
7. **信号生成**: 根据 TE 值生成交易信号
8. **可视化**: 生成分析图表和结果

## 🚀 安装方法

本项目使用 `uv` 进行依赖管理，需要 Python 3.12+ 和 uv 工具。

```bash
# 使用 uv 安装
uv sync
```

或者使用 pip：

```bash
pip install matplotlib numpy pandas pyinform ccxt retry seaborn
```

## 📦 依赖项

- **ccxt**: 从交易所（KuCoin）获取行情数据
- **numpy**: 数值计算
- **pandas**: 数据处理
- **pyinform**: 信息论计算（转移熵）
- **matplotlib**: 数据可视化
- **seaborn**: 高级数据可视化
- **retry**: 自动重试机制，提高数据下载稳定性

## 💻 使用方法

### 1. 单币种分析（main.py）

运行主程序分析 KCS 和 BTC：

```bash
python main.py
```

程序流程：
1. 下载 BTC/USDT 和 KCS/USDT 的最近 60 天、5 分钟 K 线
2. 计算最优延迟 τ*
3. 计算虚假转移熵
4. 生成交易信号
5. 保存可视化图表到 `spurious_te_kcs_btc.png`

### 2. 批量相关系数分析（corrcoef_get.py）

分析所有 USDT 交易对与 BTC 的相关系数：

```bash
python corrcoef_get.py
```

功能特点：
- 自动遍历所有 USDT 交易对
- 支持多个时间周期（1m、5m）和数据周期（1d、7d、30d、60d）的组合分析
- 对每个币种，输出按相关系数排序的结果表格
- BTC 数据智能缓存，避免重复下载

### 3. 异常数据检测（corrcoef_abnormal.py）

检测并输出相关系数异常的币种：

```bash
python corrcoef_abnormal.py
```

异常检测规则：
- 第一行最大相关系数 > 0.4，且最后一行最大相关系数 < 0.05
- 第一行最大相关系数 < 0.11，且最后一行最大相关系数 < 0.05

### 自定义分析参数

可以通过修改代码中的参数来自定义分析：

```python
analyzer = SpuriousTEAnalyzer(
    exchange_name="kucoin",           # 交易所名称
    timeout=30000,                     # 请求超时时间（毫秒）
    default_timeframes=["1m", "5m"],   # 时间周期列表
    default_periods=["1d", "7d", "30d", "60d"]  # 数据周期列表
)
analyzer.run(quote_currency="USDT")    # 分析所有 USDT 交易对
```

## 📖 API 文档

### 核心类：SpuriousTEAnalyzer

#### `SpuriousTEAnalyzer(exchange_name="kucoin", timeout=30000, default_timeframes=None, default_periods=None)`
创建一个虚假转移熵分析器实例

**参数:**
- `exchange_name`: 交易所名称，默认 "kucoin"
- `timeout`: 请求超时时间（毫秒），默认 30000
- `default_timeframes`: 默认时间周期列表，默认 `["1m", "5m"]`
- `default_periods`: 默认数据周期列表，默认 `["1d", "7d", "30d", "60d"]`

**方法:**

#### `download_ccxt_data(symbol, period, timeframe)`
从 KuCoin 通过 CCXT 下载并处理加密货币数据（带自动重试）

**参数:**
- `symbol`: 加密货币交易对，如 "BTC/USDT"
- `period`: 数据时间范围，如 "60d"
- `timeframe`: 数据间隔，如 "5m"

**返回:**
- DataFrame，包含价格数据、收益率和 USD 交易量

#### `find_optimal_delay(btc_ret, alt_ret, max_lag=48)`
通过交叉相关分析找到最优延迟

**参数:**
- `btc_ret`: BTC 收益率数组
- `alt_ret`: Altcoin 收益率数组
- `max_lag`: 最大延迟范围（单位：bars），默认 48

**返回:**
- `tau_star`: 最优延迟值
- `corrs`: 所有延迟的相关系数数组
- `max_related_matrix`: 最大相关系数值

#### `compute_spurious_te(btc_ret, alt_ret, delay, k=3)`
计算虚假转移熵 T_{ALT → BTC}(τ)

**参数:**
- `btc_ret`: BTC 收益率数组
- `alt_ret`: Altcoin 收益率数组
- `delay`: 延迟值
- `k`: 嵌入维度，默认 3

**返回:**
- 虚假转移熵值（bits）

#### `generate_signal(te_value, threshold=0.05)`
根据 TE 值生成交易信号

**参数:**
- `te_value`: 转移熵值
- `threshold`: 信号阈值，默认 0.05

**返回:**
- 信号字符串："ENTER: 延迟套利信号触发！" 或 "HOLD: 虚假 TE 不足"

#### `one_coin_analysis(coin)`
分析单个币种与 BTC 的相关系数

**参数:**
- `coin`: 币种交易对，如 "KCS/USDT"

**功能:**
- 遍历所有时间周期和数据周期的组合
- 计算每个组合的最优延迟和最大相关系数
- 输出按相关系数排序的结果表格

#### `run(quote_currency="USDT")`
批量分析所有指定计价货币的交易对

**参数:**
- `quote_currency`: 计价货币，默认 "USDT"

**功能:**
- 自动遍历交易所中所有指定计价货币的交易对
- 对每个币种调用 `one_coin_analysis`
- 币种之间自动延迟，避免 API 限流

### 独立函数（main.py）

这些函数也可以独立使用，功能与类方法类似，但仅支持单币种分析。

## 📊 输出结果

### main.py 输出：
- 最优延迟 τ* 值（单位：bars，括号内显示分钟）
- 虚假转移熵 T_{KCS→BTC}(τ*)
- 交易信号
- 可视化图表 `spurious_te_kcs_btc.png`
  - 交叉相关 vs 延迟
  - 虚假转移熵 vs 延迟

### corrcoef_get.py 输出：
- 每个币种的相关系数分析表格，包含：
  - 最大相关系数（按降序排列）
  - 对应的时间周期（timeframe）
  - 对应的数据周期（period）
  - 最优延迟（tau_star）

### corrcoef_abnormal.py 输出：
- 仅输出符合异常检测规则的币种分析结果
- 帮助识别可能存在特殊市场行为的币种

## ⚠️ 注意事项

- **数据可靠性**: 依赖交易所（KuCoin）与 CCXT 的数据质量与限频
- **计算复杂度**: 批量分析所有币种可能需要较长时间
- **API 限流**: 程序已内置重试机制和延迟策略，但仍需注意交易所 API 限制
- **信号有效性**: 仅供参考，不构成投资建议
- **BTC 数据缓存**: 批量分析时会自动缓存 BTC 数据，提高效率

## 🔧 技术特性

- **自动重试**: 使用 `@retry` 装饰器实现数据下载的自动重试（最多 10 次，指数退避）
- **数据缓存**: BTC 数据按 `(timeframe, period)` 组合缓存，避免重复下载
- **错误处理**: 完善的异常处理和数据验证机制
- **灵活配置**: 支持自定义时间周期、数据周期和交易所

## 🔄 未来改进

1. ✅ 支持更多加密货币对的分析（已实现）
2. 添加回测功能验证信号有效性
3. 优化算法提高计算速度
4. ✅ 增加错误处理和交易所 API 限流处理（已实现）
5. 支持更多交易所
6. 添加数据持久化功能

## 📚 参考资料

更多关于 Transfer Entropy 的理论和应用，请参考信息论和因果推断相关文献。本项目用于学术研究和教学目的。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！

## 📄 许可证

本项目使用 MIT 许可证
