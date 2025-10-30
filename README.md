# Calculate Fake Transfer Entropy

一个用于计算和分析虚假转移熵（Spurious Transfer Entropy）的 Python 工具，用于检测加密货币市场中的虚假因果关系和延迟套利机会。

## 📌 项目简介

本项目通过分析 KCS 和 BTC 之间的虚假转移熵，识别由共同驱动因素引起的虚假因果关系，并生成交易信号。

## 🎯 主要功能

1. **数据下载**: 使用 CCXT 从 KuCoin 获取 5m、60d 的加密货币历史 K 线（`BTC/USDT`、`KCS/USDT`）
2. **最优延迟计算**: 通过交叉相关分析找到最优延迟 τ*
3. **虚假转移熵计算**: 使用 PyInform 库计算 T_{ALT → BTC}(τ)
4. **信号生成**: 根据 TE 值生成交易信号
5. **可视化**: 生成分析图表和结果

## 🚀 安装方法

本项目使用 `uv` 进行依赖管理，需要 Python 3.12+ 和 uv 工具。

```bash
# 使用 uv 安装
uv sync
```

或者使用 pip：

```bash
pip install matplotlib numpy pandas pyinform ccxt
```

## 📦 依赖项

- **ccxt**: 从交易所（KuCoin）获取行情数据
- **numpy**: 数值计算
- **pandas**: 数据处理
- **pyinform**: 信息论计算（转移熵）
- **matplotlib**: 数据可视化

## 💻 使用方法

运行主程序：

```bash
python main.py
```

程序流程：
1. 下载 BTC/USDT 和 KCS/USDT 的最近 60 天、5 分钟 K 线
2. 计算最优延迟 τ*
3. 计算虚假转移熵
4. 生成交易信号
5. 保存可视化图表到 `spurious_te_shib_btc.png`

## 📖 API 文档

### 核心函数

#### `download_ccxt_data(symbol, period="60d", timeframe="5m")`
从 KuCoin 通过 CCXT 下载并处理加密货币数据

**参数:**
- `symbol`: 加密货币交易对，如 "BTC/USDT"
- `period`: 数据时间范围，默认 "60d"
- `timeframe`: 数据间隔，默认 "5m"

**返回:**
- DataFrame，包含价格数据和收益率（USD）

#### `find_optimal_delay(btc_ret, alt_ret, max_lag=48)`
通过交叉相关分析找到最优延迟

**参数:**
- `btc_ret`: BTC 收益率数组
- `alt_ret`: Altcoin 收益率数组
- `max_lag`: 最大延迟范围（单位：5m bars），默认 48

**返回:**
- `tau_star`: 最优延迟值
- `corrs`: 所有延迟的相关系数数组

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
- 信号字符串："ENTER" 或 "HOLD"

## 📊 输出结果

程序会生成：
- 最优延迟 τ* 值（单位：5m bars，括号内显示分钟）
- 虚假转移熵 T_{KCS→BTC}(τ*)
- 交易信号
- 可视化图表
  - 交叉相关 vs 延迟
  - 虚假转移熵 vs 延迟

## ⚠️ 注意事项

- **数据可靠性**: 依赖交易所（KuCoin）与 CCXT 的数据质量与限频
- **计算复杂度**: 转移熵计算可能需要较长时间
- **信号有效性**: 仅供参考，不构成投资建议

## 🔄 未来改进

1. 支持更多加密货币对的分析
2. 添加回测功能验证信号有效性
3. 优化算法提高计算速度
4. 增加错误处理和交易所 API 限流处理

## 📚 参考资料

更多关于 Transfer Entropy 的理论和应用，请参考信息论和因果推断相关文献。本项目用于学术研究和教学目的。

## 🤝 贡献

欢迎提交 Issue 或 Pull Request！

## 📄 许可证

本项目使用 MIT 许可证
