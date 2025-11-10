import os
import time
import requests
import pandas as pd
import hmac
import hashlib
from datetime import datetime
from threading import Thread, Event
import talib
from dotenv import load_dotenv
import warnings
import numpy as np
import math

warnings.filterwarnings('ignore')

# ==================== 加载环境变量 ====================
load_dotenv()
HORUS_API_KEY = os.getenv('HORUS_API_KEY')  

# ==================== 全局配置 ====================
TRANSACTION_FEE = 0.001
EMERGENCY_DROP_THRESHOLD = 0.15
EMERGENCY_SELL_DISCOUNT = 0.05

ALL_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'XRP/USD',
    'ADA/USD', 'DOGE/USD', 'TRX/USD', 'AVAX/USD', 'LINK/USD',
    'DOT/USD', 'TON/USD', 'LTC/USD', 'BCH/USD'
]
MAINSTREAM_COINS = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'XRP/USD']
ALT_COINS = [s for s in ALL_SYMBOLS if s not in MAINSTREAM_COINS]

stop_event = Event()

INTERVAL_SECONDS = {
    '15m': 15 * 60,
    '1h': 60 * 60,
    '1d': 24 * 60 * 60,
}

MOMENTUM_LOOKBACK_MINUTES = 15 * 50  # 确保至少50根K线以计算动量指标

# ==================== Roostoo 客户端 ====================
class RoostooClient:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv('ROOSTOO_API_KEY')
        self.secret_key = secret_key or os.getenv('ROOSTOO_SECRET_KEY')
        self.base_url = "https://mock-api.roostoo.com"
        self.session = requests.Session()

    def _get_timestamp(self):
        return str(int(time.time() * 1000))

    def _get_signed_headers(self, payload: dict = {}):
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers = {'RST-API-KEY': self.api_key, 'MSG-SIGNATURE': signature}
        return headers, payload, total_params

    def get_balance(self):
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            r = self.session.get(url, headers=headers, params=payload, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"获取余额失败: {e}")
            return {"available_USD": 50000.0}

    def get_ticker(self, pair: str):
        """获取实时最新价格"""
        url = f"{self.base_url}/v3/ticker"
        payload = {'symbol': pair.replace('/', '')}
        headers, _, _ = self._get_signed_headers(payload)
        try:
            r = self.session.get(url, headers=headers, params=payload, timeout=10)
            r.raise_for_status()
            data = r.json()
            return float(data.get('price', 0.0))
        except Exception as e:
            print(f"获取 {pair} 实时价格失败: {e}")
            return None

    def place_order(self, pair: str, side: str, quantity: float, price: float = None, order_type: str = "MARKET"):
        url = f"{self.base_url}/v3/place_order"
        payload = {'pair': pair, 'side': side.upper(), 'type': order_type.upper(), 'quantity': str(quantity)}
        if price and order_type == "LIMIT":
            payload['price'] = str(price)
        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'
        try:
            r = self.session.post(url, headers=headers, data=total_params, timeout=10)
            r.raise_for_status()
            print(f"手续费0.1%已扣除 | 下单成功: {side} {quantity:.6f} {pair} @ {price or '市价'}")
            return r.json()
        except Exception as e:
            print(f"下单失败: {e}")
            return None

# ==================== Horus 数据客户端 ====================
class HorusDataClient:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('HORUS_API_KEY')
        self.base_url = "https://api-horus.com"
        self.session = requests.Session()

    def _parse_json_to_dataframe(self, data, sym: str) -> pd.DataFrame:
        """
        通用 JSON 解析函数，支持多种 API 返回格式：
        1. 列表格式: [{"timestamp": ..., "price": ...}, ...]
        2. 对象包含数组: {"data": [...], "prices": [...]} 
        3. 嵌套结构: {"result": {"BTC": [...]}}
        """
        if isinstance(data, list):
            # 格式1: 直接是数组
            if len(data) == 0:
                return None
            return pd.DataFrame(data)
        
        elif isinstance(data, dict):
            # 格式2/3: 字典 - 需要找到包含数据的键
            
            # 尝试常见的数据键名
            for key in ['data', 'prices', 'result', 'records', 'items', 'values']:
                if key in data:
                    nested = data[key]
                    if isinstance(nested, list) and len(nested) > 0:
                        return pd.DataFrame(nested)
                    elif isinstance(nested, dict):
                        # 如果是嵌套字典，尝试提取该币种的数据
                        clean_sym = sym.split('/')[0]  # 'BTC/USD' -> 'BTC'
                        if clean_sym in nested:
                            sym_data = nested[clean_sym]
                            if isinstance(sym_data, list):
                                return pd.DataFrame(sym_data)
                            elif isinstance(sym_data, dict):
                                return pd.DataFrame([sym_data])
            
            # 如果上述都没找到，尝试把整个字典当作一行数据
            return pd.DataFrame([data])
        
        return None

    def fetch_price_series(self, symbols: list, minutes: int = 15, interval: str = '15m') -> dict:
        """从 Horus 的 /market/price 获取价格时间序列"""
        data_dict = {}
        interval_seconds = INTERVAL_SECONDS.get(interval)
        if interval_seconds is None:
            raise ValueError(f"Unsupported interval: {interval}")

        now = int(time.time())
        # 将 end_ts 对齐至最近的完整 K 线结束时间（end 为独占区间）
        end_ts = (now // interval_seconds) * interval_seconds

        window_seconds = max(minutes * 60, interval_seconds)
        num_intervals = max(1, math.ceil(window_seconds / interval_seconds))
        start_ts = end_ts - num_intervals * interval_seconds

        headers = {"X-API-KEY": self.api_key}

        for sym in symbols:
            clean_symbol = sym.replace('/USD', '')
            url = f"{self.base_url}/market/price"
            params = {
                "asset": clean_symbol,
                "interval": interval,
                "start": start_ts,
                "end": end_ts,
                "format": "json"
            }
            try:
                r = self.session.get(url, headers=headers, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                # 检查数据格式
                if not data:
                    print(f"Horus 获取 {sym} 失败: 返回数据为空")
                    continue

                # 使用通用解析函数
                df = self._parse_json_to_dataframe(data, sym)

                if df is None or df.empty:
                    print(f"Horus 获取 {sym} 失败: 无法解析数据结构。原始数据: {type(data).__name__}")
                    continue

                # 检查必需列是否存在
                if 'timestamp' not in df.columns:
                    print(f"Horus 获取 {sym} 失败: 返回数据中没有 'timestamp' 列。可用列: {list(df.columns)}")
                    continue
                
                if 'price' not in df.columns:
                    print(f"Horus 获取 {sym} 失败: 返回数据中没有 'price' 列。可用列: {list(df.columns)}")
                    continue
                
                # 处理 timestamp（可能是秒级或毫秒级）
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    except:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                df.set_index('timestamp', inplace=True)
                data_dict[sym] = df['price'].astype(float)
            except requests.exceptions.RequestException as e:
                print(f"Horus 获取 {sym} 失败 (网络错误): {e}")
            except KeyError as e:
                print(f"Horus 获取 {sym} 失败 (缺少字段): {e}")
            except Exception as e:
                print(f"Horus 获取 {sym} 失败 (未知错误): {type(e).__name__}: {e}")
        
        return data_dict

    def fetch_fear_greed_index(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=10)
            return r.json()['data'][0]['value_classification']
        except Exception:
            return "Unknown"

# ==================== 市场分析器 ====================
class MarketAnalyzer:
    def analyze(self, price_series: dict) -> dict:
        results = {}
        for symbol, prices in price_series.items():
            df = pd.DataFrame({'price': prices})
            df['close'] = df['price']
            df['volume'] = df['price'].pct_change().abs().fillna(0) * 1000.0
            df['returns'] = df['price'].pct_change()
            df['ma_5'] = df['price'].rolling(5).mean()
            df['ma_10'] = df['price'].rolling(10).mean()
            df['ma_15'] = df['price'].rolling(15).mean()
            df['ma_200'] = df['price'].rolling(200).mean()
            df['rsi'] = talib.RSI(df['price'], timeperiod=14)
            df['volatility'] = df['returns'].rolling(20).std()

            if len(df) >= 10:
                drop = (df['price'].iloc[-1] - df['price'].iloc[-10]) / df['price'].iloc[-10]
                df['emergency_sell'] = drop <= -EMERGENCY_DROP_THRESHOLD
            else:
                df['emergency_sell'] = False

            if symbol in MAINSTREAM_COINS:
                df = self._analyze_mainstream(df)
            else:
                df = self._analyze_alt(df)
            results[symbol] = df
        return results

    def _analyze_mainstream(self, df):
        df['upward_trend'] = df['price'] > df['ma_200']
        vol = df['volatility'].mean()
        grid_size = df['price'].mean() * vol * 2
        df['grid_buy']  = (df['price'] < df['ma_10'] - grid_size) & df['upward_trend']
        df['grid_sell'] = df['price'] > df['ma_10'] + grid_size
        df['reversion_buy'] = (df['price'] < df['ma_15']) & (df['rsi'] < 35)
        return df

    def _analyze_alt(self, df):
        df['whale_signal'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.2, 0.6, 0.2])
        df['high_turnover'] = np.random.random(len(df)) > 0.9
        return df

# ==================== 策略逻辑 ====================
class TradingStrategy:
    def __init__(self, roostoo_client: RoostooClient):
        self.client = roostoo_client
        self.usd_balance = None
        self.positions = {}
        self.last_account_value = None
        self.trade_history = []

    def generate_signals(self, analysis: dict, fear_greed: str) -> dict:
        signals = {}
        risk_multiplier = self._risk_multiplier_from_fng(fear_greed)

        for pair, df in analysis.items():
            if df.empty:
                signals[pair] = {
                    'pair': pair,
                    'price': None,
                    'action': 'hold',
                    'size': 0.0,
                    'reason': ['数据不足，保持观望'],
                    'fear_greed': fear_greed,
                }
                continue

            latest = df.iloc[-1]
            reference_price = float(latest.get('close', latest.get('price', 0.0)))
            signal = {
                'pair': pair,
                'price': reference_price,
                'action': 'hold',
                'size': 0.0,
                'reason': [],
                'fear_greed': fear_greed,
            }

            if 'close' not in df.columns:
                signal['reason'].append('缺少 close 数据，保持观望')
                signals[pair] = signal
                continue

            momentum_df = df[['close']].copy()
            if 'volume' in df.columns:
                momentum_df['volume'] = df['volume']

            long_signal, short_signal = self.momentum_signal(momentum_df)

            if long_signal and not short_signal:
                buy_size = min(max(0.01, 0.05 * risk_multiplier), 0.3)
                signal.update({'action': 'buy', 'size': buy_size})
                signal['reason'].append(f"动量做多信号 (F&G: {fear_greed})")
            elif short_signal and not long_signal:
                sell_size = min(max(0.05, 0.25 / max(risk_multiplier, 0.5)), 1.0)
                signal.update({'action': 'sell', 'size': sell_size})
                signal['reason'].append(f"动量做空信号 (F&G: {fear_greed})")
            else:
                signal['reason'].append("动量指标无操作信号")

            signals[pair] = signal

        return signals

    def momentum_signal(self, data: pd.DataFrame):
        if 'close' not in data.columns:
            return False, False

        close = data['close'].dropna()
        if close.shape[0] < 50:
            return False, False

        rsi = pd.Series(talib.RSI(close.values, timeperiod=14), index=close.index)
        macd, macd_signal, _ = talib.MACD(close.values, fastperiod=12, slowperiod=26, signalperiod=9)
        macd = pd.Series(macd, index=close.index)
        macd_signal = pd.Series(macd_signal, index=close.index)
        sma_20 = pd.Series(talib.SMA(close.values, timeperiod=20), index=close.index)
        sma_50 = pd.Series(talib.SMA(close.values, timeperiod=50), index=close.index)

        volume_filter_long = True
        volume_filter_short = True
        if 'volume' in data.columns:
            volume = data['volume'].reindex(close.index)
            if volume.notna().sum() >= 20:
                volume = volume.fillna(method='ffill').fillna(0)
                volume_sma_vals = talib.SMA(volume.values, timeperiod=20)
                volume_sma = pd.Series(volume_sma_vals, index=close.index)
                vol_sma_last = volume_sma.iloc[-1]
                current_volume = volume.iloc[-1]
                if not np.isnan(vol_sma_last) and vol_sma_last > 0:
                    volume_filter_long = current_volume > vol_sma_last * 1.5
                    volume_filter_short = current_volume > vol_sma_last * 1.5

        rsi_last = rsi.iloc[-1]
        macd_last = macd.iloc[-1]
        macd_signal_last = macd_signal.iloc[-1]
        sma20_last = sma_20.iloc[-1]
        sma50_last = sma_50.iloc[-1]
        price_last = close.iloc[-1]

        if any(np.isnan(val) for val in [rsi_last, macd_last, macd_signal_last, sma20_last, sma50_last, price_last]):
            return False, False

        long_signal = (
            50 < rsi_last < 70 and
            macd_last > macd_signal_last and
            price_last > sma20_last and
            sma20_last > sma50_last and
            volume_filter_long
        )

        short_signal = (
            30 < rsi_last < 50 and
            macd_last < macd_signal_last and
            price_last < sma20_last and
            sma20_last < sma50_last and
            volume_filter_short
        )

        return long_signal, short_signal

    def _risk_multiplier_from_fng(self, classification: str) -> float:
        if not classification:
            return 1.0
        normalized = classification.strip().lower()
        mapping = {
            'extreme greed': 1.3,
            'greed': 1.1,
            'neutral': 1.0,
            'fear': 0.85,
            'extreme fear': 0.7,
        }
        if normalized in mapping:
            return mapping[normalized]

        zh_mapping = {
            '极度贪婪': 1.3,
            '贪婪': 1.1,
            '中性': 1.0,
            '恐惧': 0.85,
            '极度恐惧': 0.7,
        }
        return zh_mapping.get(classification.strip(), 1.0)

    def execute(self, signals: dict):
        if self.usd_balance is None:
            balance = self.client.get_balance()
            self.usd_balance = float(balance.get('available_USD', 50000))
            self.last_account_value = self.usd_balance

        market_prices = {}

        for pair, sig in signals.items():
            if sig['action'] == 'hold' or sig['size'] <= 0:
                reasons = ", ".join(sig.get('reason', ['信号保持']))
                print(f"{pair} | HOLD | 原因: {reasons}")
                continue

            ticker_price = self.client.get_ticker(pair)
            price_candidate = ticker_price if ticker_price is not None else sig['price']
            try:
                price = float(price_candidate)
            except (TypeError, ValueError):
                price = float('nan')
            if not np.isfinite(price) or price <= 0:
                print(f"跳过 {pair}: 无有效价格")
                continue

            market_prices[pair] = price
            action = sig['action']
            size = sig['size']

            if action == 'buy':
                usd_to_use = self.usd_balance * min(size, 1.0)
                if usd_to_use <= 0:
                    continue
                qty = usd_to_use / price
                estimated_cost = qty * price * (1 + TRANSACTION_FEE)
                if estimated_cost > self.usd_balance:
                    qty = (self.usd_balance / (1 + TRANSACTION_FEE)) / price
                    estimated_cost = qty * price * (1 + TRANSACTION_FEE)
                if qty <= 0:
                    continue
                response = self.client.place_order(pair, action, qty, price=price)
                if not response:
                    continue
                self.usd_balance -= estimated_cost
                self.positions[pair] = self.positions.get(pair, 0.0) + qty
                trade_value = -estimated_cost

            elif action == 'sell':
                held_qty = self.positions.get(pair, 0.0)
                if held_qty <= 0:
                    print(f"跳过 {pair}: 无持仓可卖出")
                    continue
                qty = held_qty * min(size, 1.0)
                if qty <= 0:
                    continue
                response = self.client.place_order(pair, action, qty, price=price)
                if not response:
                    continue
                proceeds = qty * price * (1 - TRANSACTION_FEE)
                self.usd_balance += proceeds
                remaining = held_qty - qty
                if remaining <= 0:
                    self.positions.pop(pair, None)
                else:
                    self.positions[pair] = remaining
                trade_value = proceeds

            else:
                continue

            account_value = self._compute_account_value(market_prices)
            roi = 0.0
            if self.last_account_value and self.last_account_value > 0:
                roi = (account_value - self.last_account_value) / self.last_account_value
            self.last_account_value = account_value

            self._record_trade(
                pair,
                action,
                qty,
                price,
                trade_value,
                account_value,
                roi,
                sig.get('reason', []),
                sig.get('fear_greed'),
            )

    def _compute_account_value(self, market_prices: dict) -> float:
        total = self.usd_balance
        for pair, qty in self.positions.items():
            if qty <= 0:
                continue
            price = market_prices.get(pair)
            if price is None:
                price = self.client.get_ticker(pair)
                if price is None or price <= 0:
                    continue
                market_prices[pair] = price
            total += qty * price
        return total

    def _record_trade(self, pair: str, action: str, qty: float, price: float, trade_value: float,
                      account_value: float, roi: float, reasons: list, fear_greed: str = None):
        trade_info = {
            'timestamp': datetime.utcnow().isoformat(),
            'pair': pair,
            'action': action,
            'quantity': qty,
            'price': price,
            'trade_value': trade_value,
            'account_value': account_value,
            'roi': roi,
            'reasons': reasons,
            'fear_greed': fear_greed,
        }
        self.trade_history.append(trade_info)
        fg_text = f" | F&G: {fear_greed}" if fear_greed else ""
        print(
            f"{pair} | {action.upper()} {qty:.6f} @ {price:.2f} | 账户净值: {account_value:.2f} USD | 单次ROI: {roi:.4%}{fg_text}"
        )

# ==================== 系统主循环 ====================
def run_once():
    print(f"\n{'='*60}")
    print(f"3q1 quant trading bot executing - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    horus = HorusDataClient(HORUS_API_KEY)
    roostoo = RoostooClient()
    analyzer = MarketAnalyzer()
    strategy = TradingStrategy(roostoo)
    price_data = horus.fetch_price_series(ALL_SYMBOLS, minutes=MOMENTUM_LOOKBACK_MINUTES, interval='15m')
    analysis = analyzer.analyze(price_data)
    fear_greed = horus.fetch_fear_greed_index()
    signals = strategy.generate_signals(analysis, fear_greed)
    strategy.execute(signals)
    print("恐惧贪婪指数:", fear_greed)

def periodic_task():
    while not stop_event.is_set():
        try:
            run_once()
        except Exception as e:
            print("运行异常:", e)
        stop_event.wait(60)  # 改这里！60秒=每分钟执行一次 → 改成3600秒=每小时

if __name__ == "__main__":
    print("3q1 quant trading bot launching...")
    print("按 Ctrl+C 停止")
    thread = Thread(target=periodic_task, daemon=True)
    thread.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止系统...")
        stop_event.set()
        thread.join()
        print("系统已安全停止")
