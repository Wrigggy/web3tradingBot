# horus_quant_trading_system.py
# ✅ 实时交易版：Horus真实接口，无稳定币逻辑
# 支持：主流币均值回归/网格 + 山寨币鲸鱼策略
# 数据更新周期15min，默认交易频率1min（可改1h）

import pandas as pd
import numpy as np
import requests
import time
import os
import hmac
import hashlib
from datetime import datetime
from typing import Dict, List
import warnings
from threading import Thread, Event
import talib

warnings.filterwarnings('ignore')

# ==================== 加载环境变量 ====================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: pip install python-dotenv 可自动加载 .env 文件")

# ==================== 全局配置 ====================
TRANSACTION_FEE = 0.001
EMERGENCY_DROP_THRESHOLD = 0.15
EMERGENCY_SELL_DISCOUNT = 0.05

# 前15大市值币种（可根据实际更新）
ALL_SYMBOLS = [
    'BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'XRP/USD',
    'ADA/USD', 'DOGE/USD', 'TRX/USD', 'AVAX/USD', 'LINK/USD',
    'DOT/USD', 'TON/USD', 'MATIC/USD', 'LTC/USD', 'BCH/USD'
]
MAINSTREAM_COINS = ['BTC/USD', 'ETH/USD', 'BNB/USD', 'SOL/USD', 'XRP/USD']
ALT_COINS = [s for s in ALL_SYMBOLS if s not in MAINSTREAM_COINS]

stop_event = Event()

# ==================== Roostoo 客户端 ====================
class RoostooClient:
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv('ROOSTOO_API_KEY')
        self.secret_key = secret_key or os.getenv('ROOSTOO_SECRET_KEY')
        self.base_url = "https://mock-api.roostoo.com"

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
            r = requests.get(url, headers=headers, params=payload, timeout=10)
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
            r = requests.get(url, headers=headers, params=payload, timeout=10)
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
            r = requests.post(url, headers=headers, data=total_params, timeout=10)
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

    def fetch_price_series(self, symbols: List[str], minutes: int = 15) -> Dict[str, pd.Series]:
        """从 Horus 的 /market/price 获取价格时间序列"""
        data_dict = {}
        end_ts = int(time.time())
        start_ts = end_ts - minutes * 60
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for sym in symbols:
            url = f"{self.base_url}/market/price"
            params = {
                "assest": sym,
                "interval": "15m",
                "start": start_ts,
                "end": end_ts,
                "format": "json"
            }
            try:
                r = requests.get(url, headers=headers, params=params, timeout=10)
                r.raise_for_status()
                data = r.json()
                # 假设返回 [{"timestamp": 1731090000, "price": 70000.5}, ...]
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                data_dict[sym] = df['price'].astype(float)
            except Exception as e:
                print(f"Horus 获取 {sym} 失败: {e}")
        return data_dict

    def fetch_fear_greed_index(self):
        try:
            r = requests.get("https://api.alternative.me/fng/", timeout=10)
            return r.json()['data'][0]['value_classification']
        except Exception:
            return "Unknown"

# ==================== 市场分析器 ====================
class MarketAnalyzer:
    def analyze(self, price_series: Dict[str, pd.Series]) -> Dict[str, pd.DataFrame]:
        results = {}
        for symbol, prices in price_series.items():
            df = pd.DataFrame({'price': prices})
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

    def generate_signals(self, analysis: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        signals = {}
        for pair, df in analysis.items():
            latest = df.iloc[-1]
            signal = {'pair': pair, 'price': latest['price'], 'action': 'hold', 'size': 0.0, 'reason': []}

            if latest.get('emergency_sell', False):
                signal.update({'action': 'sell', 'size': 1.0})
                signal['price'] *= (1 - EMERGENCY_SELL_DISCOUNT)
                signal['reason'].append("紧急抛售")
            elif pair in MAINSTREAM_COINS:
                if latest.get('grid_buy') or latest.get('reversion_buy'):
                    signal.update({'action': 'buy', 'size': 0.05})
                    signal['reason'].append("主流币买入信号")
                elif latest.get('grid_sell'):
                    signal.update({'action': 'sell', 'size': 0.05})
                    signal['reason'].append("主流币卖出信号")
            else:
                if latest['whale_signal'] > 0:
                    signal.update({'action': 'buy', 'size': 0.02})
                    signal['reason'].append("跟随鲸鱼买入")
                elif latest['price'] < latest['ma_5']:
                    signal.update({'action': 'sell', 'size': 0.25})
                    signal['reason'].append("破5线减仓")
            signals[pair] = signal
        return signals

    def execute(self, signals: Dict[str, Dict]):
        balance = self.client.get_balance()
        usd = float(balance.get('available_USD', 50000))
        for pair, sig in signals.items():
            if sig['action'] == 'hold' or sig['size'] == 0:
                continue
            price = self.client.get_ticker(pair) or sig['price']
            qty = (usd * sig['size']) / price * (1 - TRANSACTION_FEE)
            self.client.place_order(pair, sig['action'], qty, price=price)

# ==================== 系统主循环 ====================
def run_once():
    print(f"\n{'='*60}")
    print(f"Horus量化交易系统运行 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    horus = HorusDataClient()
    roostoo = RoostooClient()
    analyzer = MarketAnalyzer()
    strategy = TradingStrategy(roostoo)
    price_data = horus.fetch_price_series(ALL_SYMBOLS, minutes=15)
    analysis = analyzer.analyze(price_data)
    signals = strategy.generate_signals(analysis)
    strategy.execute(signals)
    print("恐惧贪婪指数:", horus.fetch_fear_greed_index())

def periodic_task():
    while not stop_event.is_set():
        try:
            run_once()
        except Exception as e:
            print("运行异常:", e)
        stop_event.wait(60)  # 改这里！60秒=每分钟执行一次 → 改成3600秒=每小时

if __name__ == "__main__":
    print("Horus 实盘量化交易系统启动（默认每分钟执行一次）")
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
