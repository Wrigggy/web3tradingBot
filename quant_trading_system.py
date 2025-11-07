# test_installation.py
import pandas as pd
import numpy as np
import requests
import time
import os
import hmac
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# 从.env文件加载环境变量（如果存在）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("提示: 安装python-dotenv可以更好地管理环境变量: pip install python-dotenv")

class RoostooClient:
    """Roostoo交易所客户端 - 支持Secret Key"""
    
    def __init__(self, api_key: str = None, secret_key: str = None):
        self.api_key = api_key or os.getenv('ROOSTOO_API_KEY')
        self.secret_key = secret_key or os.getenv('ROOSTOO_SECRET_KEY')
        self.base_url = "https://mock-api.roostoo.com"
        
    def _get_timestamp(self):
        """获取13位毫秒时间戳"""
        return str(int(time.time() * 1000))
    
    def _get_signed_headers(self, payload: dict = {}):
        """生成签名头 - 使用Secret Key进行HMAC SHA256签名"""
        payload['timestamp'] = self._get_timestamp()
        sorted_keys = sorted(payload.keys())
        total_params = "&".join(f"{k}={payload[k]}" for k in sorted_keys)

        # 如果没有配置 secret_key，则使用一个 mock 签名并提示（便于本地测试）
        if not self.secret_key:
            print("Warning: ROOSTOO_SECRET_KEY 未配置，正在使用 mock 签名进行本地测试。")
            headers = {
                'RST-API-KEY': self.api_key or '',
                'MSG-SIGNATURE': 'mock-signature'
            }
            return headers, payload, total_params

        # 使用Secret Key生成签名
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            total_params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'RST-API-KEY': self.api_key,
            'MSG-SIGNATURE': signature
        }
        
        return headers, payload, total_params

    def get_ticker(self, pair="BTC/USD"):
        """获取行情数据"""
        url = f"{self.base_url}/v3/ticker"
        params = {'timestamp': self._get_timestamp(), 'pair': pair}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取Roostoo行情数据失败: {e}")
            # 返回模拟数据
            return self._get_mock_ticker_data(pair)
    
    def _get_mock_ticker_data(self, pair):
        """模拟行情数据"""
        return {
            "Data": {
                pair: {
                    "last": 45000 + np.random.normal(0, 1000),
                    "volume": 1000 + np.random.normal(0, 100)
                }
            }
        }
    
    def get_balance(self):
        """获取账户余额 - 需要签名认证"""
        url = f"{self.base_url}/v3/balance"
        headers, payload, _ = self._get_signed_headers({})
        try:
            response = requests.get(url, headers=headers, params=payload, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"获取余额失败: {e}")
            return self._get_mock_balance_data()
    
    def _get_mock_balance_data(self):
        """模拟余额数据"""
        return {
            "available_USD": 50000.0,
            "available_BTC": 1.5,
            "available_ETH": 15.0
        }
    
    def place_order(self, pair_or_coin, side, quantity, price=None, order_type=None):
        """下单 - 需要签名认证"""
        url = f"{self.base_url}/v3/place_order"
        pair = f"{pair_or_coin}/USD" if "/" not in pair_or_coin else pair_or_coin

        if order_type is None:
            order_type = "LIMIT" if price is not None else "MARKET"

        if order_type == 'LIMIT' and price is None:
            print("Error: LIMIT orders require 'price'.")
            return None

        payload = {
            'pair': pair,
            'side': side.upper(),
            'type': order_type.upper(),
            'quantity': str(quantity)
        }
        if order_type == 'LIMIT':
            payload['price'] = str(price)

        headers, _, total_params = self._get_signed_headers(payload)
        headers['Content-Type'] = 'application/x-www-form-urlencoded'

        try:
            response = requests.post(url, headers=headers, data=total_params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"下单失败: {e}")
            return None
    
    def get_klines(self, pair: str, interval: str = '1d', limit: int = 100):
        """获取K线数据"""
        # 模拟实现 - 实际应根据Roostoo API调整
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='D')
        base_price = 40000 if 'BTC' in pair else 3000
        
        # 生成模拟价格数据
        prices = []
        current_price = base_price
        for _ in range(limit):
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            current_price = current_price * (1 + change)
            prices.append(current_price)
        
        return pd.Series(prices, index=dates)


class HorusDataClient:
    """Horus数据客户端"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('HORUS_API_KEY')
        self.base_url = "https://api.horus.com/v1"  # 实际使用时替换为真实URL
        
    def fetch_bitcoin_onchain_data(self, days: int = 365) -> Dict:
        """获取比特币链上数据"""
        print(f"📊 获取比特币链上数据，时间范围: {days}天")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # 模拟数据 - 实际使用时替换为API调用
        data = {
            'transaction_count': self._generate_transaction_data(dates),
            'utxo_count': self._generate_utxo_data(dates),
            'block_size': self._generate_block_size_data(dates),
            'block_weight': self._generate_block_weight_data(dates),
            'block_count': self._generate_block_count_data(dates)
        }
        
        return data
    
    def fetch_defi_tvl_data(self, days: int = 365) -> Dict:
        """获取DeFi TVL数据"""
        print(f"🔄 获取DeFi TVL数据，时间范围: {days}天")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        data = {
            'total_value_locked': self._generate_total_tvl_data(dates),
            'chain_tvl': self._generate_chain_tvl_data(dates),
            'protocol_tvl': self._generate_protocol_tvl_data(dates)
        }
        
        return data
    
    def fetch_market_prices(self, symbols: List[str], days: int = 365) -> Dict:
        """获取市场价格数据"""
        print(f"📈 获取市场价格数据，币种: {symbols}，时间范围: {days}天")
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        price_data = {}
        
        for symbol in symbols:
            price_data[symbol] = self._generate_price_data(symbol, dates)
            
        return price_data
    
    def _generate_transaction_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成交易数量数据"""
        base_tx = 250000
        trend = np.linspace(1, 1.2, len(dates))
        seasonal = np.sin(np.arange(len(dates)) * 2 * np.pi / 30) * 0.1
        noise = np.random.normal(0, 0.05, len(dates))
        
        values = base_tx * trend * (1 + seasonal + noise)
        return pd.Series(values, index=dates)
    
    def _generate_utxo_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成UTXO数量数据"""
        base_utxo = 80000000
        growth = np.linspace(1, 1.1, len(dates))
        noise = np.random.normal(0, 0.01, len(dates))
        
        values = base_utxo * growth * (1 + noise)
        return pd.Series(values, index=dates)
    
    def _generate_block_size_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成区块大小数据"""
        base_size = 1.5
        values = np.random.normal(base_size, 0.2, len(dates))
        return pd.Series(values, index=dates)
    
    def _generate_block_weight_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成区块权重数据"""
        base_weight = 3.8
        values = np.random.normal(base_weight, 0.3, len(dates))
        return pd.Series(values, index=dates)
    
    def _generate_block_count_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成区块数量数据"""
        values = np.random.poisson(144, len(dates))
        return pd.Series(values, index=dates)
    
    def _generate_total_tvl_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成总TVL数据"""
        base_tvl = 50000000000
        trend = np.linspace(1, 0.8, len(dates))
        volatility = np.random.normal(0, 0.05, len(dates))
        
        values = base_tvl * trend * (1 + volatility)
        return pd.Series(values, index=dates)
    
    def _generate_chain_tvl_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成链TVL数据"""
        base_tvl = 20000000000
        trend = np.linspace(1, 0.85, len(dates))
        volatility = np.random.normal(0, 0.03, len(dates))
        
        values = base_tvl * trend * (1 + volatility)
        return pd.Series(values, index=dates)
    
    def _generate_protocol_tvl_data(self, dates: pd.DatetimeIndex) -> pd.Series:
        """生成协议TVL数据"""
        base_tvl = 10000000000
        trend = np.linspace(1, 0.75, len(dates))
        volatility = np.random.normal(0, 0.04, len(dates))
        
        values = base_tvl * trend * (1 + volatility)
        return pd.Series(values, index=dates)
    
    def _generate_price_data(self, symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
        """生成价格数据"""
        base_prices = {
            'BTC': 45000,
            'ETH': 3000,
            'BNB': 500,
            'ADA': 1.2,
            'SOL': 120
        }
        
        base_price = base_prices.get(symbol, 100)
        
        # 熊市特征
        bear_trend = np.linspace(1, 0.7, len(dates))
        volatility = np.random.normal(0, 0.04, len(dates))
        cycles = np.sin(np.arange(len(dates)) * 2 * np.pi / 90) * 0.15
        
        values = base_price * bear_trend * (1 + volatility + cycles)
        return pd.Series(values, index=dates)


class BearMarketAnalyzer:
    """熊市数据分析器"""
    
    def __init__(self):
        self.analysis_results = None
        
    def comprehensive_analysis(self, onchain_data: Dict, tvl_data: Dict, price_data: Dict) -> pd.DataFrame:
        """综合数据分析"""
        print("🔍 开始综合数据分析...")
        
        # 使用BTC价格作为基准
        btc_prices = price_data.get('BTC')
        if btc_prices is None:
            raise ValueError("需要BTC价格数据作为基准")
        
        df = pd.DataFrame(index=btc_prices.index)
        df['price'] = btc_prices
        
        # 1. 价格趋势分析
        print("📊 计算价格趋势指标...")
        df = self._calculate_price_indicators(df)
        
        # 2. 链上活动分析
        print("🔗 分析链上活动...")
        df = self._analyze_onchain_activity(df, onchain_data)
        
        # 3. DeFi健康状况分析
        print("🔄 分析DeFi健康状况...")
        df = self._analyze_defi_health(df, tvl_data)
        
        # 4. 市场情绪分析
        print("😊 分析市场情绪...")
        df = self._analyze_market_sentiment(df)
        
        # 5. 熊市专用指标
        print("🐻 计算熊市专用指标...")
        df = self._calculate_bear_market_indicators(df)

        self.analysis_results = df
        # 不强制删除所有含NaN的行：在样本天数少于某些长窗口(如ma_200)时，dropna会导致空结果。
        # 返回完整DataFrame，调用方可按需处理NaN
        return df
    
    def _calculate_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算价格技术指标"""
        # 移动平均线
        df['ma_7'] = df['price'].rolling(7).mean()
        df['ma_30'] = df['price'].rolling(30).mean()
        df['ma_90'] = df['price'].rolling(90).mean()
        df['ma_200'] = df['price'].rolling(200).mean()
        
        # 价格相对位置
        df['price_vs_ma30'] = (df['price'] - df['ma_30']) / df['ma_30']
        df['price_vs_ma200'] = (df['price'] - df['ma_200']) / df['ma_200']
        
        # 波动率
        df['volatility'] = df['price'].pct_change().rolling(20).std()
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['price'])
        
        # 支撑阻力水平
        df['support_level'] = df['price'].rolling(50).min()
        df['resistance_level'] = df['price'].rolling(50).max()
        
        return df
    
    def _analyze_onchain_activity(self, df: pd.DataFrame, onchain_data: Dict) -> pd.DataFrame:
        """分析链上活动"""
        # 交易数量动量
        if 'transaction_count' in onchain_data:
            tx_data = onchain_data['transaction_count']
            df['tx_momentum'] = tx_data.pct_change(7)
            df['tx_ma_ratio'] = tx_data / tx_data.rolling(30).mean()
        
        # UTXO增长分析
        if 'utxo_count' in onchain_data:
            utxo_data = onchain_data['utxo_count']
            df['utxo_growth'] = utxo_data.pct_change(30)
            df['utxo_health'] = (utxo_data - utxo_data.rolling(90).min()) / utxo_data.rolling(90).std()
        
        # 链上健康度综合评分
        onchain_health = 0
        weight_count = 0
        
        if 'tx_momentum' in df.columns:
            onchain_health += np.where(df['tx_momentum'] > 0, 1, -1)
            weight_count += 1
        
        if 'utxo_growth' in df.columns:
            onchain_health += np.where(df['utxo_growth'] > 0, 1, -1)
            weight_count += 1
            
        if weight_count > 0:
            df['onchain_health_score'] = onchain_health / weight_count
        
        return df
    
    def _analyze_defi_health(self, df: pd.DataFrame, tvl_data: Dict) -> pd.DataFrame:
        """分析DeFi健康状况"""
        # TVL动量分析
        if 'total_value_locked' in tvl_data:
            total_tvl = tvl_data['total_value_locked']
            df['tvl_momentum'] = total_tvl.pct_change(7)
            df['tvl_trend'] = total_tvl.rolling(30).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.mean(x)
            )
        
        # DeFi健康度评分
        defi_health = 0
        weight_count = 0
        
        if 'tvl_momentum' in df.columns:
            defi_health += np.where(df['tvl_momentum'] > -0.01, 1, -1)
            weight_count += 1
        
        if 'tvl_trend' in df.columns:
            defi_health += np.where(df['tvl_trend'] > -0.0001, 1, -1)
            weight_count += 1
            
        if weight_count > 0:
            df['defi_health_score'] = defi_health / weight_count
        
        return df
    
    def _analyze_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析市场情绪"""
        # 基于价格和波动率的情绪指标
        df['price_momentum'] = df['price'].pct_change(5)
        df['volatility_regime'] = np.where(
            df['volatility'] > df['volatility'].quantile(0.7), 'high', 
            np.where(df['volatility'] < df['volatility'].quantile(0.3), 'low', 'normal')
        )
        
        # 综合情绪评分
        sentiment_score = 0
        
        # RSI情绪
        sentiment_score += np.where(df['rsi'] < 35, 1, 0)
        sentiment_score += np.where(df['rsi'] > 65, -1, 0)
        
        # 价格动量
        sentiment_score += np.where(df['price_momentum'] > 0.02, 1, 0)
        sentiment_score += np.where(df['price_momentum'] < -0.02, -1, 0)
        
        # 波动率
        sentiment_score += np.where(df['volatility_regime'] == 'high', -1, 0)
        sentiment_score += np.where(df['volatility_regime'] == 'low', 1, 0)
        
        df['sentiment_score'] = sentiment_score / 4
        
        return df
    
    def _calculate_bear_market_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算熊市专用指标"""
        # 熊市阶段识别
        df['bear_market_phase'] = self._identify_bear_market_phase(df)
        
        # 投降指标
        price_drawdown = (df['price'] - df['price'].rolling(90).max()) / df['price'].rolling(90).max()
        df['capitulation_indicator'] = np.where(
            (price_drawdown < -0.4) & (df['volatility'] > df['volatility'].quantile(0.8)),
            'high', 'low'
        )
        
        # 积累区识别
        df['accumulation_zone'] = np.where(
            (df['price'] < df['support_level'] * 1.05) & 
            (df.get('onchain_health_score', 0) > 0) &
            (df['rsi'] < 40),
            True, False
        )
        
        # 熊市反弹概率
        bounce_probability = 0.3
        
        if 'onchain_health_score' in df.columns:
            bounce_probability += df['onchain_health_score'] * 0.2
        
        if 'sentiment_score' in df.columns:
            bounce_probability += df['sentiment_score'] * 0.15
            
        bounce_probability += np.where(df['rsi'] < 30, 0.2, 0)
        bounce_probability += np.where(df['accumulation_zone'], 0.15, 0)
        
        df['bounce_probability'] = np.clip(bounce_probability, 0, 1)
        
        return df
    
    def _identify_bear_market_phase(self, df: pd.DataFrame) -> pd.Series:
        """识别熊市阶段"""
        phases = []
        
        for i in range(len(df)):
            price = df['price'].iloc[i]
            ma_200 = df['ma_200'].iloc[i]
            rsi = df['rsi'].iloc[i]
            volatility = df['volatility'].iloc[i]
            
            if price < ma_200 * 0.6:
                if rsi < 25 and volatility > df['volatility'].quantile(0.8):
                    phase = 'capitulation'
                elif rsi < 40 and volatility < df['volatility'].quantile(0.6):
                    phase = 'accumulation'
                else:
                    phase = 'down_trend'
            elif price < ma_200 * 0.8:
                phase = 'early_bear'
            else:
                phase = 'transition'
                
            phases.append(phase)
            
        return pd.Series(phases, index=df.index)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))


class BearMarketTradingStrategy:
    """熊市交易策略"""
    
    def __init__(self, roostoo_client: RoostooClient):
        self.roostoo_client = roostoo_client
        self.signals_history = []
        
    def generate_trading_signals(self, analysis_df: pd.DataFrame) -> Dict:
        """生成交易信号"""
        if analysis_df is None or len(analysis_df) == 0:
            return {'error': '无分析数据'}
        
        latest = analysis_df.iloc[-1]
        
        signal = {
            'timestamp': datetime.now(),
            'price': latest['price'],
            'market_phase': latest.get('bear_market_phase', 'unknown'),
            'signals': [],
            'action': 'hold',
            'confidence': 0,
            'position_size': 0,
            'risk_level': 'medium'
        }
        
        # 收集信号
        bullish_signals = self._check_bullish_conditions(latest, analysis_df)
        bearish_signals = self._check_bearish_conditions(latest, analysis_df)
        
        signal['signals'].extend(bullish_signals['signals'])
        signal['signals'].extend(bearish_signals['signals'])
        
        # 决定行动
        bull_score = bullish_signals['score']
        bear_score = bearish_signals['score']
        
        if bull_score > 0.6 and bull_score > bear_score:
            signal['action'] = 'buy'
            signal['confidence'] = bull_score
            signal['position_size'] = self._calculate_position_size(bull_score, latest)
            signal['risk_level'] = 'low' if bull_score > 0.8 else 'medium'
        elif bear_score > 0.6 and bear_score > bull_score:
            signal['action'] = 'sell'
            signal['confidence'] = bear_score
            signal['position_size'] = self._calculate_position_size(bear_score, latest)
            signal['risk_level'] = 'high'
        else:
            signal['action'] = 'hold'
            signal['confidence'] = max(bull_score, bear_score)
        
        self.signals_history.append(signal)
        return signal
    
    def execute_trade(self, signal: Dict, symbol: str = "BTC/USD"):
        """执行交易"""
        # 使用安全访问，避免 KeyError
        action = signal.get('action', 'hold') if isinstance(signal, dict) else 'hold'
        print(f"执行交易 - 接收到的信号: {signal}")
        if action == 'hold':
            return {'status': 'no_trade', 'reason': '持有信号'}
        
        try:
            # 获取账户余额
            balance = self.roostoo_client.get_balance()
            if not balance:
                return {'status': 'error', 'message': '无法获取余额'}
            
            # 计算交易数量
            usd_balance = float(balance.get('available_USD', 10000))
            trade_amount = usd_balance * signal['position_size']
            quantity = trade_amount / signal['price']
            
            # 执行订单
            if action == 'buy':
                order_result = self.roostoo_client.place_order(
                    symbol, 'BUY', quantity, price=signal['price'] * 0.995
                )
            else:  # sell
                order_result = self.roostoo_client.place_order(
                    symbol, 'SELL', quantity, price=signal['price'] * 1.005
                )
            
            if order_result:
                return {
                    'status': 'success',
                    'action': action,
                    'quantity': quantity,
                    'trade_amount': trade_amount,
                    'order_info': order_result
                }
            else:
                return {'status': 'error', 'message': '下单失败'}
                
        except Exception as e:
            print(f"交易执行错误: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _check_bullish_conditions(self, data: pd.Series, full_df: pd.DataFrame) -> Dict:
        """检查看涨条件"""
        signals = []
        score = 0
        max_score = 0
        
        # 条件1: 积累区信号
        if data.get('accumulation_zone', False):
            signals.append("处于积累区域")
            score += 0.3
        max_score += 0.3
        
        # 条件2: RSI超卖
        if data['rsi'] < 30:
            signals.append("RSI超卖")
            score += 0.2
        max_score += 0.2
        
        # 条件3: 链上健康度
        if data.get('onchain_health_score', 0) > 0:
            signals.append("链上健康度改善")
            score += 0.15
        max_score += 0.15
        
        # 条件4: DeFi健康度
        if data.get('defi_health_score', 0) > 0:
            signals.append("DeFi健康度稳定")
            score += 0.1
        max_score += 0.1
        
        # 条件5: 反弹概率高
        if data.get('bounce_probability', 0) > 0.6:
            signals.append("反弹概率较高")
            score += 0.15
        max_score += 0.15
        
        # 条件6: 低波动率环境
        if data.get('volatility_regime') == 'low':
            signals.append("低波动率环境")
            score += 0.1
        max_score += 0.1
        
        normalized_score = score / max_score if max_score > 0 else 0
        
        return {'signals': signals, 'score': normalized_score}
    
    def _check_bearish_conditions(self, data: pd.Series, full_df: pd.DataFrame) -> Dict:
        """检查看跌条件"""
        signals = []
        score = 0
        max_score = 0
        
        # 条件1: 投降指标
        if data.get('capitulation_indicator') == 'high':
            signals.append("市场可能出现恐慌性抛售")
            score += 0.25
        max_score += 0.25
        
        # 条件2: RSI超买
        if data['rsi'] > 65:
            signals.append("RSI显示超买")
            score += 0.2
        max_score += 0.2
        
        # 条件3: 高波动率
        if data.get('volatility_regime') == 'high':
            signals.append("高波动率环境")
            score += 0.15
        max_score += 0.15
        
        # 条件4: 价格接近阻力位
        resistance_distance = (data['resistance_level'] - data['price']) / data['price']
        if resistance_distance < 0.05:
            signals.append("价格接近阻力位")
            score += 0.2
        max_score += 0.2
        
        # 条件5: 链上健康度恶化
        if data.get('onchain_health_score', 0) < -0.5:
            signals.append("链上健康度恶化")
            score += 0.2
        max_score += 0.2
        
        normalized_score = score / max_score if max_score > 0 else 0
        
        return {'signals': signals, 'score': normalized_score}
    
    def _calculate_position_size(self, signal_strength: float, market_data: pd.Series) -> float:
        """计算头寸规模"""
        base_size = 0.1
        
        # 信号强度调整
        size_by_strength = base_size * signal_strength
        
        # 波动率调整
        volatility = market_data.get('volatility', 0.03)
        if volatility > 0.05:
            size_by_vol = size_by_strength * 0.5
        elif volatility < 0.02:
            size_by_vol = size_by_strength * 1.2
        else:
            size_by_vol = size_by_strength
        
        # 熊市阶段调整
        market_phase = market_data.get('bear_market_phase', 'down_trend')
        if market_phase == 'capitulation':
            final_size = size_by_vol * 0.3
        elif market_phase == 'accumulation':
            final_size = size_by_vol * 1.5
        else:
            final_size = size_by_vol
        
        return min(final_size, 0.2)


def generate_comprehensive_report(analysis_df: pd.DataFrame, signals: Dict, trade_result: Dict = None) -> Dict:
    """生成综合分析报告"""
    if analysis_df is None:
        return {'error': '无分析数据'}
    
    latest = analysis_df.iloc[-1]
    
    report = {
        'report_time': datetime.now(),
        'market_overview': {
            'current_price': f"${latest['price']:,.2f}",
            'price_change_7d': f"{(latest['price'] / analysis_df['price'].iloc[-8] - 1):.2%}" if len(analysis_df) > 8 else "N/A",
            'market_phase': latest.get('bear_market_phase', 'unknown'),
            'volatility_regime': latest.get('volatility_regime', 'unknown'),
            'rsi_level': f"{latest['rsi']:.1f}"
        },
        'onchain_analysis': {
            'health_score': f"{latest.get('onchain_health_score', 0):.2f}",
            'transaction_trend': '上升' if latest.get('tx_momentum', 0) > 0 else '下降',
            'utxo_growth': f"{latest.get('utxo_growth', 0):.2%}" if 'utxo_growth' in latest else "N/A"
        },
        'defi_analysis': {
            'health_score': f"{latest.get('defi_health_score', 0):.2f}",
            'tvl_momentum': f"{latest.get('tvl_momentum', 0):.2%}" if 'tvl_momentum' in latest else "N/A",
            'bounce_probability': f"{latest.get('bounce_probability', 0):.1%}"
        },
        'trading_recommendation': {
            'action': signals.get('action', 'hold'),
            'confidence': f"{signals.get('confidence', 0):.1%}",
            'position_size': f"{signals.get('position_size', 0):.1%}",
            'risk_level': signals.get('risk_level', 'medium')
        },
        'key_insights': signals.get('signals', [])
    }
    
    if trade_result:
        report['trade_execution'] = {
            'status': trade_result.get('status'),
            'action': trade_result.get('action'),
            'quantity': trade_result.get('quantity', 0),
            'trade_amount': f"${trade_result.get('trade_amount', 0):,.2f}" if trade_result.get('trade_amount') else "N/A"
        }
    
    return report


def main():
    """主函数 - 完整的执行流程"""
    print("=== 🐻 Horus数据熊市量化交易系统 ===")
    print("开始初始化...\n")
    
    try:
        # 初始化客户端
        print("1. 🔑 初始化API客户端...")
        horus_client = HorusDataClient()  # 可以传入API Key: HorusDataClient(api_key="your_key")
        roostoo_client = RoostooClient()  # 从环境变量自动加载API Key和Secret Key
        
        # 测试Roostoo连接
        print("2. 🔗 测试Roostoo连接...")
        balance = roostoo_client.get_balance()
        if balance:
            print(f"   账户余额: {balance.get('available_USD', 'N/A')} USD")
        
        # 获取数据
        print("\n3. 📥 获取市场数据...")
        symbols = ['BTC', 'ETH', 'BNB']  # 主要交易对
        
        onchain_data = horus_client.fetch_bitcoin_onchain_data(180)  # 180天数据
        tvl_data = horus_client.fetch_defi_tvl_data(180)
        price_data = horus_client.fetch_market_prices(symbols, 180)
        
        # 分析数据
        print("\n4. 🔍 分析市场状况...")
        analyzer = BearMarketAnalyzer()
        analysis_results = analyzer.comprehensive_analysis(onchain_data, tvl_data, price_data)
        
        # 生成交易信号
        print("\n5. 💡 生成交易信号...")
        strategy = BearMarketTradingStrategy(roostoo_client)
        signals = strategy.generate_trading_signals(analysis_results)
        
        # 执行交易（如果信号不是hold）
        trade_result = None
        if signals.get('action') != 'hold':
            print("\n6. 💰 执行交易...")
            trade_result = strategy.execute_trade(signals)
        
        # 生成报告
        print("\n7. 📊 生成分析报告...")
        report = generate_comprehensive_report(analysis_results, signals, trade_result)
        
        # 输出结果
        print(f"\n{'='*60}")
        print("🎯 最终分析报告")
        print(f"{'='*60}")
        
        print(f"\n📊 市场概览:")
        for key, value in report['market_overview'].items():
            print(f"   {key}: {value}")
        
        print(f"\n🔗 链上分析:")
        for key, value in report['onchain_analysis'].items():
            print(f"   {key}: {value}")
        
        print(f"\n🔄 DeFi分析:")
        for key, value in report['defi_analysis'].items():
            print(f"   {key}: {value}")
        
        print(f"\n💡 交易建议:")
        for key, value in report['trading_recommendation'].items():
            print(f"   {key}: {value}")
        
        if 'trade_execution' in report:
            print(f"\n💰 交易执行:")
            for key, value in report['trade_execution'].items():
                print(f"   {key}: {value}")
        
        print(f"\n📈 关键洞察:")
        for i, insight in enumerate(report['key_insights'], 1):
            print(f"   {i}. {insight}")
        
        # 保存详细数据
        try:
            analysis_results.to_csv('horus_market_analysis.csv')
            print(f"\n💾 详细分析数据已保存至: horus_market_analysis.csv")
        except Exception as e:
            print(f"\n⚠️  无法保存文件: {e}")
        
        print(f"\n✅ 分析完成! 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ 系统执行出错: {e}")
        print("请检查网络连接或系统配置")


if __name__ == "__main__":
    # 这是程序的主入口
    main()