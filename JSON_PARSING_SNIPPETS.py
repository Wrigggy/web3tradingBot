# JSON 数据格式转换代码片段库
# 根据你的 API 返回格式，复制相应的代码片段到 fetch_price_series() 方法中

# ============================================================
# 方案 A: 如果 API 返回 {"data": [...]} 格式
# ============================================================
def fetch_price_series_format_A(self, symbols: list, minutes: int = 15) -> dict:
    """处理格式: {"data": [{"timestamp": ..., "price": ...}]}"""
    data_dict = {}
    headers = {"X-API-KEY": self.api_key}
    
    for sym in symbols:
        try:
            clean_symbol = sym.replace('/USD', '')
            url = f"{self.base_url}/market/price"
            params = {"asset": clean_symbol, "interval": "15m"}
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            
            # ⭐ 关键改动：提取 'data' 键
            data = r.json()
            if 'data' not in data:
                print(f"错误: {sym} 返回的 JSON 中没有 'data' 键，可用键: {list(data.keys())}")
                continue
            
            prices_list = data['data']
            df = pd.DataFrame(prices_list)
            
            # ... 后续处理时间戳和 DataFrame ...
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            data_dict[sym] = df['price'].astype(float)
            print(f"✓ {sym} 成功")
            
        except Exception as e:
            print(f"✗ {sym} 失败: {e}")
    
    return data_dict


# ============================================================
# 方案 B: 如果 API 返回按币种分类的嵌套格式
# ============================================================
def fetch_price_series_format_B(self, symbols: list, minutes: int = 15) -> dict:
    """处理格式: {"result": {"BTC": [...], "ETH": [...]}}"""
    data_dict = {}
    headers = {"X-API-KEY": self.api_key}
    
    for sym in symbols:
        try:
            clean_symbol = sym.replace('/USD', '')  # 'BTC/USD' -> 'BTC'
            url = f"{self.base_url}/market/price"
            params = {"assets": ",".join([s.split('/')[0] for s in symbols])}
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            
            # ⭐ 关键改动：从嵌套结构中提取该币种的数据
            data = r.json()
            if 'result' not in data:
                print(f"错误: 返回的 JSON 中没有 'result' 键")
                continue
            
            if clean_symbol not in data['result']:
                print(f"错误: {clean_symbol} 不在 result 中，可用币种: {list(data['result'].keys())}")
                continue
            
            prices_list = data['result'][clean_symbol]
            df = pd.DataFrame(prices_list)
            
            # ... 后续处理时间戳和 DataFrame ...
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            data_dict[sym] = df['price'].astype(float)
            print(f"✓ {sym} 成功")
            
        except Exception as e:
            print(f"✗ {sym} 失败: {e}")
    
    return data_dict


# ============================================================
# 方案 C: 如果 API 返回 OHLCV 格式（含开盘、最高、最低、收盘、成交量）
# ============================================================
def fetch_price_series_format_C(self, symbols: list, minutes: int = 15) -> dict:
    """处理格式: [{"timestamp": ..., "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}]"""
    data_dict = {}
    headers = {"X-API-KEY": self.api_key}
    
    for sym in symbols:
        try:
            clean_symbol = sym.replace('/USD', '')
            url = f"{self.base_url}/market/price"
            params = {"asset": clean_symbol, "interval": "15m"}
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            
            data = r.json()
            df = pd.DataFrame(data)
            
            # ⭐ 关键改动：使用 'close' 列而不是 'price'
            if 'close' not in df.columns:
                print(f"错误: {sym} 返回的 DataFrame 中没有 'close' 列，可用列: {list(df.columns)}")
                continue
            
            # ... 后续处理时间戳和 DataFrame ...
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            data_dict[sym] = df['close'].astype(float)  # ⭐ 改这里
            print(f"✓ {sym} 成功")
            
        except Exception as e:
            print(f"✗ {sym} 失败: {e}")
    
    return data_dict


# ============================================================
# 方案 D: 如果时间戳是 ISO 8601 字符串格式
# ============================================================
def fetch_price_series_format_D(self, symbols: list, minutes: int = 15) -> dict:
    """处理格式: [{"timestamp": "2025-11-09T12:00:00Z", "price": ...}]"""
    data_dict = {}
    headers = {"X-API-KEY": self.api_key}
    
    for sym in symbols:
        try:
            clean_symbol = sym.replace('/USD', '')
            url = f"{self.base_url}/market/price"
            params = {"asset": clean_symbol, "interval": "15m"}
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            
            data = r.json()
            df = pd.DataFrame(data)
            
            # ⭐ 关键改动：不指定 unit，直接转换字符串时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])  # 不需要 unit='s'
            df.set_index('timestamp', inplace=True)
            data_dict[sym] = df['price'].astype(float)
            print(f"✓ {sym} 成功")
            
        except Exception as e:
            print(f"✗ {sym} 失败: {e}")
    
    return data_dict


# ============================================================
# 方案 E: 自动检测格式（最灵活）
# ============================================================
def fetch_price_series_format_E(self, symbols: list, minutes: int = 15) -> dict:
    """自动检测并处理多种 JSON 格式"""
    data_dict = {}
    headers = {"X-API-KEY": self.api_key}
    
    for sym in symbols:
        try:
            clean_symbol = sym.replace('/USD', '')
            url = f"{self.base_url}/market/price"
            params = {"asset": clean_symbol, "interval": "15m"}
            
            r = requests.get(url, headers=headers, params=params, timeout=30)
            r.raise_for_status()
            
            data = r.json()
            
            # ⭐ 自动检测格式
            if isinstance(data, list):
                # 格式 1: 直接是列表
                df = pd.DataFrame(data)
            
            elif isinstance(data, dict):
                # 格式 2/3: 字典，需要提取数据
                if 'data' in data and isinstance(data['data'], list):
                    df = pd.DataFrame(data['data'])
                elif 'result' in data:
                    if isinstance(data['result'], list):
                        df = pd.DataFrame(data['result'])
                    elif isinstance(data['result'], dict) and clean_symbol in data['result']:
                        df = pd.DataFrame(data['result'][clean_symbol])
                    else:
                        raise ValueError(f"无法从 result 中提取 {clean_symbol} 数据")
                else:
                    df = pd.DataFrame([data])
            else:
                raise ValueError(f"未知的数据格式: {type(data)}")
            
            # 处理时间戳（尝试多种格式）
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                except:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                raise ValueError(f"数据中没有 'timestamp' 列，可用列: {list(df.columns)}")
            
            # 处理价格列（尝试多个名称）
            price_col = None
            for col in ['price', 'close', 'last', 'last_price']:
                if col in df.columns:
                    price_col = col
                    break
            
            if price_col is None:
                raise ValueError(f"数据中没有价格列，可用列: {list(df.columns)}")
            
            df.set_index('timestamp', inplace=True)
            data_dict[sym] = df[price_col].astype(float)
            print(f"✓ {sym} 成功（格式: 列表={'是' if isinstance(data, list) else '否'}，价格列: {price_col}）")
            
        except Exception as e:
            print(f"✗ {sym} 失败: {type(e).__name__}: {e}")
    
    return data_dict
