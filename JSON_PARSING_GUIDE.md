# Horus API JSON 数据解析指南

## 常见的 API 返回格式及处理方法

### 格式 1：直接数组（最简单）
```json
[
  {"timestamp": 1699600000, "price": 45000.50},
  {"timestamp": 1699603600, "price": 45100.25},
  {"timestamp": 1699607200, "price": 44900.75}
]
```

**处理方法**：
```python
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
prices = df['price'].astype(float)
```

---

### 格式 2：数据包裹在 `data` 或 `prices` 键中
```json
{
  "data": [
    {"timestamp": 1699600000, "price": 45000.50},
    {"timestamp": 1699603600, "price": 45100.25}
  ],
  "status": "success",
  "count": 2
}
```

**处理方法**：
```python
data = response.json()
df = pd.DataFrame(data['data'])  # 提取 'data' 键
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
prices = df['price'].astype(float)
```

---

### 格式 3：按币种分类的嵌套结构
```json
{
  "result": {
    "BTC": [
      {"timestamp": 1699600000, "price": 45000.50},
      {"timestamp": 1699603600, "price": 45100.25}
    ],
    "ETH": [
      {"timestamp": 1699600000, "price": 3000.25},
      {"timestamp": 1699603600, "price": 3050.75}
    ]
  }
}
```

**处理方法**：
```python
data = response.json()
btc_data = data['result']['BTC']
df = pd.DataFrame(btc_data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
prices = df['price'].astype(float)
```

---

### 格式 4：OHLCV 格式（含开盘、最高、最低、收盘、成交量）
```json
[
  {
    "timestamp": 1699600000,
    "open": 44950.00,
    "high": 45100.00,
    "low": 44900.00,
    "close": 45000.50,
    "volume": 1500.25
  },
  {
    "timestamp": 1699603600,
    "open": 45000.50,
    "high": 45200.00,
    "low": 44950.00,
    "close": 45100.25,
    "volume": 1200.50
  }
]
```

**处理方法**：
```python
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.set_index('timestamp', inplace=True)
prices = df['close'].astype(float)  # 使用 'close' 价格
```

---

### 格式 5：时间戳是字符串格式
```json
[
  {"timestamp": "2025-11-09 12:00:00", "price": 45000.50},
  {"timestamp": "2025-11-09 13:00:00", "price": 45100.25}
]
```

**处理方法**：
```python
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])  # 直接转换，不需要 unit 参数
df.set_index('timestamp', inplace=True)
prices = df['price'].astype(float)
```

---

## 时间戳格式转换

### 秒级时间戳（最常见）
```python
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
# 例: 1699600000 -> 2025-11-09 12:00:00
```

### 毫秒级时间戳
```python
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
# 例: 1699600000000 -> 2025-11-09 12:00:00
```

### ISO 8601 字符串格式
```python
df['timestamp'] = pd.to_datetime(df['timestamp'])
# 例: "2025-11-09T12:00:00Z" -> 2025-11-09 12:00:00
```

### 自定义格式
```python
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y-%m-%d %H:%M:%S')
# 例: "2025-11-09 12:00:00" -> 2025-11-09 12:00:00
```

---

## 快速诊断：确定你的 API 返回格式

运行以下代码来检查你的 API 返回格式：

```python
import requests
import json

api_key = "你的 API Key"
headers = {"X-API-KEY": api_key}
url = "https://api-horus.com/market/price"
params = {
    "asset": "BTC",
    "interval": "15m",
    "start": 1699600000,
    "end": 1699610000,
    "format": "json"
}

r = requests.get(url, headers=headers, params=params)
data = r.json()

print("=== 响应类型 ===")
print(f"类型: {type(data).__name__}")

print("\n=== 响应内容（前 500 字符）===")
print(json.dumps(data, indent=2)[:500])

if isinstance(data, list):
    print(f"\n✓ 列表格式，包含 {len(data)} 行数据")
    if len(data) > 0:
        print(f"第一行内容: {data[0]}")
        print(f"可用字段: {list(data[0].keys())}")

elif isinstance(data, dict):
    print(f"\n✓ 字典格式，顶级键: {list(data.keys())}")
    for key in ['data', 'prices', 'result', 'records']:
        if key in data:
            print(f"  - '{key}' 存在，类型: {type(data[key]).__name__}")
```

---

## 修改 `fetch_price_series()` 方法

根据你的 API 返回格式，修改第 136-160 行的数据解析部分：

### 如果是格式 2（data 键）：
```python
data = r.json()
df = pd.DataFrame(data['data'])  # 改这里
```

### 如果是格式 3（嵌套结构）：
```python
data = r.json()
clean_sym = sym.split('/')[0]
df = pd.DataFrame(data['result'][clean_sym])  # 改这里
```

### 如果是格式 4（OHLCV）：
```python
df = pd.DataFrame(data)
prices = df['close'].astype(float)  # 改这里，使用 'close' 而非 'price'
```

---

## 完整示例：自适应解析器

如果不确定 API 格式，可以使用代码中的 `_parse_json_to_dataframe()` 方法，它会自动尝试多种格式。

运行脚本后，查看 DEBUG 输出：
```
DEBUG: BTC/USD 返回的 JSON 结构: dict - {'data': [...]}
DEBUG: BTC/USD DataFrame 列: ['timestamp', 'price', 'volume']
✓ 成功获取 BTC/USD 数据，行数: 15
```

---

## 常见错误及解决方案

| 错误信息 | 原因 | 解决方案 |
|---------|------|--------|
| `'timestamp'` KeyError | 数据中没有 'timestamp' 列 | 检查可用列，改用正确的列名（如 'time', 'date', 't'） |
| `'price'` KeyError | 数据中没有 'price' 列 | 使用 'close', 'last', 'last_price' 等替代 |
| `list indices must be integers` | 尝试用整数索引访问字典 | 先检查数据类型，字典要用键（key）访问 |
| `No JSON object could be decoded` | 返回的不是有效 JSON | 检查 API 端点和请求参数是否正确 |
| 空列表 `[]` | API 没有返回数据 | 检查时间范围参数、API Key、网络连接 |

