# 快速诊断 API 返回格式的脚本
# 运行此脚本来确定 Horus API 的实际返回格式

import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('HORUS_API_KEY', "78732c7f065ebee7e63c0b313628cc3a95e0e805ae6e237f59e445c69e3a1d8d")
headers = {"X-API-KEY": api_key}

# 测试不同的 API 端点和参数组合
test_configs = [
    {
        "name": "配置 1: /market/price（原始配置）",
        "url": "https://api-horus.com/market/price",
        "params": {
            "asset": "BTC",
            "interval": "15m",
            "start": 1699600000,
            "end": 1699610000,
            "format": "json"
        }
    },
    {
        "name": "配置 2: /prices（简化版）",
        "url": "https://api-horus.com/prices",
        "params": {
            "symbols": "BTC,ETH",
            "limit": 15
        }
    },
    {
        "name": "配置 3: /ticker（实时行情）",
        "url": "https://api-horus.com/ticker",
        "params": {
            "symbol": "BTC"
        }
    },
    {
        "name": "配置 4: /ohlcv（K线数据）",
        "url": "https://api-horus.com/ohlcv",
        "params": {
            "symbol": "BTC",
            "interval": "15m",
            "limit": 15
        }
    }
]

print("="*80)
print("Horus API 返回格式诊断工具")
print("="*80)

for config in test_configs:
    print(f"\n\n{'='*80}")
    print(f"测试: {config['name']}")
    print(f"URL: {config['url']}")
    print(f"参数: {config['params']}")
    print(f"{'='*80}")
    
    try:
        r = requests.get(config['url'], headers=headers, params=config['params'], timeout=10)
        print(f"状态码: {r.status_code}")
        
        if r.status_code == 200:
            data = r.json()
            print(f"\n✓ 成功获取数据！")
            print(f"返回类型: {type(data).__name__}")
            
            if isinstance(data, list):
                print(f"列表长度: {len(data)}")
                if len(data) > 0:
                    print(f"第一行内容:\n{json.dumps(data[0], indent=2, ensure_ascii=False)}")
                    print(f"可用字段: {list(data[0].keys()) if isinstance(data[0], dict) else 'N/A'}")
            
            elif isinstance(data, dict):
                print(f"顶级键: {list(data.keys())}")
                print(f"\n完整内容（前 1000 字符）:")
                print(json.dumps(data, indent=2, ensure_ascii=False)[:1000])
            
            else:
                print(f"完整内容:\n{data}")
        
        else:
            print(f"✗ 错误: {r.status_code}")
            print(f"响应: {r.text[:500]}")
    
    except requests.exceptions.ConnectionError:
        print(f"✗ 连接错误: 无法连接到 {config['url']}")
    except requests.exceptions.Timeout:
        print(f"✗ 超时: 请求超过 10 秒")
    except Exception as e:
        print(f"✗ 错误: {type(e).__name__}: {e}")

print(f"\n\n{'='*80}")
print("诊断完成！")
print("="*80)
print("\n提示:")
print("1. 复制上面成功获取的数据格式")
print("2. 参考 JSON_PARSING_GUIDE.md 中的相应格式")
print("3. 根据数据结构修改 fetch_price_series() 方法中的数据提取逻辑")
