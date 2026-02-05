# 安装

main is newest version, you can install it by:

```bash
pip install git+https://git.garena.com/shopee/bg-logistics/logistics/atp-sim-sdk-py.git
```

# 使用


```python

from atp-sim-sdk-py import SKU, CacheOrderRolling

sku = SKU(1,2,1)
spx_ado, abo_at, actual_ado, bind_qty, rts_qty = CacheOrderRolling().rolling(sku, [1,2,3,1,2])

```