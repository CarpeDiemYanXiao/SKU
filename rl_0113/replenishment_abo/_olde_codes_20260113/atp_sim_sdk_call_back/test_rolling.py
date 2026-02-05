import ctypes
import os
import sys
from pathlib import Path

# 定义回调函数类型
REPLENISHMENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int)

# 实现回调函数
def GetReplenishment():
    return 10

# Define the structures
class SkuOutputInfo(ctypes.Structure):
    _fields_ = [
        ("end_of_stock", ctypes.c_int),
        ("transition_stock", ctypes.c_int),
        ("rtss", ctypes.c_int),
        ("binding_qty", ctypes.c_int),
        ("replenishment_qty", ctypes.c_int),
        ("overnight", ctypes.c_int)
    ]

class RollingResult(ctypes.Structure):
    _fields_ = [
        ("items", ctypes.POINTER(SkuOutputInfo)),
        ("size", ctypes.c_int)
    ]

class SkuInputInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("lead_time", ctypes.c_int),
        ("end_of_stock", ctypes.c_int),
        ("transition_stock", ctypes.c_int),
        ("predicted_demand", ctypes.c_int),
        ("actual_sale", ctypes.c_int),
        ("day_index", ctypes.c_int),
        ("begin_stock", ctypes.c_int),
        ("bind_stock", ctypes.c_int),
        ("rts_qty", ctypes.c_int),
        ("today_arrived", ctypes.c_int),
        ("reset_end_stock", ctypes.c_int)
    ]

class RollingInput(ctypes.Structure):
    _fields_ = [
        ("skus", ctypes.POINTER(SkuInputInfo)),
        ("sku_count", ctypes.c_int)
    ]

def load_rolling_sdk_lib():
    # 确定库文件的名称（根据操作系统）
    if sys.platform.startswith('win'):
        lib_name = 'rollingSdkLib.dll'
    elif sys.platform.startswith('linux'):
        lib_name = 'librollingSdkLib.so'
    elif sys.platform.startswith('darwin'):
        lib_name = 'librollingSdkLib.dylib'
    else:
        raise RuntimeError("Unsupported platform")

    # 查找库文件
    # 假设库文件在build/lib目录下
    current_dir = Path(__file__).parent
    # lib_path = current_dir.parent / 'build' / 'lib' / lib_name
    # lib_path = current_dir.parent / 'build' / 'src' / lib_name
    lib_path =  current_dir.parent / 'cpp' / 'build' / 'src' / lib_name



    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found at {lib_path}")

    # 加载库
    lib = ctypes.CDLL(str(lib_path))

    # 设置函数参数和返回类型
    lib.roll_skus.argtypes = [
        ctypes.POINTER(RollingInput),  # input
        REPLENISHMENT_CALLBACK         # callback
    ]
    lib.roll_skus.restype = ctypes.POINTER(RollingResult)

    return lib

def test_roll_skus():
    try:
        # 加载库
        lib = load_rolling_sdk_lib()
        
        # 创建回调函数的包装器
        callback = REPLENISHMENT_CALLBACK(GetReplenishment)
        
        # 创建测试数据
        sku_count = 3
        skus = (SkuInputInfo * sku_count)(
            SkuInputInfo(id=123, lead_time=1, end_of_stock=10, transition_stock=5,
                        predicted_demand=8, actual_sale=7, day_index=-1, begin_stock=10,
                        bind_stock=0, rts_qty=0, today_arrived=0, reset_end_stock=10),
            # Add similar initialization for other SKUs
        )
        
        # 创建输入结构体
        input_data = RollingInput()
        input_data.skus = skus
        input_data.sku_count = sku_count
        
        # 调用函数
        result = lib.roll_skus(ctypes.byref(input_data), callback)
        
        # 打印结果
        if result:
            result_struct = result.contents
            for i in range(min(3, result_struct.size)):  # 只打印前3个结果
                print(f"SKU ID {skus[i].id}:")
                item = result_struct.items[i]
                print(f"  end_of_stock: {item.end_of_stock}")
                print(f"  transition_stock: {item.transition_stock}")
                print(f"  rtss: {item.rtss}")
                print(f"  binding_qty: {item.binding_qty}")
                print(f"  replenishment_qty: {item.replenishment_qty}")
                print(f"  overnight: {item.overnight}")
            
            # 清理内存
            # Note: 这里应该调用C++端的删除函数，但现在我们没有实现
            # 这可能会导致内存泄漏，在实际产品中需要处理
        
        return True
    
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_roll_skus()
