import ctypes
import os
import sys
import numpy as np
from pathlib import Path

# 定义回调函数类型
REPLENISHMENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int)


# 实现回调函数
def MyReplenishmentCallback():
    print("Replenishment callback called1")
    return 111  # 与C++代码保持一致


# Define the structures - 修改以匹配C++代码
class SkuInputInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("rts_day", ctypes.c_int),
        ("lead_time", ctypes.c_int),
        ("end_of_stock", ctypes.c_int),
        ("day_index", ctypes.c_int),
        ("begin_stock", ctypes.c_int),
        ("bind_stock", ctypes.c_int),
        ("rts_qty", ctypes.c_int),
        ("today_arrived", ctypes.c_int),  # 添加这个字段
        ("abo_qty", ctypes.c_int),
        ("orders", ctypes.POINTER(ctypes.c_int)),
        ("orders_size", ctypes.c_int),
        ("ending_stock_list", ctypes.POINTER(ctypes.c_int)),
        ("ending_stock_list_size", ctypes.c_int),
        ("order_returned", ctypes.POINTER(ctypes.c_int)),
        ("order_returned_size", ctypes.c_int),
        ("lead_time_bind", ctypes.c_int),
        ("estimate_end_stock", ctypes.c_int),
        ("overnight_list", ctypes.POINTER(ctypes.c_int)),
        ("predicts", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ("sales", ctypes.POINTER(ctypes.c_int)),
    ]


class RollingInput(ctypes.Structure):
    _fields_ = [
        ("skus", ctypes.POINTER(SkuInputInfo)),
        ("sku_count", ctypes.c_int),
    ]


class RollingResult(ctypes.Structure):
    _fields_ = [
        ("skus", ctypes.POINTER(SkuInputInfo)),  # 使用相同的SkuInputInfo结构
        ("sku_count", ctypes.c_int),
    ]


def load_rolling_sdk_lib():
    # 确定库文件的名称（根据操作系统）
    if sys.platform.startswith("win"):
        lib_name = "rollingSdkLib.dll"
    elif sys.platform.startswith("linux"):
        lib_name = "librollingSdkLib.so"
    elif sys.platform.startswith("darwin"):
        lib_name = "librollingSdkLib.dylib"
    else:
        raise RuntimeError("Unsupported platform")

    # 查找库文件
    current_dir = Path(__file__).parent
    lib_path = current_dir / "build" / "src" / lib_name

    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found at {lib_path}")

    # 加载库
    lib = ctypes.CDLL(str(lib_path))

    # 更新函数签名
    lib.roll_skus.argtypes = [
        ctypes.POINTER(RollingInput),
        REPLENISHMENT_CALLBACK,
    ]
    lib.roll_skus.restype = None

    return lib


def test_roll_skus():
    try:
        # 加载库
        lib = load_rolling_sdk_lib()

        # 创建回调函数的包装器
        callback = REPLENISHMENT_CALLBACK(MyReplenishmentCallback)

        # 创建测试数据 - 与C++版本对齐
        sku_count = 1
        simulation_days = 30

        # 使用numpy创建数组 - 这样可以确保内存连续性
        # 创建数组 - 与C++版本保持一致
        ending_stock_list_np = np.zeros(simulation_days * 2, dtype=np.int32)
        ending_stock_list = ending_stock_list_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        orders_np = np.zeros(simulation_days * 2, dtype=np.int32)
        orders = orders_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        order_returned_np = np.zeros(simulation_days * 2, dtype=np.int32)
        order_returned = order_returned_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        overnight_list_np = np.zeros(simulation_days, dtype=np.int32)
        overnight_list = overnight_list_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        sales_np = np.zeros(simulation_days, dtype=np.int32)
        sales = sales_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        predicts_arrays = []
        max_lead_time = 5
        predicts_ptrs = (ctypes.POINTER(ctypes.c_int) * simulation_days)()
        for i in range(simulation_days):
            predicts_np = np.ones(max_lead_time, dtype=np.int32)  # 每天销售1个
            predicts_ptr = predicts_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            predicts_ptrs[i] = predicts_ptr
            predicts_arrays.append(predicts_np)

        # 初始化SkuInputInfo - 匹配C++代码的初始化
        skus = (SkuInputInfo * sku_count)(
            SkuInputInfo(
                id=123,
                lead_time=1,
                end_of_stock=10,
                day_index=-1,
                begin_stock=10,
                bind_stock=0,
                rts_qty=0,
                today_arrived=0,
                abo_qty=0,
                ending_stock_list_size=simulation_days * 2,
                ending_stock_list=ending_stock_list,
                orders_size=simulation_days * 2,
                orders=orders,
                order_returned_size=simulation_days * 2,
                order_returned=order_returned,
                rts_day=3,  # 与C++代码一致
                lead_time_bind=0,
                estimate_end_stock=0,
                overnight_list=overnight_list,
                sales=sales,
                predicts=predicts_ptrs,
            )
        )

        # 创建输入结构体
        input_data = RollingInput()
        input_data.skus = skus
        input_data.sku_count = sku_count

        # 调用函数
        print("before: ", input_data.skus[0].day_index)
        lib.roll_skus(ctypes.byref(input_data), callback)

        # 打印结果
        print("\nTest results:")
        for i in range(input_data.sku_count):
            print(f"SKU ID {input_data.skus[i].id}:")
            print(f"  end_of_stock: {input_data.skus[i].end_of_stock}")
            print(f"  transition_stock: {input_data.skus[i].today_arrived}")
            print(f"  rtss: {input_data.skus[i].rts_qty}")
            print(f"  binding_qty: {input_data.skus[i].bind_stock}")
            print(f"  replenishment_qty: {input_data.skus[i].abo_qty}")
            # 打印overnight_list的值
            overnight_data = [input_data.skus[i].overnight_list[j] for j in range(3)]  # 只打印前3个值
            print(f"  overnight: {overnight_data}")

        return True

    except Exception as e:
        print(f"Test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_roll_skus()
    print(f"Test {'succeeded' if success else 'failed'}")
