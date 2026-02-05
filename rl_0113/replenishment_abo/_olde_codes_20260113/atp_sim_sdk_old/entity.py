import ctypes
import sys
from pathlib import Path
import numpy as np

# 定义回调函数类型
REPLENISHMENT_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_float, ctypes.c_int)


# Define the structures - 修改以匹配C++代码
class SkuInputInfo(ctypes.Structure):
    predicts_list: list  # 防止内存回收
    sales_array: np.ndarray
    total_day: int
    _fields_ = [
        ("id", ctypes.c_char * 128),  # 使用固定长度字符数组
        ("rts_day", ctypes.c_int),
        ("lead_time", ctypes.POINTER(ctypes.c_int)),
        ("end_of_stock", ctypes.c_int),
        ("day_index", ctypes.c_int),
        ("begin_stock", ctypes.c_int),
        ("bind_stock", ctypes.c_int),
        ("rts_qty", ctypes.c_int),
        ("estimate_rts_qty", ctypes.c_int),  # 在用temp不断滚动后, 这个值会更新
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
        ("multiplier", ctypes.c_float),  # 补货乘数，直接传值替代 callback
        ("callback", REPLENISHMENT_CALLBACK),  # 保留以保持兼容性，但不再使用
    ]

    @property
    def id_str(self):
        return str(self.id.decode("utf-8"))


class RollingInput(ctypes.Structure):
    _fields_ = [
        ("skus", ctypes.POINTER(SkuInputInfo)),
        ("sku_count", ctypes.c_int),
        ("evaluate", ctypes.c_bool),
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
    lib_path = Path(__file__).parent.parent / "cpp" / "build" / "src" / lib_name
    if not lib_path.exists():
        raise FileNotFoundError(f"Library not found at {lib_path}")

    lib = ctypes.CDLL(str(lib_path))
    lib.roll_skus.argtypes = [ctypes.POINTER(RollingInput)]
    lib.roll_skus.restype = ctypes.c_int  # ErrorCode 枚举类型

    return lib
