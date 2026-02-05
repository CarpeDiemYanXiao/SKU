import ctypes
import numpy as np
import sys
from atp_sim_sdk.entity import REPLENISHMENT_CALLBACK, RollingInput, SkuInputInfo, load_rolling_sdk_lib
from utils import datasets


class MemoryKeeper:
    def __init__(self):
        self.references = []
        # 为每个SKU存储NumPy数组的引用，以便重置时使用
        self.sku_arrays = {}  # {sku_id: {'orders': array, 'order_returned': array, ...}}

    def add(self, obj):
        self.references.append(obj)
        return obj

    def add_sku_arrays(self, sku_id, array_dict):
        """添加SKU相关的数组引用"""
        self.sku_arrays[sku_id] = array_dict

    def get_sku_arrays(self, sku_id):
        """获取SKU相关的数组引用"""
        return self.sku_arrays.get(sku_id, {})

    def clear(self):
        self.references.clear()
        self.sku_arrays.clear()


# 创建一个全局实例
memory_keeper = MemoryKeeper()


def init_skus(rts_day, sku_ids, datasets: datasets.Datasets) -> ctypes.Array[SkuInputInfo]:
    skus = (SkuInputInfo * len(sku_ids))()
    for i, sku_id in enumerate(sku_ids):
        lead_time = datasets.sku_lead_time(sku_id, 0)
        total_day = datasets.end_date_map[sku_id]
        simulation_days = total_day + lead_time + 5

        # 创建核心数组
        orders_np = np.zeros(simulation_days, dtype=np.int32)
        order_returned_np = np.zeros(simulation_days, dtype=np.int32)
        overnight_list_np = np.zeros(simulation_days, dtype=np.int32)
        ending_stock_list_np = np.zeros(simulation_days, dtype=np.int32)

        # 保存到memory_keeper中，并存储特定SKU的数组引用
        sku_arrays = {
            "orders": orders_np,
            "order_returned": order_returned_np,
            "overnight_list": overnight_list_np,
            "ending_stock_list": ending_stock_list_np,
        }
        memory_keeper.add_sku_arrays(str(sku_id), sku_arrays)
        memory_keeper.add([orders_np, order_returned_np, overnight_list_np, ending_stock_list_np])

        # 获取C指针
        orders = orders_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        order_returned = order_returned_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        overnight_list = overnight_list_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        ending_stock_list = ending_stock_list_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # leadtime 全部数据 - 需要扩展数组大小，防止C++端越界访问
        # C++代码会访问 lead_time[day_index + 1]，所以需要多分配空间
        lead_time_data = datasets.range_lead_time(0, total_day, sku_id)
        # 扩展到 simulation_days 大小，用最后一个值填充
        lead_time_extended = lead_time_data + [lead_time_data[-1]] * (simulation_days - total_day)
        lead_time_np = np.array(lead_time_extended, dtype=np.int32)
        lead_time_ptr = lead_time_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # 仿真全部销量 - 扩展数组防止越界
        sales_data = datasets.range_sales(0, total_day, sku_id)
        # 扩展到 simulation_days 大小，用0填充
        sales_extended = sales_data + [0] * (simulation_days - total_day)
        sales_np = np.array(sales_extended, dtype=np.int32)
        sales_ptr = sales_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        # 仿真全部预测销量 - 扩展数组防止越界
        predicts_data = datasets.range_prdicts(0, total_day, sku_id)
        predicts_np = np.array(predicts_data, dtype=np.int32)
        predicts_ptrs = (ctypes.POINTER(ctypes.c_int) * simulation_days)()

        memory_keeper.add(lead_time_np)
        memory_keeper.add(sales_np)
        memory_keeper.add(predicts_data)
        memory_keeper.add(predicts_np)

        predicts_list = []
        # 为 simulation_days 天都创建 predicts 指针，防止越界
        for day in range(simulation_days):
            if day < total_day:
                day_predicts = memory_keeper.add(predicts_np[day].copy())
            else:
                # 超出 total_day 的部分用零数组填充
                day_predicts = memory_keeper.add(np.zeros(6, dtype=np.int32))
            ptr = day_predicts.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            predicts_ptrs[day] = ptr
            predicts_list.append(day_predicts)

        skus[i] = SkuInputInfo(
            id=str(sku_id).encode("utf-8"),
            lead_time=lead_time_ptr,  # 如果leadtime要变化, 这里需要变为数组
            day_index=-1,
            ending_stock_list_size=simulation_days,
            ending_stock_list=ending_stock_list,
            orders_size=simulation_days,
            orders=orders,
            order_returned_size=simulation_days,
            order_returned=order_returned,
            rts_day=rts_day,  # 与C++代码一致
            lead_time_bind=0,
            estimate_end_stock=0,
            overnight_list=overnight_list,
            sales=sales_ptr,
            predicts=predicts_ptrs,
            predicts_list=predicts_list,
            sales_array=sales_np,
            total_day=total_day,
        )

    return skus


def reset_skus(skus: ctypes.Array[SkuInputInfo]):
    """
    重置SKU的动态字段，使用NumPy的fill方法快速初始化
    直接访问memory_keeper中存储的数组引用进行重置
    """
    for i in range(len(skus)):
        sku = skus[i]
        sku_id = sku.id_str

        # 重置SKU标量属性
        sku.day_index = -1
        sku.end_of_stock = 0
        sku.begin_stock = 0
        sku.bind_stock = 0
        sku.rts_qty = 0
        sku.estimate_rts_qty = 0
        sku.today_arrived = 0
        sku.abo_qty = 0
        sku.lead_time_bind = 0
        sku.estimate_end_stock = 0

        # 获取NumPy数组引用并重置
        sku_arrays = memory_keeper.get_sku_arrays(sku_id)
        if sku_arrays:
            # 使用NumPy的fill方法快速初始化
            sku_arrays["orders"].fill(0)
            sku_arrays["order_returned"].fill(0)
            sku_arrays["overnight_list"].fill(0)
            sku_arrays["ending_stock_list"].fill(0)
        else:
            # 如果没有找到数组引用，则使用指针逐个元素重置
            orders_size = sku.orders_size
            for j in range(orders_size):
                sku.orders[j] = 0
                sku.order_returned[j] = 0
                sku.overnight_list[j] = 0
                sku.ending_stock_list[j] = 0

    return skus


def rolling(skus: ctypes.Array[SkuInputInfo], evaluate, action_map):
    sku_count = len(skus)

    for i in range(sku_count):
        # 直接获取 multiplier 值，而不是使用 callback
        # 这避免了从 C++ 线程调用 Python callback 时的 GIL 竞争问题
        get_action = action_map[skus[i].id_str].get("get_action")
        if not get_action:
            raise Exception("get_action callback not found")
        # 调用一次获取 multiplier 值，然后直接传给 C++
        multiplier_value = get_action(0)  # day_idx 参数实际上被忽略
        skus[i].multiplier = float(multiplier_value)
        # skus[i].callback = REPLENISHMENT_CALLBACK(get_action)



    input_data = RollingInput()
    input_data.skus = skus
    input_data.sku_count = sku_count
    input_data.evaluate = evaluate

    lib = load_rolling_sdk_lib()
    error_code = lib.roll_skus(ctypes.byref(input_data))

    # 检查执行结果，如果有错误则输出错误信息并退出
    if error_code != 0:  # 0 = SUCCESS
        # 使用错误码映射替代直接调用getErrorMessage函数
        error_messages = {-1: "空指针错误", -2: "索引越界", -3: "内存分配错误", -4: "无效参数", -99: "未预期的错误"}
        error_message = error_messages.get(error_code, f"未知错误码: {error_code}")
        print(f"错误: {error_message}", file=sys.stderr)
        sys.exit(1)

    return error_code == 0
