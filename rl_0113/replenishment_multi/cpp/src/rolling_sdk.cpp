#include "rolling_sdk.h"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

// 调试输出函数
template <typename T>
void debug_print(T value) {
    if (getenv("CPP_DEBUG") && std::string(getenv("CPP_DEBUG")) == "true") {
        std::cout << value << std::endl;
    }
}

template <typename T, typename... Args>
void debug_print(T first, Args... args) {
    if (getenv("CPP_DEBUG") && std::string(getenv("CPP_DEBUG")) == "true") {
        std::cout << first << "\t\t";
        debug_print(args...);
    }
}

// 日志文件输出函数
template <typename T>
void debug_file_log(T value) {
    AsyncLogger &logger = AsyncLogger::getInstance();
    std::ostringstream ss;
    ss << value;
    logger.log(ss.str());
}

template <typename T, typename... Args>
void debug_file_log(T first, Args... args) {
    std::ostringstream ss;

    // 处理第一个参数
    if constexpr (std::is_array<T>::value && std::is_same<std::remove_extent_t<T>, char>::value) {
        // 特殊处理 char 数组（字符串）
        ss << first;
    } else if constexpr (std::is_pointer<T>::value && std::is_same<std::remove_pointer_t<T>, char>::value) {
        // 特殊处理 char 指针（字符串）
        ss << first;
    } else if constexpr (std::is_pointer<T>::value && !std::is_same<T, const char *>::value) {
        ss << "[";
        for (int i = 0; i < 6; ++i) {
            if (i > 0) ss << ". ";
            ss << first[i];
        }
        ss << "]";
    } else {
        ss << first;
    }

    // 使用折叠表达式处理所有剩余参数
    auto append_arg = [&ss](auto arg) {
        ss << ",";
        if constexpr (std::is_array<decltype(arg)>::value && std::is_same<std::remove_extent_t<decltype(arg)>, char>::value) {
            ss << arg;
        } else if constexpr (std::is_pointer<decltype(arg)>::value && std::is_same<std::remove_pointer_t<decltype(arg)>, char>::value) {
            ss << arg;
        } else if constexpr (std::is_pointer<decltype(arg)>::value && !std::is_same<decltype(arg), const char *>::value) {
            ss << "[";
            for (int i = 0; i < 6; ++i) {
                if (i > 0) ss << ". ";
                ss << arg[i];
            }
            ss << "]";
        } else {
            ss << arg;
        }
    };

    // 应用到所有剩余参数
    (append_arg(args), ...);

    // 发送日志
    AsyncLogger &logger = AsyncLogger::getInstance();
    logger.log(ss.str());
}

// 错误消息定义
const char *getErrorMessage(ErrorCode code) {
    switch (code) {
        case SUCCESS:
            return "Operation successful";
        case NULL_POINTER_ERROR:
            return "Null pointer encountered";
        case INDEX_OUT_OF_BOUNDS:
            return "Index out of bounds";
        case MEMORY_ALLOCATION_ERROR:
            return "Memory allocation failed";
        case INVALID_PARAMETER:
            return "Invalid parameter";
        case UNEXPECTED_ERROR:
            return "Unexpected error occurred";
        default:
            return "Unknown error";
    }
}

// 验证SKU输入数据
ErrorCode validate_sku_input(const SkuInputInfo *input) {
    if (!input) return NULL_POINTER_ERROR;

    // 检查必要的数组指针
    if (!input->lead_time) return NULL_POINTER_ERROR;
    if (!input->orders) return NULL_POINTER_ERROR;
    if (!input->ending_stock_list) return NULL_POINTER_ERROR;
    if (!input->order_returned) return NULL_POINTER_ERROR;
    if (!input->overnight_list) return NULL_POINTER_ERROR;
    if (!input->predicts) return NULL_POINTER_ERROR;
    if (!input->sales) return NULL_POINTER_ERROR;

    // 检查数组大小
    if (input->orders_size <= 0) return INVALID_PARAMETER;
    if (input->ending_stock_list_size <= 0) return INVALID_PARAMETER;
    if (input->order_returned_size <= 0) return INVALID_PARAMETER;

    // 检查rts_day参数
    if (input->rts_day <= 0) return INVALID_PARAMETER;

    return SUCCESS;
}

// 验证总体输入数据
ErrorCode validate_rolling_input(const RollingInput *input) {
    if (!input) return NULL_POINTER_ERROR;
    if (!input->skus) return NULL_POINTER_ERROR;
    if (input->sku_count <= 0) return INVALID_PARAMETER;

    // 验证每个SKU
    for (int i = 0; i < input->sku_count; i++) {
        ErrorCode result = validate_sku_input(&input->skus[i]);
        if (result != SUCCESS) return result;
    }

    return SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

ErrorCode roll_sku_one_day(SkuInputInfo *input, int sales, int *predicts, float multiplier) {
    // 参数验证
    if (!input || !predicts) return NULL_POINTER_ERROR;

    try {
        // 前进到下一天
        input->day_index++;
        const int today = input->day_index;
        const int rts_day = input->rts_day;
        const int lead_time = input->lead_time[today];

        // 合并边界检查
        if (today < 0 || today >= input->orders_size || today >= input->ending_stock_list_size || lead_time < 0) {
            debug_print("Invalid day index or lead time:", today, lead_time);
            return INDEX_OUT_OF_BOUNDS;
        }

        // 处理当天到货
        const int arriving_qty = input->orders[today];
        input->ending_stock_list[today] = arriving_qty;
        input->today_arrived = arriving_qty;
        input->begin_stock = input->end_of_stock;

        // 处理销售
        input->bind_stock = std::min(input->begin_stock + input->today_arrived, sales);
        int remaining_to_sell = input->bind_stock;

        // 扣减库存(FIFO)
        for (size_t i = 0; i < input->ending_stock_list_size && remaining_to_sell > 0; i++) {
            const int deduction = std::min(remaining_to_sell, input->ending_stock_list[i]);
            input->ending_stock_list[i] -= deduction;
            remaining_to_sell -= deduction;
        }

        // 计算补货
        int end_of_stock = std::max(input->begin_stock, 0);
        // 预先检查边界
        if (today + lead_time >= input->orders_size) {
            debug_print("Future order index out of bounds:", today + lead_time);
            return INDEX_OUT_OF_BOUNDS;
        }

        for (size_t i = 0; i < lead_time; i++) {
            end_of_stock = std::max(end_of_stock + input->orders[today + i] - predicts[i], 0);
        }

        // 计算补货量
        input->abo_qty = static_cast<int>(std::max(predicts[lead_time] * multiplier - end_of_stock, 0.0f));

        // 生成自动补货订单
        input->orders[today + lead_time] = input->abo_qty;

        // 处理退货(RTS)
        input->rts_qty = 0;  // 默认值
        if (rts_day <= today + 1) {
            const int arrived_day = today + 1 - rts_day;
            if (arrived_day >= 0 && arrived_day < input->ending_stock_list_size && arrived_day < input->order_returned_size) {
                input->rts_qty = input->ending_stock_list[arrived_day];
                input->order_returned[arrived_day] = input->rts_qty;
                input->ending_stock_list[arrived_day] = 0;
            }
        }

        // 更新期末库存
        input->end_of_stock = std::max(input->begin_stock - input->rts_qty, 0) + input->today_arrived - input->bind_stock;

        // 记录日志
        debug_file_log(input->day_index, input->id, lead_time, input->begin_stock + input->today_arrived, input->bind_stock, input->end_of_stock,
                       input->abo_qty, predicts[0], sales, input->rts_qty, input->today_arrived, predicts);

        return SUCCESS;
    } catch (const std::exception &e) {
        debug_print("Exception in roll_sku_one_day:", e.what());
        return UNEXPECTED_ERROR;
    } catch (...) {
        debug_print("Unknown exception in roll_sku_one_day");
        return UNEXPECTED_ERROR;
    }
}

// 重构step_one_sku函数
ErrorCode step_one_sku(SkuInputInfo *input, bool evaluate, ReplenishmentCallback callback, int overnight_key) {
    // 参数验证
    ErrorCode result = validate_sku_input(input);
    if (result != SUCCESS) {
        debug_print("Invalid input in step_one_sku:", getErrorMessage(result));
        return result;
    }

    // 边界检查
    if (overnight_key < 0 || overnight_key >= input->ending_stock_list_size) {
        debug_print("Overnight key out of bounds:", overnight_key);
        return INDEX_OUT_OF_BOUNDS;
    }

    // 清空overnight值
    for (size_t i = 0; i < input->ending_stock_list_size; i++) {
        input->overnight_list[i] = 0;
    }

    // 使用直接传入的 multiplier 值，而不是调用 callback
    // 这避免了从 C++ 线程调用 Python callback 时的 GIL 竞争问题
    float multiplier = input->multiplier;
    (void)callback;  // callback 保留以保持兼容性，但不再使用

    // 第一天滚动
    int next_day_index = input->day_index + 1;
    if (next_day_index < 0 || next_day_index >= input->orders_size) {
        debug_print("Next day index out of bounds:", next_day_index);
        return INDEX_OUT_OF_BOUNDS;
    }

    result = roll_sku_one_day(input, input->sales[next_day_index], input->predicts[next_day_index], multiplier);
    if (result != SUCCESS) {
        return result;
    }

    input->overnight_list[0] = input->ending_stock_list[overnight_key];

    // 如果仅评估模式，提前返回
    if (evaluate) {
        return SUCCESS;
    }

    // 创建临时副本并进行后续模拟
    SkuInputInfo temp = *input;
    temp.ending_stock_list = nullptr;
    temp.orders = nullptr;
    temp.order_returned = nullptr;

    try {
        // 为数组分配内存
        temp.ending_stock_list = new int[input->ending_stock_list_size]();
        temp.orders = new int[input->orders_size]();
        temp.order_returned = new int[input->order_returned_size]();

        // 复制数组数据
        std::copy(input->ending_stock_list, input->ending_stock_list + input->ending_stock_list_size, temp.ending_stock_list);
        std::copy(input->orders, input->orders + input->orders_size, temp.orders);
        std::copy(input->order_returned, input->order_returned + input->order_returned_size, temp.order_returned);

        // 获取当前日期的lead time
        const int lead_time = input->lead_time[input->day_index + 1];
        const size_t total_days = input->rts_day + lead_time;

        // 模拟未来几天
        for (size_t i = 1; i < total_days; i++) {
            int future_day_index = input->day_index + i;

            // 边界检查
            if (future_day_index >= input->orders_size) {
                debug_print("Future day index out of bounds:", future_day_index);
                break;  // 提前结束模拟
            }

            // multiplier 使用预设的固定值（不再调用 callback）

            // 模拟下一天
            result = roll_sku_one_day(&temp, input->sales[future_day_index], input->predicts[future_day_index], multiplier);
            if (result != SUCCESS) {
                debug_print("Error in simulation for day", future_day_index, ":", getErrorMessage(result));
                break;
            }

            // 记录overnight和lead_time_bind
            input->overnight_list[i] = temp.ending_stock_list[overnight_key];
            if (i == lead_time) {
                input->lead_time_bind = temp.bind_stock;
                input->estimate_end_stock = temp.end_of_stock;
            }
        }
        input->estimate_rts_qty = temp.rts_qty;

    } catch (const std::bad_alloc &e) {
        // 释放已分配的内存
        if (temp.ending_stock_list) delete[] temp.ending_stock_list;
        if (temp.orders) delete[] temp.orders;
        if (temp.order_returned) delete[] temp.order_returned;
        debug_print("Memory allocation failed:", e.what());
        return MEMORY_ALLOCATION_ERROR;
    } catch (const std::exception &e) {
        // 释放已分配的内存
        if (temp.ending_stock_list) delete[] temp.ending_stock_list;
        if (temp.orders) delete[] temp.orders;
        if (temp.order_returned) delete[] temp.order_returned;
        debug_print("Exception in step_one_sku:", e.what());
        return UNEXPECTED_ERROR;
    } catch (...) {
        // 释放已分配的内存
        if (temp.ending_stock_list) delete[] temp.ending_stock_list;
        if (temp.orders) delete[] temp.orders;
        if (temp.order_returned) delete[] temp.order_returned;
        debug_print("Unknown exception in step_one_sku");
        return UNEXPECTED_ERROR;
    }

    // 释放临时分配的内存
    if (temp.ending_stock_list) delete[] temp.ending_stock_list;
    if (temp.orders) delete[] temp.orders;
    if (temp.order_returned) delete[] temp.order_returned;

    return SUCCESS;
}

// 重构roll_skus函数
ErrorCode roll_skus(const RollingInput *input) {
    // 验证输入参数
    ErrorCode result = validate_rolling_input(input);
    if (result != SUCCESS) {
        debug_print("Invalid input in roll_skus:", getErrorMessage(result));
        return result;
    }

    try {
        // 创建线程池
        const int thread_count = std::min(static_cast<int>(std::thread::hardware_concurrency()) + 1, input->sku_count);
        std::vector<std::thread> threads;
        std::atomic<int> current_idx(0);
        std::atomic<bool> has_error(false);
        std::string error_message;

        // Worker函数
        auto worker = [&]() {
            int idx;
            while ((idx = current_idx++) < input->sku_count && !has_error) {
                const int lead_time = input->skus[idx].lead_time[input->skus[idx].day_index + 1];
                const int overnight_key = lead_time + input->skus[idx].day_index;

                ErrorCode local_result = step_one_sku(&input->skus[idx], input->evaluate, input->skus[idx].callback, overnight_key);

                if (local_result != SUCCESS) {
                    has_error = true;
                    std::ostringstream ss;
                    ss << "Error processing SKU " << idx << ": " << getErrorMessage(local_result);
                    error_message = ss.str();
                    break;
                }
            }
        };

        // 启动线程
        for (int i = 0; i < thread_count; i++) {
            threads.emplace_back(worker);
        }

        // 等待所有线程完成
        for (auto &thread : threads) {
            thread.join();
        }

        // 检查是否有错误
        if (has_error) {
            debug_print(error_message);
            return UNEXPECTED_ERROR;
        }

    } catch (const std::exception &e) {
        debug_print("Exception in roll_skus:", e.what());
        return UNEXPECTED_ERROR;
    } catch (...) {
        debug_print("Unknown exception in roll_skus");
        return UNEXPECTED_ERROR;
    }

    // 确保日志写入
    AsyncLogger::getInstance().flush();
    debug_print("end rolling");
    return SUCCESS;
}

void free_rolling_result(RollingResult *result) {
    if (result) {
        delete[] result->skus;
        delete result;
    }
}

#ifdef __cplusplus
}
#endif
