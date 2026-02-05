// // 文件类型：h
#ifndef rolling_sdk_H
#define rolling_sdk_H

#include <string>

#define MAX_ID_LENGTH 128

// 定义错误代码
enum ErrorCode {
    SUCCESS = 0,
    NULL_POINTER_ERROR = -1,
    INDEX_OUT_OF_BOUNDS = -2,
    MEMORY_ALLOCATION_ERROR = -3,
    INVALID_PARAMETER = -4,
    UNEXPECTED_ERROR = -99
};

// 前向声明AsyncLogger类
class AsyncLogger;

// 声明cleanup_debug函数
void cleanup_debug();

typedef float (*ReplenishmentCallback)(int);

// 确保 SkuInputInfo 的定义在前面
typedef struct {
    char id[MAX_ID_LENGTH];      // SKU的唯一标识符
    int rts_day;                 // 退货时间
    int *lead_time;              // 交货时间，默认为1
    int end_of_stock;            // 期末库存
    int day_index;               // 当前天的索引，初始为-1表示昨天
    int begin_stock;             // 当天开始时的库存
    int bind_stock;              // 绑定库存
    int rts_qty;                 // 可用库存数量
    int estimate_rts_qty;        // 未来模拟完成后, 最后一天的rts
    int today_arrived;           // 今天到货的数量
    int abo_qty;                 // 今日下abo, 补货量
    int *orders;                 // Array to store all orders
    int orders_size;             // Number of orders in the array
    int *ending_stock_list;      // 数组，存储每天的剩余库存
    int ending_stock_list_size;  //
    int *order_returned;         // 已经退掉的数量
    int order_returned_size;     //
    int lead_time_bind;          // 当前补货后继续模拟, 到leadtime时候的预估绑定量
    int estimate_end_stock;      // 当前补货后继续模拟, 到leadtime时候的预估期末库存
    int *overnight_list;         // 记录当前滚动情况下, 未来产生rts后的overnight结果
    int **predicts;              // 预测销量 [days][0-5]
    int *sales;                  // 实际销量 [days]
    ReplenishmentCallback callback;
} SkuInputInfo;

// 然后是 SkuOutputInfo 的定义
typedef struct {
    int id;                 // SKU的唯一标识符
    int lead_time;          // 交货时间
    int end_of_stock;       // 期末库存
    int transition_stock;   // 在途库存
    int rtss;               // Real Time Safety Stock
    int binding_qty;        // 绑定数量
    int replenishment_qty;  // 补货数量
    int overnight;          // 过夜库存
    int day_index;          // 当前天的索引
    int begin_stock;        // 当天开始时的库存
    int rts_qty;            // 可用库存数量
    int today_arrived;      // 今天到货的数量
    int *orders;            // Array to store all orders
    int order_count;        // Number of orders in the array
    int overnight_qty;      // 某天到达的商品的库存剩余情况
} SkuOutputInfo;

// C struct for returning array of results
typedef struct {
    SkuOutputInfo *skus;  // Array of SkuOutputInfo (renamed from items)
    int sku_count;        // Number of SKUs (renamed from size)
    int *lead_time_bind;
} RollingResult;

// Input data structure
typedef struct {
    SkuInputInfo *skus;  // Array of SkuInputInfo
    int sku_count;       // Number of SKUs
    bool evaluate;       // 推理会提前返回
} RollingInput;

// 异步日志系统类声明
class AsyncLogger {
   public:
    // 单例模式获取实例
    static AsyncLogger &getInstance();

    // 将消息添加到队列
    void log(const std::string &message);

    // 启用/禁用日志
    void setEnabled(bool value);

    // 停止日志线程
    void stop();

    void flush();

    // 新增错误日志方法
    void logError(ErrorCode code, const std::string &message);

    // 析构函数
    ~AsyncLogger();

   private:
    // 私有构造函数，实现单例模式
    AsyncLogger();

    // 日志处理线程函数
    void processLogs();

    // 私有成员变量（具体实现在CPP文件中）
    class Impl;
    Impl *pImpl;
};

template <typename T>
void debug_file_log(T value);

template <typename T, typename... Args>
void debug_file_log(T first, Args... args);

// 用于错误处理的辅助函数
const char *getErrorMessage(ErrorCode code);

#ifdef __cplusplus
extern "C" {
#endif

// 修改函数签名，添加错误码返回
ErrorCode roll_skus(const RollingInput *input);
void free_rolling_result(RollingResult *result);

ErrorCode step_one_sku(SkuInputInfo *input, bool evaluate, ReplenishmentCallback callback, int overnight_key);
ErrorCode roll_sku_one_day(SkuInputInfo *input, int sales, int *predicts, float multiplier);

// 新增验证函数
ErrorCode validate_sku_input(const SkuInputInfo *input);
ErrorCode validate_rolling_input(const RollingInput *input);

#ifdef __cplusplus
}
#endif

#endif  // rolling_sdk_H
