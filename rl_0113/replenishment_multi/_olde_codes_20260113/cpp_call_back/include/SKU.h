#ifndef SKU_H
#define SKU_H

#include <string>
#include <vector>
#include <unordered_map>

class SKU {
public:
    explicit SKU(const std::string& sku_id);
    
    // 基础成员变量
    int beginStock;      // 期初库存
    int endStock;        // 期末库存
    int todayArrived;    // 今日到货
    int replenishQty;    // 补货数量
    int leadTime;        // 提前期
    int dayIndex;        // 当前日期索引
    int bindStock;       // 绑定库存
    int id;             // SKU ID

    // 库存相关的映射
    std::unordered_map<std::string, int> endingStockGroup;  // 结束库存组

    // 成员函数
    int arrivingStock(int day) const;  // 获取指定日期的到货量
    void nextDay();                    // 进入下一天
    int revertStock(int day);          // 恢复库存
    void selling(int quantity);        // 销售
    void bookOrder(int quantity);      // 下订单
    
    // 原有的数据相关方法
    void add_data(double value);
    void clear_data();
    std::vector<double> get_data() const;
    std::string get_id() const;
    
private:
    std::string sku_id_;
    std::vector<double> data_;
};

#endif // SKU_H 