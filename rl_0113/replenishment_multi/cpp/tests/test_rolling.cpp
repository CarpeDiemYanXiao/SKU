#include <algorithm>
#include <cstring>
#include <iostream>

#include "../include/rolling_sdk.h"

// 定义回调函数
float MyReplenishmentCallback(int day_index) {
    std::cout << "Replenishment callback called: " << day_index << std::endl;
    return 1.1;
}

// 创建销售数据
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// Function to load sales and prediction data from CSV
bool loadDataFromCSV(const std::string &filename, SkuInputInfo *skus, int sku_count, int simulation_days) {
    // Read sales and prediction data from CSV file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << filename << std::endl;
        return false;
    }

    std::string line;

    // 用于存储每天的销售数据和预测数据
    std::vector<std::pair<int, std::vector<int>>> data;  // pair<actual_sale, prediction_vector>

    // Skip header
    std::getline(file, line);

    // 读取CSV的每一行
    while (std::getline(file, line)) {
        std::istringstream ss(line);
        std::string field;

        std::string sku_id, date, leadtime, actual, pred_y, predicted_demand, initial_stock, a;

        // 解析字段（用逗号分隔）
        std::getline(ss, sku_id, ',');
        std::getline(ss, date, ',');
        std::getline(ss, leadtime, ',');
        std::getline(ss, actual, ',');
        std::getline(ss, pred_y, ',');
        std::getline(ss, predicted_demand, ']');
        std::getline(ss, a, ',');
        std::getline(ss, initial_stock, ',');

        // 去除引号
        actual.erase(std::remove(actual.begin(), actual.end(), '\"'), actual.end());
        predicted_demand = predicted_demand.substr(2);

        // 解析actual销售数据
        int actual_sale = std::stoi(actual);

        // 解析预测数据字符串，例如 "[22, 36, 47, 49, 50, 55]"
        std::vector<int> prediction;
        if (predicted_demand.size() > 0) {  // 确保至少有 "[]"
            // 按逗号分隔
            std::istringstream pred_ss(predicted_demand);
            std::string pred_value;
            while (std::getline(pred_ss, pred_value, ',')) {
                // 修复: 正确处理带空格的数字，只移除前后空格
                pred_value.erase(0, pred_value.find_first_not_of(" \t"));  // 移除前导空格
                pred_value.erase(pred_value.find_last_not_of(" \t") + 1);  // 移除尾随空格

                if (!pred_value.empty()) {
                    try {
                        prediction.push_back(std::stoi(pred_value));
                    } catch (const std::exception &e) {
                        std::cerr << "Failed to parse prediction value: '" << pred_value << "'" << std::endl;
                    }
                }
            }
        }

        // 收集数据
        data.push_back({actual_sale, prediction});
    }

    // 确保我们有足够的数据
    if (data.empty()) {
        std::cerr << "No data found in CSV file" << std::endl;
        return false;
    }

    // 填充销售数据
    for (int i = 0; i < sku_count; i++) {
        int days = std::min(static_cast<int>(data.size()), simulation_days);
        for (int j = 0; j < days; j++) {
            skus[i].sales[j] = data[j].first;  // 实际销售数据
        }
    }

    // 填充预测数据
    for (int i = 0; i < sku_count; i++) {
        int days = std::min(static_cast<int>(data.size()), simulation_days);
        for (int j = 0; j < days; j++) {
            // 为每天创建预测数组
            int max_predictions = std::min(simulation_days, static_cast<int>(data[j].second.size()));
            for (int k = 0; k < max_predictions; k++) {
                skus[i].predicts[j][k] = data[j].second[k];
            }

            // 如果预测数据不足，用最后一个值填充
            if (max_predictions > 0 && max_predictions < simulation_days) {
                int last_value = data[j].second[max_predictions - 1];
                for (int k = max_predictions; k < simulation_days; k++) {
                    skus[i].predicts[j][k] = last_value;
                }
            }
        }
    }

    std::cout << "Data loaded successfully: " << data.size() << " days of data" << std::endl;
    return true;
}

int main() {
    // Create test SKUs
    const int sku_count = 1;
    const int simulation_days = 30;
    SkuInputInfo skus[sku_count] = {{
        .id = "111-1",
        .lead_time = new int[simulation_days * 2](),
        .end_of_stock = 0,
        .day_index = -1,
        .begin_stock = 0,
        .bind_stock = 0,
        .rts_qty = 0,
        .today_arrived = 0,
        .orders = new int[simulation_days * 2](),
        .orders_size = simulation_days * 2,
        .ending_stock_list = new int[simulation_days * 2](),
        .ending_stock_list_size = simulation_days * 2,
        .order_returned = new int[simulation_days * 2](),
        .order_returned_size = simulation_days * 2,
        .overnight_list = new int[simulation_days * 2](),
        .predicts = new int *[simulation_days * 2](),
        .sales = new int[simulation_days * 2](),
        .callback = MyReplenishmentCallback,
    }};

    // Initialize the 2D predicts array
    for (int i = 0; i < sku_count; i++) {
        for (int j = 0; j < simulation_days * 2; j++) {
            skus[i].predicts[j] = new int[7]();
        }
    }

    // Create input structure
    // 在test_rolling.cpp中添加
    // 确保rts_day有合理的值
    for (int i = 0; i < sku_count; i++) {
        skus[i].rts_day = 14;  // 例如设置为3天
    }

    // Call the function to load data
    if (!loadDataFromCSV("/Users/jelech/codes/py/algo/algorithm/replenishment/data/simu_data_v1_case.csv", skus, sku_count, simulation_days)) {
        return 1;
    }

    // 初始化RollingInput
    RollingInput *input = new RollingInput();
    input->skus = skus;
    input->sku_count = sku_count;

    // Call function with new structure
    RollingResult *result = new RollingResult();
    roll_skus(input);

    // Print results
    std::cout << "Test results:" << std::endl;
    for (int i = 0; i < input->sku_count; i++) {
        std::cout << "SKU ID " << skus[i].id << ":" << std::endl;
        std::cout << "  end_of_stock: " << input->skus[i].end_of_stock << std::endl;
        std::cout << "  transition_stock: " << input->skus[i].today_arrived << std::endl;
        std::cout << "  rtss: " << input->skus[i].rts_qty << std::endl;
        std::cout << "  binding_qty: " << input->skus[i].bind_stock << std::endl;
        std::cout << "  replenishment_qty: " << input->skus[i].abo_qty << std::endl;
        std::cout << "  overnight: " << input->skus[i].overnight_list << std::endl;
    }

    // Clean up
    delete result;  // 释放之前分配的result
    delete input;   // 只删除input，不删除input->skus，因为它指向栈上的数组

    return 0;
}