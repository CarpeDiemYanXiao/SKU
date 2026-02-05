#define TESTING
#include <gtest/gtest.h>

#include "rolling_sdk.h"

float testReplenishmentCallback(int day) {
    return 2.0f;
}

TEST(StepOneSkuTest, BasicCalculation) {
    // 定义测试所需的销售和预测数组
    const int days = 30;
    int sales[days] = {10};  // 第一天销售10个
    for (size_t i = 0; i < days; i++) {
        sales[i] = 10;
    }
    int lead_time[days] = {1};

    // 创建预测数组
    int* predicts[days];
    const int prediction_feature_day = 6;  // 从今天开始算起第几天
    for (int i = 0; i < days; i++) {
        predicts[i] = new int[prediction_feature_day];
        for (int j = 0; j < prediction_feature_day; j++) {
            predicts[i][j] = 20;  // 预测值设为20
        }
    }

    // 测试用例1：正常输入
    // 准备数据结构
    int orders[days] = {0};
    int ending_stock_list[days] = {0};
    int order_returned[days] = {0};
    int overnight_list[days] = {0};

    SkuInputInfo input1 = {.id = "111-1",
                           .rts_day = 14,
                           .lead_time = lead_time,
                           .end_of_stock = 100,
                           .day_index = -1,
                           .begin_stock = 100,
                           .bind_stock = 0,
                           .rts_qty = 2,
                           .today_arrived = 0,
                           .abo_qty = 0,
                           .orders = orders,
                           .orders_size = days,
                           .ending_stock_list = ending_stock_list,
                           .ending_stock_list_size = days,
                           .order_returned = order_returned,
                           .order_returned_size = days,
                           .lead_time_bind = 0,
                           .overnight_list = overnight_list};

    // 执行函数
    int overnight_key = lead_time[0] + input1.day_index;
    step_one_sku(&input1, false, testReplenishmentCallback, overnight_key);

    // 验证结果
    EXPECT_EQ(input1.id, "111-1");
    EXPECT_EQ(input1.lead_time[0], 1);
    EXPECT_EQ(input1.day_index, 0);                                // 应该更新为0
    EXPECT_GE(input1.abo_qty, 0);                                  // 补货量应该非负
    EXPECT_EQ(input1.ending_stock_list[0], input1.today_arrived);  // 当天的结束库存应该等于到货量

    // 测试用例2：零值输入
    // 重置数组
    for (int i = 0; i < days; i++) {
        orders[i] = 0;
        ending_stock_list[i] = 0;
        order_returned[i] = 0;
        overnight_list[i] = 0;
        sales[i] = 0;
        for (int j = 0; j < prediction_feature_day; j++) {
            predicts[i][j] = 0;
        }
    }

    SkuInputInfo input2 = {.id = "112-1",
                           .rts_day = 14,
                           .lead_time = lead_time,
                           .end_of_stock = 0,
                           .day_index = -1,
                           .begin_stock = 0,
                           .bind_stock = 0,
                           .rts_qty = 0,
                           .today_arrived = 0,
                           .abo_qty = 0,
                           .orders = orders,
                           .orders_size = days,
                           .ending_stock_list = ending_stock_list,
                           .ending_stock_list_size = days,
                           .order_returned = order_returned,
                           .order_returned_size = days,
                           .lead_time_bind = 0,
                           .overnight_list = overnight_list};

    // 执行函数
    overnight_key = lead_time[0] + input2.day_index;
    step_one_sku(&input2, false, testReplenishmentCallback, overnight_key);

    // 验证结果
    EXPECT_EQ(input2.id, "112-1");
    EXPECT_EQ(input2.lead_time[0], 1);
    EXPECT_EQ(input2.day_index, 0);  // 应该更新为0
    EXPECT_EQ(input2.abo_qty, 0);    // 零输入应该有零补货

    // 清理分配的内存
    for (int i = 0; i < days; i++) {
        delete[] predicts[i];
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}