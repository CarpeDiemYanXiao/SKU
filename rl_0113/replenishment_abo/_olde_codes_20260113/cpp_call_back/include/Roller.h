#pragma once
#include "SKU.h"
#include "Strategy.h"
#include <functional>
#include <vector>

class Roller {
public:
    explicit Roller(int rtsDay = 14, bool debug = false);
    void reset();
    
    std::tuple<int, int, int, int> rolling(
        int rollingDay,
        SKU& sku,
        const std::vector<std::vector<float>>& predicts,
        const std::vector<int>& salesList,
        std::function<float(int)> getMultiplier,
        int overnightKey
    );
    
    std::tuple<int, int, int, int> rollingOneDay(
        SKU& sku,
        const std::vector<float>& predicts,
        int sales,
        float multiplier,
        int overnightKey
    );
    
    std::tuple<int, int, int> summaryResult() const;

private:
    int rtsDay;
    bool debug;
    StrategyB strategy;
    std::vector<int> overnight;
    
    int rtsQty = 0;
    int bindQty = 0;
    int aboQty = 0;
    int totalSales = 0;
    
    void setState();
}; 