#pragma once
#include "SKU.h"
#include <vector>

class Strategy {
public:
    virtual ~Strategy() = default;
    virtual int beginStock(const SKU& sku) const;
    virtual int bindStock(const SKU& sku, int sales) = 0;
    virtual int endStock(const SKU& sku) const;
    virtual int calculateRep(const SKU& sku, const std::vector<float>& predicts, float multiplier) = 0;
};

class StrategyA : public Strategy {
public:
    int bindStock(const SKU& sku, int sales) override;
    int calculateRep(const SKU& sku, const std::vector<float>& predicts, float multiplier) override;
};

class StrategyB : public Strategy {
public:
    int bindStock(const SKU& sku, int sales) override;
    int endStock(const SKU& sku) const override;
    int calculateRep(const SKU& sku, const std::vector<float>& predicts, float multiplier) override;
};

class StrategyC : public Strategy {
public:
    int bindStock(const SKU& sku, int sales) override;
    int endStock(const SKU& sku) const override;
    int calculateRep(const SKU& sku, const std::vector<float>& predicts, float multiplier) override;
private:
    int endStockImpl(int begin, float bind, int arrived) const;
}; 