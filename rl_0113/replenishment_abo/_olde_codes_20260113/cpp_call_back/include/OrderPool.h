#ifndef ORDER_POOL_H
#define ORDER_POOL_H

#include "SKU.h"
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

class OrderPool {
public:
    OrderPool();
    
    void add_sku(const std::string& sku_id);
    void add_data(const std::string& sku_id, double value);
    void clear_data(const std::string& sku_id);
    void clear_all_data();
    std::vector<double> get_data(const std::string& sku_id) const;
    std::vector<std::string> get_all_skus() const;
    bool has_sku(const std::string& sku_id) const;
    
private:
    std::unordered_map<std::string, std::shared_ptr<SKU>> skus_;
};

#endif // ORDER_POOL_H 