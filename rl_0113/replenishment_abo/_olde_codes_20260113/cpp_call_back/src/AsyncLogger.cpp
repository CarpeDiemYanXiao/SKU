#include <cxxabi.h>
#include <execinfo.h>
#include <libunwind.h>
#include <signal.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>

#include "rolling_sdk.h"

// AsyncLogger的实现
class AsyncLogger::Impl {
   public:
    Impl() : enabled(false), message_count(0) {
        // 检查是否启用调试
        if (getenv("DEBUG") && std::string(getenv("DEBUG")) == "true") {
            enabled = true;
            std::ifstream check_file("debug.csv");
            bool file_exists = check_file.good();
            check_file.close();

            log_file.open("debug.csv", std::ios::app);
            if (!log_file.is_open()) {
                std::cerr << "Failed to open log file" << std::endl;
                enabled = false;
                return;
            }

            // TODO: 暂时不写头, python会写. 保持代码统一
            // 如果文件是新创建的，写入标题行
            if (!file_exists || check_file.peek() == std::ifstream::traits_type::eof()) {
                // log_file << "Date,ModelID,LeadTime,BeginningQty,BindingQty,EndingQty,ReplenishQty,PredictQty,SellingQty,RtsQty,PredictArriveStock,"
                //             "PredictQtyList"
                //          << std::endl;
            }
        }
    }

    ~Impl() {
        if (log_file.is_open()) {
            log_file.close();
        }
    }

    void log(const std::string &message) {
        if (!enabled) return;

        std::lock_guard<std::mutex> lock(file_mutex);
        log_file << message << std::endl;

        // 计数并每1000条日志执行一次flush
        message_count++;
        if (message_count % 1000 == 0) {
            log_file.flush();
        }
    }

    void logError(ErrorCode code, const std::string &message) {
        if (!enabled) return;

        std::ostringstream ss;
        ss << "ERROR(" << code << "): " << getErrorMessage(code) << " - " << message;

        std::lock_guard<std::mutex> lock(file_mutex);
        std::cerr << ss.str() << std::endl;  // 输出到stderr

        if (log_file.is_open()) {
            log_file << "# " << ss.str() << std::endl;  // 用#标记错误行
            log_file.flush();                           // 立即刷新错误日志
        }
    }

    void setEnabled(bool value) {
        enabled = value;
    }

    void flush() {
        // 强制将缓冲区数据写入文件
        std::lock_guard<std::mutex> lock(file_mutex);
        if (log_file.is_open()) {
            log_file.flush();
        }
    }

    void stop() {
        // 先刷新，再关闭文件
        std::lock_guard<std::mutex> lock(file_mutex);
        if (log_file.is_open()) {
            log_file.flush();
            log_file.close();
            std::cerr << "AsyncLogger: Stopped after processing " << message_count << " messages." << std::endl;
        }
    }

   private:
    bool enabled;
    std::ofstream log_file;
    std::mutex file_mutex;
    std::atomic<size_t> message_count;  // 添加消息计数器
};

// 实现AsyncLogger的方法
AsyncLogger &AsyncLogger::getInstance() {
    static AsyncLogger instance;
    return instance;
}

AsyncLogger::AsyncLogger() : pImpl(new Impl()) {}

AsyncLogger::~AsyncLogger() {
    delete pImpl;
}

void AsyncLogger::log(const std::string &message) {
    if (pImpl) {
        pImpl->log(message);
    }
}

void AsyncLogger::logError(ErrorCode code, const std::string &message) {
    if (pImpl) {
        pImpl->logError(code, message);
    }
}

void AsyncLogger::setEnabled(bool value) {
    if (pImpl) {
        pImpl->setEnabled(value);
    }
}

// 添加新方法
void AsyncLogger::flush() {
    if (pImpl) {
        pImpl->flush();
    }
}

void AsyncLogger::stop() {
    if (pImpl) {
        pImpl->stop();
    }
}

// 清理函数
void cleanup_debug() {
    AsyncLogger::getInstance().stop();
}
