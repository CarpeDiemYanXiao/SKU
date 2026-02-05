# 在文件开头添加 Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 添加测试可执行文件
add_executable(test_rolling test_rolling.cpp)

# 链接测试目标与库
target_link_libraries(test_rolling PRIVATE rolling_sdk_lib)
target_include_directories(test_rolling PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_step_one_sku test_step_one_sku.cpp)
target_link_libraries(test_step_one_sku PRIVATE 
    GTest::gtest_main
    rolling_sdk_lib
)
target_include_directories(test_step_one_sku PRIVATE ${CMAKE_SOURCE_DIR}/include)

# 添加测试
add_test(NAME test_rolling COMMAND test_rolling)
add_test(NAME test_step_one_sku COMMAND test_step_one_sku)

# 确保测试程序输出到正确位置
set_target_properties(test_rolling PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set_target_properties(test_step_one_sku PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
