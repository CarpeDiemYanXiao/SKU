# 添加源文件
add_library(rolling_sdk_lib SHARED
    rolling_sdk.cpp
    AsyncLogger.cpp
)


# 设置头文件目录
target_include_directories(rolling_sdk_lib 
    PUBLIC 
        ${CMAKE_SOURCE_DIR}/include
)

# 设置输出名称
set_target_properties(rolling_sdk_lib PROPERTIES
    OUTPUT_NAME "rollingSdkLib"
    PREFIX "lib"
)

# 可执行文件
# add_executable(rollingSdkApp rolling.cpp)
# target_link_libraries(rollingSdkApp PRIVATE rollingSdkLib)


# 确保继承顶层输出路径
# set_target_properties(rollingSdkApp PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

