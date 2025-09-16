#!/bin/bash

echo "=========================================="
echo "开始构建Qt网络通信工具 (Debug模式)"
echo "=========================================="

# Debug模式强制使用本地编译
echo "Debug模式强制使用本地编译模式"
USE_CROSS_COMPILE=false

# 创建debug构建目录
BUILD_DIR="build_debug"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
    echo "创建debug构建目录: $BUILD_DIR"
fi

cd "$BUILD_DIR"

# 运行CMake配置 (Debug模式，强制本地编译)
echo "运行CMake配置 (Debug模式，强制本地编译)..."
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_NATIVE_COMPILE=ON -DUSE_CROSS_COMPILE=OFF

if [ $? -ne 0 ]; then
    echo "CMake配置失败"
    exit 1
fi

# 编译项目
echo "开始编译 (Debug模式)..."
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "编译失败"
    exit 1
fi

echo "=========================================="
echo "Debug模式编译完成！"
echo "=========================================="
echo "可执行文件位置: $BUILD_DIR/bin/xfeatcpp"

# 检查可执行文件是否存在
if [ -f "bin/xfeatcpp" ]; then
    echo "构建成功！"
    echo ""
    echo "调试信息:"
    echo "- 可执行文件包含调试符号 (-g)"
    echo "- 优化级别设置为 -O0 (无优化)"
    echo "- 启用了DEBUG宏定义"
    echo "- 启用了QT_DEBUG宏定义"
    echo "- 启用了警告标志 (-Wall -Wextra)"
    echo ""
    echo "运行程序: ./bin/xfeatcpp"
    echo "使用GDB调试: gdb ./bin/xfeatcpp"
else
    echo "警告: 未找到可执行文件"
fi 