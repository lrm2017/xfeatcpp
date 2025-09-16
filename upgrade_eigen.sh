#!/bin/bash

# 升级Eigen3到支持C++17的版本

echo "=========================================="
echo "升级Eigen3到支持C++17的版本"
echo "=========================================="

# 检查当前Eigen3版本
echo "当前Eigen3版本："
if [ -f "/usr/include/eigen3/Eigen/src/Core/util/Macros.h" ]; then
    grep -E "EIGEN_(WORLD|MAJOR|MINOR)_VERSION" /usr/include/eigen3/Eigen/src/Core/util/Macros.h
fi

echo "正在下载Eigen3 3.4.0..."

# 创建临时目录
mkdir -p /tmp/eigen_build
cd /tmp/eigen_build

# 下载Eigen3 3.4.0
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0

# 创建构建目录
mkdir build && cd build

# 配置和编译
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local/eigen3
make -j$(nproc)
sudo make install

# 更新库路径
sudo ldconfig

echo "=========================================="
echo "Eigen3 3.4.0安装完成！"
echo "=========================================="

# 验证安装
if [ -f "/usr/local/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h" ]; then
    echo "新版本Eigen3信息："
    grep -E "EIGEN_(WORLD|MAJOR|MINOR)_VERSION" /usr/local/eigen3/include/eigen3/Eigen/src/Core/util/Macros.h
fi
