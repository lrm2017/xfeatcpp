#!/bin/bash

# 安装Eigen3库的脚本

echo "正在安装Eigen3库..."

# 检查系统类型
if command -v apt-get &> /dev/null; then
    # Ubuntu/Debian系统
    echo "检测到Ubuntu/Debian系统，使用apt安装Eigen3..."
    sudo apt-get update
    sudo apt-get install -y libeigen3-dev
elif command -v yum &> /dev/null; then
    # CentOS/RHEL系统
    echo "检测到CentOS/RHEL系统，使用yum安装Eigen3..."
    sudo yum install -y eigen3-devel
elif command -v pacman &> /dev/null; then
    # Arch Linux系统
    echo "检测到Arch Linux系统，使用pacman安装Eigen3..."
    sudo pacman -S eigen
else
    echo "未识别的系统，请手动安装Eigen3库"
    echo "Ubuntu/Debian: sudo apt-get install libeigen3-dev"
    echo "CentOS/RHEL: sudo yum install eigen3-devel"
    echo "Arch Linux: sudo pacman -S eigen"
    exit 1
fi

echo "Eigen3库安装完成！"
echo "现在可以编译项目了："
echo "  mkdir build && cd build"
echo "  cmake .."
echo "  make"
