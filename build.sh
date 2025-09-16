#!/bin/bash

# XFeat C++ 构建脚本
# 使用方法: ./build.sh [clean|debug|release]

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查依赖
check_dependencies() {
    print_info "检查依赖..."
    
    # 检查cmake
    if ! command -v cmake &> /dev/null; then
        print_error "CMake 未安装，请先安装 CMake"
        exit 1
    fi
    
    # 检查OpenCV
    if ! pkg-config --exists opencv4; then
        print_warning "OpenCV4 未找到，尝试检查 OpenCV..."
        if ! pkg-config --exists opencv; then
            print_error "OpenCV 未安装，请先安装 OpenCV"
            exit 1
        fi
    fi
    
    # 检查ONNX Runtime
    if [ ! -d "/usr/local/onnxruntime" ] && [ ! -d "/opt/onnxruntime" ]; then
        print_warning "ONNX Runtime 未在标准路径找到"
        print_info "请确保 ONNX Runtime 已安装，或设置 ONNXRUNTIME_ROOT_DIR 环境变量"
    fi
    
    print_info "依赖检查完成"
}

# 清理构建目录
clean_build() {
    print_info "清理构建目录..."
    if [ -d "build" ]; then
        rm -rf build
        print_info "构建目录已清理"
    else
        print_info "构建目录不存在，无需清理"
    fi
}

# 构建项目
build_project() {
    local build_type=$1
    
    print_info "开始构建项目 (${build_type})..."
    
    # 创建构建目录
    mkdir -p build
    cd build
    
    # 配置CMake
    print_info "配置CMake..."
    cmake .. -DCMAKE_BUILD_TYPE=${build_type} \
             -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
    
    # 编译
    print_info "编译项目..."
    make -j$(nproc)
    
    print_info "构建完成!"
    print_info "可执行文件位置: build/bin/xfeatcpp"
    
    cd ..
}

# 安装ONNX Runtime (可选)
install_onnxruntime() {
    print_info "安装 ONNX Runtime..."
    
    # 下载ONNX Runtime
    local version="1.16.3"
    local platform="linux-x64"
    local filename="onnxruntime-linux-x64-${version}.tgz"
    
    if [ ! -f "${filename}" ]; then
        print_info "下载 ONNX Runtime ${version}..."
        wget "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${filename}"
    fi
    
    # 解压并安装
    print_info "解压并安装 ONNX Runtime..."
    tar -xzf "${filename}"
    sudo mv "onnxruntime-linux-x64-${version}" "/usr/local/onnxruntime"
    
    # 设置环境变量
    echo 'export ONNXRUNTIME_ROOT_DIR=/usr/local/onnxruntime' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/onnxruntime/lib' >> ~/.bashrc
    
    print_info "ONNX Runtime 安装完成"
    print_warning "请重新加载环境变量: source ~/.bashrc"
}

# 运行测试
run_test() {
    print_info "运行测试..."
    
    if [ ! -f "build/bin/xfeatcpp" ]; then
        print_error "可执行文件不存在，请先构建项目"
        exit 1
    fi
    
    # 检查测试图像是否存在
    if [ ! -f "assets/move1.png" ] || [ ! -f "assets/big6.png" ]; then
        print_warning "测试图像不存在，请确保 assets/ 目录中有测试图像"
        return
    fi
    
    cd build/bin
    ./xfeatcpp
    cd ../..
}

# 主函数
main() {
    local command=${1:-"release"}
    
    case $command in
        "clean")
            clean_build
            ;;
        "debug")
            check_dependencies
            clean_build
            build_project "Debug"
            ;;
        "release")
            check_dependencies
            clean_build
            build_project "Release"
            ;;
        "test")
            run_test
            ;;
        "install-onnx")
            install_onnxruntime
            ;;
        "help"|"-h"|"--help")
            echo "XFeat C++ 构建脚本"
            echo ""
            echo "使用方法:"
            echo "  ./build.sh [command]"
            echo ""
            echo "命令:"
            echo "  clean       清理构建目录"
            echo "  debug       构建调试版本"
            echo "  release     构建发布版本 (默认)"
            echo "  test        运行测试"
            echo "  install-onnx 安装 ONNX Runtime"
            echo "  help        显示此帮助信息"
            ;;
        *)
            print_error "未知命令: $command"
            echo "使用 './build.sh help' 查看帮助信息"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@"