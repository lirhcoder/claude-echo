#!/bin/bash
# Claude Voice Assistant - Quick Start Script for Unix/Linux/macOS
# Unix/Linux/macOS快速启动脚本

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} Claude Voice Assistant Alpha 测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}❌ 虚拟环境不存在，请先运行 ./install.sh${NC}"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "src/main.py" ]; then
    echo -e "${RED}❌ 源代码文件不存在，请检查项目完整性${NC}"
    exit 1
fi

# Activate virtual environment
echo "激活虚拟环境..."
source venv/bin/activate

# Create logs directory if not exists
mkdir -p logs

# Set environment variables
export PYTHONPATH="src:$PYTHONPATH"
export CLAUDE_VOICE_ENV="alpha_testing"

echo ""
echo -e "${GREEN}🚀 启动 Claude Voice Assistant...${NC}"
echo -e "${YELLOW}⚠️  Alpha测试版本 - Mock模式运行${NC}"
echo -e "${BLUE}📋 测试指南: testing/alpha_test_checklist.md${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${BLUE}📊 应用已退出${NC}"
    echo -e "${BLUE}📝 查看日志: logs/alpha_test.log${NC}"
    echo ""
}

# Set trap to call cleanup function on script exit
trap cleanup EXIT

# Start the application with error handling
if python src/main.py --config config/test_config.yaml; then
    echo -e "${GREEN}✅ 应用正常退出${NC}"
else
    echo -e "${RED}❌ 应用异常退出，错误代码: $?${NC}"
    echo -e "${YELLOW}💡 请检查日志文件获取详细信息${NC}"
fi