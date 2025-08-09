#!/bin/bash
# Claude Voice Assistant - Quick Installation Script for Unix/Linux/macOS
# 快速安装和环境配置脚本

set -e

echo ""
echo "============================================"
echo "  Claude Voice Assistant Alpha 测试安装"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

# Check if Python is installed
echo "[1/8] 检查 Python 环境..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python 未安装，请先安装 Python 3.9 或更高版本"
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c "import sys; print('.'.join(map(str, sys.version_info[:2])))")
REQUIRED_VERSION="3.9"

if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" &> /dev/null; then
    print_error "Python 版本过低 ($PYTHON_VERSION)，需要 3.9 或更高版本"
fi

print_status "Python 版本检查通过 ($PYTHON_VERSION)"

# Check if running on macOS for special handling
OS_TYPE=$(uname -s)
print_status "检测到操作系统: $OS_TYPE"

# Create virtual environment
echo ""
echo "[2/8] 创建虚拟环境..."
if [ -d "venv" ]; then
    print_warning "虚拟环境已存在，跳过创建"
else
    $PYTHON_CMD -m venv venv
    print_status "虚拟环境创建成功"
fi

# Activate virtual environment
echo ""
echo "[3/8] 激活虚拟环境..."
source venv/bin/activate
print_status "虚拟环境已激活"

# Upgrade pip
echo ""
echo "[4/8] 更新 pip..."
python -m pip install --upgrade pip
print_status "pip 更新完成"

# Install system dependencies for audio (Linux only)
echo ""
echo "[5/8] 安装系统依赖..."
if [[ "$OS_TYPE" == "Linux" ]]; then
    echo "检测到 Linux 系统，检查音频依赖..."
    
    # Check if apt is available (Debian/Ubuntu)
    if command -v apt-get &> /dev/null; then
        echo "使用 apt 包管理器..."
        if ! dpkg -l | grep -q portaudio19-dev; then
            print_warning "需要安装 portaudio19-dev，请运行："
            echo "sudo apt-get update && sudo apt-get install portaudio19-dev python3-pyaudio"
        fi
        if ! dpkg -l | grep -q ffmpeg; then
            print_warning "建议安装 ffmpeg，请运行："
            echo "sudo apt-get install ffmpeg"
        fi
    # Check if yum is available (RedHat/CentOS)
    elif command -v yum &> /dev/null; then
        echo "使用 yum 包管理器..."
        print_warning "请确保已安装音频开发库："
        echo "sudo yum install portaudio-devel python3-devel"
    fi
    
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "检测到 macOS 系统，检查 Homebrew..."
    if command -v brew &> /dev/null; then
        echo "检查 portaudio..."
        if ! brew list portaudio &> /dev/null; then
            print_warning "建议安装 portaudio："
            echo "brew install portaudio"
        fi
        if ! brew list ffmpeg &> /dev/null; then
            print_warning "建议安装 ffmpeg："
            echo "brew install ffmpeg"
        fi
    else
        print_warning "建议安装 Homebrew 以便管理依赖"
    fi
fi

print_status "系统依赖检查完成"

# Install core dependencies
echo ""
echo "[6/8] 安装核心依赖..."
echo "这可能需要几分钟时间，请耐心等待..."

# Install basic requirements first
pip install pydantic loguru pyyaml aiofiles
print_status "基础依赖安装完成"

# Install speech dependencies with error handling
echo ""
echo "[7/8] 安装语音处理依赖..."
echo "正在安装 openai-whisper..."

# Install whisper with specific version to avoid conflicts
pip install openai-whisper

# Install TTS dependencies
echo "正在安装语音合成依赖..."
pip install pyttsx3

# Try to install pyaudio with better error handling
echo "正在安装 pyaudio (音频处理)..."
if ! pip install pyaudio; then
    print_warning "pyaudio 安装失败，将在静默模式下运行"
    print_warning "如需音频功能，请参考系统特定的安装说明"
    
    if [[ "$OS_TYPE" == "Darwin" ]]; then
        echo "macOS 用户可以尝试："
        echo "brew install portaudio"
        echo "pip install pyaudio"
    elif [[ "$OS_TYPE" == "Linux" ]]; then
        echo "Linux 用户可以尝试："
        echo "sudo apt-get install portaudio19-dev"
        echo "pip install pyaudio"
    fi
fi

# Install additional dependencies
echo "正在安装其他依赖..."
pip install requests aiohttp numpy wave

print_status "依赖安装完成"

# Create necessary directories
echo ""
echo "[8/8] 初始化项目结构..."
mkdir -p logs
mkdir -p temp
mkdir -p test_projects
mkdir -p .voice-assistant-backups
mkdir -p .voice-assistant-cache
mkdir -p config

print_status "项目结构初始化完成"

# Create test configuration
echo ""
echo "创建测试配置..."
cat > config/test_config.yaml << 'EOF'
system:
  log_level: INFO
  environment: testing
  enable_mock_mode: true

speech:
  enabled: false
  mock_mode: true
  recognizer:
    model_size: "base"
    language: "auto"
  
adapters:
  claude_code:
    enabled: true
    mock_mode: true
  
  voice_interface:
    enabled: false  # Disabled for testing
    
security:
  risk_levels:
    low: ["read_file", "list_directory"]
    medium: ["edit_file", "run_command"]
    high: ["delete_file", "system_command"]
    
  policies:
    testing_mode: ["low", "medium"]  # Only allow safe operations in testing
EOF

print_status "测试配置文件已生成"

# Test installation
echo ""
echo "验证安装..."
if python -c "
import sys
sys.path.append('src')
try:
    from core.types import CommandResult, AdapterStatus
    from core.config_manager import ConfigManager
    print('✅ 核心模块导入成功')
    
    # Test config loading
    config = ConfigManager('config/test_config.yaml')
    print('✅ 配置管理器测试通过')
    
except ImportError as e:
    print(f'❌ 模块导入失败: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ 配置测试失败: {e}')
    sys.exit(1)
"; then
    print_status "安装验证通过"
else
    print_error "安装验证失败"
fi

# Create startup script
echo ""
echo "创建快速启动脚本..."
cat > start_claude_voice.sh << 'EOF'
#!/bin/bash
# Claude Voice Assistant 启动脚本

echo "启动 Claude Voice Assistant..."
cd "$(dirname "$0")"

# 激活虚拟环境
source venv/bin/activate

# 启动应用
python src/main.py --config config/test_config.yaml

echo "应用已退出"
EOF

chmod +x start_claude_voice.sh

# Create requirements.txt for reference
echo ""
echo "生成 requirements.txt..."
pip freeze > requirements_alpha.txt
print_status "依赖列表已保存到 requirements_alpha.txt"

# Final success message
echo ""
echo "============================================"
echo "             🎉 安装完成！"
echo "============================================"
echo ""
echo "Alpha 测试环境已准备就绪！"
echo ""
echo "📋 下一步:"
echo "   1. 运行 ./start_claude_voice.sh 启动系统"
echo "   2. 阅读 testing/alpha_test_checklist.md 开始测试"
echo "   3. 遇到问题请查看 logs/claude_voice.log"
echo ""
echo "📞 技术支持:"
echo "   - GitHub Issues: https://github.com/claude-voice/issues"
echo "   - 测试指南: docs/testing_guide.md"
echo ""
echo "⚠️  注意: 当前在Mock模式下运行，语音功能已禁用"
echo "   如需完整功能，请确保音频依赖正确安装"
echo ""

# Check if we can enable voice features
if python -c "import pyaudio, pyttsx3, whisper" &> /dev/null; then
    print_status "语音依赖检测成功，可启用完整功能"
    echo "   如要启用语音功能，编辑 config/test_config.yaml："
    echo "   将 speech.enabled 设置为 true"
else
    print_warning "部分语音依赖缺失，将在静默模式运行"
fi

echo ""
echo "安装脚本执行完毕，按任意键继续..."
read -n 1 -s