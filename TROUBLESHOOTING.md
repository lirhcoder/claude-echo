# Claude Voice Assistant - 故障排除指南

## 🚨 安装错误解决方案

### 问题1: MySQL-python编译错误

**错误信息**:
```
Building wheel for MySQL-python (pyproject.toml) ... error
fatal error C1083: 无法打开包括文件: "config-win.h": No such file or directory
```

**原因**: `wave` 包错误地依赖了 `MySQL-python`，在Windows环境下无法编译。

**解决方案**:
1. **使用Alpha安装脚本** (推荐):
   ```cmd
   install_alpha.bat
   ```
   这个脚本避免了有问题的依赖包。

2. **手动安装最小依赖**:
   ```cmd
   pip install pydantic loguru pyyaml aiofiles requests aiohttp numpy
   ```

3. **跳过语音依赖**:
   - Alpha版本专注于核心架构测试
   - 语音功能已在Mock模式下运行

### 问题2: Python代码在批处理中执行失败

**错误信息**:
```
'import' is not recognized as an internal or external command
'from' is not recognized as an internal or external command
```

**原因**: 多行Python代码在Windows批处理中解析错误。

**解决方案**: 已在新版本中修复：
- 使用临时Python文件替代内联代码
- 改进错误处理和清理机制

### 问题3: 虚拟环境激活失败

**错误信息**:
```
[ERROR] Failed to activate virtual environment
```

**解决方案**:
1. **检查Python安装**:
   ```cmd
   python --version
   python -m venv --help
   ```

2. **手动创建虚拟环境**:
   ```cmd
   python -m venv venv
   call venv\Scripts\activate.bat
   ```

3. **权限问题** (Windows):
   - 以管理员身份运行命令提示符
   - 检查执行策略设置

## 🔧 常见问题快速修复

### 依赖相关

#### 问题: 无法安装PyAudio
```cmd
# 解决方案: 使用预编译的wheel文件
pip install pipwin
pipwin install pyaudio
```

#### 问题: Whisper安装失败
```cmd
# 解决方案: 使用CPU版本
pip install openai-whisper --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 环境配置

#### 问题: PATH环境变量
```cmd
# 检查Python路径
where python
echo %PATH%

# 添加Python到PATH (临时)
set PATH=%PATH%;C:\Users\YourUser\AppData\Local\Programs\Python\Python312
```

#### 问题: 字符编码错误
```cmd
# 设置UTF-8编码
chcp 65001
set PYTHONIOENCODING=utf-8
```

## 🎯 Alpha测试专用解决方案

### 使用简化安装

如果标准安装失败，使用Alpha专用安装：

```cmd
# 1. 运行简化安装
install_alpha.bat

# 2. 验证核心功能
python -c "import pydantic, loguru, yaml; print('Core dependencies OK')"

# 3. 启动Alpha测试
start_alpha.bat
```

### 手动安装步骤

如果自动安装完全失败：

```cmd
# 1. 创建虚拟环境
python -m venv venv
call venv\Scripts\activate.bat

# 2. 安装最小依赖
pip install pydantic==2.11.7 loguru==0.7.3 pyyaml==6.0.2

# 3. 创建测试配置
mkdir config
echo system: > config\alpha_config.yaml
echo   log_level: INFO >> config\alpha_config.yaml
echo   environment: alpha_testing >> config\alpha_config.yaml

# 4. 测试导入
python -c "import pydantic; print('Ready for Alpha testing')"
```

## 📊 系统要求验证

### 最小系统要求

- **操作系统**: Windows 10+ (1903或更高)
- **Python**: 3.9+ (推荐3.12)
- **内存**: 4GB RAM可用
- **存储**: 1GB磁盘空间
- **网络**: 稳定网络连接(安装依赖)

### 兼容性检查

```cmd
# Python版本检查
python -c "import sys; print(f'Python {sys.version}')"

# 依赖兼容性检查
python -c "import platform; print(f'Platform: {platform.platform()}')"

# 编码支持检查
python -c "import locale; print(f'Encoding: {locale.getpreferredencoding()}')"
```

## 🔍 日志分析

### 重要日志位置

- **安装日志**: `install-error-001.txt` (如果存在)
- **应用日志**: `logs/alpha_test.log`
- **系统日志**: Windows事件查看器

### 常见日志错误

1. **ImportError**: 模块导入失败 → 检查依赖安装
2. **FileNotFoundError**: 配置文件缺失 → 重新运行安装
3. **PermissionError**: 权限不足 → 以管理员身份运行

## 🆘 获取技术支持

### 自助资源

1. **GitHub Issues**: https://github.com/lirhcoder/claude-echo/issues
2. **测试指南**: `testing/alpha_test_checklist.md`
3. **快速开始**: `docs/QUICK_START_GUIDE.md`

### 报告问题时请提供

```cmd
# 收集系统信息
python -c "import sys, platform; print(f'Python: {sys.version}'); print(f'Platform: {platform.platform()}')" > system_info.txt
pip list > installed_packages.txt
```

- 错误日志完整内容
- 系统信息 (`system_info.txt`)
- 已安装包列表 (`installed_packages.txt`)
- 重现步骤

### 紧急修复模式

如果所有方法都失败：

```cmd
# 完全重置环境
rmdir /s /q venv
del /q config\*.yaml
del /q logs\*.log

# 重新开始
install_alpha.bat
```

---

**记住**: Alpha版本专注于核心架构测试，不需要完整的语音功能。如果遇到语音相关的错误，可以忽略并继续进行核心功能测试。