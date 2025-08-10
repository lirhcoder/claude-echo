# Claude Echo 在 Claude Code 中的使用指南

## 🎯 在 Claude Code 命令行中使用语音助手

### 方法一：直接在Claude Code中调用

在Claude Code的命令行界面中，您可以直接运行语音助手：

```bash
# 进入项目目录
cd c:\development\claude-echo

# 激活虚拟环境 (如果需要)
call venv\Scripts\activate

# 启动简化语音测试
python simple_voice_test.py

# 或启动完整语音助手
python src\main.py
```

### 方法二：集成到Claude Code工作流

您可以在Claude Code会话中直接请求使用语音助手：

```
用户: "请启动语音助手测试"
Claude Code: [会自动执行] python simple_voice_test.py
```

## 🖥️ UI界面选项

### 1. 终端UI界面（推荐）

我来为您创建一个专用的终端UI界面：