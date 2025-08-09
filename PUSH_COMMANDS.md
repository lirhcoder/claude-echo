# 一键推送命令 - 复制粘贴执行

## Windows 命令提示符 (CMD)

```cmd
cd C:\development\claude-echo
git init
git branch -M main
git remote add origin https://github.com/lirhcoder/claude-echo.git
git add .
git commit -m "🎉 Claude Voice Assistant Alpha版本完整实现

✨ 新增功能:
- 4层架构设计 (UI层、智能中枢层、适配器层、执行层)
- 完整Speech模块 (语音识别、合成、意图解析)
- Claude Code深度集成适配器
- 7个核心AI代理系统
- 事件驱动异步架构
- 插件化适配器系统

📁 项目结构:
- src/: 核心源代码 (8,000+ 行Python代码)
- docs/: 完整技术文档
- config/: 配置文件和模板
- testing/: Alpha测试材料

🧪 Alpha测试准备:
- 自动安装脚本 (Windows + Unix)
- 详细测试检查清单和指南
- 完整配置和环境设置

🤖 Generated with Claude Code"

git push -u origin main
```

## PowerShell

```powershell
cd C:\development\claude-echo
git init
git branch -M main
git remote add origin https://github.com/lirhcoder/claude-echo.git
git add .
git commit -m "🎉 Claude Voice Assistant Alpha版本完整实现`n`n✨ 新增功能:`n- 4层架构设计`n- 完整Speech模块`n- Claude Code深度集成`n- 7个核心AI代理系统`n`n🤖 Generated with Claude Code"
git push -u origin main
```

## 认证说明

当执行 `git push` 时，系统会要求输入认证信息：

- **用户名**: `lirhcoder`  
- **密码**: 使用您的Personal Access Token（不是GitHub密码）

## 完成后验证

推送成功后，访问以下链接验证：
- https://github.com/lirhcoder/claude-echo

## 如果遇到问题

1. **权限错误**: 确认token有repo权限
2. **网络问题**: 检查防火墙设置
3. **仓库冲突**: 如果仓库已有内容，先pull再push

```cmd
git pull origin main --allow-unrelated-histories
git push origin main
```