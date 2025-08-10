# 🎤 Claude Echo 语音控制系统 - 快速开始

## 🚀 立即使用

```bash
cd c:\development\claude-echo

# 启动语音控制系统 (推荐)
python voice_to_claude_fixed.py
```

## 📝 基本操作

1. **开始语音控制**：
   ```
   > r          # 按回车开始5秒录音
   ```

2. **说话内容示例**：
   - "创建一个Python计算器程序"
   - "写一个快速排序算法" 
   - "分析这个项目的代码结构"
   - "生成单元测试代码"

3. **确认发送**：
   ```
   是否发送到Claude Code? (y/n): y
   ```

4. **查看结果**：系统会显示Claude Code的完整响应

## ⚙️ 实用功能

- `history` - 查看最近5条命令历史
- `test` - 测试Claude Code连接
- `quiet` - 开启简洁模式 (减少录音时的显示干扰)
- `help` - 显示帮助信息
- `quit` - 退出程序

## 💡 语音识别技巧

### ✅ 最佳实践：
- **安静环境**：选择无背景噪音的地方
- **清晰发音**：正常语速，发音标准
- **编程术语**：使用标准编程词汇
- **简洁表达**：避免过长或复杂的句子

### 🎯 推荐命令格式：
```
"创建" + "一个" + "具体内容" + "程序/函数/文件"
"写" + "一个" + "具体算法/功能"  
"分析" + "具体对象"
"检查/修复" + "具体问题"
```

### ❌ 避免的问题：
- 说话太快或太慢
- 背景有音乐或其他声音
- 句子过于复杂或含糊
- 方言或口音过重

## 🔧 故障排除

### 录音显示混乱？
```bash
> quiet        # 开启简洁模式
> r            # 重新录音
```

### 识别准确率低？
1. 检查环境是否安静
2. 靠近麦克风说话  
3. 使用标准普通话
4. 尝试更简洁的表达

### Claude CLI连接问题？
```bash
# 测试连接
> test

# 或运行诊断
python diagnose_claude.py
```

## 📊 系统状态

- ✅ **Claude CLI**: v1.0.72 已连接
- ✅ **语音识别**: Whisper Base模型
- ✅ **音频设备**: 16个输入设备可用  
- ✅ **识别语言**: 中文+英文支持

## 🎉 示例对话

```
> r
[录音] 5秒录音开始...
[完成] 录音结束
[识别] 结果: '创建一个Python网页爬虫' (置信度: 0.87)
是否发送到Claude Code? (y/n): y

[发送] 命令到Claude Code: '创建一个Python网页爬虫'
[使用] 真实Claude Code CLI
[响应] Claude Code响应:
--------------------------------------------------
I'll help you create a Python web scraper. Here's a basic example using requests and BeautifulSoup:

```python
import requests
from bs4 import BeautifulSoup
import time

def web_scraper(url):
    """Simple web scraper function"""
    try:
        # Send GET request
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract title
        title = soup.find('title').text if soup.find('title') else 'No title'
        
        return {
            'title': title,
            'status_code': response.status_code,
            'content_length': len(response.content)
        }
        
    except Exception as e:
        return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    url = "https://example.com"
    result = web_scraper(url)
    print(result)
```
--------------------------------------------------

>
```

现在您可以完全通过语音控制Claude Code进行编程了！🎤✨