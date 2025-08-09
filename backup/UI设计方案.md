# Claude Echo - 最小化UI设计方案

## 设计原则

### 1. 极简主义
- **一屏完成**：所有信息在一个屏幕区域显示
- **最少元素**：只显示必要的状态和反馈信息
- **无装饰**：去除所有非功能性的视觉元素

### 2. 零输入设计
- **语音优先**：所有操作通过语音完成
- **最少按键**：只保留必要的系统按键（空格、ESC）
- **自动操作**：程序主动处理，用户被动接收

### 3. 实时反馈
- **即时显示**：语音识别结果立即显示
- **状态明确**：当前程序状态一目了然
- **进度可见**：操作进度实时更新

## UI布局设计

### 方案一：单行状态栏（最小）
```
Claude Echo ● 录制中... > "打开设置文件" → claude open settings.json ✓
```
- **优点**：占用空间最小（1行）
- **缺点**：信息密集，可读性稍差

### 方案二：三行布局（推荐）
```
Claude Echo [录制中]
> "打开设置文件"  
✓ claude open settings.json
```
- **优点**：信息层次清晰，易读
- **缺点**：占用3行空间

### 方案三：卡片式布局（清晰）
```
┌─ Claude Echo ──────────────┐
│ ● 录制中                   │
│ > "打开设置文件"           │
│ ✓ claude open settings.json │
└────────────────────────────┘
```
- **优点**：视觉边界清晰，专业感
- **缺点**：占用空间相对较大

## 详细设计规范

### 颜色方案
```python
COLORS = {
    'recording': '\033[91m●\033[0m',      # 红色圆点-录制中
    'listening': '\033[93m●\033[0m',      # 黄色圆点-等待中  
    'processing': '\033[94m●\033[0m',     # 蓝色圆点-处理中
    'success': '\033[92m✓\033[0m',        # 绿色勾-成功
    'error': '\033[91m✗\033[0m',          # 红色叉-错误
    'command': '\033[96m→\033[0m',        # 青色箭头-命令
    'text': '\033[97m{}\033[0m',          # 白色文字-普通文本
}
```

### 状态指示器
- **●** 录制中（红色）
- **○** 等待中（黄色）  
- **◐** 处理中（蓝色，可旋转动画）
- **✓** 成功（绿色）
- **✗** 失败（红色）
- **→** 执行命令（青色）

### 文本显示规则
1. **用户语音**：使用引号包围 `"打开文件"`
2. **系统命令**：使用等宽字体 `claude open file.py`
3. **状态信息**：简短动词短语 `录制中...` `处理中...`
4. **错误信息**：简明扼要 `文件不存在` `识别失败`

## 交互流程设计

### 启动流程
```
1. Claude Echo 启动中...
2. ● 准备就绪，请说话（空格开始录制）
```

### 正常操作流程
```
1. ● 录制中...
2. ○ 处理语音："打开设置"
3. → claude open settings.json
4. ✓ 完成
```

### 错误处理流程
```
1. ● 录制中...
2. ○ 处理语音："asdfgh"  
3. ✗ 无法理解，请重试
```

## 快捷键设计

### 基础操作
- **空格键**：开始/停止录制
- **ESC键**：退出程序
- **?键**：显示帮助（临时覆盖当前界面）

### 高级操作（可选）
- **R键**：重新识别上一段语音
- **C键**：清除屏幕历史
- **H键**：显示命令历史

## 响应性设计

### 终端适配
```python
def get_terminal_size():
    """获取终端大小并适配显示"""
    import shutil
    width, height = shutil.get_terminal_size()
    
    if width < 40:
        # 超窄屏幕：单行模式
        return "single_line"
    elif width < 80:
        # 窄屏幕：简化模式
        return "compact"
    else:
        # 正常屏幕：完整模式
        return "full"
```

### 动态布局
- **超窄屏 (<40字符)**：单行显示所有信息
- **窄屏 (40-80字符)**：两行显示，文字截断
- **正常屏 (>80字符)**：完整三行显示

## 具体实现代码

### UI控制器类
```python
class MinimalUI:
    def __init__(self):
        self.mode = self.detect_terminal_mode()
        self.clear_screen()
        
    def show_status(self, status, text="", command="", result=""):
        """显示当前状态"""
        if self.mode == "single_line":
            self._show_single_line(status, text, command, result)
        else:
            self._show_multi_line(status, text, command, result)
    
    def _show_single_line(self, status, text, command, result):
        """单行显示模式"""
        line = f"Claude Echo {status}"
        if text:
            line += f" > \"{text[:20]}...\""
        if command:
            line += f" → {command[:30]}..."
        if result:
            line += f" {result}"
        
        print(f"\r{line:<80}", end="", flush=True)
    
    def _show_multi_line(self, status, text, command, result):
        """多行显示模式"""
        self.clear_line(0)
        self.clear_line(1) 
        self.clear_line(2)
        
        print(f"\r Claude Echo [{status}]")
        if text:
            print(f" > \"{text}\"")
        if command:
            print(f" → {command}")
        if result:
            print(f" {result}")
```

### 动画效果
```python
class StatusAnimator:
    def __init__(self):
        self.frames = ['◐', '◓', '◑', '◒']
        self.current_frame = 0
    
    def get_processing_indicator(self):
        """获取处理中动画指示器"""
        indicator = self.frames[self.current_frame]
        self.current_frame = (self.current_frame + 1) % len(self.frames)
        return f"\033[94m{indicator}\033[0m"
```

## 可用性测试方案

### 测试场景
1. **新用户首次使用**
   - 5秒内理解界面含义
   - 10秒内完成第一个语音命令

2. **日常使用场景**
   - 连续使用30分钟不感到疲劳
   - 错误发生时能快速理解并恢复

3. **极限环境测试**
   - 超小终端窗口 (24x8)
   - 高DPI显示器
   - 深色/浅色主题

### 测试指标
- **可读性**：2秒内理解当前状态
- **响应性**：按键响应延迟 < 100ms
- **错误恢复**：错误后3秒内恢复正常操作
- **内存占用**：UI刷新不造成内存泄漏

## 无障碍访问

### 视觉辅助
- 高对比度颜色方案
- 字体大小自适应
- 状态使用符号+颜色双重指示

### 听觉反馈（可选）
```python
def speak_status(status_text):
    """语音播报状态变化"""
    import pyttsx3
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)    # 语速
    engine.setProperty('volume', 0.7)  # 音量
    engine.say(status_text)
    engine.runAndWait()
```

## 性能优化

### 减少重绘
- 只更新变化的区域
- 使用缓存避免重复计算
- 异步更新UI状态

### 内存管理
- 限制历史记录数量
- 及时清理无用的显示缓存
- 避免字符串重复创建

## 总结

这个最小化UI设计方案专注于：

1. **极简视觉**：最少的视觉元素，最大的信息密度
2. **零学习成本**：界面含义直观明了，无需说明文档
3. **快速反馈**：每个操作都有即时的视觉反馈
4. **适应性强**：在不同大小的终端中都能正常显示

**推荐实现方案**：采用三行布局（方案二），在功能完整性和占用空间之间取得最佳平衡。