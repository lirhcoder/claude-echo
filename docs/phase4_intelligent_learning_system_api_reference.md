# 第四阶段智能学习系统 API 参考文档

## 概述

本文档提供Claude Echo第四阶段智能学习系统的完整API参考，包括所有学习组件、代理和接口的详细说明。

## 目录

1. [核心学习框架 API](#核心学习框架-api)
2. [学习代理 API](#学习代理-api)
3. [语音学习系统 API](#语音学习系统-api)
4. [数据管理 API](#数据管理-api)
5. [事件系统 API](#事件系统-api)
6. [配置管理 API](#配置管理-api)

---

## 核心学习框架 API

### BaseLearner

抽象基类，定义了所有学习算法的统一接口。

#### 类定义

```python
from src.learning.base_learner import BaseLearner, LearningMode, LearningStage

class BaseLearner(ABC):
    """学习算法基类"""
```

#### 核心方法

##### `initialize(config: Dict[str, Any]) -> bool`

**描述**: 初始化学习器

**参数**:
- `config: Dict[str, Any]` - 学习器配置参数

**返回**: `bool` - 初始化是否成功

**示例**:
```python
config = {
    "learning_mode": "online",
    "model_path": "./models/custom_learner",
    "batch_size": 100
}
success = await learner.initialize(config)
```

##### `learn(data: List[Dict[str, Any]], context: LearningContext) -> LearningResult`

**描述**: 执行学习操作

**参数**:
- `data: List[Dict[str, Any]]` - 学习数据
- `context: LearningContext` - 学习上下文

**返回**: `LearningResult` - 学习结果

**示例**:
```python
from src.learning.base_learner import LearningContext, LearningResult

context = LearningContext(
    user_id="user123",
    agent_id="agent456",
    session_id="session789"
)

data = [
    {"interaction_type": "command", "result": "success"},
    {"interaction_type": "query", "result": "failure"}
]

result = await learner.learn(data, context)
print(f"Learning confidence: {result.confidence}")
```

##### `get_insights() -> Dict[str, Any]`

**描述**: 获取学习洞察和统计信息

**返回**: `Dict[str, Any]` - 包含学习洞察的字典

**示例**:
```python
insights = await learner.get_insights()
print(f"Total learnings: {insights['total_learnings']}")
print(f"Success rate: {insights['success_rate']}")
```

#### 学习模式枚举

```python
class LearningMode(Enum):
    ONLINE = "online"              # 实时学习
    BATCH = "batch"                # 批处理学习
    REINFORCEMENT = "reinforcement" # 强化学习
    SUPERVISED = "supervised"       # 监督学习
    UNSUPERVISED = "unsupervised"   # 无监督学习
```

#### 学习阶段枚举

```python
class LearningStage(Enum):
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    VALIDATION = "validation"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
```

### AdaptiveBehaviorManager

自适应行为管理器，负责系统行为的智能调整。

#### 类定义

```python
from src.learning.adaptive_behavior import AdaptiveBehaviorManager

class AdaptiveBehaviorManager:
    """自适应行为管理器"""
```

#### 核心方法

##### `analyze_behavior_patterns() -> List[BehaviorPattern]`

**描述**: 分析系统行为模式

**返回**: `List[BehaviorPattern]` - 检测到的行为模式列表

**示例**:
```python
patterns = await behavior_manager.analyze_behavior_patterns()
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}, Confidence: {pattern.confidence}")
```

##### `generate_adaptation_strategies(patterns: List[BehaviorPattern]) -> List[AdaptationStrategy]`

**描述**: 根据行为模式生成适应策略

**参数**:
- `patterns: List[BehaviorPattern]` - 行为模式列表

**返回**: `List[AdaptationStrategy]` - 适应策略列表

**示例**:
```python
strategies = await behavior_manager.generate_adaptation_strategies(patterns)
for strategy in strategies:
    print(f"Strategy: {strategy.name}, Expected Impact: {strategy.expected_impact}")
```

##### `apply_adaptation_strategy(strategy: AdaptationStrategy) -> bool`

**描述**: 应用适应策略

**参数**:
- `strategy: AdaptationStrategy` - 要应用的策略

**返回**: `bool` - 应用是否成功

**示例**:
```python
success = await behavior_manager.apply_adaptation_strategy(strategy)
if success:
    print(f"Successfully applied strategy: {strategy.name}")
```

---

## 学习代理 API

### LearningAgent

学习统筹代理，负责协调所有学习功能。

#### 类定义

```python
from src.agents.learning_agent import LearningAgent

class LearningAgent(BaseAgent):
    """学习统筹代理"""
```

#### 核心方法

##### `analyze_user_behavior_patterns(user_id: str) -> Dict[str, Any]`

**描述**: 分析用户行为模式

**参数**:
- `user_id: str` - 用户ID

**返回**: `Dict[str, Any]` - 用户行为分析结果

**示例**:
```python
patterns = await learning_agent.analyze_user_behavior_patterns("user123")
print(f"User patterns: {patterns}")
```

##### `optimize_agent_performance(agent_ids: List[str]) -> Dict[str, float]`

**描述**: 优化指定代理的性能

**参数**:
- `agent_ids: List[str]` - 要优化的代理ID列表

**返回**: `Dict[str, float]` - 优化结果，键为代理ID，值为性能改善比例

**示例**:
```python
agent_ids = ["coordinator", "task_planner", "auto_worker"]
improvements = await learning_agent.optimize_agent_performance(agent_ids)
for agent_id, improvement in improvements.items():
    print(f"{agent_id}: {improvement:.2%} improvement")
```

##### `coordinate_multi_agent_learning(agent_ids: List[str], objective: str, mode: str) -> Dict[str, Any]`

**描述**: 协调多代理学习

**参数**:
- `agent_ids: List[str]` - 参与学习的代理ID列表
- `objective: str` - 学习目标
- `mode: str` - 学习模式

**返回**: `Dict[str, Any]` - 协调结果

**示例**:
```python
result = await learning_agent.coordinate_multi_agent_learning(
    ["user_profile_agent", "correction_agent"],
    "improve_user_experience",
    "collaborative"
)
print(f"Coordination result: {result}")
```

### UserProfileAgent

用户配置代理，负责用户档案和偏好管理。

#### 类定义

```python
from src.agents.user_profile_agent import UserProfileAgent

class UserProfileAgent(BaseAgent):
    """用户配置代理"""
```

#### 核心方法

##### `identify_user_by_voice(voice_data: bytes) -> Optional[str]`

**描述**: 基于声纹识别用户

**参数**:
- `voice_data: bytes` - 语音数据

**返回**: `Optional[str]` - 用户ID，如果识别失败返回None

**示例**:
```python
with open("voice_sample.wav", "rb") as f:
    voice_data = f.read()

user_id = await user_profile_agent.identify_user_by_voice(voice_data)
if user_id:
    print(f"Identified user: {user_id}")
```

##### `create_user_profile(user_data: Dict[str, Any]) -> UserProfile`

**描述**: 创建用户档案

**参数**:
- `user_data: Dict[str, Any]` - 用户数据

**返回**: `UserProfile` - 创建的用户档案

**示例**:
```python
user_data = {
    "name": "张三",
    "preferences": {
        "language": "zh-CN",
        "voice_speed": 1.2
    }
}

profile = await user_profile_agent.create_user_profile(user_data)
print(f"Created profile for: {profile.name}")
```

##### `enroll_voice_profile(user_id: str, voice_samples: List[bytes]) -> bool`

**描述**: 注册用户声纹档案

**参数**:
- `user_id: str` - 用户ID
- `voice_samples: List[bytes]` - 声纹样本列表

**返回**: `bool` - 注册是否成功

**示例**:
```python
voice_samples = []
for i in range(5):  # 收集5个样本
    with open(f"voice_sample_{i}.wav", "rb") as f:
        voice_samples.append(f.read())

success = await user_profile_agent.enroll_voice_profile("user123", voice_samples)
if success:
    print("Voice profile enrolled successfully")
```

### CorrectionAgent

错误纠正代理，负责交互式错误纠正和学习。

#### 类定义

```python
from src.agents.correction_agent import CorrectionAgent

class CorrectionAgent(BaseAgent):
    """错误纠正代理"""
```

#### 核心方法

##### `process_user_correction(correction_data: Dict[str, Any]) -> CorrectionResult`

**描述**: 处理用户纠正反馈

**参数**:
- `correction_data: Dict[str, Any]` - 纠正数据

**返回**: `CorrectionResult` - 纠正处理结果

**示例**:
```python
correction_data = {
    "original_response": "原始回复",
    "corrected_response": "纠正后的回复",
    "correction_type": "RESPONSE_CONTENT",
    "user_id": "user123",
    "context": {"session_id": "session456"}
}

result = await correction_agent.process_user_correction(correction_data)
print(f"Correction processed: {result.success}")
```

##### `identify_correction_patterns(user_id: Optional[str] = None) -> List[CorrectionPattern]`

**描述**: 识别纠正模式

**参数**:
- `user_id: Optional[str]` - 用户ID，如果为None则分析所有用户

**返回**: `List[CorrectionPattern]` - 纠正模式列表

**示例**:
```python
patterns = await correction_agent.identify_correction_patterns("user123")
for pattern in patterns:
    print(f"Pattern: {pattern.pattern_type}, Frequency: {pattern.frequency}")
```

---

## 语音学习系统 API

### SpeechLearningManager

语音学习管理器，统筹所有语音学习功能。

#### 类定义

```python
from src.speech.speech_learning_manager import SpeechLearningManager

class SpeechLearningManager:
    """语音学习管理器"""
```

#### 核心方法

##### `register_learner(learner_type: str, learner: Any) -> bool`

**描述**: 注册语音学习器

**参数**:
- `learner_type: str` - 学习器类型
- `learner: Any` - 学习器实例

**返回**: `bool` - 注册是否成功

**示例**:
```python
from src.speech.voice_profile_learner import VoiceProfileLearner

voice_learner = VoiceProfileLearner()
success = await speech_manager.register_learner("voice_profile", voice_learner)
```

##### `process_speech_interaction(audio_data: bytes, user_id: str) -> Dict[str, Any]`

**描述**: 处理语音交互并学习

**参数**:
- `audio_data: bytes` - 音频数据
- `user_id: str` - 用户ID

**返回**: `Dict[str, Any]` - 处理结果

**示例**:
```python
with open("speech.wav", "rb") as f:
    audio_data = f.read()

result = await speech_manager.process_speech_interaction(audio_data, "user123")
print(f"Recognition result: {result['text']}")
```

### VoiceProfileLearner

声纹学习器，负责用户声纹特征学习。

#### 类定义

```python
from src.speech.voice_profile_learner import VoiceProfileLearner

class VoiceProfileLearner(BaseLearner):
    """声纹学习器"""
```

#### 核心方法

##### `extract_voice_features(audio_data: bytes) -> np.ndarray`

**描述**: 提取语音特征

**参数**:
- `audio_data: bytes` - 音频数据

**返回**: `np.ndarray` - 语音特征向量

**示例**:
```python
features = await voice_learner.extract_voice_features(audio_data)
print(f"Feature vector shape: {features.shape}")
```

##### `update_voice_profile(user_id: str, voice_features: np.ndarray) -> bool`

**描述**: 更新用户声纹档案

**参数**:
- `user_id: str` - 用户ID
- `voice_features: np.ndarray` - 语音特征

**返回**: `bool` - 更新是否成功

**示例**:
```python
success = await voice_learner.update_voice_profile("user123", features)
if success:
    print("Voice profile updated")
```

---

## 数据管理 API

### LearningDataManager

学习数据管理器，负责学习数据的存储和管理。

#### 类定义

```python
from src.learning.learning_data_manager import LearningDataManager

class LearningDataManager:
    """学习数据管理器"""
```

#### 核心方法

##### `store_learning_data(data: LearningData) -> str`

**描述**: 存储学习数据

**参数**:
- `data: LearningData` - 学习数据对象

**返回**: `str` - 数据ID

**示例**:
```python
from src.learning.learning_data_manager import LearningData, DataPrivacyLevel

learning_data = LearningData(
    user_id="user123",
    agent_id="learning_agent",
    data_type="user_interaction",
    data_content={"action": "query", "result": "success"},
    privacy_level=DataPrivacyLevel.PRIVATE
)

data_id = await data_manager.store_learning_data(learning_data)
print(f"Data stored with ID: {data_id}")
```

##### `retrieve_learning_data(user_id: Optional[str] = None, data_type: Optional[str] = None, limit: int = 100) -> List[LearningData]`

**描述**: 检索学习数据

**参数**:
- `user_id: Optional[str]` - 用户ID过滤器
- `data_type: Optional[str]` - 数据类型过滤器
- `limit: int` - 最大返回数量

**返回**: `List[LearningData]` - 学习数据列表

**示例**:
```python
data_list = await data_manager.retrieve_learning_data(
    user_id="user123",
    data_type="user_interaction",
    limit=50
)
print(f"Retrieved {len(data_list)} data points")
```

##### `cleanup_expired_data() -> Dict[str, int]`

**描述**: 清理过期数据

**返回**: `Dict[str, int]` - 清理统计信息

**示例**:
```python
cleanup_stats = await data_manager.cleanup_expired_data()
print(f"Cleaned up {cleanup_stats['deleted_count']} expired records")
```

#### 数据隐私级别

```python
class DataPrivacyLevel(Enum):
    PUBLIC = "public"              # 公开数据
    INTERNAL = "internal"          # 内部数据
    PRIVATE = "private"            # 私有数据
    CONFIDENTIAL = "confidential"  # 机密数据
```

---

## 事件系统 API

### 学习事件类型

```python
from src.learning.learning_events import LearningEventType

class LearningEventType(Enum):
    # 用户学习事件
    USER_PATTERN_DETECTED = "learning.user_pattern_detected"
    USER_PREFERENCE_UPDATED = "learning.user_preference_updated"
    USER_BEHAVIOR_ANALYZED = "learning.user_behavior_analyzed"
    
    # 模型训练事件
    MODEL_TRAINING_STARTED = "learning.model_training_started"
    MODEL_TRAINING_COMPLETED = "learning.model_training_completed"
    MODEL_UPDATED = "learning.model_updated"
    
    # 系统适应事件
    ADAPTATION_APPLIED = "learning.adaptation_applied"
    PERFORMANCE_IMPROVEMENT_DETECTED = "learning.performance_improvement_detected"
    SYSTEM_ADAPTATION_COMPLETED = "learning.system_adaptation_completed"
```

### LearningEventFactory

学习事件工厂，用于创建标准化的学习事件。

#### 方法

##### `create_user_pattern_event(user_id: str, pattern_data: Dict[str, Any]) -> LearningEvent`

**描述**: 创建用户模式检测事件

**参数**:
- `user_id: str` - 用户ID
- `pattern_data: Dict[str, Any]` - 模式数据

**返回**: `LearningEvent` - 学习事件

**示例**:
```python
from src.learning.learning_events import LearningEventFactory

pattern_data = {
    "pattern_type": "usage_frequency",
    "confidence": 0.85,
    "details": {"peak_hours": [9, 14, 20]}
}

event = LearningEventFactory.create_user_pattern_event("user123", pattern_data)
await event_system.emit_event(event)
```

---

## 配置管理 API

### 学习系统配置结构

学习系统使用YAML格式的配置文件，主要配置节点包括：

#### 核心配置节点

```yaml
learning:
  enabled: true
  
  data_manager:
    db_path: "./data/learning.db"
    encryption_enabled: true
    cache_size: 1000
  
  learners:
    adaptive_behavior:
      enabled: true
      learning_mode: "online"
      batch_size: 100
```

#### 访问配置

```python
from src.core.config_manager import ConfigManager

config = ConfigManager()

# 获取学习系统配置
learning_config = config.get_section("learning")
print(f"Learning enabled: {learning_config['enabled']}")

# 获取特定学习器配置
adaptive_config = config.get_nested("learning.learners.adaptive_behavior")
print(f"Adaptive learning mode: {adaptive_config['learning_mode']}")
```

---

## 错误处理和状态码

### 常见错误类型

#### LearningError

```python
from src.learning.base_learner import LearningError

class LearningError(Exception):
    """学习系统基础异常"""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
```

#### 错误代码

| 错误代码 | 描述 | 处理建议 |
|---------|------|----------|
| `LEARNING_001` | 学习器初始化失败 | 检查配置文件和依赖 |
| `LEARNING_002` | 数据存储失败 | 检查数据库连接和权限 |
| `LEARNING_003` | 模型训练失败 | 检查训练数据质量 |
| `LEARNING_004` | 用户未找到 | 验证用户ID的有效性 |
| `LEARNING_005` | 权限不足 | 检查用户权限设置 |

---

## 性能监控和指标

### 学习系统指标

#### 获取指标数据

```python
from src.learning.adaptive_behavior import AdaptiveBehaviorManager

behavior_manager = AdaptiveBehaviorManager()
metrics = await behavior_manager.get_performance_metrics()

print(f"Learning metrics: {metrics}")
```

#### 关键性能指标

| 指标名称 | 描述 | 单位 |
|---------|------|------|
| `total_learnings` | 总学习次数 | 次数 |
| `successful_learnings` | 成功学习次数 | 次数 |
| `average_confidence` | 平均置信度 | 0-1 |
| `learning_rate` | 学习速率 | 次/小时 |
| `adaptation_success_rate` | 适应成功率 | 百分比 |

---

## 使用示例

### 完整学习流程示例

```python
import asyncio
from src.agents.learning_agent import LearningAgent
from src.agents.user_profile_agent import UserProfileAgent
from src.learning.learning_data_manager import LearningDataManager, LearningData, DataPrivacyLevel

async def complete_learning_example():
    # 初始化组件
    learning_agent = LearningAgent("learning_agent")
    user_profile_agent = UserProfileAgent("user_profile_agent")
    data_manager = LearningDataManager()
    
    await learning_agent.initialize({})
    await user_profile_agent.initialize({})
    await data_manager.initialize({})
    
    # 创建用户档案
    user_data = {
        "name": "测试用户",
        "preferences": {"language": "zh-CN"}
    }
    profile = await user_profile_agent.create_user_profile(user_data)
    
    # 存储学习数据
    learning_data = LearningData(
        user_id=profile.user_id,
        agent_id="learning_agent",
        data_type="user_interaction",
        data_content={"action": "query", "result": "success"},
        privacy_level=DataPrivacyLevel.PRIVATE
    )
    
    data_id = await data_manager.store_learning_data(learning_data)
    
    # 执行学习分析
    patterns = await learning_agent.analyze_user_behavior_patterns(profile.user_id)
    print(f"Detected patterns: {patterns}")
    
    # 优化系统性能
    improvements = await learning_agent.optimize_agent_performance(["coordinator"])
    print(f"Performance improvements: {improvements}")

# 运行示例
if __name__ == "__main__":
    asyncio.run(complete_learning_example())
```

### 语音学习示例

```python
async def speech_learning_example():
    from src.speech.speech_learning_manager import SpeechLearningManager
    
    speech_manager = SpeechLearningManager()
    await speech_manager.initialize({})
    
    # 处理语音交互
    with open("user_speech.wav", "rb") as f:
        audio_data = f.read()
    
    result = await speech_manager.process_speech_interaction(audio_data, "user123")
    
    print(f"Recognition result: {result['text']}")
    print(f"Learning applied: {result['learning_applied']}")
```

---

## 版本信息

- **API版本**: 1.0.0
- **文档版本**: 1.0.0
- **最后更新**: 2025-08-09
- **兼容性**: Python 3.9+

## 相关文档

- [智能学习系统架构文档](learning_system_architecture.md)
- [用户使用手册](phase4_user_manual.md)
- [开发者指南](phase4_developer_guide.md)
- [系统部署手册](phase4_deployment_manual.md)