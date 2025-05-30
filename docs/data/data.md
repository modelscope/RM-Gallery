# RM Gallery: 大语言模型数据标注框架

## 🌟 主要特性

## 🚀 快速开始

### 1. 安装依赖

### 2. 启动Label Studio标注服务

```bash
# 使用Docker启动（推荐）
python rm_gallery/tools/label_studio_sevice.py start

# 或使用pip安装方式
python rm_gallery/tools/label_studio_sevice.py start --use-pip

# 检查服务状态
python rm_gallery/tools/label_studio_sevice.py status
```

启动成功后将显示：
```
🚀 Label Studio Successfully Started!
============================
🌐 Web Interface: http://localhost:8080
📧 Username: admin@example.com
🔐 Password: admin123
🔑 API Token: your_api_token_here
📁 Data Directory: ./label_studio_data
🐳 Deployment: Docker
============================
```

### 3. 创建标注项目

```python
from rm_gallery.core.data.annotation import create_annotation_module
from rm_gallery.core.data.config.label_studio_config import REWARD_BENCH_LABEL_CONFIG

# 创建标注模块
annotation_module = create_annotation_module(
    name="rm_gallery_annotation",
    api_token="your_api_token_here",  # 从Label Studio服务获取
    project_title="RM Gallery Quality Annotation",
    label_config=REWARD_BENCH_LABEL_CONFIG,
    config_processor="reward_bench"
)

# 创建标注项目
result = annotation_module.create_project()
print(f"标注项目已创建: {result.metadata['annotation_server_url']}/projects/{result.metadata['annotation_project_id']}")
```

### 4. 导出标注结果

```python
# 在完成标注后导出结果
annotated_dataset = annotation_module.export_annotations_to_dataset(
    filename="exported_annotations.json",
    include_original_data=True
)

print(f"导出了 {len(annotated_dataset.datas)} 个已标注样本")
```

## 📋 标注配置类型


## 🎯 标注使用场景


### 1. 奖励模型数据标注
```python
# 专门为奖励模型训练准备标注数据
annotation_module = create_annotation_module(
    name="reward_model_annotation",
    label_config=REWARD_BENCH_LABEL_CONFIG,
    config_processor="reward_bench"
)
```

### 2. 对话质量评估
```python
# 评估对话系统的回复质量
annotation_module = create_annotation_module(
    name="dialogue_quality",
    label_config=CONVERSATION_QUALITY_CONFIG
)
```

### 3. 偏好学习数据准备
```python
# 为RLHF训练准备偏好数据
annotation_module = create_annotation_module(
    name="preference_annotation",
    label_config=PREFERENCE_RANKING_CONFIG
)
```

## 🔧 自定义标注配置

### 创建自定义标注界面



## 📊 支持的标注数据格式




## 📈 标注性能特性


## 🔍 故障排除

### 常见问题

1. **Label Studio服务启动失败**
```bash
# 检查端口占用
lsof -i :8080

# 使用不同端口
python rm_gallery/tools/label_studio_sevice.py start --port 8081
```

2. **API Token获取失败**
```bash
# 检查服务状态
python rm_gallery/tools/label_studio_sevice.py status

# 重新启动服务
python rm_gallery/tools/label_studio_sevice.py stop
python rm_gallery/tools/label_studio_sevice.py start
```

## 📚 文档

