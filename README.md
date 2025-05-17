# 电子商务客户服务助手框架（RAG技术实现）

```markdown
## 环境搭建
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # 可选的NLP依赖

## 数据准备
- 文本数据存放路径：`data/documents/`（支持.txt格式）
- 商品图片存放路径：`data/images/`

## 索引构建
```bash
python scripts/preprocess.py  # 生成向量并索引到ES

## 启动服务
```bash
python app.py  # 访问http://localhost:8888/chat

## API 调用示例
```bash
curl -X POST http://localhost:8888/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "这款羽绒服的填充物是什么？", "image": "/data/images/down-jacket-123.jpg"}'

## 注意事项

### 模型选择
- 文本嵌入模型替换：`sentence-transformers/LaBSE`（多语言支持）
- 图像模型升级：`openai/clip-vit-large-patch14`

### 性能优化
1. 大模型量化：
   ```python
   # 示例：Llama 2 GPTQ量化
   model.quantize(gptq_config)
2. 向量检索优化：
   ```json
   // ES索引配置
   "hnsw": {
     "ef_construction": 128,
     "m": 16
   }

### 多语言支持
```python
# 添加翻译模块示例
from transformers import MarianMT
translator = MarianMT.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
