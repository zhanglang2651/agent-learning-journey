# P3 · RAG 知识库问答助手

给 Agent 接上「知识库」：从裸语义检索到防幻觉问答，让模型拿着资料答话。

## 功能

- 语义检索：BGE-M3 向量化 + ChromaDB，字面不同也能按「意思」命中
- 文档切分：300 字窗口 + 50 字重叠，长文自动分块
- 防幻觉问答：四原则 prompt（只准用资料 / 没有就说没有 / 带引用 / 医疗问题拒答并引导就医）
- 引用溯源：答案带 [资料N] 编号，可人工核对
- RAG + Agent 融合：knowledge_qa 作为工具，Agent 自动路由健康问题

## 技术栈

Python 3.13 · 硅基流动 DeepSeek-V3 + BGE-M3 · ChromaDB · numpy

## 运行

1. `pip install openai python-dotenv numpy chromadb`
2. 复制 `.env.example` 为 `.env`，填入 `SILICONFLOW_API_KEY`
3. `python test_p3_3.py`（首次灌库）
4. `python agent_rag.py`

## 项目结构

```
chat.py       —— API 封装（对话 + embedding）
kb.py         —— 知识库：切分 / 入库 / 检索
rag.py        —— RAG pipeline：检索 → 增强 → 防幻觉生成
agent_rag.py  —— 融合 Agent：计算 / 时间 / 知识库三工具路由
test_p3_1~4.py —— 各课验证实验（含 S1 幻觉病历对照）
```

## 踩坑记录

- chromadb 在 Python 3.14 上无匹配版本（依赖链不兼容）→ 换 3.13
- 四包联合安装 ResolutionImpossible → 分开装逐个击破
- 「嗓子疼缓解办法」被过度拒答 → 知识库覆盖不足，补资料后修复——**RAG 是迭代出来的，不是一次写对的**
