# rag.py —— RAG 完整闭环：检索 → 增强 → 生成
from chat import chat
from kb import search

# 防幻觉 prompt（医疗级，P4 直接复用）
SYSTEM_PROMPT = """你是健康科普助手。严格遵守：
1. 只能根据「参考资料」回答问题，禁止使用你自己的知识补充
2. 参考资料中没有的信息，直接说「根据现有资料无法回答」，不要编造
3. 回答末尾列出你引用了哪几条资料的编号
4. 涉及诊断结论、用药建议，拒绝回答并建议就医"""

def rag_answer(question, top_k=3):
    # 检索：问题 → top-k 资料切片
    results = search(question, top_k=top_k)

    # 增强：拼成带编号的上下文（编号是为了让模型能标注引用）
    context = "\n".join(f"[资料{i+1}] {doc}" for i, (doc, dist) in enumerate(results))

    # 生成：资料 + 问题一起发给模型
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n问题：{question}"}
    ]
    answer = chat(messages)
    return answer, results
