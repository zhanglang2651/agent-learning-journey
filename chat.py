# chat.py —— 和大模型对话的统一入口
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 客户端创建一次，全局复用（每次调用都新建是浪费）
client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

MODEL = "deepseek-ai/DeepSeek-V3"

def chat(messages):
    """messages: 完整对话历史列表 → 返回模型回答文本"""
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages
    )
    return resp.choices[0].message.content
def chat_stream(messages):
    """流式输出：边生成边打印，返回完整回答文本"""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True                      # 开启流式
    )
    full = ""
    for chunk in stream:                # 逐个「包裹」读取
        piece = chunk.choices[0].delta.content   # delta=增量：每个 chunk 只带新生成的一小段
        if piece:                       # 防御：首尾 chunk 可能没有内容（None）
            print(piece, end="", flush=True)     # end="" 不换行；flush=True 立即刷新
            full += piece              # 同时攒完整文本（要存进历史）
    print()                             # 收尾换行
    return full
def embed(texts):
    """把文本或文本列表变成向量（embedding）"""
    resp = client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [item.embedding for item in resp.data]
