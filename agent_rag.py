# agent_rag.py —— P3 收官：会动手又会查资料的 Agent
import json
from datetime import datetime
from chat import client, MODEL
from rag import rag_answer

# ===== 执行层 =====
def calculator(expression):
    return str(eval(expression, {"__builtins__": {}}))

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def knowledge_qa(question):
    """基于健康知识库回答问题（RAG 内部已完成检索+防幻觉+引用）"""
    answer, _ = rag_answer(question)
    return answer

TOOLS = {
    "calculator": calculator,
    "get_current_time": get_current_time,
    "knowledge_qa": knowledge_qa,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "执行数学计算。需要算术、乘方、百分比等任何数值计算时使用。",
        "parameters": {"type": "object",
            "properties": {"expression": {"type": "string", "description": "数学表达式，如 '27 * 43'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "get_current_time",
        "description": "获取当前日期和时间。当用户问现在几点、今天日期时使用。",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "knowledge_qa",
        "description": "基于健康知识库回答健康科普问题。当用户询问疾病症状、体检指标含义、用药常识、饮食运动睡眠等健康话题时使用此工具。",
        "parameters": {"type": "object",
            "properties": {"question": {"type": "string", "description": "用户的健康问题原话"}},
            "required": ["question"]}}},
]

def run_agent(question, max_steps=8):
    messages = [
        {"role": "system", "content": "你是智能助手。可以使用计算、时间、健康知识库三个工具。健康相关问题交给 knowledge_qa，不要自己编造医学内容。"},
        {"role": "user", "content": question}
    ]

    for step in range(max_steps):
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = resp.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:                      # 停止条件：不点菜了 = 收工
            print(f"[第 {step + 1} 轮] 模型直接回答，收工")
            return msg.content

        print(f"[第 {step + 1} 轮] 模型点了 {len(msg.tool_calls)} 个菜")
        for call in msg.tool_calls:                 # 炒菜 + 上菜
            name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"    → 调用 {name}({args})")
            try:
                result = TOOLS[name](**args)        # 炒菜
            except Exception as e:                  # 炒糊了也是经验
                result = f"工具执行出错：{e}"
            print(f"    ← 返回：{result[:80]}")     # 无论成败，日志要记
            messages.append({                        # 无论成败，结果要回传
                "role": "tool",
                "tool_call_id": call.id,
                "content": result
            })

    return "（达到最大轮次，强制收工）"              # 安全阀

# ===== CLI =====
print("=== RAG Agent 已启动，quit 退出 ===")
while True:
    q = input("你: ")
    if q.strip().lower() == "quit":
        break
    print("助手:", run_agent(q))
