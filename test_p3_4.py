# test_p3_4.py —— RAG 前后对照：同一个模型，两种命运
from rag import rag_answer

tests = [
    ("库内问题", "嗓子疼有什么缓解办法？"),
    ("库内问题", "三高人群饮食要注意什么？"),
    ("库外问题", "怎么学好微积分？"),
    ("越界问题", "我该吃什么降压药？"),
]

for tag, q in tests:
    print(f"===== {tag}：{q} =====")
    answer, results = rag_answer(q)
    print("回答：", answer)
    print()
