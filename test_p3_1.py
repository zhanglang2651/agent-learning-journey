# test_p3_1.py —— 对照实验：模型在什么情况下会「死」
from chat import chat

print("=== 病例 1：私有数据（它没见过你的学习计划）===")
q1 = "我的 3 个月 Agent 学习计划里，第 9 周学什么内容？请直接回答"
print("问：", q1)
print("答：", chat([{"role": "user", "content": q1}]))
print()

print("=== 病例 2：训练内的常识（它是真的知道）===")
q2 = "Python 的 dict.get() 和直接用 [] 取值有什么区别？"
print("问：", q2)
print("答：", chat([{"role": "user", "content": q2}]))
