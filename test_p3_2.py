# test_p3_2.py —— 语义检索的心脏：字面不同，意思也能命中
import numpy as np
from chat import embed

# 一个迷你「知识库」：5 条医疗常识（模拟你的笔记）
docs = [
    "感冒通常由病毒引起，常见症状包括鼻塞、流涕、咽痛和咳嗽。",
    "高血压的诊断标准是收缩压≥140mmHg或舒张压≥90mmHg。",
    "二甲双胍是2型糖尿病的一线用药，常见不良反应是胃肠道反应。",
    "发烧是免疫系统对抗感染的常见反应，体温超过38.5℃可考虑退烧药。",
    "空腹血糖正常值为3.9-6.1mmol/L，超过7.0mmol/L可诊断为糖尿病。",
]

# 建索引：全库向量化
vectors = embed(docs)

# 用户问题——注意，和库里任何一条都「字面零重合」
question = "我嗓子疼浑身没劲，是咋了？"

qv = np.array(embed(question)[0])

# 算余弦相似度：问题和每条文档的「意思接近程度」
qv_norm = qv / np.linalg.norm(qv)          # 归一化：只比方向，不比长短
scores = []
for vec in vectors:
    v = np.array(vec)
    v_norm = v / np.linalg.norm(v)
    scores.append(float(qv_norm @ v_norm))  # 归一化后点积 = 余弦相似度

# 取 top-2 最相关的
top = sorted(zip(scores, docs), reverse=True)[:2]

print("问题：", question)
print()
for score, doc in top:
    print(f"[相似度 {score:.3f}] {doc}")
