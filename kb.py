# kb.py —— 知识库：切分 + 向量化 + 入库 + 检索
import chromadb
from chat import embed

# ===== 1. 文档切分 =====
def chunk_text(text, size=300, overlap=50):
    """按长度切分，保留重叠区（防止语义在切口断裂）"""
    chunks = []
    step = size - overlap                    # 步长 = 窗口 - 重叠
    for i in range(0, len(text), step):
        chunks.append(text[i:i + size])
    return [c for c in chunks if c.strip()]

# ===== 2. 向量库：持久化到本地 kb_db 目录 =====
client = chromadb.PersistentClient(path="kb_db")
collection = client.get_or_create_collection("medical_kb")

# ===== 3. 入库：文档 → 切分 → 向量化 → 写入 =====
def add_docs(docs, source="notes"):
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc))
    vectors = embed(all_chunks)              # 批量向量化（一次 API 调用）
    ids = [f"{source}-{i}" for i in range(len(all_chunks))]
    collection.upsert(ids=ids, documents=all_chunks, embeddings=vectors)
    print(f"入库完成：{len(all_chunks)} 个切片，库存共 {collection.count()} 条")

# ===== 4. 检索：问题 → 向量 → top-k =====
def search(question, top_k=3):
    qv = embed([question])[0]
    res = collection.query(query_embeddings=[qv], n_results=top_k)
    return list(zip(res["documents"][0], res["distances"][0]))
