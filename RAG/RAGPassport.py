from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List
import os

AGICTO_API_KEY = os.getenv("AGICTO_API_KEY")
AGICTO_BASE_URL = os.getenv("AGICTO_BASE_URL")


def extract_pages_as_documents(pdf) -> List[Document]:
    """
    从PDF中提取文本，每页生成一个Document对象，页码存入metadata

    参数:
        pdf: PDF文件对象

    返回:
        documents: Document对象列表，每个包含page_content和metadata中的page_number
    """
    documents = []

    for page_number, page in enumerate(pdf.pages, start=1):
        extracted_text = page.extract_text()
        if extracted_text:
            documents.append(
                Document(
                    page_content=extracted_text,
                    metadata={"page_number": page_number},
                )
            )
        else:
            print(f"No text found on page {page_number}.")

    return documents


def process_documents_with_splitter(
    documents: List[Document], save_path: str = None
) -> FAISS:
    """
    处理文档并创建向量存储，页码信息自动保留在Document.metadata中

    参数:
        documents: Document对象列表（已包含page_number元数据）
        save_path: 可选，保存向量数据库的路径

    返回:
        knowledgeBase: 基于FAISS的向量存储对象
    """

    # 创建文本分割器，用于将长文本分割成小块
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "◆", "●", "-", ".", " "],
        chunk_size=300,
        chunk_overlap=50,
    )

    # 分割文档（metadata自动继承到子文档）
    chunks = text_splitter.split_documents(documents)
    print(f"文本被分割成 {len(chunks)} 个块。")

    # 创建嵌入模型
    embeddings = OllamaEmbeddings(model="bge-m3:latest")

    # 从文档块创建知识库
    knowledgeBase = FAISS.from_documents(chunks, embeddings)
    print("已从文本块创建知识库。")

    # 如果提供了保存路径，则保存向量数据库
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        knowledgeBase.save_local(save_path)
        print(f"向量数据库已保存到: {save_path}")

    return knowledgeBase


def load_knowledge_base(load_path: str, embeddings=None) -> FAISS:
    """
    从磁盘加载向量数据库（页码信息已内嵌在Document.metadata中）

    参数:
        load_path: 向量数据库的保存路径
        embeddings: 可选，嵌入模型。如果为None，将创建一个新的OllamaEmbeddings实例

    返回:
        knowledgeBase: 加载的FAISS向量数据库对象
    """
    if embeddings is None:
        embeddings = OllamaEmbeddings(model="bge-m3:latest")

    knowledgeBase = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    print(f"向量数据库已从 {load_path} 加载。")
    return knowledgeBase


# 读取PDF文件
pdf_reader = PdfReader("./700纳税办理——日本单次签证须知.pdf")

# 提取文档（每页一个Document，页码在metadata中）
documents = extract_pages_as_documents(pdf_reader)

print(f"提取了 {len(documents)} 页文档。")

# 处理文档并创建知识库，同时保存到磁盘
save_dir = "./vector_db"
knowledgeBase = process_documents_with_splitter(documents, save_path=save_dir)


llm = ChatOpenAI(
    api_key=AGICTO_API_KEY,
    base_url=AGICTO_BASE_URL,
    model_name="deepseek-v4-flash",
    temperature=0.9,
)

# 设置查询问题
query = "身份证需要吗？有什么要求？"
# query = "需要纳税清单吗？有什么要求？"

if query:
    # 执行相似度搜索，找到与查询相关的文档
    docs = knowledgeBase.similarity_search(query, k=2)

    # 构建上下文
    context = "\n\n".join([doc.page_content for doc in docs])

    # 构建提示
    prompt = f"""根据以下上下文回答问题:

{context}

问题: {query}"""

    # 直接调用 LLM
    response = llm.invoke(prompt)
    print(response.content)
    print("来源:")

    # 记录唯一的页码
    unique_pages = set()

    # 显示每个文档块的来源页码（页码在metadata中）
    for doc in docs:
        source_page = doc.metadata.get("page_number", "未知")

        if source_page not in unique_pages:
            unique_pages.add(source_page)
            print(f"文本块页码: {source_page}")
