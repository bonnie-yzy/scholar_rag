import os
import re
import requests
import fitz
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from src.utils.logger import setup_logger
from src.config import settings  # <--- 引入配置
import openai
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)

# --- 1. 修改 Embedding 函数，支持配置化的 Batch Size ---
class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str, model_name: str, batch_size: int = 32):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.batch_size = batch_size  # <--- 从配置接收 batch_size

    def __call__(self, input: Documents) -> Embeddings:
        input = [text.replace("\n", " ") for text in input]
        all_embeddings = []

        # 使用配置中的 self.batch_size
        for i in range(0, len(input), self.batch_size):
            batch = input[i : i + self.batch_size]
            try:
                response = self.client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"❌ Embedding API Error at batch {i}: {e}")
                raise e
        return all_embeddings

# --- 2. 修改主类，应用 Chunk 参数 ---
class LocalVectorStore:
    def __init__(self, persist_dir="data/vector_store"):
        self.logger = setup_logger("VectorStore")
        self.pdf_dir = "data/cache/pdfs"
        
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.logger.info(f"Init Embedding: {settings.EMBEDDING_MODEL_NAME} | Batch: {settings.EMBEDDING_BATCH_SIZE}")
        
        # [修改] 传入配置的 batch_size
        self.ef = SiliconFlowEmbeddingFunction(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model_name=settings.EMBEDDING_MODEL_NAME,
            batch_size=settings.EMBEDDING_BATCH_SIZE
        )
        
        self.collection = self.client.get_or_create_collection(
            name="research_papers",
            embedding_function=self.ef
        )

    def _sanitize_filename(self, title):
        clean_name = re.sub(r'[\\/*?:"<>|]', "", title)
        clean_name = " ".join(clean_name.split())
        return clean_name[:50]

    def _download_pdf(self, url, paper_id, title):
        """下载 PDF (带详细 Debug)"""
        if not url:
            return None
        
        safe_title = self._sanitize_filename(title)
        short_id = paper_id.split("/")[-1]
        file_name = f"{short_id}_{safe_title}.pdf"
        save_path = os.path.join(self.pdf_dir, file_name)
        
        if os.path.exists(save_path):
            return save_path

        try:
            # 稍微增强一点 Headers，模拟真实浏览器
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
                'Referer': 'https://scholar.google.com/' 
            }
            # verify=False 解决证书问题，stream=True 优化大文件下载
            response = requests.get(url, headers=headers, timeout=20, verify=False, stream=True)
            
            # [新增] 检查状态码，如果不是 200，主动抛出异常以便被 except 捕获并打印
            if response.status_code != 200:
                raise ValueError(f"HTTP {response.status_code}")

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return save_path

        except Exception as e:
            # [新增] 打印具体的 URL 和错误原因，方便你排查
            self.logger.warning(f"⚠️ Download Failed: {str(e)} | URL: {url}")
            # 如果是 IEEE 这种反爬严重的，这里会打印 HTTP 418 或 403
            return None

    def _parse_pdf(self, pdf_path):
        if not pdf_path: return ""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text
        except Exception as e:
            self.logger.error(f"PDF parse error: {e}")
            return ""

    # [关键修改] 让 _chunk_text 读取配置参数
    def _chunk_text(self, text):
        """使用 .env 配置进行切片"""
        chunk_size = settings.RAG_CHUNK_SIZE
        overlap = settings.RAG_CHUNK_OVERLAP
        
        if not text: return []
        chunks = []
        # 滑动窗口切片逻辑
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) > 100: # 过滤太短的噪音
                chunks.append(chunk)
        return chunks

    def add_papers(self, papers_metadata):
        self.logger.info(f"Processing papers (Chunk Size: {settings.RAG_CHUNK_SIZE}, Batch: {settings.EMBEDDING_BATCH_SIZE})")
        new_count = 0
        
        for paper in tqdm(papers_metadata, desc="Building VectorDB"):
            paper_id = paper['id']
            title = paper['title']
            
            # 增量检查
            existing = self.collection.get(where={"paper_id": paper_id}, limit=1)
            if existing['ids']:
                continue

            # 下载逻辑不变...
            pdf_path = self._download_pdf(paper.get('pdf_url'), paper_id, title)
            if not pdf_path:
                continue 
            
            full_text = self._parse_pdf(pdf_path)
            chunks = self._chunk_text(full_text)
            if not chunks:
                continue

            ids = [f"{paper_id}_chk_{i}" for i in range(len(chunks))]
            
            # [修改点 1]：在 metadata 中存入 url 和 pdf_url
            metadatas = [{
                "paper_id": paper_id,
                "title": title,
                "url": paper.get('url', ''),          # <--- 新增：DOI或落地页链接
                "pdf_url": paper.get('pdf_url', ''),  # <--- 新增：PDF下载链接
                "year": paper['year'] or 0,
                "chunk_index": i
            } for i in range(len(chunks))]
            
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            new_count += len(chunks)

        if new_count > 0:
            self.logger.info(f"Success: Added {new_count} new chunks to local DB.")
        else:
            self.logger.info("Skipped: All papers already exist in DB or download failed.")

    def search(self, query: str, top_k: int = 5):
        if self.collection.count() == 0:
            return []
        
        # 检索时也会自动调用 Embedding API 将 query 向量化
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        structured_results = []
        for i in range(len(docs)):
            item = metas[i]           # 获取 metadata (含 title, url, year)
            item['content'] = docs[i] # 将正文内容塞进去
            structured_results.append(item)
        return structured_results # 返回 List[Dict]