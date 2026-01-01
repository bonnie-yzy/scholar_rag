import os
import re
import asyncio
import aiohttp
import fitz
import chromadb
from datetime import datetime  # <--- 新增
from urllib.parse import urlparse
from collections import defaultdict
from chromadb import Documents, EmbeddingFunction, Embeddings
from tqdm import tqdm
from src.utils.logger import setup_logger
from src.config import settings
import openai
import warnings
from urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter('ignore', InsecureRequestWarning)

# --- 1. Embedding 函数 (保持不变) ---
class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, base_url: str, model_name: str, batch_size: int = 32):
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        input = [text.replace("\n", " ") for text in input]
        all_embeddings = []
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

# --- 2. 异步 VectorStore (支持 Debug 文件写入) ---
class LocalVectorStore:
    def __init__(self, persist_dir="data/vector_store"):
        self.logger = setup_logger("VectorStore")
        
        # [修改] 保存 root_dir 为实例变量，方便后续写入 debug 文件
        self.root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.pdf_dir = os.path.join(self.root_dir, "data", "pdfs")
        self.debug_file_path = os.path.join(self.root_dir, "data", "papers_debug.txt") # <--- 定位 Debug 文件
        
        os.makedirs(self.pdf_dir, exist_ok=True)
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)
        
        self.logger.info(f"Init Embedding: {settings.EMBEDDING_MODEL_NAME} | Batch: {settings.EMBEDDING_BATCH_SIZE}")
        
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
        
        self.semaphore = asyncio.Semaphore(5)

    def _sanitize_filename(self, title):
        clean_name = re.sub(r'[\\/*?:"<>|]', "", title)
        clean_name = " ".join(clean_name.split())
        return clean_name[:50]

    def _get_headers(self):
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/pdf,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Referer': 'https://scholar.google.com/'
        }

    async def _download_one_pdf(self, session: aiohttp.ClientSession, url: str, paper_id: str, title: str) -> dict:
        result = {
            "id": paper_id, 
            "url": url, 
            "path": None, 
            "status": "failed", 
            "code": 0, 
            "domain": "unknown"
        }
        
        if not url: return result

        try:
            domain = urlparse(url).netloc
            result["domain"] = domain
        except:
            pass
        
        safe_title = self._sanitize_filename(title)
        short_id = paper_id.split("/")[-1]
        file_name = f"{short_id}_{safe_title}.pdf"
        save_path = os.path.join(self.pdf_dir, file_name)
        
        if os.path.exists(save_path):
            result["path"] = save_path
            result["status"] = "cached"
            result["code"] = 200
            return result

        async with self.semaphore:
            try:
                async with session.get(url, headers=self._get_headers(), timeout=30, ssl=False) as response:
                    result["code"] = response.status
                    
                    if response.status == 200:
                        content_type = response.headers.get('Content-Type', '').lower()
                        if 'pdf' not in content_type and 'octet-stream' not in content_type:
                            self.logger.warning(f"⚠️ [Type Mismatch] {domain} | {url} | Type: {content_type}")
                            result["status"] = "invalid_type"
                            return result

                        with open(save_path, "wb") as f:
                            while True:
                                chunk = await response.content.read(8192)
                                if not chunk: break
                                f.write(chunk)
                        
                        result["path"] = save_path
                        result["status"] = "success"
                        self.logger.info(f"⬇️ [200 OK] {domain} | {title[:30]}...")
                    else:
                        self.logger.warning(f"❌ [{response.status}] {domain} | {url}")
                        
            except asyncio.TimeoutError:
                self.logger.warning(f"⏳ [Timeout] {result['domain']} | {url}")
                result["status"] = "timeout"
            except Exception as e:
                self.logger.warning(f"💥 [Error] {result['domain']} | {url} | {str(e)}")
                result["status"] = "error"
                
        return result

    def _print_download_stats(self, results: list):
        """[修改] 生成报告并同时：1.打印到控制台 2.追加到 papers_debug.txt"""
        total = len(results)
        if total == 0: return

        stats = defaultdict(lambda: {"total": 0, "success": 0})
        failures = []

        for r in results:
            domain = r["domain"]
            stats[domain]["total"] += 1
            if r["status"] in ["success", "cached"]:
                stats[domain]["success"] += 1
            else:
                failures.append(r)

        # 构建报告字符串
        lines = []
        lines.append(f"\n{'='*60}")
        lines.append(f"📊 PDF DOWNLOAD REPORT (Total: {total})")
        lines.append(f"Timestamp: {datetime.now()}")
        lines.append("="*60)
        lines.append(f"{'DOMAIN':<30} | {'TOTAL':<8} | {'SUCCESS':<8} | {'RATE':<8}")
        lines.append("-" * 60)
        
        for domain, data in sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True):
            rate = (data['success'] / data['total']) * 100
            lines.append(f"{domain:<30} | {data['total']:<8} | {data['success']:<8} | {rate:.1f}%")
        
        if failures:
            lines.append("-" * 60)
            lines.append("❌ FAILURES (Top 5):")
            for f in failures[:5]:
                lines.append(f"   • [{f['code']}] {f['domain']} -> {f['url']}")
            if len(failures) > 5:
                lines.append(f"   ... and {len(failures)-5} more.")
        lines.append("="*60 + "\n")

        report_str = "\n".join(lines)

        # 1. 打印到控制台
        print(report_str)

        # 2. 追加到 Debug 文件
        try:
            # 使用 'a' 模式追加，确保不覆盖 OpenAlexRetriever 写入的内容
            with open(self.debug_file_path, "a", encoding="utf-8") as f:
                f.write(report_str)
            self.logger.info(f"Appended download report to {self.debug_file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to write stats to debug file: {e}")

    async def _download_batch(self, papers_to_download: list) -> dict:
        """批量下载调度器"""
        if not papers_to_download: return {}

        self.logger.info(f"Starting async download for {len(papers_to_download)} papers...")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for p in papers_to_download:
                tasks.append(
                    self._download_one_pdf(session, p['pdf_url'], p['id'], p['title'])
                )
            results_list = await asyncio.gather(*tasks)
            
        # 统计并写入文件
        self._print_download_stats(results_list)

        path_map = {}
        for r in results_list:
            if r["path"]:
                path_map[r["id"]] = r["path"]
        
        return path_map

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

    def _chunk_text(self, text):
        chunk_size = settings.RAG_CHUNK_SIZE
        overlap = settings.RAG_CHUNK_OVERLAP
        if not text: return []
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i : i + chunk_size]
            if len(chunk) > 100:
                chunks.append(chunk)
        return chunks

    def add_papers(self, papers_metadata):
        self.logger.info(f"Processing {len(papers_metadata)} papers...")
        
        papers_to_process = []
        for paper in papers_metadata:
            # 简单检查去重
            existing = self.collection.get(where={"paper_id": paper['id']}, limit=1)
            if not existing['ids']:
                papers_to_process.append(paper)
        
        if not papers_to_process:
            self.logger.info("All papers already indexed. Skipping.")
            return

        # 1. 批量下载 PDF
        downloadable_papers = [p for p in papers_to_process if p.get('pdf_url')]
        download_map = {}
        if downloadable_papers:
            try:
                download_map = asyncio.run(self._download_batch(downloadable_papers))
            except Exception as e:
                self.logger.error(f"Async loop failed: {e}")

        new_count = 0
        
        # 2. 混合处理 (PDF Full Text vs Abstract)
        for paper in tqdm(papers_to_process, desc="Indexing"):
            paper_id = paper['id']
            pdf_path = download_map.get(paper_id)
            
            text_content = ""
            is_full_text = False
            
            # A. 尝试 PDF
            if pdf_path:
                parsed_text = self._parse_pdf(pdf_path)
                if parsed_text and len(parsed_text) > 100:
                    text_content = parsed_text
                    is_full_text = True
            
            # B. 兜底摘要
            if not text_content:
                text_content = paper.get('abstract', '')
                is_full_text = False
            
            if not text_content or len(text_content) < 10: 
                continue

            # 3. 切片
            chunks = self._chunk_text(text_content)
            if not chunks: continue

            ids = [f"{paper_id}_chk_{i}" for i in range(len(chunks))]
            
            # [CRITICAL FIX] 强制清洗 Metadata，防止 None 进入 ChromaDB
            metadatas = [{
                "paper_id": str(paper_id),
                "title": str(paper.get('title') or "Unknown Title"),
                "url": str(paper.get('url') or ""),        # 关键修改: None -> ""
                "pdf_url": str(paper.get('pdf_url') or ""),# 关键修改: None -> ""
                "year": int(paper.get('year') or 0),
                "chunk_index": i,
                "is_full_text": bool(is_full_text)
            } for i in range(len(chunks))]

            try:
                self.collection.add(
                    documents=chunks,
                    metadatas=metadatas,
                    ids=ids
                )
                new_count += len(chunks)
            except Exception as e:
                self.logger.error(f"Failed to add paper {paper_id}: {e}")

        if new_count > 0:
            self.logger.info(f"Success: Added {new_count} new chunks (Mixed Full/Abstract).")

    def search(self, query: str, top_k: int = None):
        if top_k is None:
            top_k = getattr(settings, "RAG_RETRIEVAL_K", 5)
        if self.collection.count() == 0: return []
        
        results = self.collection.query(query_texts=[query], n_results=top_k)
        if not results['documents']: return []
        
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        structured_results = []
        for i in range(len(docs)):
            item = metas[i]
            item['content'] = docs[i]
            structured_results.append(item)
        return structured_results