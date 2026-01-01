import math
import datetime
import numpy as np
import os
import json
from collections import defaultdict
from src.core.llm import LLMService

class RecommendationEngine:
    def __init__(self, users_data, posts_data, likes_data):
        self.users = {u['username']: u for u in users_data}
        self.posts = {p['id']: p for p in posts_data}
        self.llm_service = LLMService()
        
        # 1. 整理交互历史
        self.user_history = defaultdict(list)     # username -> [(pid, days_ago)]
        self.post_likes_users = defaultdict(set)  # pid -> {username}
        self._process_history(likes_data)

        # 2. [核心] 准备 Post Embeddings (带缓存)
        self.post_embeddings = self._load_or_compute_embeddings()

    def _process_history(self, likes_data):
        """处理点赞历史和时间衰减"""
        now = datetime.datetime.now()
        for l in likes_data:
            pid = l['post_id']
            uid = l['username']
            
            # 只有当帖子存在时才处理
            if pid not in self.posts: continue
            
            self.post_likes_users[pid].add(uid)
            
            # 计算天数差
            try:
                # 尝试解析 ISO 格式
                if 'T' in l['created_at']:
                    dt = datetime.datetime.fromisoformat(l['created_at'])
                else:
                    # 兼容可能得旧格式
                    dt = datetime.datetime.strptime(l['created_at'], "%Y-%m-%d %H:%M:%S")
                days_diff = (now - dt).days
            except:
                days_diff = 30 # 默认 30 天前
            
            self.user_history[uid].append((pid, max(0, days_diff)))

    def _load_or_compute_embeddings(self):
        """
        加载或计算所有帖子的 Embedding。
        为了节省 Token，会优先读取本地缓存文件。
        """
        cache_file = "data/embeddings_cache.json"
        cache = {}
        
        # A. 尝试加载缓存
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    # JSON key 是 string，需要转回 int (post_id)
                    raw_cache = json.load(f)
                    cache = {int(k): np.array(v) for k, v in raw_cache.items()}
                print(f"✅ Loaded {len(cache)} embeddings from cache.")
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")

        # B. 计算缺失的 Embedding
        updates_needed = False
        post_embeddings = {}
        
        for pid, post in self.posts.items():
            if pid in cache:
                post_embeddings[pid] = cache[pid]
            else:
                # 构造语义文本：标题 + 摘要 + 标签
                text = f"{post.get('title', '')} {post.get('summary', '')} {post.get('domain_tag', '')}"
                print(f"🔄 Computing embedding for Post {pid}...")
                
                vec = self.llm_service.get_embedding(text)
                
                if vec:
                    vec_np = np.array(vec)
                    post_embeddings[pid] = vec_np
                    cache[pid] = vec_np # 更新内存缓存用于后续保存
                    updates_needed = True
                else:
                    print(f"❌ Failed to embed Post {pid}")

        # C. 如果有更新，保存回文件
        if updates_needed:
            try:
                # numpy array 不可序列化，需转 list
                serializable_cache = {str(k): v.tolist() for k, v in cache.items()}
                # 确保存储目录存在
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(serializable_cache, f)
                print("💾 Embeddings cache updated.")
            except Exception as e:
                print(f"⚠️ Failed to save cache: {e}")
                
        return post_embeddings

    def _get_time_decay(self, days_ago, half_life=30):
        """时间衰减权重"""
        return math.pow(0.5, days_ago / half_life)

    def _build_user_vector(self, username):
        """
        构建用户向量：
        User_Vec = 加权平均(历史喜欢的 Post_Vec)
        如果用户没有历史，尝试用 Bio 计算，否则返回 None (冷启动)
        """
        history = self.user_history.get(username, [])
        
        # 1. 冷启动：无历史，尝试 Embedding Bio
        if not history:
            bio = self.users[username].get('bio', '')
            if bio and len(bio) > 5:
                return np.array(self.llm_service.get_embedding(bio))
            return None

        # 2. 有历史：加权平均
        user_vec = None
        total_weight = 0.0
        
        for pid, days_ago in history:
            if pid not in self.post_embeddings: continue
            
            # 越近的点赞权重越高
            weight = self._get_time_decay(days_ago)
            vec = self.post_embeddings[pid]
            
            if user_vec is None:
                user_vec = vec * weight
            else:
                user_vec += vec * weight
            
            total_weight += weight
            
        if user_vec is not None and total_weight > 0:
            user_vec /= total_weight
            
        return user_vec

    def _cosine_similarity(self, vec_a, vec_b):
        """计算余弦相似度"""
        if vec_a is None or vec_b is None: return 0.0
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0: return 0.0
        return np.dot(vec_a, vec_b) / (norm_a * norm_b)

    def recommend(self, username, top_k=5):
        """
        执行推荐：向量召回 -> 过滤 -> 排序
        """
        if username not in self.users:
            return []

        # 1. 获取用户向量
        user_vec = self._build_user_vector(username)
        
        # 2. 获取已读列表 (用于过滤)
        viewed_ids = {pid for pid, _ in self.user_history.get(username, [])}
        
        candidates = []
        
        # 3. 遍历所有候选帖子
        for pid, post_vec in self.post_embeddings.items():
            if pid in viewed_ids: continue
            
            # 计算语义相似度
            if user_vec is not None:
                score = self._cosine_similarity(user_vec, post_vec)
            else:
                score = 0.0 # 冷启动且无 Bio
            
            # 引入一点点随机性/热度因子防止完全死板 (可选)
            score += len(self.post_likes_users[pid]) * 0.001 

            candidates.append({
                "data": (
                    self.posts[pid]['id'], 
                    self.posts[pid]['owner'], 
                    self.posts[pid]['title'], 
                    self.posts[pid].get('content_json') or self.posts[pid].get('content_raw'), 
                    self.posts[pid]['mode'], 
                    self.posts[pid]['likes']
                ),
                "score": score
            })
            
        # 4. 排序与返回
        # 如果所有分数都是0 (冷启动失败)，则回退到热门推荐
        if not candidates or all(c['score'] == 0 for c in candidates):
            return [x['data'] for x in self._get_popular_fallback(top_k, viewed_ids)]

        candidates.sort(key=lambda x: x['score'], reverse=True)
        return [c['data'] for c in candidates[:top_k]]

    def _get_popular_fallback(self, k, viewed_ids):
        """兜底策略：全站热门"""
        popular = []
        for pid, users in self.post_likes_users.items():
            if pid in viewed_ids: continue
            post = self.posts[pid]
            popular.append({
                "data": (
                    post['id'], 
                    post['owner'], 
                    post['title'], 
                    post.get('content_json') or post.get('content_raw'), 
                    post['mode'], 
                    post['likes']
                ),
                "score": len(users)
            })
        popular.sort(key=lambda x: x['score'], reverse=True)
        return popular[:k]