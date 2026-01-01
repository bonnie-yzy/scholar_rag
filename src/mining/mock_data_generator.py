import json
import random
import datetime
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ==========================================
# 1. 导入你的基础设施
# 假设你的目录结构是标准的，且 llm.py 在 src/core/ 或 src/ 下
# 如果 llm.py 和此脚本在同一级，直接 import 即可
# 这里按照你 generator.py 的风格尝试导入
# ==========================================

# 尝试从 src 导入 (推荐)
from src.core.llm import LLMService 


class MockContentGenerator:
    """负责与 LLM 交互生成文本"""
    def __init__(self):
        self.llm = LLMService() # 复用 llm.py 中的初始化逻辑 (读取 config.py)

    def generate_post_data(self, domain, keywords):
        """生成符合特定格式的帖子内容"""
        
        system_prompt = f"""
        You are a specialized data generator for a social media platform focusing on academic and tech topics.
        You must output a valid JSON object.
        """

        user_prompt = f"""
        Please generate a social media post for the domain "{domain}" based on these keywords: {', '.join(keywords)}.
        
        The output must be a JSON object with the following structure:
        {{
            "title": "String starting with [{domain}]",
            "summary": "String with emojis and bullet points",
            "messages": [
                {{"role": "user", "content": "..."}},
                {{"role": "assistant", "content": "..."}}
            ]
        }}

        Style Requirements:
        1. Title: Short and catchy. Example: "[AI] TTS Future Outlook"
        2. Summary: 
           - Start with "💡" for a core insight.
           - Use "📌" for 2-3 key bullet points.
           - End with 3 relevant hashtags (e.g. #AI #Tech).
           - Language: Chinese (Simplified).
           - Example: 
             "💡语音合成的未来不在“更像人”，而在“更少数据”！\n\n📌扩散模型打破平衡困局\n📌零样本合成降低依赖\n\n#AI语音 #科研 #扩散模型"
        3. Messages: 
           - A conversation where the user asks about the topic and the assistant explains details, OR a detailed analysis.
           - Keep it professional but engaging.
        """

        # 调用 llm.py 中的 chat_json 方法
        # 它会自动处理 JSON 解析和 ```json 清洗
        data = self.llm.chat_json(system_prompt, user_prompt)
        
        if not data:
            # Fallback if LLM fails
            return {
                "title": f"[{domain}] 关于 {keywords[0]} 的思考",
                "summary": f"💡 关于 {keywords[0]} 的一些想法...\n📌 这是一个模拟数据 (LLM调用失败)\n#{domain}",
                "messages": [{"role": "assistant", "content": "暂无详细内容。"}]
            }
        return data

class MockDataGenerator:
    def __init__(self, use_llm=False):
        self.use_llm = use_llm
        if self.use_llm:
            self.content_gen = MockContentGenerator()
            
        self.domains = {
            "AI": ["Deep Learning", "LLM", "Transformer", "Agent", "CoT", "RAG", "Generative", "Diffusion", "Reinforcement Learning", "VLM", "Embodied AI"],
            "Bio": ["Genomics", "CRISPR", "Protein", "Cell", "DNA", "AlphaFold", "Medicine"],
            "Physics": ["Quantum", "Dark Matter", "Gravity", "String Theory", "Particle", "Universe"],
            "Startup": ["SaaS", "Product", "Growth", "VC", "Market", "Efficiency", "Tool"],
            "Art": ["Design", "Aesthetics", "Color", "Composition", "Midjourney", "Digital Art"]
        }
        # 预定义一些数据结构
        self.data = {"users": [], "posts": [], "likes": []}

    def _random_date(self, days_back=30):
        now = datetime.datetime.now()
        delta = datetime.timedelta(days=random.randint(0, days_back), 
                                   seconds=random.randint(0, 86400))
        return (now - delta).isoformat()

    def generate(self, user_count=20, post_count=20):
        print("🚀 开始生成数据...")
        
        # 1. Generate Users
        # -------------------------------------------------
        themes = ["Nature一作", "AI天才", "我想创业", "理科男", "文艺青年"]
        domain_keys = list(self.domains.keys())
        
        # 统计各领域人数，用于后续“点赞均衡算法”
        domain_user_counts = {d: 0 for d in domain_keys}

        for i in range(user_count):
            domain = random.choice(domain_keys)
            domain_user_counts[domain] += 1
            
            keywords = random.sample(self.domains[domain], 3)
            user = {
                "username": f"User_{domain}_{i}",
                "password": "123",
                "bio": f"Focus on {domain}. Interested in {', '.join(keywords)}.",
                "theme": random.choice(themes),
                "domain_affinity": domain 
            }
            self.data["users"].append(user)
        
        print(f"👤 用户生成完毕。领域分布: {domain_user_counts}")

        # 2. Generate Posts (支持 LLM)
        # -------------------------------------------------
        modes = ["review", "explain", "inspire"]
        
        for i in range(post_count):
            domain = random.choice(domain_keys)
            keywords = random.sample(self.domains[domain], 2)
            
            # 随机找个作者
            potential_owners = [u['username'] for u in self.data["users"] if u['domain_affinity'] == domain]
            owner = random.choice(potential_owners) if potential_owners else self.data["users"][0]['username']
            
            created_at = self._random_date(days_back=30)
            
            # === LLM 内容生成逻辑 ===
            if self.use_llm:
                print(f"🤖 [LLM] 正在生成第 {i+1}/{post_count} 条帖子 ({domain})...")
                # 调用我们的生成器
                content_data = self.content_gen.generate_post_data(domain, keywords)
                
                title = content_data.get("title", f"[{domain}] {keywords[0]}")
                summary = content_data.get("summary", "Summary...")
                messages = content_data.get("messages", [])
            else:
                # 快速生成的假数据
                title = f"[{domain}] Thoughts on {keywords[0]} & {keywords[1]}"
                summary = f"💡 Insights about {keywords[0]}...\n📌 Point 1: ...\n📌 Point 2: ...\n#{domain} #{keywords[0]}"
                messages = [{"role": "user", "content": "Mock content..."}]

            # 封装 JSON
            content_json_obj = {
                "summary": summary,
                "messages": messages
            }

            post = {
                "id": i + 100,
                "owner": owner,
                "title": title,
                "content_json": json.dumps(content_json_obj, ensure_ascii=False),
                "mode": random.choice(modes),
                "domain_tag": domain,
                "likes": 0,
                "created_at": created_at
            }
            self.data["posts"].append(post)

        # 3. Generate Likes (均衡分布算法)
        # -------------------------------------------------
        print("❤️ 正在生成点赞 (使用均衡分布算法)...")
        
        # 设定：每个帖子平均期望获得的“同领域”点赞数
        TARGET_AVG_LIKES = 3.5 
        
        for post in self.data["posts"]:
            post_domain = post['domain_tag']
            # 该领域有多少潜在点赞者
            domain_population = domain_user_counts.get(post_domain, 1)
            
            # 动态计算概率：
            # 如果领域人多(User_Physics=10)，单人点赞概率降低 (3.5/10 = 35%)
            # 如果领域人少(User_Art=2)，单人点赞概率升高 (3.5/2 = 100% -> max 0.8)
            raw_prob = TARGET_AVG_LIKES / max(1, domain_population)
            dynamic_prob = min(0.9, raw_prob) # 上限 90%

            for user in self.data["users"]:
                user_domain = user['domain_affinity']
                
                is_same_domain = (user_domain == post_domain)
                
                # 决定点赞概率
                if is_same_domain:
                    prob = dynamic_prob
                else:
                    prob = 0.05 # 跨领域由于“不明觉厉”产生的随机点赞
                
                if random.random() < prob:
                    # 确保点赞时间在发帖之后
                    post_time = datetime.datetime.fromisoformat(post['created_at'])
                    like_time = post_time + datetime.timedelta(hours=random.randint(1, 48))
                    
                    if like_time > datetime.datetime.now():
                        like_time = datetime.datetime.now() - datetime.timedelta(minutes=random.randint(1, 60))

                    self.data["likes"].append({
                        "username": user['username'],
                        "post_id": post['id'],
                        "created_at": like_time.isoformat()
                    })
                    post["likes"] += 1

        return self.data

if __name__ == "__main__":
    # 开关：设置为 True 使用 config.py 配置的 LLM
    USE_LLM = True 
    
    # 注意：LLM 生成较慢，测试时建议 post_count 设小一点 (比如 5-10)
    gen = MockDataGenerator(use_llm=USE_LLM)
    
    # 这里生成 20 个用户，10 个 LLM 生成的帖子用于测试
    data = gen.generate(user_count=30, post_count=100)
    
    # 保存文件
    root_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root_dir, "mock_data.json")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        
    print(f"✅ 生成完成！")
    print(f"   用户数: {len(data['users'])}")
    print(f"   帖子数: {len(data['posts'])}")
    print(f"   点赞数: {len(data['likes'])}")
    print(f"📂 数据已保存至: {path}")