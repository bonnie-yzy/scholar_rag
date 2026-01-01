# ui/db.py
import sqlite3
import hashlib
import json
from datetime import datetime
import json, os
DB_PATH = "data/scholar_ui.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Update users table to include avatar, bio, theme, font
    # We use ALTER TABLE to be safe if table exists, or create new with columns
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, created_at TEXT)''')
    
    # Check if columns exist (sqlite handling for migrations)
    try:
        c.execute("ALTER TABLE users ADD COLUMN avatar BLOB")
    except: pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN bio TEXT")
    except: pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN theme TEXT")
    except: pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN font TEXT")
    except: pass

    # 私人历史表
    c.execute('''CREATE TABLE IF NOT EXISTS private_chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  summary TEXT, 
                  messages JSON, 
                  updated_at TEXT)''')

    # 灵感广场表
    c.execute('''CREATE TABLE IF NOT EXISTS shared_chats
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  username TEXT, 
                  title TEXT, 
                  content JSON, 
                  mode TEXT,
                  likes INTEGER DEFAULT 0,
                  created_at TEXT)''')
    
    # [新增] 点赞记录表：用于记录谁给哪个帖子点了赞
    # 联合主键 (username, post_id) 确保每人对每贴只能点赞一次
    c.execute('''CREATE TABLE IF NOT EXISTS post_likes
                 (username TEXT, 
                  post_id INTEGER, 
                  created_at TEXT,
                  PRIMARY KEY (username, post_id))''')

    conn.commit()
    conn.close()

def hash_pass(password):
    return hashlib.sha256(password.encode()).hexdigest()

def seed_from_json(json_path="mock_data.json"):
    if not os.path.exists(json_path):
        return False, "mock_data.json not found"
        
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 1. 导入用户
        for u in data.get("users", []):
            try:
                c.execute("INSERT OR IGNORE INTO users (username, password, created_at, bio, theme) VALUES (?, ?, ?, ?, ?)",
                          (u['username'], hash_pass(u.get('password', '123')), datetime.now().isoformat(), u.get('bio'), u.get('theme')))
            except: pass

        # 2. 导入帖子并建立 ID 映射 (Mock ID -> Real DB ID)
        # [CRITICAL] 必须建立映射，否则点赞数据无法关联
        id_map = {} 
        new_posts_count = 0
        
        for p in data.get("posts", []):
            # 检查标题去重
            c.execute("SELECT id FROM shared_chats WHERE title=?", (p['title'],))
            existing = c.fetchone()
            
            if existing:
                real_id = existing[0]
            else:
                # [NEW] 使用 json 里的 created_at，如果没有则用当前时间
                p_time = p.get('created_at', datetime.now().isoformat())
                
                c.execute("INSERT INTO shared_chats (username, title, content, mode, likes, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                          (p['owner'], p['title'], p.get('content_json') or json.dumps(p), p['mode'], p.get('likes', 0), p_time))
                real_id = c.lastrowid
                new_posts_count += 1
            
            # 记录映射: json_id (e.g. 100) -> db_id (e.g. 1)
            id_map[p['id']] = real_id

        # 3. 导入点赞 (使用映射后的 ID)
        new_likes_count = 0
        for l in data.get("likes", []):
            mock_pid = l['post_id']
            real_pid = id_map.get(mock_pid) # 获取真实 ID
            
            if real_pid:
                try:
                    c.execute("INSERT OR IGNORE INTO post_likes (username, post_id, created_at) VALUES (?, ?, ?)", 
                              (l['username'], real_pid, l.get('created_at', datetime.now().isoformat())))
                    new_likes_count += 1
                except: pass # 忽略重复点赞
        
        # 4. 强制修正 shared_chats 表里的 likes 计数
        # 因为 mock 数据的 likes 计数可能和实际插入 post_likes 表的数量不一致（比如有些点赞因为重复被 ignore 了）
        c.execute("""
            UPDATE shared_chats 
            SET likes = (SELECT COUNT(*) FROM post_likes WHERE post_likes.post_id = shared_chats.id)
        """)
        
        conn.commit()
        conn.close()
        return True, f"成功注入: {new_posts_count} 新帖子, {new_likes_count} 条点赞关联 (含时间戳)."
    except Exception as e:
        return False, str(e)

def fetch_recommendation_data():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # 1. Users
    c.execute("SELECT username, bio FROM users")
    users = [dict(row) for row in c.fetchall()]
    
    # 2. Posts (关键修改：增加 username, likes, content)
    c.execute("SELECT id, title, content, mode, username, likes FROM shared_chats")
    posts_raw = c.fetchall()
    posts = []
    for row in posts_raw:
        # 为了 content score 计算，我们需要 summary
        summary = ""
        try:
            content_obj = json.loads(row['content'])
            if isinstance(content_obj, dict):
                summary = content_obj.get("summary", "")
        except: pass
        
        posts.append({
            "id": row['id'], 
            "title": row['title'], 
            "summary": summary,     # 供推荐算法计算相似度
            "mode": row['mode'],
            "owner": row['username'], # [新增] 供 UI 显示作者
            "likes": row['likes'],    # [新增] 供 UI 显示点赞数
            "content_raw": row['content'] # [新增] 供 UI 解析详情
        })
        
    # 3. Likes
    c.execute("SELECT username, post_id, created_at FROM post_likes")
    likes = [dict(row) for row in c.fetchall()]
    
    conn.close()
    return users, posts, likes

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users (username, password, created_at, theme, font) VALUES (?, ?, ?, ?, ?)", 
                  (username, hash_pass(password), datetime.now().isoformat(), "Science Geek", "Sans-Serif"))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    res = c.fetchone()
    conn.close()
    if res and res[0] == hash_pass(password):
        return True
    return False

# [NEW] Get User Profile
def get_user_profile(username):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT bio, theme, font, avatar FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {}

# [NEW] Update User Profile
def update_user_profile(username, bio=None, theme=None, font=None, avatar_bytes=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if bio is not None:
        c.execute("UPDATE users SET bio=? WHERE username=?", (bio, username))
    if theme is not None:
        c.execute("UPDATE users SET theme=? WHERE username=?", (theme, username))
    if font is not None:
        c.execute("UPDATE users SET font=? WHERE username=?", (font, username))
    if avatar_bytes is not None:
        c.execute("UPDATE users SET avatar=? WHERE username=?", (avatar_bytes, username))
        
    conn.commit()
    conn.close()

# [新增] 保存私人历史
def save_private_chat(username, summary, messages):
    if not messages: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO private_chats (username, summary, messages, updated_at) VALUES (?, ?, ?, ?)",
              (username, summary, json.dumps(messages), datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_private_history_list(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 查询了 4 个字段: id (0), summary (1), messages (2), updated_at (3)
    c.execute("SELECT id, summary, messages, updated_at FROM private_chats WHERE username=? ORDER BY updated_at DESC LIMIT 20", (username,))
    rows = c.fetchall()
    conn.close()
    
    # 🔴 修复：这里必须要把 updated_at (索引3) 映射进去
    return [{
        "id": r[0], 
        "summary": r[1], 
        "msgs": json.loads(r[2]), 
        "updated_at": r[3] 
    } for r in rows]

def share_chat_to_square(username, title, chat_history, mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO shared_chats (username, title, content, mode, created_at) VALUES (?, ?, ?, ?, ?)",
              (username, title, json.dumps(chat_history), mode, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_inspiration_posts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 按点赞数倒序
    c.execute("SELECT id, username, title, content, mode, likes FROM shared_chats ORDER BY likes DESC LIMIT 20")
    posts = c.fetchall()
    conn.close()
    return posts

def save_or_update_chat(chat_id, username, summary, messages):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # 确保 summary 不为空，否则历史列表很难看
    if not summary: summary = "新对话..."
    
    if chat_id is None:
        c.execute("INSERT INTO private_chats (username, summary, messages, updated_at) VALUES (?, ?, ?, ?)",
                  (username, summary, json.dumps(messages), datetime.now().isoformat()))
        new_id = c.lastrowid
        conn.commit()
        conn.close()
        return new_id
    else:
        c.execute("UPDATE private_chats SET summary=?, messages=?, updated_at=? WHERE id=?",
                  (summary, json.dumps(messages), datetime.now().isoformat(), chat_id))
        conn.commit()
        conn.close()
        return chat_id
    
# [新增] 根据 ID 删除指定的对话记录
def delete_private_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM private_chats WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()


def share_chat_to_square(username, title, chat_history, mode):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO shared_chats (username, title, content, mode, created_at) VALUES (?, ?, ?, ?, ?)",
              (username, title, json.dumps(chat_history), mode, datetime.now().isoformat()))
    conn.commit()
    conn.close()

def get_inspiration_posts(sort_by="hot", limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if sort_by == "new":
        # 🆕 按时间倒序 (最新发布在最前)
        c.execute("SELECT id, username, title, content, mode, likes FROM shared_chats ORDER BY created_at DESC LIMIT ?", (limit,))
    else:
        # 🔥 默认按热度 (点赞数倒序)
        c.execute("SELECT id, username, title, content, mode, likes FROM shared_chats ORDER BY likes DESC LIMIT ?", (limit,))
        
    posts = c.fetchall()
    conn.close()
    return posts

# [重写] 点赞逻辑：增加权限检查
def like_post(post_id, username):
    """
    返回: (Success: bool, Message: str)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. 检查帖子归属
    c.execute("SELECT username FROM shared_chats WHERE id=?", (post_id,))
    row = c.fetchone()
    if not row:
        conn.close()
        return False, "帖子不存在"
    
    owner = row[0]
    if owner == username:
        conn.close()
        return False, "不能给自己点赞 (保持谦虚!)"
    
    # 2. 检查是否已点赞
    c.execute("SELECT 1 FROM post_likes WHERE username=? AND post_id=?", (username, post_id))
    if c.fetchone():
        conn.close()
        return False, "你已经点过赞了"

    # 3. 执行点赞 (事务)
    try:
        # 记录点赞人
        c.execute("INSERT INTO post_likes (username, post_id, created_at) VALUES (?, ?, ?)", 
                  (username, post_id, datetime.now().isoformat()))
        # 增加计数
        c.execute("UPDATE shared_chats SET likes = likes + 1 WHERE id=?", (post_id,))
        conn.commit()
        msg = "❤️ 点赞成功！"
        success = True
    except Exception as e:
        msg = f"点赞失败: {e}"
        success = False
        
    conn.close()
    return success, msg

def get_academic_star():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT username, SUM(likes) as total_likes FROM shared_chats GROUP BY username ORDER BY total_likes DESC LIMIT 1")
    res = c.fetchone()
    conn.close()
    return res if res else ("暂无", 0)

def delete_shared_chat(post_id, username):
    """
    删除分享的帖子。
    安全检查：只有当帖子属于 username 时才执行删除。
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. 验证归属权
    c.execute("SELECT username FROM shared_chats WHERE id=?", (post_id,))
    row = c.fetchone()
    
    success = False
    if row and row[0] == username:
        # 2. 删除帖子
        c.execute("DELETE FROM shared_chats WHERE id=?", (post_id,))
        # 3. (可选) 删除关联的点赞记录，保持数据整洁
        c.execute("DELETE FROM post_likes WHERE post_id=?", (post_id,))
        conn.commit()
        success = True
    
    conn.close()
    return success