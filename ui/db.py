# ui/db.py
import sqlite3
import hashlib
import json
from datetime import datetime

DB_PATH = "data/scholar_ui.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, created_at TEXT)''')
    
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

def register_user(username, password):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO users VALUES (?, ?, ?)", 
                  (username, hash_pass(password), datetime.now().isoformat()))
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

# [新增] 保存私人历史
def save_private_chat(username, summary, messages):
    if not messages: return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO private_chats (username, summary, messages, updated_at) VALUES (?, ?, ?, ?)",
              (username, summary, json.dumps(messages), datetime.now().isoformat()))
    conn.commit()
    conn.close()

# [新增] 获取用户的历史列表 (按时间倒序)
def get_private_history_list(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, summary, messages, updated_at FROM private_chats WHERE username=? ORDER BY updated_at DESC LIMIT 20", (username,))
    rows = c.fetchall()
    conn.close()
    # 返回格式: [{"id":..., "summary":..., "msgs":...}]
    return [{"id": r[0], "summary": r[1], "msgs": json.loads(r[2])} for r in rows]

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

def get_inspiration_posts():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, username, title, content, mode, likes FROM shared_chats ORDER BY likes DESC LIMIT 20")
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