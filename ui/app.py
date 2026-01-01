# ui/app.py
import streamlit as st
import time
import json
import base64
import random
from db import (
    init_db, register_user, login_user, share_chat_to_square, 
    get_inspiration_posts, like_post, get_academic_star, 
    save_private_chat, get_private_history_list, save_or_update_chat,
    delete_shared_chat, get_user_profile, update_user_profile, seed_from_json, 
    fetch_recommendation_data
)
from logic import get_engine, recursive_summarize, perform_retrieval, get_response_stream, generate_viral_copy
from src.mining.recommendation import RecommendationEngine

# 初始化数据库
init_db()

# 页面配置
st.set_page_config(page_title="ScholarRAG", page_icon="🎓", layout="wide")

# --- Theme & Style Injection (已修改：全员暗黑模式) ---
def load_style(theme_key, font_key):
    # 基础字体映射
    FONT_MAP = {
        "Sans-Serif": "'Google Sans', sans-serif",
        "Serif": "'Georgia', serif",
        "Monospace": "'Courier New', monospace"
    }
    font_fam = FONT_MAP.get(font_key, "'Google Sans', sans-serif")

    # 读取基础 CSS (ui/style.css)
    try:
        with open("ui/style.css") as f:
            base_css = f.read()
    except FileNotFoundError:
        base_css = ""

    # 统一深灰底座（所有主题共用）
    base_vars = """
        --bg-color: #121316;
        --sidebar-bg: #1B1F24;
        --panel-bg: rgba(255, 255, 255, 0.06);
        --input-bg: rgba(255, 255, 255, 0.08);
        --card-bg: rgba(255, 255, 255, 0.05);
        --border-color: rgba(255, 255, 255, 0.14);
    """

    # 主题调色板：只变“前景系统”（文字/强调/链接等）
    THEME_PRESETS = {
        "理科男": {
            "text": "#E6E9EF",
            "muted": "#AAB3C0",
            "accent": "#4DA6FF",
            "accent2": "#7C3AED",
            "accent3": "#22C55E",
            "link": "#7AB7FF",
        },
        "Nature一作": {
            "text": "#E9E1D3",
            "muted": "#BFAF98",
            "accent": "#6FBF73",   # 绿
            "accent2": "#B07A4A",  # 棕
            "accent3": "#D9B26F",  # 金棕（可选）
            "link": "#8EDB95",
        },
        "我想创业": {
            "text": "#F2F6FF",
            "muted": "#B7C2D6",
            "accent": "#3B82F6",   # 蓝
            "accent2": "#F59E0B",  # 橙
            "accent3": "#06B6D4",  # 青（可选）
            "link": "#7FB2FF",
        },
        "AI天才": {
            # 更荧光的“青绿字 + 粉色按钮”
            "text":   "#7CFFEE",
            "muted":  "#38FFE2",
            "accent": "#FF2DAA",   # 主 accent：粉色（按钮/hover 主色）
            "accent2":"#39FF14",   # 荧光绿
            "accent3":"#B026FF",   # 荧光紫
            "link":   "#7CFFEE",

            # 让霓虹更突出：底更深、面板更亮一点
            "bg":      "#0B0D10",
            "sidebar": "#10131A",
            "panel":   "rgba(255, 255, 255, 0.075)",
            "card":    "rgba(255, 255, 255, 0.06)",
            "input":   "rgba(255, 255, 255, 0.10)",
            "border":  "rgba(255, 255, 255, 0.18)",
        },

        "文艺青年": {
            "text":   "#E9E6DF",
            "muted":  "#C9C2B7",
            "accent": "#4A78B8",   # 牛仔蓝
            "accent2":"#D07A57",   # 陶土橙（比土黄更耐看）
            "accent3":"#F2E7D6",   # 米色高光
            "link":   "#86AEE8",

            # 背景稍浅一点（仍是深色系）
            "bg":      "#171A1F",
            "sidebar": "#20242C",
            "panel":   "rgba(255, 255, 255, 0.065)",
            "card":    "rgba(255, 255, 255, 0.055)",
            "input":   "rgba(255, 255, 255, 0.095)",
            "border":  "rgba(255, 255, 255, 0.16)",
        }
    }

    preset = THEME_PRESETS.get(theme_key, THEME_PRESETS["理科男"])

    # 统一深灰底座（默认值）
    DEFAULTS = {
        "bg": "#121316",
        "sidebar": "#1B1F24",
        "panel": "rgba(255, 255, 255, 0.06)",
        "input": "rgba(255, 255, 255, 0.08)",
        "card": "rgba(255, 255, 255, 0.05)",
        "border": "rgba(255, 255, 255, 0.14)",
    }

    bg      = preset.get("bg", DEFAULTS["bg"])
    sidebar = preset.get("sidebar", DEFAULTS["sidebar"])
    panel   = preset.get("panel", DEFAULTS["panel"])
    inp     = preset.get("input", DEFAULTS["input"])
    card    = preset.get("card", DEFAULTS["card"])
    border  = preset.get("border", DEFAULTS["border"])

    theme_vars = f"""
        --bg-color: {bg};
        --sidebar-bg: {sidebar};
        --panel-bg: {panel};
        --input-bg: {inp};
        --card-bg: {card};
        --border-color: {border};

        --text-color: {preset['text']};
        --muted-text: {preset['muted']};
        --accent-color: {preset['accent']};
        --accent-2: {preset['accent2']};
        --accent-3: {preset['accent3']};
        --link-color: {preset['link']};
    """

    final_css = f"""
    <style>
        {base_css}
        :root {{
            {theme_vars}
            --font-family: {font_fam};
        }}

        /* App base */
        .stApp {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: var(--font-family);
        }}
        .stSidebar {{
            background-color: var(--sidebar-bg);
        }}
        .stChatInputContainer {{
            background-color: var(--sidebar-bg);
        }}

        /* Links */
        a {{
            color: var(--link-color) !important;
            text-decoration: none;
        }}
        a:hover {{
            color: var(--accent-2) !important;
            text-decoration: underline;
        }}

        /* Inputs / Panels */
        textarea, input, .stTextInput input, .stTextArea textarea {{
            background: var(--input-bg) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
        }}

        /* Buttons */
        .stButton button {{
            border: 1px solid var(--border-color);
            color: var(--text-color);
            background-color: var(--panel-bg);
        }}
        .stButton button:hover {{
            border-color: var(--accent-color);
            color: var(--accent-color);
            box-shadow: 0 0 0 1px var(--accent-color) inset;
        }}

        /* Cards (例如 inspiration-card) */
        .inspiration-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 14px;
        }}

        /* Muted text helper */
        .muted, .stCaption, .stMarkdown p small {{
            color: var(--muted-text) !important;
        }}

        /* Default avatar */
        .default-avatar {{
            background: rgba(255,255,255,0.10) !important;
            color: var(--text-color) !important;
            border: 1px solid var(--border-color) !important;
        }}
    </style>
    """
    st.markdown(final_css, unsafe_allow_html=True)

def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "messages" not in st.session_state:
        st.session_state.messages = [] 

    if "current_suggestions" not in st.session_state:
        st.session_state.current_suggestions = []
    
    # [新增] 递归摘要状态
    if "current_summary" not in st.session_state:
        st.session_state.current_summary = "" # 当前的全局摘要
    if "last_summarized_idx" not in st.session_state:
        st.session_state.last_summarized_idx = 0 # 指针：messages中多少条已被总结

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None # None 表示这是个新对话，还没入库
    
    if "page" not in st.session_state:
        st.session_state.page = "chat"

    if "language" not in st.session_state:
        st.session_state.language = "Chinese"

init_session()

# --- Helper: Render Avatar ---
def render_avatar(username, avatar_bytes, size=100):
    if avatar_bytes:
        b64_img = base64.b64encode(avatar_bytes).decode('utf-8')
        html = f"""
        <img src="data:image/png;base64,{b64_img}" class="user-avatar-circle" style="width:{size}px; height:{size}px;">
        """
    else:
        # Default Avatar: White background, Black text initials
        initial = username[0].upper() if username else "?"
        html = f"""
        <div class="default-avatar" style="width:{size}px; height:{size}px;">
            {initial}
        </div>
        """
    st.markdown(html, unsafe_allow_html=True)

# --- 登录/注册页 ---
def login_page():
    st.title("🎓 ScholarRAG - 登录")
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        username = st.text_input("用户名", key="login_user")
        password = st.text_input("密码", type="password", key="login_pass")
        if st.button("登录"):
            if login_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                # Load Profile Preferences
                profile = get_user_profile(username)
                st.session_state.user_theme = profile.get("theme", "理科男")
                st.session_state.user_font = profile.get("font", "Sans-Serif")
                
                st.rerun()
            else:
                st.error("用户名或密码错误")

    with tab2:
        new_user = st.text_input("新用户名", key="reg_user")
        new_pass = st.text_input("新密码", type="password", key="reg_pass")
        if st.button("注册"):
            if register_user(new_user, new_pass):
                st.success("注册成功！请登录")
            else:
                st.error("用户已存在")

# --- 侧边栏 ---
def sidebar():
    with st.sidebar:
        st.markdown('<div class="rainbow-text">ScholarRAG</div>', unsafe_allow_html=True)
        # [Update] Display Avatar in Sidebar
        profile = get_user_profile(st.session_state.username)
        render_avatar(st.session_state.username, profile.get("avatar"), size=80)
        st.caption(f"🚀 Current User: **{st.session_state.username}**")
        if profile.get("bio"):
            st.info(f"📝 {profile['bio']}")
        st.divider()
        
        # [修改] 发起新对话 -> 存入数据库
        if st.button("➕ 发起新对话", use_container_width=True):
            # 不需要再手动 save 了，因为每句话都自动 save 过
            # 清空状态，准备迎接新对话
            st.session_state.messages = []
            st.session_state.current_summary = ""
            st.session_state.last_summarized_idx = 0
            st.session_state.current_chat_id = None # 重置 ID，下次说话会创建新记录
            st.session_state.page = "chat"
            st.rerun()
        
        st.divider()
        st.subheader("🕒 历史归档 (DB)")
        
        # [修改] 从数据库读取历史
        # 注意：每次刷新页面这里都会读库，量大时可以加 cache
        history_list = get_private_history_list(st.session_state.username)
        
        if not history_list:
            st.caption("暂无历史记录")

        for item in history_list:
            # [修改] 调整列比例，让删除按钮贴在最右边
            # 使用 container 将其包裹，虽然 Streamlit 的 columns 本身就是 block，
            # 但为了确保 CSS 能够精准捕获 hover，我们保持结构简单
            col1, col2 = st.columns([5, 1])
            
            with col1:
                # 截断标题，防止换行破坏布局
                display_title = (item['summary'][:16] + '..') if len(item['summary']) > 16 else item['summary']
                
                # [关键] 加载按钮：使用 use_container_width=True 让它填满左侧空间
                # 这样用户的鼠标只要在左侧区域，都能触发 Hover
                if st.button(f"📄 {display_title}", key=f"hist_load_{item['id']}", use_container_width=True):
                    st.session_state.messages = item['msgs']
                    st.session_state.current_summary = item['summary']
                    st.session_state.last_summarized_idx = len(item['msgs'])
                    st.session_state.current_chat_id = item['id']
                    st.session_state.page = "chat"
                    st.rerun()
            
            with col2:
                # [关键] 删除按钮：只放一个图标
                # CSS 会负责默认隐藏它，只有 Hover 时显示
                if st.button("🗑️", key=f"hist_del_{item['id']}", use_container_width=True):
                    from db import delete_private_chat
                    delete_private_chat(item['id'])
                    
                    if st.session_state.get("current_chat_id") == item['id']:
                        st.session_state.messages = []
                        st.session_state.current_summary = ""
                        st.session_state.current_chat_id = None
                        st.toast("对话已删除")
                    
                    st.rerun()

        st.divider()
        st.subheader("🛠️ 功能区")
        mode = st.radio("选择模式", ["review (综述)", "explain (深度)", "inspire (脑暴)"], index=0)
        mode_key = mode.split(" ")[0] # 提取 'review' 等
        
        use_graph = st.checkbox("启用知识图谱增强", value=True)

        st.divider()
        if st.button("✨ 灵感广场", use_container_width=True):
            st.session_state.page = "square"
            st.rerun()
            
        if st.button("⚙️ 设置 / 个人信息", use_container_width=True):
            st.session_state.page = "profile"
            st.rerun()
            
        if st.button("退出登录"):
            st.session_state.logged_in = False
            st.rerun()
            
        return mode_key, use_graph

def get_smart_suggestions(username):
    """
    基于推荐系统，生成 4 个学术提问建议
    """
    suggestions = []
    
    # 1. 尝试获取个性化推荐
    try:
        users_data, posts_data, likes_data = fetch_recommendation_data()
        rec_engine = RecommendationEngine(users_data, posts_data, likes_data)
        # 获取 Top 5 推荐帖子
        recs = rec_engine.recommend(username, top_k=5)
        
        # 2. 将帖子转化为问题
        templates = [
            "帮我深度解析 '{title}' 的核心理论",
            "我想了解关于 '{title}' 的最新研究进展",
            "请综述 '{title}' 相关的技术路线",
            "'{title}' 在实际应用中有哪些挑战？",
            "基于 '{title}' 写一段研究灵感"
        ]
        
        for rec in recs:
            # rec 结构: (id, owner, title, content, mode, likes)
            title = rec[2]
            # 去掉标题中的标签前缀 (如 [AI]) 以便句子更通顺
            clean_title = title.split(']')[-1].strip() if ']' in title else title
            
            question = random.choice(templates).format(title=clean_title)
            suggestions.append(question)
            
    except Exception as e:
        print(f"Suggestion Error: {e}")
    
    # 3. 兜底逻辑：如果推荐系统没返回（新用户），或者不够4个，补充通用热门问题
    fallback_questions = [
        "解释一下 RAG (Retrieval-Augmented Generation) 的原理",
        "Transformer 架构相比 RNN 有什么核心优势？",
        "最新的 AI Agent 包含哪些核心组件？",
        "如何利用 Deep Learning 进行蛋白质结构预测？",
        "量子计算对密码学有哪些潜在威胁？"
    ]
    
    # 补齐到 4 个
    needed = 4 - len(suggestions)
    if needed > 0:
        suggestions.extend(random.sample(fallback_questions, min(needed, len(fallback_questions))))
    
    return suggestions[:4]

def render_welcome_screen():
    """渲染空状态下的欢迎页和猜你想问"""
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; margin-bottom: 30px;">
        <h1 style="color: var(--text-color); opacity: 0.9;">👋 Hi, Scholar!</h1>
        <p style="color: var(--text-color); opacity: 0.6;">我是你的科研助手。你可以查询文献、生成综述或寻找灵感。</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h4 style="text-align: center; opacity: 0.7; margin-bottom: 20px;">✨ 猜你想问 (Based on your interest)</h4>', unsafe_allow_html=True)

    # 获取建议
    questions = get_smart_suggestions(st.session_state.username)
    
    # [修改点 1] 定义回调函数：点击后只更新状态，不负责 Rerun (Streamlit 会自动处理)
    def start_chat_callback(q_text):
        st.session_state.messages.append({"role": "user", "content": q_text})
        # 清空建议，防止下次还显示
        st.session_state.current_suggestions = []

    # 创建 2x2 网格
    col1, col2 = st.columns(2)
    
    # [修改点 2] 将 if st.button + st.rerun 改为 on_click 回调模式
    with col1:
        st.button(
            f"💡 {questions[0]}", 
            use_container_width=True, 
            on_click=start_chat_callback, 
            args=(questions[0],)
        )
        st.button(
            f"🔬 {questions[1]}", 
            use_container_width=True, 
            on_click=start_chat_callback, 
            args=(questions[1],)
        )
            
    with col2:
        st.button(
            f"📚 {questions[2]}", 
            use_container_width=True, 
            on_click=start_chat_callback, 
            args=(questions[2],)
        )
        st.button(
            f"🚀 {questions[3]}", 
            use_container_width=True, 
            on_click=start_chat_callback, 
            args=(questions[3],)
        )

def chat_page(mode, use_graph):
    if not st.session_state.messages:
        render_welcome_screen()

    # 2. 渲染历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                 with st.expander("📚 参考来源"):
                    for p in msg["sources"]:
                        st.write(f"- [{p['year']}] **{p['title']}** [PDF]({p.get('pdf_url', '#')})")

    if st.session_state.messages and \
       st.session_state.messages[-1]["role"] == "assistant" and \
       st.session_state.current_suggestions:
        
        st.markdown('<p style="font-size: 0.8em; color: var(--text-color); opacity: 0.6; margin-top: 10px;">✨ 猜你想问 (Follow-up):</p>', unsafe_allow_html=True)
        
        # 定义回调：点击后续问题 -> 上屏 -> 触发生成
        def click_suggestion(q_text):
            st.session_state.messages.append({"role": "user", "content": q_text})
            # 点击后，清空当前的建议，防止重复点击
            st.session_state.current_suggestions = []
            # 回调结束后 Streamlit 会自动 rerun，进入下方的生成逻辑

        # 使用列布局渲染按钮
        cols = st.columns(len(st.session_state.current_suggestions))
        for i, q in enumerate(st.session_state.current_suggestions):
            with cols[i]:
                # 使用 on_click 回调
                st.button(q, key=f"sugg_{len(st.session_state.messages)}_{i}", on_click=click_suggestion, args=(q,), use_container_width=True)

    if prompt := st.chat_input("输入你的研究问题..."):
        # 用户手动输入时，清除旧的推荐建议
        st.session_state.current_suggestions = []
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

    # =========================================================
    # 4. 自动触发回复生成 (Core Loop)
    # =========================================================
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        with st.chat_message("assistant"):
            context_chunks = []
            papers = []
            
            # --- RAG 检索阶段 ---
            with st.status("🚀 正在启动学术引擎...", expanded=True) as status:
                # 构造上下文
                unsummarized_msgs = st.session_state.messages[st.session_state.last_summarized_idx:-1]
                recent_context_str = "\n".join([f"{m['role']}: {m['content']}" for m in unsummarized_msgs])
                full_context_str = f"[Previous Summary]: {st.session_state.current_summary}\n[Recent Context]: {recent_context_str}"
                
                status.write("🔍 分析意图并检索文献...")
                try:
                    user_query = st.session_state.messages[-1]["content"]
                    context_chunks, papers, logs = perform_retrieval(user_query, use_graph, full_context_str)
                    for log in logs: st.write(f"   ↳ {log}")
                    status.update(label="✅ 文献阅读完成，正在撰写报告...", state="running", expanded=False)
                except Exception as e:
                    status.update(label="❌ 检索过程发生错误", state="error")
                    st.error(f"Error: {e}")
                    st.stop()

            # --- LLM 生成阶段 ---
            response_text = ""
            try:
                # 引入新函数 generate_follow_up_questions
                from logic import generate_follow_up_questions 
                
                user_query = st.session_state.messages[-1]["content"]
                stream_gen = get_response_stream(
                    user_query, 
                    mode, 
                    full_context_str, 
                    context_chunks, 
                    language=st.session_state.language,
                    papers_metadata=papers # <--- 新增：这是从 perform_retrieval 返回的
                )
                response_text = st.write_stream(stream_gen)
                
                if not response_text: response_text = "生成似乎中断了..."
                
            except Exception as e:
                st.error(f"生成失败: {e}")
                response_text = "生成失败。"

            # 显示来源
            if papers:
                with st.expander("📚 参考来源"):
                    for p in papers: st.write(f"- [{p['year']}] **{p['title']}** [PDF]({p.get('pdf_url', '#')})")
            
            # --- 保存助手回复 ---
            st.session_state.messages.append({"role": "assistant", "content": response_text, "sources": papers})

            # --- [关键步骤] 生成下一轮的猜你想问 ---
            # 在回答生成完毕后，立即根据最新的上下文生成建议
            # 此时的 full_context_str 包含了之前的信息，但我们需要把最新的问答也加进去生成建议
            latest_interaction = f"User: {user_query}\nAssistant: {response_text}"
            suggestion_context = f"{full_context_str}\n{latest_interaction}"
            
            # 异步/后台生成建议（为了体验，这里是同步的，但通常很快）
            suggestions = generate_follow_up_questions(suggestion_context)
            st.session_state.current_suggestions = suggestions

            # --- 数据库持久化 ---
            current_sum = st.session_state.current_summary
            if not current_sum and st.session_state.messages:
                current_sum = st.session_state.messages[0]['content'][:30] + "..."

            new_id = save_or_update_chat(st.session_state.current_chat_id, st.session_state.username, current_sum, st.session_state.messages)
            st.session_state.current_chat_id = new_id

            # 递归摘要更新 (Optional)
            _, _, _, generator = get_engine()
            new_msgs = st.session_state.messages[st.session_state.last_summarized_idx:]
            if len(new_msgs) >= 2:
                try:
                    new_summary = recursive_summarize(generator, st.session_state.current_summary, new_msgs)
                    st.session_state.current_summary = new_summary
                    st.session_state.last_summarized_idx = len(st.session_state.messages)
                except: pass
            
            # Rerun 刷新界面，此时 st.session_state.current_suggestions 已有值，会在上方被渲染出来
            st.rerun()

    
    # 3. 分享按钮 (修改逻辑：点击生成金句)
    if st.session_state.messages:
        st.divider()
        col1, col2 = st.columns([8, 2])
        with col2:
            if st.button("📤 生成灵感海报并分享", use_container_width=True):
                # 1. 确保已保存
                if st.session_state.current_chat_id is None:
                    summary_fallback = st.session_state.messages[0]['content'][:30]
                    new_id = save_or_update_chat(None, st.session_state.username, summary_fallback, st.session_state.messages)
                    st.session_state.current_chat_id = new_id

                # 2. 调用 LLM 生成 Social Summary
                with st.spinner("✨ 正在提炼金句..."):
                    full_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages])
                    viral_copy = generate_viral_copy(full_text)
                
                # 3. 准备 Payload，默认全选消息
                st.session_state.share_payload = {
                    "original_summary": viral_copy, # 初始生成的文案
                    "msgs": st.session_state.messages,
                    "mode": mode,
                    "selected_indices": list(range(len(st.session_state.messages))) # 默认全选
                }
                
                st.session_state.page = "share_confirm" 
                st.rerun()

# --- [重写] share_confirm_page (支持编辑历史和摘要) ---
def share_confirm_page():
    st.header("📤 分享到灵感广场")
    
    if "share_payload" not in st.session_state or not st.session_state.share_payload:
        st.warning("无内容。")
        if st.button("⬅️ 返回"):
            st.session_state.page = "chat"
            st.rerun()
        return

    payload = st.session_state.share_payload
    msgs = payload['msgs']

    with st.form("share_form"):
        # 1. 标题与摘要编辑
        st.subheader("1. 编辑摘要 (用于广场展示)")
        st.info("💡 这是一个“小红书”风格的短摘要，建议包含 Emoji 和 3 个核心点。")
        
        # 默认标题从摘要第一行提取，或者用户自己写
        default_title = "我的学术灵感"
        if payload.get("original_summary"):
            first_line = payload["original_summary"].split('\n')[0]
            if len(first_line) < 30: default_title = first_line.replace("#", "").strip()

        new_title = st.text_input("标题", value=default_title)
        new_summary = st.text_area("摘要内容 (金句)", value=payload['original_summary'], height=150, max_chars=300)

        st.divider()

        # 2. 对话历史选择
        st.subheader("2. 选择要公开的对话片段")
        st.caption("取消勾选以隐藏特定的问答。")
        
        # 这一步有点 trick，因为在 form 里不能动态 update session_state 的 list
        # 我们使用 key 来记录 checkbox 的状态
        
        selected_msgs_mask = []
        for i, msg in enumerate(msgs):
            # 默认都是 True，除非用户改了
            is_checked = st.checkbox(
                f"**{msg['role']}**: {msg['content'][:60]}...", 
                value=True, 
                key=f"chk_msg_{i}"
            )
            selected_msgs_mask.append(is_checked)

        st.divider()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            submitted = st.form_submit_button("✅ 确认发布", type="primary", use_container_width=True)
        
    if submitted:
        if not new_title.strip():
            st.error("标题不能为空！")
        else:
            # 过滤消息
            final_msgs = [m for i, m in enumerate(msgs) if selected_msgs_mask[i]]
            
            if not final_msgs:
                st.error("请至少保留一条消息内容。")
            else:
                # 写入 Shared Chats 表
                # 这里我们把 new_summary 存入 content 还是 title? 
                # 通常 title 存 title, content 存 json。
                # 现在的表结构是 title, content(json), mode.
                # 我们可以把 new_summary 放在 content 的一个特殊字段里，或者作为 content 的一部分。
                # 为了兼容，我们把 new_summary 插入到 final_msgs 的前面作为系统提示？
                # 不，最好是不破坏 msg 结构。
                # 既然 `share_chat_to_square` 只存 content，我们可以把 summary 放在 content 的 metadata 里？
                # 或者：我们修改 `share_chat_to_square` 逻辑？
                # 为了简单起见，我们把 `new_summary` 仅仅作为 UI 展示用的摘要？ 
                # 实际上 `square_page` 列表里展示的是 title。
                # 让我们把 new_summary 拼接到 title 后面？太长。
                # 💡 方案：我们在 json 里存 {"summary": "...", "messages": [...]}
                # 这样需要在 square_page 解析时做兼容。
                
                final_content = {
                    "summary": new_summary,
                    "messages": final_msgs
                }

                share_chat_to_square(
                    st.session_state.username, 
                    new_title, 
                    final_content,  # 存入 Dict，稍后 json.dumps
                    payload['mode']
                )
                st.toast("🎉 发布成功！正在前往广场...")
                time.sleep(1.5)
                del st.session_state.share_payload
                st.session_state.page = "square"
                st.rerun()

    if st.button("❌ 取消"):
        st.session_state.page = "chat"
        st.rerun()

# --- [重写] square_page (支持直接发布 & 新的数据结构解析) ---
def square_page():
    if st.button("⬅️ 返回对话", key="back_to_chat"):
        st.session_state.page = "chat"
        st.rerun()
        
    st.header("✨ 灵感广场")

    recommended_posts = []
    if st.session_state.logged_in:
        try:
            # 1. 抓取全量数据
            users_data, posts_data, likes_data = fetch_recommendation_data()
            # 2. 实例化引擎
            rec_engine = RecommendationEngine(users_data, posts_data, likes_data)
            # 3. 计算推荐 (返回格式已调整为 tuple list)
            recommended_posts = rec_engine.recommend(st.session_state.username, top_k=10)
        except Exception as e:
            st.error(f"推荐系统初始化失败: {e}")
    
    with st.expander("🛠️ 刷新", expanded=False):
        col_dbg_1, col_dbg_2 = st.columns([1, 1])
        with col_dbg_1:
            if st.button("🔄 载入/刷新 Mock 数据", help="读取 mock_data.json 并注入数据库（自动去重）", use_container_width=True):
                with st.spinner("正在注入模拟数据..."):
                    # 1. 尝试注入
                    success, msg = seed_from_json("mock_data.json")
                    if success:
                        st.success(f"操作完成！\n{msg}")
                        time.sleep(1.5)
                        st.rerun()
                    else:
                        st.error(f"失败: {msg}\n请确保根目录下有 mock_data.json 文件")
        
        with col_dbg_2:
            st.caption("ℹ️ 说明：此操作会将 JSON 中的新帖子和点赞记录同步到数据库。已存在的标题不会重复插入。")

    # --- [新增] 发布新想法入口 ---
    with st.expander("✍️ 发布新想法 / Post Idea", expanded=False):
        with st.form("new_post_form"):
            p_title = st.text_input("标题", placeholder="例如：关于 Transformer 的一点思考...")
            p_summary = st.text_area("核心观点 (金句)", placeholder="一句话总结你的想法，支持 Emoji 💡", height=80)
            p_content = st.text_area("详细内容 (Markdown)", height=200)
            p_tag = st.selectbox("标签", ["inspire (脑暴)", "review (综述)", "explain (深度)"])
            
            if st.form_submit_button("🚀 发布"):
                if not p_title or not p_content:
                    st.error("标题和内容不能为空")
                else:
                    # 构造伪造的消息列表，以便查看器渲染
                    fake_msgs = [{"role": "user", "content": p_content}]
                    final_data = {
                        "summary": p_summary,
                        "messages": fake_msgs
                    }
                    share_chat_to_square(st.session_state.username, p_title, final_data, p_tag)
                    st.success("发布成功！")
                    time.sleep(1)
                    st.rerun()
    
    # 榜单
    star_user, star_likes = get_academic_star()
    if star_user != "暂无":
        st.info(f"🏆 本周学术之星: **{star_user}** (总获赞 {star_likes})")

    st.divider()
    
    # 1. 定义渲染函数 (避免代码重复)
    def render_feed(posts, source_tab):
        current_user = st.session_state.username
        if not posts:
            st.info("这里空空如也~")
            return

        for pid, post_owner, title, content_json, p_mode, likes in posts:
            with st.container():
                # 解析内容
                try:
                    data = json.loads(content_json)
                    if isinstance(data, list):
                        summary_text = data[0]['content'][:100] + "..."
                        messages = data
                    else:
                        summary_text = data.get("summary", "")
                        messages = data.get("messages", [])
                except:
                    summary_text = "数据解析错误"
                    messages = []

                # 卡片 UI
                st.markdown(f"""
                <div class="inspiration-card">
                    <h3>{title}</h3>
                    <p style="font-size: 0.9em; color: var(--text-color); opacity: 0.8; margin-bottom: 8px;">
                        {summary_text.replace(chr(10), '<br>')}
                    </p>
                    <p style="font-size: 0.8em; opacity: 0.6;">
                        👤 <b>{post_owner}</b> | 🏷️ {p_mode} | ❤️ {likes}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                is_my_post = (post_owner == current_user)
                
                # 交互按钮区
                if is_my_post:
                    col1, col2, col3 = st.columns([1.5, 1.5, 7])
                else:
                    col1, col3 = st.columns([1.5, 8.5])
                    col2 = None

                with col1:
                    # [修复] 加上唯一的 key 前缀，防止 Tab 切换时 key 冲突
                    btn_key = f"like_{pid}_{p_mode}_{source_tab}"
                    if st.button(f"❤️ ({likes})", key=btn_key, use_container_width=True, disabled=is_my_post):
                        if not is_my_post:
                            success, msg = like_post(pid, current_user)
                            if success:
                                st.balloons()
                                st.toast(msg)
                                time.sleep(0.5)
                                st.rerun()
                            else:
                                st.toast(msg, icon="🚫")

                if is_my_post and col2:
                    with col2:
                        del_key = f"del_{pid}_{p_mode}_{source_tab}"
                        if st.button("🗑️ 删除", key=del_key, type="primary", use_container_width=True):
                            if delete_shared_chat(pid, current_user):
                                st.toast("已删除", icon="✅")
                                time.sleep(1)
                                st.rerun()

                with col3:
                    with st.expander("查看详情"):
                        for msg in messages:
                            role_icon = "🧑‍💻" if msg['role'] == "user" else "🤖"
                            st.markdown(f"**{role_icon} {msg['role']}**: {msg['content']}")
                
                st.divider()

    tab_rec, tab_hot, tab_new = st.tabs(["✨ 猜你喜欢", "🔥 热门精选", "🆕 最新发布"])
    
    with tab_rec:
        if not st.session_state.logged_in:
            st.warning("请登录后查看个性化推荐")
        else:
            if recommended_posts:
                st.caption(f"基于你的 Bio 和近期点赞行为生成的推荐 (Top {len(recommended_posts)})")
                # 调用现有的渲染函数，传入 distinct tab name
                render_feed(recommended_posts, "rec")
            else:
                st.info("暂无推荐，去给其他帖子点点赞吧！")

    with tab_hot:
        st.caption("按点赞数排序，发现社区共识")
        hot_posts = get_inspiration_posts(sort_by="hot")
        render_feed(hot_posts, "hot")
        
    with tab_new:
        st.caption("按时间倒序，发现新鲜灵感")
        new_posts = get_inspiration_posts(sort_by="new")
        render_feed(new_posts, "new")

# 个人中心
def profile_page():
    # 1. 返回按钮
    if st.button("⬅️ 返回对话", key="back_from_profile"):
        st.session_state.page = "chat"
        st.rerun()

    st.title("⚙️ 个人中心")
    
    profile = get_user_profile(st.session_state.username)
    current_bio = profile.get("bio") or ""
    current_avatar_blob = profile.get("avatar")
    
    st.subheader("头像设置")
    
    # --- [修改] 布局优化：左侧头像，右侧紧凑上传 ---
    # 使用 1:3 的比例，让头像列变窄 (col1)，上传组件在右侧 (col2)
    col1, col2 = st.columns([1, 4], gap="medium")
    
    with col1:
        # 渲染头像，稍微改小一点 size 以适应窄列
        render_avatar(st.session_state.username, current_avatar_blob, size=100)
    
    with col2:
        # 使用 vertical_alignment 让上传按钮和头像垂直对齐 (需要 Streamlit 1.37+ 支持，如果不支持可忽略)
        st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True) # 简单的垂直对齐 Hack
        
        # label_visibility="collapsed" 隐藏 "Browse files" 上面的文字标签，节省空间
        uploaded_file = st.file_uploader("更换头像", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            bytes_data = uploaded_file.getvalue()
            # 按钮也设置 use_container_width=False 让它变小
            if st.button("✅ 确认上传", key="save_avatar_btn"):
                update_user_profile(st.session_state.username, avatar_bytes=bytes_data)
                st.success("已更新")
                time.sleep(1)
                st.rerun()

    # 2. 资料区域 (垂直在下方)
    st.subheader("基本资料")
    new_bio = st.text_area("个人简介 / Bio", value=current_bio, height=100)
    
    # 把保存按钮放在右边，符合操作习惯
    bc1, bc2 = st.columns([4, 1]) 
    with bc2:
        if st.button("保存简介", use_container_width=True):
            update_user_profile(st.session_state.username, bio=new_bio)
            st.success("简介已保存")
            time.sleep(1)
            st.rerun()

    st.divider()
    
    st.subheader("🎨 主题与外观")
    # 1. 语言设置
    st.write("🌐 **界面语言 / Language**")
    LANG_MAP = {"中文": "Chinese", "英文": "English"}
    curr_lang_idx = 0 if st.session_state.language in ["中文", "Chinese"] else 1
    # 使用 horizontal=True 横向排列
    lang_choice = st.radio(
        "选择语言", 
        ["中文", "英文"], 
        index=curr_lang_idx, 
        horizontal=True,
        label_visibility="collapsed" # 隐藏自带的 label，用上面 markdown 写的更好看
    )
    if LANG_MAP[lang_choice] != st.session_state.language:
        st.session_state.language = LANG_MAP[lang_choice]
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True) # 加点间距

    # 2. 主题风格
    st.write("🌈 **主题风格 / Theme**")
    THEMES = ["Nature一作", "AI天才", "我想创业", "理科男", "文艺青年"]
    curr_theme = st.session_state.user_theme
    try:
        theme_idx = THEMES.index(curr_theme)
    except:
        theme_idx = 3 
    
    new_theme = st.radio(
        "选择主题", 
        THEMES, 
        index=theme_idx, 
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # 3. 字体选择
    st.write("🔤 **字体 / Font**")
    FONTS = ["Sans-Serif", "Serif", "Monospace"]
    curr_font = st.session_state.user_font
    try:
        font_idx = FONTS.index(curr_font)
    except:
        font_idx = 0
        
    new_font = st.radio(
        "选择字体", 
        FONTS, 
        index=font_idx, 
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # Apply Theme Button
    if new_theme != st.session_state.user_theme or new_font != st.session_state.user_font:
        if st.button("💾 应用主题设置", type="primary"):
            update_user_profile(st.session_state.username, theme=new_theme, font=new_font)
            st.session_state.user_theme = new_theme
            st.session_state.user_font = new_font
            st.toast("主题已更新！")
            time.sleep(1)
            st.rerun()

    

# --- 主逻辑 ---
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
        curr_theme = st.session_state.get("user_theme", "理科男")
        curr_font = st.session_state.get("user_font", "Sans-Serif")
        load_style(curr_theme, curr_font)
        # 1. 渲染侧边栏 (始终显示)
        mode, use_graph = sidebar()
        
        # 2. 页面路由分发 (Routing)
        if st.session_state.page == "chat":
            chat_page(mode, use_graph)
            
        elif st.session_state.page == "square":
            square_page()
            
        elif st.session_state.page == "profile":
            profile_page()
            
        # [关键修复] 之前漏掉了这个路由，导致跳转后无函数可执行，显示空白
        elif st.session_state.page == "share_confirm":
            share_confirm_page()

if __name__ == "__main__":
    main()