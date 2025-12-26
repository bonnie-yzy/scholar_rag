# ui/app.py
import streamlit as st
import time
import json
from db import (
    init_db, register_user, login_user, share_chat_to_square, 
    get_inspiration_posts, like_post, get_academic_star, 
    save_private_chat, get_private_history_list, save_or_update_chat,
    delete_shared_chat  # <--- 新增这个
)
from logic import process_query, get_engine, recursive_summarize

# 初始化数据库
init_db()

# 页面配置
st.set_page_config(page_title="ScholarRAG", page_icon="🎓", layout="wide")

# 加载 CSS
with open("ui/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- 状态管理 ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "page" not in st.session_state:
    st.session_state.page = "chat" # chat, square, profile
if "messages" not in st.session_state:
    st.session_state.messages = [] # 当前对话历史
if "chat_history_list" not in st.session_state:
    st.session_state.chat_history_list = [] # 历史会话列表 (模拟)

def init_session():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "messages" not in st.session_state:
        st.session_state.messages = [] 
    
    # [新增] 递归摘要状态
    if "current_summary" not in st.session_state:
        st.session_state.current_summary = "" # 当前的全局摘要
    if "last_summarized_idx" not in st.session_state:
        st.session_state.last_summarized_idx = 0 # 指针：messages中多少条已被总结

    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = None # None 表示这是个新对话，还没入库

init_session()

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
        st.caption(f"🚀 Current User: **{st.session_state.username}**")
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
            # 截断一下 summary 防止太长
            display_title = (item['summary'][:200] + '..') if len(item['summary']) > 200 else item['summary']
            if st.button(f"📄 {display_title}", key=f"hist_{item['id']}"):
                st.session_state.messages = item['msgs']
                # 恢复摘要状态 (为了简单，恢复历史时，默认摘要就是数据库存的那个，指针指向末尾)
                st.session_state.current_summary = item['summary']
                st.session_state.last_summarized_idx = len(item['msgs'])
                # [关键] 加载历史时，必须把 ID 也加载进来，这样继续聊就是在旧记录上追加
                st.session_state.current_chat_id = item['id']
                st.session_state.page = "chat"
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

# --- 聊天主逻辑 (集成递归摘要) ---
def chat_page(mode, use_graph):
    st.header("💬 学术对话")
    
    # 1. 渲染历史
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 参考来源"):
                    for p in msg["sources"]:
                        st.write(f"- [{p['year']}] **{p['title']}** [PDF]({p['pdf_url']})")

    # 2. 处理输入
    if prompt := st.chat_input("输入你的研究问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("🤔 思考中...")
            
            # --- [A] 准备上下文 ---
            # 策略：拿 "当前的全局摘要" + "尚未总结的最近几轮对话"
            # 这样既不会丢失很久以前的信息，也保留了最近的鲜活上下文
            
            # 为了给 LLM 最好的 Prompt，我们这里把未总结的 raw text 也拼进去
            unsummarized_msgs = st.session_state.messages[st.session_state.last_summarized_idx:]
            # 这里的 unsummarized_msgs 其实包含了刚刚用户发的 prompt
            # 我们只需要把 prompt 之前的拿出来做 context 即可，但为了简单，全部给 logic 处理
            
            recent_context_str = "\n".join([f"{m['role']}: {m['content']}" for m in unsummarized_msgs[:-1]]) # 不含当前prompt
            
            full_context_str = f"""
            [Previous Summary]: {st.session_state.current_summary}
            [Recent Context]: {recent_context_str}
            """
            
            # --- [B] 生成回答 ---
            response, sources = process_query(prompt, mode, use_graph, full_context_str)
            
            # 显示
            placeholder.markdown(response)
            if sources:
                with st.expander("📚 参考来源"):
                    for p in sources:
                         st.write(f"- [{p['year']}] **{p['title']}** [PDF]({p['pdf_url']})")
            
            # 存入历史
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources
            })

            # ---------------------------------------------------------
            # [新增] 自动保存逻辑 (Auto-Save)
            # ---------------------------------------------------------
            # 1. 确定摘要 (如果没有摘要，暂时用第一句话代替)
            current_sum = st.session_state.current_summary
            if not current_sum and st.session_state.messages:
                current_sum = st.session_state.messages[0]['content'][:30] + "..."

            # 2. 写入数据库 (Upsert)
            new_id = save_or_update_chat(
                chat_id=st.session_state.current_chat_id,
                username=st.session_state.username,
                summary=current_sum,
                messages=st.session_state.messages
            )
            
            # 3. 更新当前 ID (这样下一轮对话就会走 Update 逻辑而不是 Insert)
            st.session_state.current_chat_id = new_id
            # ---------------------------------------------------------

            # --- [C] 异步/延迟更新摘要 ---
            # 回答生成完后，默默更新一下摘要，为下一轮做准备
            # 获取 LLM 引擎
            _, _, _, generator = get_engine()
            
            # 找出所有尚未总结的消息 (包含刚才的 User Prompt 和 Assistant Response)
            new_msgs = st.session_state.messages[st.session_state.last_summarized_idx:]
            
            # 如果累积了超过 2 轮对话 (4条消息)，就触发一次总结更新
            # 这样可以减少 LLM 调用频率，不必每条都总结
            if len(new_msgs) >= 2:
                with st.status("📝 正在整理记忆...", expanded=False) as status:
                    new_summary = recursive_summarize(generator, st.session_state.current_summary, new_msgs)
                    st.session_state.current_summary = new_summary
                    st.session_state.last_summarized_idx = len(st.session_state.messages)
                    status.update(label="记忆已更新", state="complete", expanded=False)

    # 3. 分享按钮
    if st.session_state.messages:
        st.divider()
        col1, col2 = st.columns([8, 2])
        with col2:
            if st.button("📤 分享到广场", use_container_width=True):
                # A. 准备摘要
                summary_to_share = st.session_state.current_summary
                if not summary_to_share:
                    first_msg = st.session_state.messages[0]['content']
                    summary_to_share = first_msg[:200] + ("..." if len(first_msg)>200 else "")

                # B. [关键] 在跳转前，确保当前对话已保存并获取到 ID
                # 这样可以防止跳转回来后 ID 丢失变成 None，从而导致新建重复记录
                if st.session_state.current_chat_id is None:
                    new_id = save_or_update_chat(
                        chat_id=None,
                        username=st.session_state.username,
                        summary=summary_to_share,
                        messages=st.session_state.messages
                    )
                    st.session_state.current_chat_id = new_id

                # C. 存入 Payload
                st.session_state.share_payload = {
                    "summary": summary_to_share,
                    "msgs": st.session_state.messages,
                    "mode": mode
                }
                
                # D. 页面跳转
                st.session_state.page = "share_confirm" 
                st.rerun()

def share_confirm_page():
    st.header("📤 分享到灵感广场")
    
    # [安全检查] 防止直接访问此页面导致报错
    if "share_payload" not in st.session_state or not st.session_state.share_payload:
        st.warning("没有待分享的内容，请返回对话页。")
        if st.button("⬅️ 返回"):
            st.session_state.page = "chat"
            st.rerun()
        return

    payload = st.session_state.share_payload
    
    # 使用 Form 容器，这样看起来更整洁，且不会一修改标题就自动刷新
    with st.form("share_form"):
        st.subheader("编辑发布信息")
        
        # 允许用户修改标题
        new_title = st.text_input("为这段对话起个标题", value=payload['summary'])
        
        st.write("👀 **内容预览:**")
        # 仅展示前几条作为预览
        preview_len = min(3, len(payload['msgs']))
        for i in range(preview_len):
            m = payload['msgs'][i]
            st.caption(f"**{m['role']}**: {m['content'][:100]}...")
        if len(payload['msgs']) > 3:
            st.caption(f"... (共 {len(payload['msgs'])} 条消息)")

        st.divider()
        
        c1, c2 = st.columns([1, 1])
        with c1:
            # 提交按钮
            submitted = st.form_submit_button("✅ 确认发布")
        
    # 表单提交后的逻辑
    if submitted:
        if not new_title.strip():
            st.error("标题不能为空！")
        else:
            # 写入 Shared Chats 表
            share_chat_to_square(
                st.session_state.username, 
                new_title, 
                payload['msgs'], 
                payload['mode']
            )
            st.toast("🎉 发布成功！正在前往广场...")
            time.sleep(1.5)
            # 清除 payload 释放内存
            del st.session_state.share_payload
            st.session_state.page = "square"
            st.rerun()

    # 取消按钮 (在 Form 外面，否则会触发 Form 提交)
    if st.button("❌ 取消"):
        st.session_state.page = "chat"
        st.rerun()

# --- 灵感广场页面 ---
def square_page():
    if st.button("⬅️ 返回对话", key="back_to_chat"):
        st.session_state.page = "chat"
        st.rerun()
        
    st.header("✨ 灵感广场")
    
    # 榜单
    star_user, star_likes = get_academic_star()
    if star_user != "暂无":
        st.info(f"🏆 本周学术之星: **{star_user}** (总获赞 {star_likes})")
    
    posts = get_inspiration_posts()
    
    if not posts:
        st.write("广场暂时空空如也，快去分享你的第一个灵感吧！")
    
    current_user = st.session_state.username

    for pid, post_owner, title, content_json, p_mode, likes in posts:
        with st.container():
            # 卡片样式
            st.markdown(f"""
            <div class="inspiration-card">
                <h3>{title}</h3>
                <p>👤 <b>{post_owner}</b> | 🏷️ 模式: {p_mode} | ❤️ {likes}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # 判断是否是自己的帖子
            is_my_post = (post_owner == current_user)
            
            # 布局调整：根据是否是自己的帖子，分配列宽
            if is_my_post:
                # 如果是自己的，分三栏：点赞(展示用) | 删除按钮 | 详情
                col1, col2, col3 = st.columns([1.5, 1.5, 7])
            else:
                # 如果是别人的，分两栏：点赞按钮 | 详情
                col1, col3 = st.columns([1.5, 8.5])
                col2 = None

            # --- 第一列：点赞 (功能相同) ---
            with col1:
                btn_label = f"❤️ ({likes})"
                # 只有非本人才能点赞，且通过数据库校验
                if st.button(btn_label, key=f"like_{pid}", use_container_width=True, disabled=is_my_post):
                    if is_my_post:
                        st.toast("不能给自己点赞哦", icon="🚫")
                    else:
                        success, msg = like_post(pid, current_user)
                        if success:
                            st.balloons()
                            st.toast(msg)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.toast(msg, icon="🚫")

            # --- 第二列：删除 (仅作者可见) ---
            if is_my_post and col2:
                with col2:
                    # 使用红色按钮区分
                    if st.button("🗑️ 删除", key=f"del_share_{pid}", type="primary", use_container_width=True):
                        if delete_shared_chat(pid, current_user):
                            st.toast("已删除你的分享", icon="✅")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("删除失败，可能权限不足")

            # --- 第三列：详情展开 ---
            with col3:
                with st.expander("查看对话详情"):
                    try:
                        chat_data = json.loads(content_json)
                        for msg in chat_data:
                            role_icon = "🧑‍💻" if msg['role'] == "user" else "🤖"
                            # 限制一下过长的内容显示
                            content_display = msg['content']
                            st.markdown(f"**{role_icon} {msg['role']}**: {content_display}")
                    except:
                        st.error("数据解析失败")
            
            st.divider()

# 个人中心
def profile_page():
    # 1. [交互优化] 返回按钮
    if st.button("⬅️ 返回对话", key="back_from_profile"):
        st.session_state.page = "chat"
        st.rerun()

    st.header("⚙️ 个人中心")
    st.write(f"当前用户: **{st.session_state.username}**")
    st.write("个性化设置接口预留位置...")
    st.divider()
    
    st.subheader("📊 我的数据")
    
    # 2. [严谨逻辑] 从数据库获取真实数据
    # 使用 db.py 中已导入的 get_private_history_list 函数
    try:
        # 获取真实的历史记录列表
        # 注意: db.py 中该函数默认 LIMIT 20，这里显示的是最近的记录数
        history_list = get_private_history_list(st.session_state.username)
        real_count = len(history_list)
        
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric(label="最近归档会话", value=str(real_count))
        
        with col2:
            if real_count > 0:
                with st.expander("📄 查看最近归档记录 (预览)"):
                    for item in history_list:
                        # 格式化时间显示，仅保留日期和时间的前半部分
                        time_str = item['updated_at'].replace("T", " ")[:16]
                        st.caption(f"**{time_str}** | {item['summary']}")
            else:
                st.info("暂无归档记录，快去开始你的第一次学术对话吧！")
                
    except Exception as e:
        st.error(f"读取数据库失败: {e}")

# --- 主逻辑 ---
def main():
    if not st.session_state.logged_in:
        login_page()
    else:
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