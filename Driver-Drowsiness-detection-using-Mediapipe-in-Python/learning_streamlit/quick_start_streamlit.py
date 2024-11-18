import streamlit as st

# 设置全局属性
st.set_page_config(
    page_title="welcome to our first streamlit_app",
    page_icon='☆☆☆ ', # set页面的图标
    layout='wide'      # 控制app的页面布局，设置为宽屏模式
)

# 正文
st.title('hello world')
st.markdown('> streamlit 支持通过st.markdown 直接渲染markdown')
st.markdown('nice to meet you, teachers, I am Brian!')