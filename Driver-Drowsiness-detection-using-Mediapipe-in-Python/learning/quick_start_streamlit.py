import streamlit as st

# 设置全局属性
st.set_page_config(
    page_title="welcome to our first streamlit_app",
    page_icon='☆☆☆ ',
    layout='wide'
)

# 正文
st.title('hello world')
st.markdown('> streamlit 支持通过st.markdown 直接渲染markdown')