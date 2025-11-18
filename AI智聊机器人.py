import ollama
import streamlit as st
from streamlit import spinner

client = ollama.Client(host="http://127.0.0.1:11434")

#初始化消息记录
if 'message' not in st.session_state:
    st.session_state['message'] = []

st.title("AI智聊机器人")
prompt = st.chat_input("请输入你的问题")

if prompt:
    # st.session_state["message"] = []
    st.session_state['message'].append({"role":"user","content":prompt})

    #将历史消息 全部从容器中 加载出来
    for message in st.session_state['message']:
        st.chat_message(message['role']).markdown(message['content'])

    #调用ollama将用户输入传回 然后接收消息 放入 st session_state中
    with spinner():
        response = client.chat(
            model= "deepseek-r1:1.5b",
            messages= [{"role":"user","content":prompt}]
        )

        print(response.message.content)
        st.session_state['message'].append({"role":"assistant","content":response['message']['content']})
        st.chat_message("assistant").markdown(response['message']['content'])

