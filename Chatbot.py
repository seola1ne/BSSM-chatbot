from numpy import squeeze
from torch import embedding
from yaml import unsafe_load_all
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# 모델 및 데이터 가져오기
# 모델을 최초로 한번만 가져와서 캐시로 계속 사용함

@st.cache(allow_output_mutation = True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation = True)
def get_dataset():
    df = pd.read_csv('Chatbot_embedding_data.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def local_css(style):
    with open(style) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html = True)
        
local_css("style.css")


model = cached_model()
df = get_dataset()

st.header('BSSM 입학 안내 및 홍보 챗봇 🐤')
st.subheader('안녕하세요! 부산소마고 챗봇입니다.')
st.markdown("[부산소마고 홈페이지 바로가기 🏫](https://school.busanedu.net/bssm-h/main.do)")
    
tab1, tab2, tab3 = st.tabs(["학교 소개", "입학 안내", "문의"])

with tab1:
    st.subheader("저희 소마고를 소개합니다")
    
with tab2:
    st.subheader("입학 안내")

with tab3:
    st.subheader("챗봇에게 무엇이든 물어보세요!")

    # 응답
    if 'generated' not in st.session_state:
        st.session_state['generated'] = [] # 챗봇이 대화한 내역
        
    # 질문
    if 'past' not in st.session_state:
        st.session_state['past'] = [] # 내가 대화한 내역
        
    with st.form('form', clear_on_submit = True):
        user_input = st.text_input('사용자 : ', '')
        submitted = st.form_submit_button('전송')
        

    # 응답 예외처리
    if submitted and user_input:
        embedding = model.encode(user_input)

        df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
        answer = df.loc[df['distance'].idxmax()]
        
        st.session_state.past.append(user_input) # 사용자 질문 채팅에 추가
        
        if answer['distance'] > 0.5: # 만약 정확도가 0.5보다 높다면
            st.session_state.generated.append(answer['챗봇']) # 해당하는 챗봇의 대답 채팅에 추가
        elif answer['distance'] <= 0.5: 
            st.session_state.generated.append("죄송합니다, 적절한 답변이 존재하지 않아요. 😢 051-971-2153로 문의해 주세요!")
        
    for i in range(len(st.session_state['past'])):
        # message(st.session_state['past'][i], is_user = True, key = str(i) + '_user')
        # if len(st.session_state['generated']) > i:
        #     message(st.session_state['generated'][i], key = str(i) + '_bot')
        
        st.markdown("""
                    <div class="myMsg">
                        <img 
                            class="user-profile"
                            src="https://item.kakaocdn.net/do/55637fa74e97ecbe8c86ac5e8d1d9f628b566dca82634c93f811198148a26065">
                        <div class="msg-info">
                            <span class="msg-info-name">사용자</span>
                            <span class="msg-info-time">12:45</span>
                        </div>
                        <p class="msg">{0}</p>
                    </div>
                    <div class="anotherMsg">
                        <img 
                            class="user-profile"
                            src="https://mblogthumb-phinf.pstatic.net/MjAxOTA0MDZfMTk1/MDAxNTU0NDc2Njg3NDUy.089kJiam91Af7LfHnaWHF7lUXwUiyiaEJaBKHf2Odj4g.iU_6CgwmfSkDstaA5CUcctJFeuw0GlbPZiiBK9aMW64g.JPEG.xvx404/1535826348.jpg?type=w800">
                        <div class="msg-info">
                            <span class="msg-info-name">소마고 챗봇</span>
                            <span class="msg-info-time">12:46</span>
                        </div>
                        <p class="msg">{1}</p>
                    </div>
                    """.format(st.session_state['past'][i], st.session_state['generated'][i]), unsafe_allow_html=True)