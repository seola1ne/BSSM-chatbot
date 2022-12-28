from numpy import squeeze
from torch import embedding
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
st.subheader('안녕하세요! 무엇이든 물어보세요 :)')

st.sidebar.title("Infomation")
st.sidebar.info(
    """
    [학교 홈페이지](https://school.busanedu.net/bssm-h/main.do)\n
    [재학생 홈페이지](https://bssm.kro.kr/)\n
    [인스타그램](https://www.instargram.com/bssm.hs)\n
    [페이스북](https://www.facebook.com/BusanSoftwareMeisterHighschool)
    """
)

st.sidebar.title("Contact")
st.sidebar.info(
    """
    051-971-2153
    """
) 
    
tab1, tab2, tab3 = st.tabs(["학교 소개", "입학 안내", "문의"])

with tab1:
    st.subheader("저희 학교를 소개합니다 🧑‍💻")
    st.markdown(
    """
    <div class="infomation">
        <img
        class="school-img"
        src="https://newsimg.sedaily.com/2022/03/15/263FDQYMFA_1.png">
        <p>
            <span class="category">주소</span> | 부산광역시 강서구 가락대로 1393<br>
            <span class="category">전화</span> | 051-971-2153<br>
            <span class="category">설립</span> | 1970년 3월 26일<br>
            <span class="category">학생</span> | 125명 (남 : 89명, 여 : 36명)<br>
            <span class="category">교원</span> | 33명 (남 : 13명, 여 : 20명)<br>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("[소마고 길찾기 바로가기 🗺️](https://map.naver.com/v5/directions/-/14349459.146333452,4189553.8356889966,%EB%B6%80%EC%82%B0%EC%86%8C%ED%94%84%ED%8A%B8%EC%9B%A8%EC%96%B4%EB%A7%88%EC%9D%B4%EC%8A%A4%ED%84%B0%EA%B3%A0%EB%93%B1%ED%95%99%EA%B5%90,632131102,PLACE_POI/-/transit?c=14349459.1463332,4189553.8356889,15,0,0,0,dh)")
    
with tab2:
    st.subheader("입학 안내 📚")
    st.markdown(
        """
        <img
            class="school-logo-img"
            src="https://www.smartsocial.co.kr/public/storage/images/partner/pxuCXVkEs0014wOxpUGtlEdbxp6PPPj9XGRrDMtu.png">
        """, unsafe_allow_html=True)
    st.markdown("[부산소마고 입학요강 바로가기 📑](https://school.busanedu.net/bssm-h/cm/cntnts/cntntsView.do?mi=1032596&cntntsId=13617)")

with tab3:
    st.subheader("챗봇에게 무엇이든 물어보세요 🌟")

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