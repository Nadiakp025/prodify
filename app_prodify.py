
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

st.set_page_config(page_title="PRODIFY", layout="wide")

st.title("🚀 PRODIFY")
st.subheader("Dashboard Prediksi dan Optimasi Produktivitas Harian")

data = pd.read_csv("prodify_dummy_data.csv")

X = data[["jam_tidur","jam_fokus","distraksi_hp","olahraga","kafein","mood","deadline"]]
y = data["skor_produktivitas"]

model = LinearRegression()
model.fit(X,y)

st.sidebar.header("Input Aktivitas Harian")

jam_tidur = st.sidebar.slider("Jam Tidur",0.0,12.0,7.0)
jam_fokus = st.sidebar.slider("Jam Fokus",0.0,12.0,4.0)
distraksi_hp = st.sidebar.slider("Distraksi HP",0.0,10.0,2.0)
olahraga = st.sidebar.slider("Olahraga",0.0,3.0,0.5)
kafein = st.sidebar.slider("Konsumsi Kafein",0,5,1)
mood = st.sidebar.slider("Mood",1,10,7)
deadline = st.sidebar.selectbox("Ada Deadline?", [0,1])

pred = model.predict([[jam_tidur,jam_fokus,distraksi_hp,olahraga,kafein,mood,deadline]])[0]

col1,col2 = st.columns(2)

with col1:
    st.metric("Prediksi Produktivitas", f"{pred:.1f}/100")

with col2:
    st.write("### Optimasi Waktu Harian")

    c = [-5,-1,-2,3]  # max 5F+R+2O-3H

    A_eq = [[1,1,1,1]]
    b_eq = [10]

    A_ub = [
        [1,0,0,0],
        [0,0,0,1],
        [0,0,-1,0]
    ]
    b_ub = [6,2,-0.5]

    bounds = [(0,None),(0,None),(0,None),(0,None)]

    res = linprog(c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=bounds)

    if res.success:
        F,R,O,H = res.x
        st.write(f"Fokus : {F:.2f} jam")
        st.write(f"Istirahat : {R:.2f} jam")
        st.write(f"Olahraga : {O:.2f} jam")
        st.write(f"Hiburan : {H:.2f} jam")

st.write("### Data Dummy")
st.dataframe(data)
