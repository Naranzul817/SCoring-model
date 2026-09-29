"""
Зээлийн оноо тооцоолох Streamlit апп.

Эхлээд:   python train.py
Дараа нь: streamlit run app.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Сургалттай ижил цэвэрлэгээ, хөрвүүлэлт, шийдвэрийн дүрмийг train.py-аас авна
from train import (
    APPROVE_MIN, METRICS_PATH, MODEL_PATH, PII_COLUMNS, REVIEW_MIN, SCORE_MAX, SCORE_MIN,
    clean_data, decide, encode_features,
)

st.set_page_config(page_title="Зээлийн оноо", page_icon="📊", layout="centered")


@st.cache_resource
def load_bundle():
    """Загвар + хадгалсан текст→тоо хөрвүүлэлтийг ачаална."""
    return joblib.load(MODEL_PATH)


def predict(df: pd.DataFrame, bundle: dict) -> pd.Series:
    # Хадгалсан хөрвүүлэлтээр тоо болгоод сургалтын баганын дарааллаар эрэмбэлнэ
    X = encode_features(df, bundle["encoders"])[bundle["feature_columns"]]
    return pd.Series(bundle["model"].predict(X).clip(SCORE_MIN, SCORE_MAX).round().astype(int), index=df.index)


if not Path(MODEL_PATH).exists():
    st.error("Загвар олдсонгүй. Эхлээд `python train.py` ажиллуулна уу.")
    st.stop()

bundle = load_bundle()
enc = bundle["encoders"]

st.title("Зээлийн оноо (300–850)")
st.caption(
    f"Шийдвэр: **{APPROVE_MIN}+ Зөвшөөрөх**, **{REVIEW_MIN}–{APPROVE_MIN - 1} Нягтлах**, "
    f"**{REVIEW_MIN}-аас доош Татгалзах**. Энэ нь зөвлөмж бөгөөд эцсийн шийдвэрийг "
    "эрх бүхий ажилтан зээлийн бодлогын дагуу гаргана."
)

# --- Загварын үзүүлэлт --------------------------------------------------------
if Path(METRICS_PATH).exists():
    m = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (оноо)", m["MAE"])
    c2.metric("R2", m["R2"])
    c3.metric("Хасагдсан мөр", m["цэвэрлэгээ"]["нийт_хасагдсан"])
    with st.expander("Цэвэрлэгээний тайлан"):
        st.json(m["цэвэрлэгээ"])

tab_one, tab_batch = st.tabs(["Нэг харилцагч", "CSV файлаар"])

# --- Нэг харилцагчийн оноо ----------------------------------------------------
with tab_one:
    with st.form("applicant"):
        a, b = st.columns(2)
        age = a.number_input("Нас", 18, 100, 35)
        edu = b.selectbox("Боловсрол", list(enc["Боловсрол"]))
        job = a.selectbox("Ажлын төрөл", enc["Ажлын төрөл"])
        years = b.number_input("Ажилласан жил", 0, 60, 5)
        income = a.number_input("Сарын орлого (₮)", 1, 100_000_000, 2_000_000, step=50_000)
        balance = b.number_input("Зээлийн үлдэгдэл (₮)", 0, 1_000_000_000, 5_000_000, step=100_000)
        ratio = a.number_input("Өр орлогын харьцаа", 0.0, 100.0, 12.0, step=0.1)
        overdue = b.number_input("Хугацаа хэтрүүлсэн тоо", 0, 50, 0)
        active = a.number_input("Идэвхтэй зээлийн тоо", 0, 50, 1)
        history = b.number_input("Зээлийн түүх (сар)", 0, 600, 60)
        collateral = a.selectbox("Барьцаа хөрөнгө", list(enc["Барьцаа хөрөнгө"]))
        submitted = st.form_submit_button("Оноо тооцох", type="primary")

    if submitted:
        row = pd.DataFrame([{
            "Нас": age, "Боловсрол": edu, "Ажлын төрөл": job, "Ажилласан жил": years,
            "Сарын орлого": income, "Зээлийн үлдэгдэл": balance, "Өр орлогын харьцаа": ratio,
            "Хугацаа хэтрүүлсэн тоо": overdue, "Идэвхтэй зээлийн тоо": active,
            "Зээлийн түүх сараар": history, "Барьцаа хөрөнгө": collateral,
        }])
        score = int(predict(row, bundle).iloc[0])
        decision = decide(score)
        show = {"Зөвшөөрөх": st.success, "Нягтлах": st.warning, "Татгалзах": st.error}[decision]
        st.metric("Зээлийн оноо", score)
        show(f"Шийдвэр: **{decision}**")

# --- CSV файлаар олноор нь ----------------------------------------------------
with tab_batch:
    st.write("credit_data.csv-тэй ижил баганатай файл оруулна уу. Алдаатай мөрийг эхлээд цэвэрлэнэ.")
    file = st.file_uploader("CSV файл", type="csv")
    if file is not None:
        try:
            raw = pd.read_csv(file, encoding="utf-8")
            df, report = clean_data(raw)
            st.write("Цэвэрлэгээ:", report)
            df["Таамагласан оноо"] = predict(df, bundle)
            df["Шийдвэр"] = df["Таамагласан оноо"].map(decide)
        except (KeyError, ValueError) as e:
            st.error(f"Файлыг боловсруулж чадсангүй: {e}")
            st.stop()

        st.bar_chart(df["Шийдвэр"].value_counts())
        # Дэлгэц дээр нэр, имэйл харуулахгүй – зөвхөн ID ба үр дүн
        shown = [c for c in df.columns if c not in PII_COLUMNS[1:]]
        st.dataframe(df[shown], use_container_width=True)
        st.download_button(
            "Үр дүн татах (CSV)",
            df[shown].to_csv(index=False).encode("utf-8-sig"),
            file_name="scored.csv",
            mime="text/csv",
        )
