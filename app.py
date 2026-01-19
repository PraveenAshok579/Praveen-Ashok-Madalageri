# ======================================================
# CBR PREDICTION WEB APP – POLISHED VERSION
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import io

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from docx import Document

# ------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------
st.set_page_config(
    page_title="CBR Prediction Tool",
    page_icon="🚦",
    layout="centered"
)

st.markdown(
    "<h1 style='text-align:center;'>🚦 CBR Prediction Tool</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Machine Learning based estimation for geotechnical engineering</p>",
    unsafe_allow_html=True
)

# ------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_excel("CBR Data.xlsx")

df = load_data()

TARGET = "CBR"
X = df.drop(columns=[TARGET])
y = df[TARGET]

# ------------------------------------------------------
# MODEL
# ------------------------------------------------------
X_train, _, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        random_state=42
    ))
])

model.fit(X_train, y_train)

# ------------------------------------------------------
# INPUT SECTION
# ------------------------------------------------------
st.subheader("🔢 Enter Soil Properties")

input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(
        label=col,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(df[col].mean())
    )

input_df = pd.DataFrame([input_data])

# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
if st.button("🔍 Predict CBR"):
    prediction = float(model.predict(input_df)[0])

    # Traffic light logic
    if prediction < 3:
        color, quality = "red", "Very Poor Subgrade"
    elif prediction < 5:
        color, quality = "orange", "Poor Subgrade"
    elif prediction < 10:
        color, quality = "yellow", "Fair Subgrade"
    else:
        color, quality = "green", "Good Subgrade"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        number={"suffix": " %"},
        gauge={
            "axis": {"range": [0, 20]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 3], "color": "red"},
                {"range": [3, 5], "color": "orange"},
                {"range": [5, 10], "color": "yellow"},
                {"range": [10, 20], "color": "green"}
            ]
        },
        title={"text": "Predicted CBR"}
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Soil Classification:** {quality}")

    # --------------------------------------------------
    # REPORTS
    # --------------------------------------------------
    now = datetime.now().strftime("%d-%m-%Y %H:%M")

    # PDF
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Times-Roman", 12)
    c.drawString(50, 800, "CBR Prediction Report")
    c.drawString(50, 780, f"Generated on: {now}")
    c.drawString(50, 750, f"Predicted CBR: {prediction:.2f} %")
    c.drawString(50, 730, f"Soil Quality: {quality}")

    y_pos = 700
    for k, v in input_data.items():
        c.drawString(50, y_pos, f"{k}: {v}")
        y_pos -= 18

    c.save()
    pdf_buffer.seek(0)

    # Word
    doc = Document()
    doc.add_heading("CBR Prediction Report", level=1)
    doc.add_paragraph(f"Generated on: {now}")
    doc.add_paragraph(f"Predicted CBR: {prediction:.2f} %")
    doc.add_paragraph(f"Soil Quality: {quality}")
    doc.add_heading("Input Parameters", level=2)

    for k, v in input_data.items():
        doc.add_paragraph(f"{k}: {v}")

    word_buffer = io.BytesIO()
    doc.save(word_buffer)
    word_buffer.seek(0)

    col1, col2 = st.columns(2)
    col1.download_button(
        "📄 Download PDF Report",
        pdf_buffer,
        "CBR_Report.pdf",
        "application/pdf"
    )
    col2.download_button(
        "📝 Download Word Report",
        word_buffer,
        "CBR_Report.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Developed for academic & professional use</p>",
    unsafe_allow_html=True
)

