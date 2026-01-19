# ======================================================
# CBR PREDICTION WEB APP – FINAL PROFESSIONAL VERSION
# (WITH LANDFILL APPLICATION CONCLUSION)
# ======================================================

import streamlit as st
import pandas as pd
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
# PAGE CONFIGURATION
# ------------------------------------------------------
st.set_page_config(
    page_title="CBR Prediction Tool",
    page_icon="🚦",
    layout="centered"
)

# ------------------------------------------------------
# DARK THEME STYLING
# ------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3 {
    text-align: center;
    color: white;
}
input {
    background-color: #1e222a !important;
    color: white !important;
}
button {
    background-color: #00c853 !important;
    color: black !important;
    font-size: 18px !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------
# TITLE
# ------------------------------------------------------
st.markdown("<h1>🚦 CBR Prediction Tool</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#b0b0b0;'>"
    "Machine learning based decision-support tool for landfill liner and cover applications"
    "</p>",
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
# MODEL TRAINING
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
# INPUT SECTION (COLORED CARDS)
# ------------------------------------------------------
st.subheader("🔢 Soil Input Parameters")

colors = ["#ff5252", "#ffa000", "#ffee58", "#69f0ae", "#40c4ff", "#b388ff"]
input_data = {}
cols = st.columns(2)

for i, col in enumerate(X.columns):
    with cols[i % 2]:
        st.markdown(
            f"<div style='border-left:5px solid {colors[i % len(colors)]}; "
            f"padding-left:10px;margin-bottom:5px;'>"
            f"<strong style='color:{colors[i % len(colors)]}'>{col}</strong>"
            f"</div>",
            unsafe_allow_html=True
        )
        input_data[col] = st.number_input(
            "",
            min_value=float(df[col].min()),
            max_value=float(df[col].max()),
            value=float(df[col].mean()),
            key=col
        )

input_df = pd.DataFrame([input_data])

st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
if st.button("🔍 Predict CBR"):

    prediction = float(model.predict(input_df)[0])

    # Traffic-light logic
    if prediction < 3:
        color, quality = "red", "Very Poor Subgrade"
    elif prediction < 5:
        color, quality = "orange", "Poor Subgrade"
    elif prediction < 10:
        color, quality = "yellow", "Fair Subgrade"
    else:
        color, quality = "green", "Good Subgrade"

    # Gauge
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
                {"range": [10, 20], "color": "green"},
            ]
        },
        title={"text": "Predicted CBR"}
    ))

    st.plotly_chart(fig, use_container_width=True)
    st.success(f"**Soil Classification:** {quality}")

    # --------------------------------------------------
    # LANDFILL APPLICATION CONCLUSION
    # --------------------------------------------------
    st.subheader("🧾 Engineering Conclusion for Landfill Applications")

    if prediction < 3:
        conclusion = (
            "The predicted CBR value indicates a very weak soil condition. "
            "Such soils are not suitable for direct use in landfill liner or cover systems "
            "due to poor load-bearing capacity and high deformation potential. "
            "Soil stabilization or blending with stronger materials is strongly recommended."
        )
    elif prediction < 5:
        conclusion = (
            "The predicted CBR suggests marginal strength characteristics. "
            "The soil may be used for landfill cover applications only after improvement measures "
            "such as stabilization or controlled compaction. "
            "Direct use as a landfill liner is not recommended."
        )
    elif prediction < 10:
        conclusion = (
            "The predicted CBR represents moderate strength. "
            "The soil is suitable for landfill cover applications and may be considered for liner use "
            "with appropriate design controls, compaction specifications, and quality assurance."
        )
    else:
        conclusion = (
            "The predicted CBR indicates good load-bearing capacity. "
            "The soil is suitable for both landfill liner and cover applications, "
            "providing adequate resistance to deformation and ensuring constructability "
            "under landfill operational conditions."
        )

    st.info(conclusion)

    # --------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------
    now = datetime.now().strftime("%d-%m-%Y %H:%M")

    # PDF
    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Times-Roman", 12)

    c.drawString(50, 800, "CBR Prediction Report")
    c.drawString(50, 780, f"Generated on: {now}")
    c.drawString(50, 760, f"Predicted CBR: {prediction:.2f} %")
    c.drawString(50, 740, f"Soil Classification: {quality}")

    y_pos = 710
    for k, v in input_data.items():
        c.drawString(50, y_pos, f"{k}: {v}")
        y_pos -= 16

    c.drawString(50, y_pos - 10, "Engineering Conclusion:")
    c.drawString(50, y_pos - 30, conclusion[:90])
    c.drawString(50, y_pos - 45, conclusion[90:])

    c.save()
    pdf_buffer.seek(0)

    # Word
    doc = Document()
    doc.add_heading("CBR Prediction Report", level=1)
    doc.add_paragraph(f"Generated on: {now}")
    doc.add_paragraph(f"Predicted CBR: {prediction:.2f} %")
    doc.add_paragraph(f"Soil Classification: {quality}")
    doc.add_heading("Input Parameters", level=2)

    for k, v in input_data.items():
        doc.add_paragraph(f"{k}: {v}")

    doc.add_heading("Engineering Conclusion", level=2)
    doc.add_paragraph(conclusion)

    word_buffer = io.BytesIO()
    doc.save(word_buffer)
    word_buffer.seek(0)

    col1, col2 = st.columns(2)
    col1.download_button("📄 Download PDF Report", pdf_buffer, "CBR_Report.pdf", "application/pdf")
    col2.download_button("📝 Download Word Report", word_buffer, "CBR_Report.docx")

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;'>"
    "Developed for academic, research, and landfill engineering applications"
    "</p>",
    unsafe_allow_html=True
)


