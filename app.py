# ======================================================
# CBR PREDICTION WEB APP – FINAL REGULATORY VERSION
# (EPA / CPCB COMPLIANT + WHITE CONCLUSION TEXT)
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
# DARK THEME + WHITE TEXT ENFORCEMENT
# ------------------------------------------------------
st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
    color: white;
}
h1, h2, h3, p {
    color: white !important;
    text-align: center;
}
div[data-testid="stAlert"] p {
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
    "<p style='color:#b0b0b0;'>"
    "Machine-learning-based regulatory decision-support tool "
    "for landfill liner and cover applications (EPA / CPCB context)"
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
# INPUT SECTION
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

# ------------------------------------------------------
# PREDICTION
# ------------------------------------------------------
if st.button("🔍 Predict CBR"):

    prediction = float(model.predict(input_df)[0])

    if prediction < 3:
        color, quality = "red", "Very Poor Subgrade"
        compliance = "❌ Not compliant with EPA / CPCB landfill requirements"
    elif prediction < 5:
        color, quality = "orange", "Poor Subgrade"
        compliance = "⚠️ Conditionally compliant (stabilization required)"
    elif prediction < 10:
        color, quality = "yellow", "Fair Subgrade"
        compliance = "✅ Compliant for landfill cover; liner with design controls"
    else:
        color, quality = "green", "Good Subgrade"
        compliance = "✅ Fully compliant for landfill liner and cover systems"

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
    st.success(f"**Regulatory Status:** {compliance}")

    # --------------------------------------------------
    # REGULATORY CONCLUSION (WHITE TEXT)
    # --------------------------------------------------
    st.subheader("🧾 Engineering & Regulatory Conclusion (EPA / CPCB)")

    if prediction < 3:
        conclusion = (
            "Based on the predicted CBR value, the soil exhibits very low strength and "
            "poor trafficability. According to EPA landfill construction guidance and "
            "CPCB solid waste management practice, such soils are not suitable for direct "
            "use in landfill liner or cover systems. Soil stabilization, blending, or "
            "replacement is mandatory before application."
        )
    elif prediction < 5:
        conclusion = (
            "The predicted CBR indicates marginal strength characteristics. In line with "
            "EPA and CPCB landfill engineering practice, the soil may be conditionally "
            "used for landfill cover applications only after stabilization or improvement. "
            "Direct use as a landfill liner is not recommended."
        )
    elif prediction < 10:
        conclusion = (
            "The predicted CBR represents moderate strength. The soil satisfies EPA and "
            "CPCB requirements for landfill cover systems and may be considered for liner "
            "applications with appropriate compaction control, protection layers, and "
            "quality assurance measures."
        )
    else:
        conclusion = (
            "The predicted CBR indicates good load-bearing capacity and structural stability. "
            "The soil complies with EPA and CPCB landfill engineering requirements and is "
            "suitable for both landfill liner and cover applications, ensuring constructability "
            "and long-term performance."
        )

    st.markdown(
        f"<div style='color:white; background-color:#1e222a; "
        f"padding:15px; border-radius:10px;'>"
        f"{conclusion}</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------
    # REPORT GENERATION
    # --------------------------------------------------
    now = datetime.now().strftime("%d-%m-%Y %H:%M")

    pdf_buffer = io.BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=A4)
    c.setFont("Times-Roman", 12)

    c.drawString(50, 800, "CBR Prediction Report (EPA / CPCB Context)")
    c.drawString(50, 780, f"Generated on: {now}")
    c.drawString(50, 760, f"Predicted CBR: {prediction:.2f} %")
    c.drawString(50, 740, f"Regulatory Status: {compliance}")

    y_pos = 710
    for k, v in input_data.items():
        c.drawString(50, y_pos, f"{k}: {v}")
        y_pos -= 16

    c.drawString(50, y_pos - 10, "Engineering Conclusion:")
    c.drawString(50, y_pos - 30, conclusion[:90])
    c.drawString(50, y_pos - 45, conclusion[90:])

    c.save()
    pdf_buffer.seek(0)

    doc = Document()
    doc.add_heading("CBR Prediction Report (EPA / CPCB)", level=1)
    doc.add_paragraph(f"Generated on: {now}")
    doc.add_paragraph(f"Predicted CBR: {prediction:.2f} %")
    doc.add_paragraph(f"Regulatory Status: {compliance}")
    doc.add_heading("Engineering Conclusion", level=2)
    doc.add_paragraph(conclusion)

    word_buffer = io.BytesIO()
    doc.save(word_buffer)
    word_buffer.seek(0)

    c1, c2 = st.columns(2)
    c1.download_button("📄 Download PDF Report", pdf_buffer, "CBR_Report.pdf", "application/pdf")
    c2.download_button("📝 Download Word Report", word_buffer, "CBR_Report.docx")

# ------------------------------------------------------
# FOOTER
# ------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;'>"
    "Decision-support tool aligned with EPA and CPCB landfill engineering practices"
    "</p>",
    unsafe_allow_html=True
)
