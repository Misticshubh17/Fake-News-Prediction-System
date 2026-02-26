import streamlit as st
import requests
import examples

API_URL = "http://127.0.0.1:8000/predict"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
        .result-box {
            padding: 30px;
            border-radius: 18px;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-top: 30px;
        }
        .real {
            background-color: #e8f5e9;
            color: #2e7d32;
            border: 2px solid #a5d6a7;
        }
        .fake {
            background-color: #ffebee;
            color: #c62828;
            border: 2px solid #ef9a9a;
        }
        .confidence-text {
            font-size: 35px;
            font-weight: bold;
            color: #3b3b3b;
            margin-top: 10px;
        }
        .center {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Fake News Detection")
st.write("Enter a news title and text to check whether it is **Fake or Real**.")

# ---------------- TEST EXAMPLES ----------------
examples = examples.EXAMPLES

selected_example = st.selectbox("Try a test example", list(examples.keys()))

# ---------------- INPUTS ----------------
title = st.text_input("Enter News Title", value=examples[selected_example]["title"])
text = st.text_area("Enter News Text", value=examples[selected_example]["text"], height=150)

# ---------------- BUTTON ----------------
if st.button("Predict"):

    if not title or not text:
        st.warning("⚠ Please enter both title and text.")
        st.stop()

    payload = {
        "title": title,
        "text": text
    }

    with st.spinner("Analyzing news..."):

        try:
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                confidence = float(result["confidence"])

                col1, col2 = st.columns(2)

                # ---------------- LEFT: CONFIDENCE RING ----------------
                with col1:
                    st.markdown("### Model Confidence")
                    ring_color = "#2e7d32" if prediction.lower() == "true" else "#c62828"

                    st.markdown(f"""
                        <div style="display:flex; justify-content:center; margin-top:20px;">
                            <div style="
                                width:180px;
                                height:180px;
                                border-radius:50%;
                                background: conic-gradient(
                                    {ring_color} {confidence * 100}%,
                                    #e0e0e0 {confidence * 100}%
                                );
                                display:flex;
                                align-items:center;
                                justify-content:center;
                                font-size:30px;
                                font-weight:bold;
                            ">
                                <div style="
                                    width:135px;
                                    height:135px;
                                    border-radius:50%;
                                    background:white;
                                    display:flex;
                                    align-items:center;
                                    justify-content:center;                                                                   
                                ">
                                <div class="confidence-text">    
                                    {confidence:.2f}%                                
                                </div>                                    
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<p class='center'>Confidence Score</p>", unsafe_allow_html=True)

                # ---------------- RIGHT: PREDICTION RESULT ----------------
                with col2:
                    st.markdown("### Prediction Result")

                    if prediction.lower() == "true":
                        st.markdown(f"""
                            <div class="result-box real">
                                ✅ REAL NEWS                               
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="result-box fake">
                                ❌ FAKE NEWS                                
                            </div>
                        """, unsafe_allow_html=True)

            else:
                st.error(f"API Error: {response.status_code}")
                st.write(response.text)

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to FastAPI.")
