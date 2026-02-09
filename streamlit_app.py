import streamlit as st



# --- Project Metadata ---
PROJECT_TITLE = "BioPrint 3D: Holder Predictor"
COLLABORATORS = "Bhavi & Yamini (aka Bhavi_Yams)"

# --- Page Setup ---
st.set_page_config(page_title=PROJECT_TITLE, layout="wide")

# --- Header ---
st.title(PROJECT_TITLE)
st.markdown(f"**Collaborators:** {COLLABORATORS}")

# --- Tabs ---
tab1, tab2 = st.tabs(["Workflow", "Results"])

with tab1:
    st.header("Workflow")
    st.write("""
    Here you can describe the steps:
    1. Data preprocessing
    2. Model training
    3. Evaluation
    4. Deployment
    """)

with tab2:
    st.header("Results")
    st.subheader("RMSE")
    st.write("👉 Images go here (plots, charts, etc.)")
    # Example placeholder
    st.image("Charts/rmse_vs_epochs_2026-02-01_16-23-36.png", caption="RMSE over iterations", width=800)

