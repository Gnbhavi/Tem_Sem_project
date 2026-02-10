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
    # Load README.md content
    with open("README.md", "r") as f:
        readme_text = f.read()
    st.markdown(readme_text)

with tab2:
    st.header("Results")
    # --- Graph 2: RMSE chart ---
    st.subheader("RMSE")
    st.image("Charts/rmse_vs_epochs_2026-02-09.png",
             caption="RMSE over iterations", width=800)

    # --- Graph 3: PCA comparison ---
    st.subheader("PCA Comparison")
    st.image("Charts/pca_scatter_2026-02-09.png",
             caption="PCA projection comparison", width=800)

    st.subheader("Scatter plot")
    # Example placeholder
    st.image("Charts/hu_moments_pairplot_2026-02-09.png", caption="RMSE over iterations", width=800)

