import streamlit as st

st.set_page_config(
    page_title="Terra Sports Dashboard",
    layout="wide"
)

st.title("Terra Sports Performance Dashboard")

st.markdown("""
Welcome to the **Terra Sports** analytics dashboard.

This dashboard is connected to a secure cloud database and is used to
explore performance, biomechanics, and training data.
""")

st.success("Streamlit app loaded successfully.")