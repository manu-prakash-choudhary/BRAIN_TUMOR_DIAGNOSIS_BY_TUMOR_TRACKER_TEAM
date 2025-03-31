import streamlit as st

# Set page config
st.set_page_config(
    page_title="BrainScanAI - Test",
    page_icon="🧠",
    layout="wide"
)

# Main application header
st.title("🧠 BrainScanAI - Test Page")
st.markdown("This is a test page to verify Streamlit is working properly.")

# Display some basic information
st.write("If you can see this, Streamlit is working!")