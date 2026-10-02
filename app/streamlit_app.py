import streamlit as st


st.set_page_config(
    page_title="PitchVision",
    page_icon="⚽",
    layout="wide",
)


st.title("⚽ PitchVision")

st.markdown(
    """
    ### Football Video Analysis

    Upload a football match video and use the PitchVision AI model
    to identify important match events and generate a searchable
    event timeline.
    """
)

st.divider()

uploaded_video = st.file_uploader(
    "Upload a football match video",
    type=["mp4", "mov", "avi", "mkv"],
)

if uploaded_video is not None:
    st.success(f"Video uploaded: {uploaded_video.name}")

    st.video(uploaded_video)

    st.divider()

    st.subheader("Analysis")

    if st.button("🔍 Analyze Match", type="primary"):
        st.info("PitchVision analysis will be connected here.")