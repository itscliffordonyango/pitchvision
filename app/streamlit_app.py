import json
import os
import sys
import tempfile

import streamlit as st


# ---------------------------------------------------------
# Make the model package available to the application
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "model",
)

sys.path.insert(0, MODEL_DIR)

from scripts.spot_video import run_inference


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="PitchVision",
    page_icon="⚽",
    layout="wide",
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("⚽ PitchVision")

st.markdown(
    """
    ### Football Video Analysis

    Upload a football match video and use the PitchVision AI
    model to identify important football events and generate
    a searchable event timeline.
    """
)

st.divider()


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    checkpoint_path = st.text_input(
        "Model checkpoint path",
        value="model/results_v2/ckpt_W7_seed0.pt",
        help="Path to the trained PitchVision .pt checkpoint.",
    )

    threshold = st.slider(
        "Detection threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    max_seconds = st.number_input(
        "Maximum video duration (seconds)",
        min_value=0,
        value=0,
        step=30,
        help="Set to 0 to process the entire video.",
    )


# ---------------------------------------------------------
# Video upload
# ---------------------------------------------------------

uploaded_video = st.file_uploader(
    "Upload a football match video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv",
    ],
)


if uploaded_video is not None:

    st.success(
        f"Video uploaded: {uploaded_video.name}"
    )

    st.video(uploaded_video)

    st.divider()

    # -----------------------------------------------------
    # Analysis
    # -----------------------------------------------------

    if st.button(
        "🔍 Analyze Match",
        type="primary",
        use_container_width=True,
    ):

        if not os.path.exists(checkpoint_path):

            st.error(
                "Model checkpoint not found."
            )

            st.info(
                f"Expected checkpoint: `{checkpoint_path}`"
            )

            st.stop()

        with tempfile.TemporaryDirectory() as temp_dir:

            video_path = os.path.join(
                temp_dir,
                uploaded_video.name,
            )

            with open(
                video_path,
                "wb",
            ) as f:
                f.write(
                    uploaded_video.getbuffer()
                )

            output_dir = os.path.join(
                temp_dir,
                "output",
            )

            duration = (
                None
                if max_seconds == 0
                else max_seconds
            )

            # -------------------------------------------------
            # Run model
            # -------------------------------------------------

            with st.spinner(
                "PitchVision is analyzing the match..."
            ):

                try:

                    result = run_inference(
                        video_path=video_path,
                        ckpt_path=checkpoint_path,
                        out_dir=output_dir,
                        threshold=threshold,
                        max_seconds=duration,
                    )

                except Exception as e:

                    st.error(
                        "Analysis failed."
                    )

                    st.exception(e)

                    st.stop()

            # -------------------------------------------------
            # Results
            # -------------------------------------------------

            timeline = result["timeline"]

            events = timeline["events"]

            st.success(
                "Analysis completed successfully."
            )

            # -------------------------------------------------
            # Summary metrics
            # -------------------------------------------------

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Events Detected",
                len(events),
            )

            col2.metric(
                "Processing Time",
                f"{result['processing_seconds']:.1f}s",
            )

            col3.metric(
                "Device",
                result["device"].upper(),
            )

            st.divider()

            # -------------------------------------------------
            # Event timeline
            # -------------------------------------------------

            st.subheader("📍 Event Timeline")

            if not events:

                st.warning(
                    "No events were detected using the current threshold."
                )

            else:

                for event in events:

                    st.markdown(
                        f"""
                        **{event['mmss']} — {event['class']}**

                        Confidence: `{event['score']:.2f}`
                        """
                    )

            st.divider()

            # -------------------------------------------------
            # Event table
            # -------------------------------------------------

            st.subheader("📊 Detected Events")

            if events:

                st.dataframe(
                    events,
                    use_container_width=True,
                )

            else:

                st.info(
                    "No detected events to display."
                )

            # -------------------------------------------------
            # Download timeline
            # -------------------------------------------------

            timeline_json = json.dumps(
                timeline,
                indent=2,
            )

            st.download_button(
                "⬇️ Download Timeline JSON",
                data=timeline_json,
                file_name="timeline.json",
                mime="application/json",
                use_container_width=True,
            )

else:

    st.info(
        "Upload a football match video to begin analysis."
    )