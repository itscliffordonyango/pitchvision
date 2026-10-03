"""PitchVision video inference.

Can be used from the command line or imported by the Streamlit application.
The underlying model architecture and inference logic are unchanged.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pitchvision.data import build_labels
from pitchvision.features import load_backbone, extract_video
from pitchvision.model import TemporalActionSpotter
from pitchvision.metrics import TOLERANCES, evaluate, peaks


def run_inference(
    video_path,
    ckpt_path,
    out_dir="spot_output",
    labels_path=None,
    max_seconds=None,
    threshold=0.5,
    device=None,
):
    """Run PitchVision event spotting on a football video.

    Returns:
        dict containing the generated timeline and optional evaluation.
    """

    os.makedirs(out_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------
    # Load trained checkpoint
    # ---------------------------------------------------------

    ck = torch.load(ckpt_path, map_location=device)

    W = ck["W"]
    CLASSES = ck["classes"]
    h = W // 2

    model = TemporalActionSpotter(
        num_classes=len(CLASSES)
    ).to(device)

    model.load_state_dict(ck["state_dict"])
    model.eval()

    # ---------------------------------------------------------
    # Extract video features
    # ---------------------------------------------------------

    bb, tf = load_backbone(device)

    X, fps, secs = extract_video(
        video_path,
        bb,
        tf,
        device,
        max_seconds,
    )

    print(
        f"Features: {X.shape} "
        f"({fps:.2f} fps video) "
        f"in {secs:.1f}s on {device}"
    )

    # ---------------------------------------------------------
    # Create temporal windows
    # ---------------------------------------------------------

    Xt = torch.tensor(X)

    wins = [
        Xt[t - h:t + h + 1].T
        for t in range(h, len(X) - h)
    ]

    # ---------------------------------------------------------
    # Model prediction
    # ---------------------------------------------------------

    P = []

    with torch.no_grad():
        for i in range(0, len(wins), 256):
            batch = torch.stack(wins[i:i + 256]).to(device)

            P.append(
                torch.sigmoid(
                    model(batch)
                ).cpu().numpy()
            )

    P = np.vstack(P)

    # Save scores
    np.save(
        os.path.join(out_dir, "scores.npy"),
        P,
    )

    # ---------------------------------------------------------
    # Detect event peaks
    # ---------------------------------------------------------

    pk = peaks(P)

    events = [
        {
            "time_s": int(t + h),
            "mmss": f"{(t + h) // 60:02d}:{(t + h) % 60:02d}",
            "class": CLASSES[c],
            "score": round(float(pk[t, c]), 4),
        }
        for t, c in zip(
            *np.where(pk >= threshold)
        )
    ]

    events.sort(
        key=lambda e: (
            e["time_s"],
            -e["score"],
        )
    )

    # ---------------------------------------------------------
    # Create timeline
    # ---------------------------------------------------------

    timeline = {
        "video": video_path,
        "checkpoint": ckpt_path,
        "W": W,
        "threshold": threshold,
        "events": events,
    }

    timeline_path = os.path.join(
        out_dir,
        "timeline.json",
    )

    with open(timeline_path, "w") as f:
        json.dump(
            timeline,
            f,
            indent=1,
        )

    print(
        f"{len(events)} timeline entries written "
        f"to {timeline_path}"
    )

    result = {
        "timeline": timeline,
        "timeline_path": timeline_path,
        "fps": fps,
        "processing_seconds": secs,
        "device": device,
    }

    # ---------------------------------------------------------
    # Optional evaluation
    # ---------------------------------------------------------

    if labels_path:

        Y, _, _ = build_labels(
            labels_path,
            len(X),
            CLASSES,
        )

        G = Y[h:len(X) - h]

        a = evaluate(
            G,
            P,
            class_names=CLASSES,
        )

        b = evaluate(
            G,
            peaks(P),
            cap=None,
            class_names=CLASSES,
        )

        rng = np.random.default_rng(0)

        ra = []
        rb = []

        for _ in range(20):

            Pr = rng.random(
                G.shape
            ).astype(np.float32)

            ra.append(
                evaluate(
                    G,
                    Pr,
                )["avg"]
            )

            rb.append(
                evaluate(
                    G,
                    peaks(Pr),
                    cap=None,
                )["avg"]
            )

        evaluation = {
            "n_eval_seconds": len(G),

            "events_per_class": {
                c: int(G[:, k].sum())
                for k, c in enumerate(CLASSES)
            },

            "A": {
                str(d): a[d]["mAP"]
                for d in TOLERANCES
            },

            "A_avg": a["avg"],

            "B": {
                str(d): b[d]["mAP"]
                for d in TOLERANCES
            },

            "B_avg": b["avg"],

            "random_A_avg": float(
                np.mean(ra)
            ),

            "random_A_avg_std": float(
                np.std(ra)
            ),

            "random_B_avg": float(
                np.mean(rb)
            ),

            "random_B_avg_std": float(
                np.std(rb)
            ),
        }

        evaluation_path = os.path.join(
            out_dir,
            "evaluation.json",
        )

        with open(evaluation_path, "w") as f:
            json.dump(
                evaluation,
                f,
                indent=1,
            )

        result["evaluation"] = evaluation
        result["evaluation_path"] = evaluation_path

        print(
            f"Protocol A Avg-mAP "
            f"{a['avg']:.4f} "
            f"(random {np.mean(ra):.4f}±{np.std(ra):.4f}) | "
            f"Protocol B Avg-mAP "
            f"{b['avg']:.4f} "
            f"(random {np.mean(rb):.4f}±{np.std(rb):.4f})"
        )

    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--labels",
        default=None,
        help="Optional Labels-ball.json for evaluation",
    )

    parser.add_argument(
        "--out",
        default="spot_output",
    )

    parser.add_argument(
        "--max_seconds",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        ckpt_path=args.ckpt,
        out_dir=args.out,
        labels_path=args.labels,
        max_seconds=args.max_seconds,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()