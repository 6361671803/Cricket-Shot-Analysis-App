"""Interactive tool to trim labeled shot clips out of a raw recording.

Turns a longer recording (e.g. a nets session) into individual clips
saved under data/<shot_name>/, ready for the future ML classifier
training pipeline (see data/README.md). Local dev tool - not part of
the Flask app, and needs a display (run it on your own machine).

Keyboard controls (interactive mode):
    d           step forward one frame
    a           step backward one frame
    s           mark the current frame as the clip start
    e           mark the current frame as the clip end
    l           label + save the marked [start, end] range
    q           quit
"""

import argparse
import os
import uuid

import cv2

from shots import SHOT_DB

DEFAULT_DATA_ROOT = "data"


def write_clip(source_path, start_frame, end_frame, shot_name, data_root=DEFAULT_DATA_ROOT):
    """
    Trim frames [start_frame, end_frame) out of source_path and save them
    as a new .mp4 under data_root/<shot_name>/. Returns the output path.
    Raises ValueError for an invalid shot_name, an empty/inverted frame
    range, or an unreadable source video.
    """
    if shot_name not in SHOT_DB:
        raise ValueError(f"'{shot_name}' is not a known shot - see shots.SHOT_DB.keys().")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame.")

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open source video: {source_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_dir = os.path.join(data_root, shot_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"clip_{uuid.uuid4().hex[:8]}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(end_frame - start_frame):
                ok, frame = cap.read()
                if not ok:
                    break
                writer.write(frame)
        finally:
            writer.release()
    finally:
        cap.release()

    return out_path


def _choose_shot_label():
    """Console menu so labels can't be mistyped - always a valid shots.json key."""
    names = list(SHOT_DB.keys())
    print("\nShot types:")
    for i, name in enumerate(names, start=1):
        print(f"  {i:2d}. {name}")
    while True:
        choice = input("Pick a shot number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid choice, try again.")


def run_labeling_session(source_path, data_root=DEFAULT_DATA_ROOT):
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        print(f"Could not open {source_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    start_frame = None
    end_frame = None

    print(__doc__)
    print(f"{total_frames} frames in {source_path}")

    window = "label_clips - a/d step, s/e mark, l label+save, q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    try:
        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                frame_idx = max(0, frame_idx - 1)
                continue

            overlay = frame.copy()
            status = f"frame {frame_idx}/{total_frames}  start={start_frame} end={end_frame}"
            cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(window, overlay)

            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("d"):
                frame_idx = min(max(0, total_frames - 1), frame_idx + 1)
            elif key == ord("a"):
                frame_idx = max(0, frame_idx - 1)
            elif key == ord("s"):
                start_frame = frame_idx
                print(f"start marked at frame {start_frame}")
            elif key == ord("e"):
                end_frame = frame_idx
                print(f"end marked at frame {end_frame}")
            elif key == ord("l"):
                if start_frame is None or end_frame is None or end_frame <= start_frame:
                    print("Mark a valid start (s) and end (e) before labeling.")
                    continue
                shot_name = _choose_shot_label()
                try:
                    out_path = write_clip(source_path, start_frame, end_frame + 1, shot_name, data_root)
                    print(f"Saved {out_path}")
                except ValueError as exc:
                    print(f"Could not save clip: {exc}")
                start_frame = None
                end_frame = None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Trim labeled shot clips out of a raw recording.")
    parser.add_argument("source_video", help="Path to the raw recording to label from.")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Where to save labeled clips (default: data/).")
    args = parser.parse_args()
    run_labeling_session(args.source_video, args.data_root)


if __name__ == "__main__":
    main()
