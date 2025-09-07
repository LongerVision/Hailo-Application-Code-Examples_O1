#!/usr/bin/env python3
"""Example module for Hailo Detection + ByteTrack + Supervision (with H.264 output)."""

import argparse
import supervision as sv
import numpy as np
from tqdm import tqdm
import cv2
import queue
import sys
import os
from typing import Dict, List
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference


def initialize_arg_parser() -> argparse.ArgumentParser:
    """Initialize argument parser for the script."""
    parser = argparse.ArgumentParser(
        description="Detection Example - Tracker with ByteTrack and Supervision"
    )
    parser.add_argument(
        "-n", "--net", help="Path for the HEF model.", default="yolov5m_wo_spp.hef"
    )
    parser.add_argument(
        "-i", "--input_video", default="input_video.mp4", help="Path to the input video."
    )
    parser.add_argument(
        "-o", "--output_video", default="output_video.mp4", help="Path to the output video."
    )
    parser.add_argument(
        "-l", "--labels", default="coco.txt", help="Path to a text file containing labels."
    )
    parser.add_argument(
        "-s", "--score_thresh", type=float, default=0.5, help="Score threshold - between 0 and 1."
    )
    return parser


def preprocess_frame(
    frame: np.ndarray, model_h: int, model_w: int, video_h: int, video_w: int
) -> np.ndarray:
    """Preprocess the frame to match the model's input size."""
    if model_h != video_h or model_w != video_w:
        return cv2.resize(frame, (model_w, model_h))
    return frame


def extract_detections(
    hailo_output: List[np.ndarray], h: int, w: int, threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """Extract detections from the HailoRT-postprocess output."""
    xyxy: List[np.ndarray] = []
    confidence: List[float] = []
    class_id: List[int] = []
    num_detections: int = 0

    for i, detections in enumerate(hailo_output):
        if len(detections) == 0:
            continue
        for detection in detections:
            bbox, score = detection[:4], detection[4]

            if score < threshold:
                continue

            # Convert bbox to xyxy absolute pixel values
            bbox[0], bbox[1], bbox[2], bbox[3] = (
                bbox[1] * w,
                bbox[0] * h,
                bbox[3] * w,
                bbox[2] * h,
            )

            xyxy.append(bbox)
            confidence.append(score)
            class_id.append(i)
            num_detections += 1

    # 保证维度正确
    return {
        "xyxy": np.array(xyxy, dtype=np.float32).reshape(-1, 4),
        "confidence": np.array(confidence, dtype=np.float32).reshape(-1),
        "class_id": np.array(class_id, dtype=np.int32).reshape(-1),
        "num_detections": num_detections,
    }


def postprocess_detections(
    frame: np.ndarray,
    detections: Dict[str, np.ndarray],
    class_names: List[str],
    tracker: sv.ByteTrack,
    box_annotator: sv.RoundBoxAnnotator,
    label_annotator: sv.LabelAnnotator,
) -> np.ndarray:
    """Annotate frame with bounding boxes and labels."""
    if detections["num_detections"] == 0:
        return frame

    sv_detections = sv.Detections(
        xyxy=detections["xyxy"],
        confidence=detections["confidence"],
        class_id=detections["class_id"],
    )

    sv_detections = tracker.update_with_detections(sv_detections)

    labels: List[str] = [
        f"#{tracker_id} {class_names[class_id]}"
        for class_id, tracker_id in zip(sv_detections.class_id, sv_detections.tracker_id)
    ]

    annotated_frame: np.ndarray = box_annotator.annotate(
        scene=frame.copy(), detections=sv_detections
    )
    annotated_labeled_frame: np.ndarray = label_annotator.annotate(
        scene=annotated_frame, detections=sv_detections, labels=labels
    )
    return annotated_labeled_frame


def main() -> None:
    args = initialize_arg_parser().parse_args()

    input_queue: queue.Queue = queue.Queue()
    output_queue: queue.Queue = queue.Queue()

    hailo_inference = HailoAsyncInference(
        hef_path=args.net,
        input_queue=input_queue,
        output_queue=output_queue,
    )
    model_h, model_w, _ = hailo_inference.get_input_shape()

    frame_generator = sv.get_video_frames_generator(source_path=args.input_video)
    video_info = sv.VideoInfo.from_video_path(video_path=args.input_video)
    video_w, video_h = video_info.resolution_wh
    box_annotator = sv.RoundBoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    tracker = sv.ByteTrack()

    with open(args.labels, "r", encoding="utf-8") as f:
        class_names: List[str] = f.read().splitlines()

    inference_thread: threading.Thread = threading.Thread(target=hailo_inference.run)
    inference_thread.start()

    # 使用 OpenCV VideoWriter (H.264 压缩)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    out = cv2.VideoWriter(args.output_video, fourcc, video_info.fps, (video_w, video_h))

    for frame in tqdm(frame_generator, total=video_info.total_frames):
        preprocessed_frame: np.ndarray = preprocess_frame(
            frame, model_h, model_w, video_h, video_w
        )
        input_queue.put([preprocessed_frame])

        _, results = output_queue.get()
        if len(results) == 1:
            results = results[0]

        detections: Dict[str, np.ndarray] = extract_detections(
            results, video_h, video_w, args.score_thresh
        )

        annotated_labeled_frame: np.ndarray = postprocess_detections(
            frame, detections, class_names, tracker, box_annotator, label_annotator
        )

        out.write(annotated_labeled_frame)

    out.release()
    input_queue.put(None)
    inference_thread.join()


if __name__ == "__main__":
    main()

