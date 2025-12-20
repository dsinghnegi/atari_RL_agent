"""
Track humans in a video and export visibility segments for selected identities.

The script consumes the JSON produced by `face_indexer.py`, which contains face
embeddings and an explicit `selected` flag for each identity. It runs YOLOv8 for
person detection and DeepSORT for multi-object tracking with re-identification
support. When a track yields a face crop whose embedding matches a selected
identity, the track contributes to that person's visibility segments.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from insightface.app import FaceAnalysis
from ultralytics import YOLO


class SegmentState:
    def __init__(self, start_frame: int) -> None:
        self.start_frame = start_frame
        self.total_frames = 0
        self.visible_frames = 0
        self.last_seen = start_frame

    def register_presence(self, frame_idx: int) -> None:
        self.total_frames += 1
        self.visible_frames += 1
        self.last_seen = frame_idx

    def register_absence(self) -> None:
        self.total_frames += 1

    def build(self) -> Optional[Tuple[int, int, float]]:
        if self.total_frames == 0:
            return None
        end_frame = self.start_frame + self.total_frames - 1
        visibility = self.visible_frames / float(self.total_frames)
        return self.start_frame, end_frame, visibility


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))


def load_selected_faces(face_json: Path) -> Dict[int, np.ndarray]:
    with open(face_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    selected: Dict[int, np.ndarray] = {}
    for person in data.get("people", []):
        if person.get("selected", False):
            emb = np.array(person.get("embedding", []), dtype=np.float32)
            if emb.size > 0:
                selected[int(person["face_id"])] = emb
    return selected


def match_face(
    crop: np.ndarray,
    matcher: FaceAnalysis,
    selected_embeddings: Dict[int, np.ndarray],
    threshold: float,
) -> Optional[int]:
    detections = matcher.get(crop)
    if not detections:
        return None

    best_id: Optional[int] = None
    best_score = -1.0
    for det in detections:
        if det.embedding is None:
            continue
        embedding = det.embedding
        for face_id, ref in selected_embeddings.items():
            score = cosine_similarity(embedding, ref)
            if score > best_score:
                best_score = score
                best_id = face_id
    if best_score >= threshold:
        return best_id
    return None


def update_segments(
    frame_idx: int,
    matched_ids: List[int],
    active_segments: Dict[int, SegmentState],
    finished: Dict[int, List[Tuple[int, int, float]]],
    max_gap: int,
    min_visible_ratio: float,
) -> None:
    for face_id, segment in list(active_segments.items()):
        if face_id in matched_ids:
            segment.register_presence(frame_idx)
        else:
            if frame_idx - segment.last_seen <= max_gap:
                segment.register_absence()
            else:
                record_segment(face_id, segment, finished, min_visible_ratio)
                active_segments.pop(face_id)

    for face_id in matched_ids:
        if face_id not in active_segments:
            seg = SegmentState(frame_idx)
            seg.register_presence(frame_idx)
            active_segments[face_id] = seg


def record_segment(
    face_id: int,
    segment: SegmentState,
    finished: Dict[int, List[Tuple[int, int, float]]],
    min_visible_ratio: float,
) -> None:
    built = segment.build()
    if built is None:
        return
    start, end, visibility = built
    if visibility >= min_visible_ratio:
        finished.setdefault(face_id, []).append((start, end, visibility))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track humans and export visibility segments.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--face-json", required=True, help="JSON produced by face_indexer.py.")
    parser.add_argument("--output-json", required=True, help="Path to save the tracking segments JSON.")
    parser.add_argument("--detector", default="yolov8x.pt", help="YOLO model checkpoint for person detection.")
    parser.add_argument("--tracking-max-age", type=int, default=30, help="Frames to keep lost tracks alive in DeepSORT.")
    parser.add_argument(
        "--embedding-threshold",
        type=float,
        default=0.4,
        help="Minimum cosine similarity to link a track to a selected face.",
    )
    parser.add_argument("--max-gap", type=int, default=15, help="Allowed gap before closing a segment.")
    parser.add_argument("--min-visible-ratio", type=float, default=0.8, help="Visibility fraction required for a segment.")
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size for insightface.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    face_json = Path(args.face_json)

    selected_embeddings = load_selected_faces(face_json)
    if not selected_embeddings:
        raise RuntimeError("No selected faces found in the provided JSON.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    detector = YOLO(args.detector)
    tracker = DeepSort(max_age=args.tracking_max_age, n_init=1, max_iou_distance=0.7)
    matcher = FaceAnalysis(name="buffalo_l")
    matcher.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    active_segments: Dict[int, SegmentState] = {}
    finished_segments: Dict[int, List[Tuple[int, int, float]]] = {}

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = detector(frame, verbose=False)[0]
        detections = []
        for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy(), results.boxes.cls.cpu().numpy()):
            if int(cls) != 0:  # person class
                continue
            x1, y1, x2, y2 = box
            detections.append(((x1, y1, x2 - x1, y2 - y1), conf, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)
        matched_ids: List[int] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame[y1:y2, x1:x2]
            matched_face = match_face(crop, matcher, selected_embeddings, args.embedding_threshold)
            if matched_face is not None:
                matched_ids.append(matched_face)

        update_segments(
            frame_idx,
            matched_ids,
            active_segments,
            finished_segments,
            max_gap=args.max_gap,
            min_visible_ratio=args.min_visible_ratio,
        )

    for face_id, segment in list(active_segments.items()):
        record_segment(face_id, segment, finished_segments, args.min_visible_ratio)

    cap.release()

    output = {
        "video_path": str(video_path),
        "source_faces": str(face_json),
        "max_gap": args.max_gap,
        "min_visible_ratio": args.min_visible_ratio,
        "people": [
            {
                "face_id": face_id,
                "segments": [
                    {"start_frame": s, "end_frame": e, "visibility_ratio": vis}
                    for s, e, vis in finished_segments.get(face_id, [])
                ],
            }
            for face_id in sorted(selected_embeddings.keys())
        ],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved tracking segments to {args.output_json}")


if __name__ == "__main__":
    main()
