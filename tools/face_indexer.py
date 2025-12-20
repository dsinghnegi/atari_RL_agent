"""
Index faces in a video, export representative crops, and build visibility segments.

The script uses the insightface FaceAnalysis pipeline to detect faces and generate
embeddings. Faces are grouped by cosine similarity and assigned incremental IDs.
For each identified person, we collect visibility segments where the face is
visible in at least `--min-visible-ratio` fraction of frames within the segment.
A collage of best-quality face crops is saved for quick review.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image


@dataclass
class SegmentState:
    start_frame: int
    total_frames: int = 0
    visible_frames: int = 0
    last_seen: int = 0

    def register_presence(self, frame_idx: int) -> None:
        self.total_frames += 1
        self.visible_frames += 1
        self.last_seen = frame_idx

    def register_absence(self) -> None:
        self.total_frames += 1

    def build(self) -> Optional[Tuple[int, int, float]]:
        if self.total_frames == 0:
            return None
        visibility = self.visible_frames / float(self.total_frames)
        end_frame = self.start_frame + self.total_frames - 1
        return self.start_frame, end_frame, visibility


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    return float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-8))


def assign_face_id(embedding: np.ndarray, registry: Dict[int, np.ndarray], threshold: float) -> int:
    best_id: Optional[int] = None
    best_score = -1.0
    for face_id, ref_emb in registry.items():
        score = cosine_similarity(embedding, ref_emb)
        if score > best_score:
            best_score = score
            best_id = face_id
    if best_id is None or best_score < threshold:
        return max(registry.keys(), default=0) + 1
    return best_id


def update_segments(
    frame_idx: int,
    present_ids: List[int],
    active_segments: Dict[int, SegmentState],
    finished: Dict[int, List[Tuple[int, int, float]]],
    max_gap: int,
    min_visible_ratio: float,
) -> None:
    # Update existing segments with presence/absence information
    for face_id, segment in list(active_segments.items()):
        if face_id in present_ids:
            segment.register_presence(frame_idx)
        else:
            if frame_idx - segment.last_seen <= max_gap:
                segment.register_absence()
            else:
                record_segment(face_id, segment, finished, min_visible_ratio)
                active_segments.pop(face_id)

    # Start new segments for newly appeared faces
    for face_id in present_ids:
        if face_id not in active_segments:
            active_segments[face_id] = SegmentState(
                start_frame=frame_idx, total_frames=1, visible_frames=1, last_seen=frame_idx
            )


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


def build_collage(crops: Dict[int, np.ndarray], tile_size: int = 224, padding: int = 10) -> Image.Image:
    if not crops:
        return Image.new("RGB", (tile_size, tile_size), color=(0, 0, 0))

    ids = sorted(crops.keys())
    grid_cols = min(5, len(ids))
    grid_rows = math.ceil(len(ids) / grid_cols)
    collage_width = grid_cols * tile_size + (grid_cols + 1) * padding
    collage_height = grid_rows * tile_size + (grid_rows + 1) * padding

    collage = Image.new("RGB", (collage_width, collage_height), color=(30, 30, 30))
    for idx, face_id in enumerate(ids):
        crop = crops[face_id]
        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        pil_crop = pil_crop.resize((tile_size, tile_size))

        row, col = divmod(idx, grid_cols)
        x = padding + col * (tile_size + padding)
        y = padding + row * (tile_size + padding)
        collage.paste(pil_crop, (x, y))

        label_bg = Image.new("RGBA", (50, 30), color=(255, 255, 255, 200))
        label = Image.new("RGBA", (50, 30))
        label.paste(label_bg, (0, 0))
        overlay = Image.new("RGBA", collage.size)
        overlay.paste(label, (x, y))
        collage = Image.alpha_composite(collage.convert("RGBA"), overlay)
        collage = collage.convert("RGB")

        cv2.putText(
            np.array(collage),
            f"ID {face_id}",
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        collage = Image.fromarray(np.array(collage))

    return collage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index faces in a video and build visibility segments.")
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output-json", required=True, help="Path to save the face index JSON.")
    parser.add_argument("--collage", required=True, help="Path to save the collage image of faces.")
    parser.add_argument("--similarity-threshold", type=float, default=0.4, help="Cosine similarity threshold to group faces.")
    parser.add_argument("--max-gap", type=int, default=10, help="Allowed gap (frames) before closing a segment.")
    parser.add_argument(
        "--min-visible-ratio", type=float, default=0.8, help="Minimum fraction of frames that must contain the face for a segment to be kept."
    )
    parser.add_argument("--det-size", type=int, default=640, help="Detector input size for insightface.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_path = Path(args.video)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    detector = FaceAnalysis(name="buffalo_l")
    detector.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    face_registry: Dict[int, np.ndarray] = {}
    active_segments: Dict[int, SegmentState] = {}
    finished_segments: Dict[int, List[Tuple[int, int, float]]] = {}
    best_crops: Dict[int, np.ndarray] = {}
    crop_scores: Dict[int, float] = {}

    frame_idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        detections = detector.get(frame)
        present_ids: List[int] = []
        for det in detections:
            if det.bbox is None or det.embedding is None:
                continue
            embedding = det.embedding
            face_id = assign_face_id(embedding, face_registry, args.similarity_threshold)
            face_registry.setdefault(face_id, embedding)
            present_ids.append(face_id)

            bbox = det.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            score = float(det.det_score)
            current_best = crop_scores.get(face_id, -1)
            if crop.size > 0 and score > current_best:
                best_crops[face_id] = crop.copy()
                crop_scores[face_id] = score
                face_registry[face_id] = embedding

        update_segments(
            frame_idx,
            present_ids,
            active_segments,
            finished_segments,
            max_gap=args.max_gap,
            min_visible_ratio=args.min_visible_ratio,
        )

    # Finalize remaining open segments
    for face_id, segment in list(active_segments.items()):
        record_segment(face_id, segment, finished_segments, args.min_visible_ratio)

    cap.release()

    collage_image = build_collage(best_crops)
    collage_image.save(args.collage)

    people_records = []
    for face_id, embedding in sorted(face_registry.items()):
        segments = [
            {"start_frame": s, "end_frame": e, "visibility_ratio": vis}
            for s, e, vis in finished_segments.get(face_id, [])
        ]
        people_records.append(
            {
                "face_id": face_id,
                "selected": True,
                "embedding": embedding.tolist(),
                "segments": segments,
            }
        )

    output = {
        "video_path": str(video_path),
        "collage": os.path.abspath(args.collage),
        "min_visible_ratio": args.min_visible_ratio,
        "max_gap": args.max_gap,
        "people": people_records,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved collage to {args.collage}")
    print(f"Saved face index to {args.output_json}")


if __name__ == "__main__":
    main()
