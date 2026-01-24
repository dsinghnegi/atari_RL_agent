#!/usr/bin/env python3
"""
Dance Pose Tracker - Multi-Person Pose Tracking Pipeline for Dance Videos

This pipeline tracks multiple dancers across video frames using AlphaPose outputs,
handling complex occlusions, frame exits/re-entries, and non-linear motion patterns.

Key Features:
- Uses AlphaPose JSON outputs (bounding boxes + keypoints) instead of separate detection
- Multi-modal feature extraction (body appearance, face embeddings, pose descriptors)
- OC-SORT-style observation-centric short-term tracking
- Global tracklet association via hierarchical clustering
- Outputs per-dancer pose sequences with 0 confidence for invisible frames

Designed for batch processing on Jetson Orin 32GB with PyTorch 2.3+

Usage:
    python dance_pose_tracker.py \
        --alphapose-json /path/to/alphapose_results.json \
        --video /path/to/video.mp4 \
        --output-dir /path/to/output \
        --num-dancers 8

Author: Generated for dance video tracking pipeline
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

# Suppress some warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TrackerConfig:
    """Configuration for the dance pose tracker pipeline."""

    # Short-term tracking parameters
    max_age: int = 60  # Max frames to keep lost tracks (2 sec at 30fps)
    min_hits: int = 3  # Min detections before track is confirmed
    iou_threshold: float = 0.3  # IoU threshold for matching

    # Appearance parameters
    appearance_weight: float = 0.6  # Weight for appearance vs motion
    body_feature_weight: float = 0.4  # Weight for body features when face unavailable
    face_feature_weight: float = 0.6  # Weight for face features when available

    # Feature thresholds
    min_face_keypoint_confidence: float = 0.3  # Min confidence for face keypoints
    min_face_keypoints_visible: int = 20  # Min visible face keypoints to extract face
    face_match_threshold: float = 0.4  # Cosine similarity threshold for face matching
    body_match_threshold: float = 0.3  # Cosine similarity threshold for body matching

    # Tracklet analysis
    min_tracklet_length: int = 10  # Min frames for valid tracklet
    anchor_min_duration: int = 90  # Min frames for anchor tracklet (3 sec at 30fps)
    anchor_min_detection_rate: float = 0.8
    anchor_min_face_visibility: float = 0.3

    # Global association
    affinity_threshold: float = 0.4  # Min affinity to merge tracklets
    temporal_overlap_threshold: float = 0.3  # Max overlap to merge tracklets
    max_spatial_velocity: float = 50.0  # Max pixels/frame for plausible motion

    # Affinity weights
    appearance_affinity_weight: float = 0.6
    pose_affinity_weight: float = 0.3
    spatial_affinity_weight: float = 0.1

    # Output generation
    interpolate_gaps: bool = True
    max_interpolation_gap: int = 15  # Max frames to interpolate
    interpolation_confidence_factor: float = 0.5  # Confidence multiplier for interpolated

    # Halpe-136 keypoint configuration
    num_keypoints: int = 136
    face_keypoint_start: int = 26  # Face keypoints start at index 26
    face_keypoint_end: int = 94  # Face keypoints end at index 94

    # Body keypoint indices for pose descriptor
    body_keypoints: List[int] = field(default_factory=lambda: [
        0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    ])


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Detection:
    """Single person detection from AlphaPose output."""
    frame_id: int
    detection_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # [num_keypoints, 3] - x, y, confidence
    score: float

    # Features (populated later)
    body_embedding: Optional[np.ndarray] = None
    face_embedding: Optional[np.ndarray] = None
    face_confidence: float = 0.0
    pose_descriptor: Optional[np.ndarray] = None
    body_center: Optional[np.ndarray] = None
    body_scale: float = 0.0

    def __post_init__(self):
        """Compute derived features after initialization."""
        if self.body_center is None:
            self._compute_body_center()

    def _compute_body_center(self):
        """Compute body center from keypoints."""
        visible_kpts = self.keypoints[self.keypoints[:, 2] > 0.1]
        if len(visible_kpts) > 0:
            self.body_center = np.mean(visible_kpts[:, :2], axis=0)
            self.body_scale = np.std(visible_kpts[:, :2])
        else:
            # Fallback to bbox center
            self.body_center = np.array([
                (self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2
            ])
            self.body_scale = max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


@dataclass
class TrackState:
    """State for active track during short-term tracking."""
    track_id: int
    detections: List[Detection]
    last_detection: Detection
    age: int = 0  # Frames since last detection
    hits: int = 1  # Number of detections
    state: str = "tentative"  # tentative, confirmed, deleted

    # Aggregated features
    body_embeddings: List[np.ndarray] = field(default_factory=list)
    face_embeddings: List[np.ndarray] = field(default_factory=list)

    @property
    def start_frame(self) -> int:
        return self.detections[0].frame_id

    @property
    def end_frame(self) -> int:
        return self.detections[-1].frame_id

    def get_mean_body_embedding(self) -> Optional[np.ndarray]:
        if not self.body_embeddings:
            return None
        return np.mean(self.body_embeddings, axis=0)

    def get_mean_face_embedding(self) -> Optional[np.ndarray]:
        if not self.face_embeddings:
            return None
        return np.mean(self.face_embeddings, axis=0)

    def predict_position(self) -> np.ndarray:
        """Predict next position using simple velocity model."""
        if len(self.detections) < 2:
            return self.last_detection.body_center

        # Use last two positions for velocity estimate
        pos1 = self.detections[-2].body_center
        pos2 = self.detections[-1].body_center
        velocity = pos2 - pos1

        # Predict with damping for non-linear motion
        damping = 0.5
        predicted = pos2 + velocity * damping * (self.age + 1)
        return predicted

    def predict_bbox(self) -> np.ndarray:
        """Predict next bounding box."""
        if len(self.detections) < 2:
            return self.last_detection.bbox.copy()

        # Simple velocity-based prediction with damping
        bbox1 = self.detections[-2].bbox
        bbox2 = self.detections[-1].bbox
        velocity = bbox2 - bbox1

        damping = 0.3  # Lower damping for bbox (more conservative)
        predicted = bbox2 + velocity * damping * (self.age + 1)
        return predicted


@dataclass
class Tracklet:
    """Completed tracklet from short-term tracking."""
    tracklet_id: int
    start_frame: int
    end_frame: int
    detections: List[Detection]

    # Aggregated features
    mean_body_embedding: Optional[np.ndarray] = None
    mean_face_embedding: Optional[np.ndarray] = None
    median_pose_descriptor: Optional[np.ndarray] = None

    # Quality metrics
    detection_rate: float = 0.0
    avg_confidence: float = 0.0
    face_visibility_rate: float = 0.0
    appearance_consistency: float = 0.0

    def __post_init__(self):
        """Compute aggregated features and quality metrics."""
        self._aggregate_features()
        self._compute_quality_metrics()

    def _aggregate_features(self):
        """Aggregate features from all detections."""
        # Body embeddings - confidence-weighted mean
        body_embs = [d.body_embedding for d in self.detections if d.body_embedding is not None]
        if body_embs:
            confidences = np.array([d.score for d in self.detections if d.body_embedding is not None])
            weights = softmax(confidences / 0.1)
            self.mean_body_embedding = np.average(body_embs, axis=0, weights=weights)

        # Face embeddings - select highest confidence ones
        face_data = [(d.face_embedding, d.face_confidence)
                     for d in self.detections if d.face_embedding is not None]
        if face_data:
            face_data.sort(key=lambda x: x[1], reverse=True)
            top_faces = [f[0] for f in face_data[:5]]
            self.mean_face_embedding = np.mean(top_faces, axis=0)

        # Pose descriptors - median
        pose_descs = [d.pose_descriptor for d in self.detections if d.pose_descriptor is not None]
        if pose_descs:
            self.median_pose_descriptor = np.median(pose_descs, axis=0)

    def _compute_quality_metrics(self):
        """Compute quality metrics for tracklet."""
        duration = self.end_frame - self.start_frame + 1

        # Detection rate
        self.detection_rate = len(self.detections) / max(duration, 1)

        # Average confidence
        self.avg_confidence = np.mean([d.score for d in self.detections])

        # Face visibility rate
        faces_visible = sum(1 for d in self.detections if d.face_embedding is not None)
        self.face_visibility_rate = faces_visible / max(len(self.detections), 1)

        # Appearance consistency
        if len(self.detections) > 1 and self.mean_body_embedding is not None:
            body_embs = [d.body_embedding for d in self.detections if d.body_embedding is not None]
            if len(body_embs) > 1:
                pairwise_sims = []
                for i in range(min(10, len(body_embs))):
                    for j in range(i + 1, min(10, len(body_embs))):
                        sim = cosine_similarity(body_embs[i], body_embs[j])
                        pairwise_sims.append(sim)
                if pairwise_sims:
                    self.appearance_consistency = np.mean(pairwise_sims)

    @property
    def anchor_score(self) -> float:
        """Compute anchor quality score (higher = more reliable)."""
        duration_score = min((self.end_frame - self.start_frame) / 150, 1.0)
        return (
            0.3 * duration_score +
            0.2 * self.detection_rate +
            0.2 * self.face_visibility_rate +
            0.2 * self.appearance_consistency +
            0.1 * self.avg_confidence
        )

    @property
    def is_anchor(self) -> bool:
        """Check if tracklet qualifies as anchor."""
        return self.anchor_score > 0.6


@dataclass
class DancerPoseSequence:
    """Complete pose sequence for a single dancer across all frames."""
    person_id: int
    name: Optional[str]
    total_frames: int
    keypoints: np.ndarray  # [total_frames, num_keypoints, 3]
    visibility: np.ndarray  # [total_frames] boolean
    confidence: np.ndarray  # [total_frames] float
    bboxes: np.ndarray  # [total_frames, 4]
    interpolated: np.ndarray  # [total_frames] boolean - which frames are interpolated

    def get_visible_segments(self) -> List[Tuple[int, int]]:
        """Return list of (start, end) frame ranges where dancer is visible."""
        segments = []
        in_segment = False
        start = 0

        for i, vis in enumerate(self.visibility):
            if vis and not in_segment:
                start = i
                in_segment = True
            elif not vis and in_segment:
                segments.append((start, i - 1))
                in_segment = False

        if in_segment:
            segments.append((start, len(self.visibility) - 1))

        return segments

    def get_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for this dancer's sequence."""
        visible_frames = int(self.visibility.sum())
        interpolated_frames = int(self.interpolated.sum())
        segments = self.get_visible_segments()

        return {
            "person_id": self.person_id,
            "name": self.name,
            "total_frames": self.total_frames,
            "visible_frames": visible_frames,
            "interpolated_frames": interpolated_frames,
            "visibility_ratio": visible_frames / max(self.total_frames, 1),
            "num_segments": len(segments),
            "mean_confidence": float(self.confidence[self.visibility].mean()) if visible_frames > 0 else 0.0,
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for array x."""
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-8)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    return intersection / (union + 1e-8)


def compute_iou_matrix(bboxes1: List[np.ndarray], bboxes2: List[np.ndarray]) -> np.ndarray:
    """Compute IoU matrix between two lists of bounding boxes."""
    n, m = len(bboxes1), len(bboxes2)
    iou_matrix = np.zeros((n, m))
    for i, bbox1 in enumerate(bboxes1):
        for j, bbox2 in enumerate(bboxes2):
            iou_matrix[i, j] = compute_iou(bbox1, bbox2)
    return iou_matrix


def temporal_overlap(t1_start: int, t1_end: int, t2_start: int, t2_end: int) -> float:
    """Compute temporal overlap ratio between two time ranges."""
    overlap_start = max(t1_start, t2_start)
    overlap_end = min(t1_end, t2_end)
    overlap = max(0, overlap_end - overlap_start + 1)

    min_duration = min(t1_end - t1_start + 1, t2_end - t2_start + 1)
    return overlap / max(min_duration, 1)


def extract_frame_number(image_id: str) -> int:
    """Extract frame number from AlphaPose image_id string."""
    # Handle various formats: "frame_123.png", "123.jpg", "frame123", etc.
    match = re.search(r'(\d+)', image_id)
    if match:
        return int(match.group(1))
    return 0


# =============================================================================
# ALPHAPOSE PARSER
# =============================================================================

class AlphaPoseParser:
    """Parse AlphaPose JSON output into Detection objects."""

    def __init__(self, config: TrackerConfig):
        self.config = config

    def parse(self, json_path: str) -> Dict[int, List[Detection]]:
        """
        Parse AlphaPose JSON file.

        Returns:
            Dict mapping frame_id to list of Detection objects
        """
        logger.info(f"Parsing AlphaPose results from {json_path}")

        with open(json_path, "r") as f:
            data = json.load(f)

        # Group by frame
        frame_detections: Dict[int, List[Detection]] = defaultdict(list)

        for idx, entry in enumerate(data):
            frame_id = extract_frame_number(entry.get("image_id", str(idx)))

            # Parse bounding box [x, y, w, h] -> [x1, y1, x2, y2]
            box = entry.get("box", [0, 0, 0, 0])
            bbox = np.array([
                box[0],
                box[1],
                box[0] + box[2],
                box[1] + box[3]
            ], dtype=np.float32)

            # Parse keypoints (flattened list) -> [num_keypoints, 3]
            kpts_flat = entry.get("keypoints", [])
            num_kpts = len(kpts_flat) // 3
            keypoints = np.array(kpts_flat, dtype=np.float32).reshape(num_kpts, 3)

            # Pad or truncate to expected number of keypoints
            if num_kpts < self.config.num_keypoints:
                padding = np.zeros((self.config.num_keypoints - num_kpts, 3), dtype=np.float32)
                keypoints = np.vstack([keypoints, padding])
            elif num_kpts > self.config.num_keypoints:
                keypoints = keypoints[:self.config.num_keypoints]

            score = entry.get("score", 0.5)

            detection = Detection(
                frame_id=frame_id,
                detection_id=len(frame_detections[frame_id]),
                bbox=bbox,
                keypoints=keypoints,
                score=score
            )

            frame_detections[frame_id].append(detection)

        logger.info(f"Parsed {sum(len(v) for v in frame_detections.values())} detections "
                   f"across {len(frame_detections)} frames")

        return dict(frame_detections)


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class PoseDescriptorExtractor:
    """Extract pose-based identity descriptors from keypoints."""

    def __init__(self, config: TrackerConfig):
        self.config = config

    def extract(self, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose descriptor from keypoints.

        Includes:
        - Normalized joint positions relative to torso
        - Limb length ratios (body proportions)
        """
        # Check if we have enough visible keypoints
        visible_mask = keypoints[:, 2] > 0.1
        if visible_mask.sum() < 5:
            return None

        # Key body points (Halpe-136 indices)
        # 0: nose, 5: l_shoulder, 6: r_shoulder, 11: l_hip, 12: r_hip, 18: neck
        key_indices = [0, 5, 6, 11, 12, 18] if keypoints.shape[0] > 18 else [0, 5, 6, 11, 12]

        # Extract key points
        key_pts = []
        for idx in key_indices:
            if idx < keypoints.shape[0] and keypoints[idx, 2] > 0.1:
                key_pts.append(keypoints[idx, :2])

        if len(key_pts) < 4:
            return None

        # Compute torso center and scale
        visible_body_kpts = keypoints[:26, :2][keypoints[:26, 2] > 0.1]
        if len(visible_body_kpts) < 3:
            return None

        torso_center = np.mean(visible_body_kpts, axis=0)
        torso_scale = np.std(visible_body_kpts) + 1e-6

        # Normalize body keypoints
        normalized_kpts = (keypoints[:26, :2] - torso_center) / torso_scale
        normalized_kpts = normalized_kpts * keypoints[:26, 2:3]  # Weight by confidence

        # Compute limb ratios if possible
        ratios = []
        if keypoints.shape[0] > 12:
            # Shoulder width
            if keypoints[5, 2] > 0.1 and keypoints[6, 2] > 0.1:
                shoulder_width = np.linalg.norm(keypoints[5, :2] - keypoints[6, :2])
                ratios.append(shoulder_width / torso_scale)

            # Hip width
            if keypoints[11, 2] > 0.1 and keypoints[12, 2] > 0.1:
                hip_width = np.linalg.norm(keypoints[11, :2] - keypoints[12, :2])
                ratios.append(hip_width / torso_scale)

        # Combine features
        features = normalized_kpts.flatten()
        if ratios:
            features = np.concatenate([features, np.array(ratios)])

        return features


class BodyFeatureExtractor:
    """Extract body appearance features using lightweight CNN or simple histogram features."""

    def __init__(self, config: TrackerConfig, use_deep_features: bool = True):
        self.config = config
        self.use_deep_features = use_deep_features
        self.model = None
        self.transform = None

        if use_deep_features:
            self._init_model()

    def _init_model(self):
        """Initialize the feature extraction model."""
        try:
            import torch
            import torchvision.transforms as T
            from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

            # Use MobileNetV3 for efficiency on Jetson
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {self.device}")

            self.model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # Remove classifier to get features
            self.model.classifier = torch.nn.Identity()
            self.model = self.model.to(self.device)
            self.model.eval()

            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((224, 112)),  # Height x Width for person crops
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            logger.info("Body feature extractor initialized with MobileNetV3")

        except ImportError:
            logger.warning("PyTorch or torchvision not available, using histogram features")
            self.use_deep_features = False

    def extract(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """Extract body appearance features from image crop."""
        # Crop and validate
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        if self.use_deep_features and self.model is not None:
            return self._extract_deep_features(crop)
        else:
            return self._extract_histogram_features(crop)

    def _extract_deep_features(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract deep features using CNN."""
        import torch

        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Transform
            input_tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                features = self.model(input_tensor)

            return features.cpu().numpy().flatten()

        except Exception as e:
            logger.debug(f"Deep feature extraction failed: {e}")
            return self._extract_histogram_features(crop)

    def _extract_histogram_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract simple color histogram features."""
        # Resize for consistency
        crop_resized = cv2.resize(crop, (64, 128))

        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)

        # Compute histograms for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256]).flatten()

        # Normalize
        hist_h = hist_h / (hist_h.sum() + 1e-8)
        hist_s = hist_s / (hist_s.sum() + 1e-8)
        hist_v = hist_v / (hist_v.sum() + 1e-8)

        # Also compute spatial color features (divide into 3 vertical parts)
        h_third = crop_resized.shape[0] // 3
        spatial_features = []
        for i in range(3):
            part = crop_resized[i * h_third:(i + 1) * h_third]
            part_hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
            part_hist = cv2.calcHist([part_hsv], [0], None, [16], [0, 180]).flatten()
            part_hist = part_hist / (part_hist.sum() + 1e-8)
            spatial_features.append(part_hist)

        features = np.concatenate([hist_h, hist_s, hist_v] + spatial_features)
        return features


class FaceFeatureExtractor:
    """Extract face embeddings using face region from pose keypoints."""

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.face_model = None
        self._init_model()

    def _init_model(self):
        """Initialize InsightFace model for face embedding."""
        try:
            from insightface.app import FaceAnalysis

            self.face_model = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.face_model.prepare(ctx_id=0, det_size=(320, 320))
            logger.info("Face feature extractor initialized with InsightFace")

        except ImportError:
            logger.warning("InsightFace not available, face features will not be extracted")
            self.face_model = None
        except Exception as e:
            logger.warning(f"Failed to initialize InsightFace: {e}")
            self.face_model = None

    def extract_face_region(self, keypoints: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
        """
        Extract face bounding box from Halpe-136 keypoints.

        Returns:
            Tuple of (bbox [x1, y1, x2, y2], confidence) or None
        """
        # Face keypoints are indices 26-93 (68 points) in Halpe-136
        face_start = self.config.face_keypoint_start
        face_end = self.config.face_keypoint_end

        if keypoints.shape[0] < face_end:
            return None

        face_kpts = keypoints[face_start:face_end]
        visible = face_kpts[:, 2] > self.config.min_face_keypoint_confidence

        if visible.sum() < self.config.min_face_keypoints_visible:
            return None

        visible_kpts = face_kpts[visible, :2]
        x1, y1 = visible_kpts.min(axis=0)
        x2, y2 = visible_kpts.max(axis=0)

        # Expand for context
        w, h = x2 - x1, y2 - y1
        x1 = x1 - 0.3 * w
        y1 = y1 - 0.3 * h
        x2 = x2 + 0.3 * w
        y2 = y2 + 0.3 * h

        confidence = float(face_kpts[:, 2].mean())

        return np.array([x1, y1, x2, y2]), confidence

    def extract(self, image: np.ndarray, keypoints: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
        """
        Extract face embedding from image using keypoints to locate face.

        Returns:
            Tuple of (embedding, confidence) or (None, 0.0)
        """
        face_region = self.extract_face_region(keypoints)
        if face_region is None:
            return None, 0.0

        bbox, confidence = face_region

        # Crop face region
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
            return None, 0.0

        face_crop = image[y1:y2, x1:x2]

        if self.face_model is None:
            return None, 0.0

        try:
            # Resize for InsightFace
            face_resized = cv2.resize(face_crop, (112, 112))

            # Get face embeddings
            faces = self.face_model.get(face_resized)

            if faces and len(faces) > 0 and faces[0].embedding is not None:
                return faces[0].embedding, confidence

        except Exception as e:
            logger.debug(f"Face feature extraction failed: {e}")

        return None, 0.0


class FeatureExtractor:
    """Combined feature extractor for body, face, and pose."""

    def __init__(self, config: TrackerConfig, use_deep_features: bool = True):
        self.config = config
        self.pose_extractor = PoseDescriptorExtractor(config)
        self.body_extractor = BodyFeatureExtractor(config, use_deep_features)
        self.face_extractor = FaceFeatureExtractor(config)

    def extract_all(self, image: np.ndarray, detection: Detection) -> Detection:
        """Extract all features for a detection and update it in place."""
        # Pose descriptor
        detection.pose_descriptor = self.pose_extractor.extract(detection.keypoints)

        # Body embedding
        detection.body_embedding = self.body_extractor.extract(image, detection.bbox)

        # Face embedding
        face_emb, face_conf = self.face_extractor.extract(image, detection.keypoints)
        detection.face_embedding = face_emb
        detection.face_confidence = face_conf

        return detection


# =============================================================================
# SHORT-TERM TRACKER
# =============================================================================

class ShortTermTracker:
    """
    OC-SORT-style observation-centric tracker with appearance features.

    Key differences from DeepSORT:
    - Uses observation-centric approach (actual observations vs predictions)
    - Better handles non-linear motion through lower reliance on Kalman filter
    - Virtual trajectory for temporarily lost tracks
    """

    def __init__(self, config: TrackerConfig):
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        self.next_track_id = 0
        self.finished_tracklets: List[Tracklet] = []

    def update(self, frame_id: int, detections: List[Detection]) -> List[TrackState]:
        """
        Update tracker with new frame detections.

        Returns:
            List of active confirmed tracks
        """
        # Predict positions for existing tracks
        for track in self.tracks.values():
            track.age += 1

        # Match detections to tracks
        matched, unmatched_tracks, unmatched_dets = self._associate(detections)

        # Update matched tracks
        for track_id, det_idx in matched:
            self._update_track(track_id, detections[det_idx])

        # Handle unmatched tracks
        for track_id in unmatched_tracks:
            track = self.tracks[track_id]
            if track.age > self.config.max_age:
                self._finish_track(track_id)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx])

        # Return confirmed tracks
        return [t for t in self.tracks.values() if t.state == "confirmed"]

    def _associate(self, detections: List[Detection]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.

        Returns:
            (matched_pairs, unmatched_track_ids, unmatched_detection_indices)
        """
        if not self.tracks or not detections:
            return [], list(self.tracks.keys()), list(range(len(detections)))

        track_ids = list(self.tracks.keys())
        cost_matrix = np.full((len(track_ids), len(detections)), np.inf)

        for i, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            predicted_bbox = track.predict_bbox()
            track_body_emb = track.get_mean_body_embedding()
            track_face_emb = track.get_mean_face_embedding()

            for j, det in enumerate(detections):
                # Motion cost (IoU-based)
                iou = compute_iou(predicted_bbox, det.bbox)
                motion_cost = 1 - iou

                # Appearance cost
                appearance_cost = 1.0

                if det.body_embedding is not None and track_body_emb is not None:
                    body_sim = cosine_similarity(track_body_emb, det.body_embedding)

                    if det.face_embedding is not None and track_face_emb is not None:
                        face_sim = cosine_similarity(track_face_emb, det.face_embedding)
                        appearance_sim = (
                            self.config.face_feature_weight * face_sim +
                            self.config.body_feature_weight * body_sim
                        )
                    else:
                        appearance_sim = body_sim

                    appearance_cost = 1 - appearance_sim

                # Combined cost
                cost = (
                    (1 - self.config.appearance_weight) * motion_cost +
                    self.config.appearance_weight * appearance_cost
                )

                # Gate by IoU threshold
                if iou > self.config.iou_threshold or motion_cost < 0.8:
                    cost_matrix[i, j] = cost

        # Hungarian matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched = []
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1.0:  # Valid match
                track_id = track_ids[i]
                matched.append((track_id, j))
                unmatched_tracks.discard(track_id)
                unmatched_dets.discard(j)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def _update_track(self, track_id: int, detection: Detection):
        """Update track with new detection."""
        track = self.tracks[track_id]
        track.detections.append(detection)
        track.last_detection = detection
        track.age = 0
        track.hits += 1

        # Update feature buffers
        if detection.body_embedding is not None:
            track.body_embeddings.append(detection.body_embedding)
            # Keep last N embeddings for memory efficiency
            if len(track.body_embeddings) > 30:
                track.body_embeddings = track.body_embeddings[-30:]

        if detection.face_embedding is not None:
            track.face_embeddings.append(detection.face_embedding)
            if len(track.face_embeddings) > 10:
                track.face_embeddings = track.face_embeddings[-10:]

        # Promote to confirmed if enough hits
        if track.state == "tentative" and track.hits >= self.config.min_hits:
            track.state = "confirmed"

    def _create_track(self, detection: Detection):
        """Create new track from detection."""
        track_id = self.next_track_id
        self.next_track_id += 1

        body_embs = [detection.body_embedding] if detection.body_embedding is not None else []
        face_embs = [detection.face_embedding] if detection.face_embedding is not None else []

        track = TrackState(
            track_id=track_id,
            detections=[detection],
            last_detection=detection,
            age=0,
            hits=1,
            state="tentative",
            body_embeddings=body_embs,
            face_embeddings=face_embs
        )

        self.tracks[track_id] = track

    def _finish_track(self, track_id: int):
        """Finish track and convert to tracklet."""
        track = self.tracks[track_id]

        # Only keep tracks with enough detections
        if len(track.detections) >= self.config.min_tracklet_length:
            tracklet = Tracklet(
                tracklet_id=track_id,
                start_frame=track.start_frame,
                end_frame=track.end_frame,
                detections=track.detections
            )
            self.finished_tracklets.append(tracklet)

        del self.tracks[track_id]

    def finalize(self) -> List[Tracklet]:
        """Finalize all remaining tracks and return all tracklets."""
        # Finish all remaining tracks
        for track_id in list(self.tracks.keys()):
            self._finish_track(track_id)

        return self.finished_tracklets


# =============================================================================
# GLOBAL TRACKLET ASSOCIATION
# =============================================================================

class TrackletAssociator:
    """
    Associate tracklets into consistent person identities using hierarchical clustering.
    """

    def __init__(self, config: TrackerConfig):
        self.config = config

    def compute_affinity(self, t1: Tracklet, t2: Tracklet) -> float:
        """Compute affinity between two tracklets."""
        # 1. Temporal compatibility - tracklets cannot overlap significantly
        overlap = temporal_overlap(t1.start_frame, t1.end_frame, t2.start_frame, t2.end_frame)
        if overlap > self.config.temporal_overlap_threshold:
            return 0.0

        # 2. Spatial plausibility
        spatial_score = self._compute_spatial_score(t1, t2)

        # 3. Appearance similarity
        appearance_score = self._compute_appearance_score(t1, t2)

        # 4. Pose consistency
        pose_score = self._compute_pose_score(t1, t2)

        # Combined affinity
        affinity = (
            self.config.spatial_affinity_weight * spatial_score +
            self.config.appearance_affinity_weight * appearance_score +
            self.config.pose_affinity_weight * pose_score
        )

        return affinity

    def _compute_spatial_score(self, t1: Tracklet, t2: Tracklet) -> float:
        """Compute spatial plausibility score."""
        # Determine which tracklet comes first
        if t1.end_frame < t2.start_frame:
            first, second = t1, t2
        elif t2.end_frame < t1.start_frame:
            first, second = t2, t1
        else:
            return 0.5  # Overlapping, unclear

        gap_frames = second.start_frame - first.end_frame

        # Get positions at boundary
        pos1 = first.detections[-1].body_center
        pos2 = second.detections[0].body_center

        if pos1 is None or pos2 is None:
            return 0.5

        spatial_dist = np.linalg.norm(pos1 - pos2)
        max_plausible_dist = min(gap_frames * self.config.max_spatial_velocity, 500)

        if spatial_dist > max_plausible_dist:
            return 0.0

        return 1 - (spatial_dist / max_plausible_dist)

    def _compute_appearance_score(self, t1: Tracklet, t2: Tracklet) -> float:
        """Compute appearance similarity score."""
        # Body similarity
        body_sim = 0.0
        if t1.mean_body_embedding is not None and t2.mean_body_embedding is not None:
            body_sim = cosine_similarity(t1.mean_body_embedding, t2.mean_body_embedding)

        # Face similarity
        face_sim = 0.0
        if t1.mean_face_embedding is not None and t2.mean_face_embedding is not None:
            face_sim = cosine_similarity(t1.mean_face_embedding, t2.mean_face_embedding)
            return self.config.face_feature_weight * face_sim + self.config.body_feature_weight * body_sim

        return body_sim

    def _compute_pose_score(self, t1: Tracklet, t2: Tracklet) -> float:
        """Compute pose descriptor similarity score."""
        if t1.median_pose_descriptor is None or t2.median_pose_descriptor is None:
            return 0.5  # Neutral score if no pose info

        # Truncate to same length
        min_len = min(len(t1.median_pose_descriptor), len(t2.median_pose_descriptor))
        p1 = t1.median_pose_descriptor[:min_len]
        p2 = t2.median_pose_descriptor[:min_len]

        return cosine_similarity(p1, p2)

    def associate(
        self,
        tracklets: List[Tracklet],
        num_persons: Optional[int] = None
    ) -> Dict[int, List[int]]:
        """
        Associate tracklets into person identities.

        Args:
            tracklets: List of tracklets to associate
            num_persons: Optional known number of persons

        Returns:
            Dict mapping person_id to list of tracklet indices
        """
        if len(tracklets) == 0:
            return {}

        if len(tracklets) == 1:
            return {0: [0]}

        logger.info(f"Associating {len(tracklets)} tracklets...")

        # Build affinity matrix
        n = len(tracklets)
        affinity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                aff = self.compute_affinity(tracklets[i], tracklets[j])
                affinity_matrix[i, j] = aff
                affinity_matrix[j, i] = aff

        # Convert to distance matrix
        distance_matrix = 1 - affinity_matrix
        np.fill_diagonal(distance_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_dist, method="average")

        # Cut dendrogram
        if num_persons is not None:
            clusters = fcluster(linkage_matrix, num_persons, criterion="maxclust")
        else:
            clusters = fcluster(linkage_matrix, 1 - self.config.affinity_threshold, criterion="distance")

        # Build result
        result: Dict[int, List[int]] = {}
        for tracklet_idx, cluster_id in enumerate(clusters):
            result.setdefault(int(cluster_id) - 1, []).append(tracklet_idx)

        # Resolve temporal conflicts within each cluster
        result = self._resolve_temporal_conflicts(tracklets, result, affinity_matrix)

        logger.info(f"Associated into {len(result)} person identities")

        return result

    def _resolve_temporal_conflicts(
        self,
        tracklets: List[Tracklet],
        assignments: Dict[int, List[int]],
        affinity_matrix: np.ndarray
    ) -> Dict[int, List[int]]:
        """Resolve temporal conflicts within each person assignment."""
        resolved = {}
        removed_tracklets = []

        for person_id, tracklet_indices in assignments.items():
            if len(tracklet_indices) <= 1:
                resolved[person_id] = tracklet_indices
                continue

            # Sort by start time
            sorted_indices = sorted(tracklet_indices, key=lambda i: tracklets[i].start_frame)

            # Greedily select non-overlapping tracklets
            selected = [sorted_indices[0]]

            for idx in sorted_indices[1:]:
                t_new = tracklets[idx]
                conflict = False

                for sel_idx in selected:
                    t_sel = tracklets[sel_idx]
                    overlap = temporal_overlap(
                        t_new.start_frame, t_new.end_frame,
                        t_sel.start_frame, t_sel.end_frame
                    )
                    if overlap > 0.1:
                        conflict = True
                        # Keep the one with higher affinity to the cluster
                        aff_new = np.mean([affinity_matrix[idx, i] for i in selected if i != idx])
                        aff_sel = np.mean([affinity_matrix[sel_idx, i] for i in selected if i != sel_idx])

                        if aff_new > aff_sel:
                            selected.remove(sel_idx)
                            selected.append(idx)
                            removed_tracklets.append(sel_idx)
                        else:
                            removed_tracklets.append(idx)
                        break

                if not conflict:
                    selected.append(idx)

            resolved[person_id] = selected

        # Reassign removed tracklets to best matching cluster or create new
        for removed_idx in removed_tracklets:
            best_person = None
            best_aff = -1

            for person_id, indices in resolved.items():
                # Check no temporal conflict
                t_removed = tracklets[removed_idx]
                has_conflict = False

                for idx in indices:
                    overlap = temporal_overlap(
                        t_removed.start_frame, t_removed.end_frame,
                        tracklets[idx].start_frame, tracklets[idx].end_frame
                    )
                    if overlap > 0.1:
                        has_conflict = True
                        break

                if not has_conflict:
                    aff = np.mean([affinity_matrix[removed_idx, i] for i in indices])
                    if aff > best_aff:
                        best_aff = aff
                        best_person = person_id

            if best_person is not None and best_aff > self.config.affinity_threshold:
                resolved[best_person].append(removed_idx)
            else:
                # Create new person
                new_person_id = max(resolved.keys()) + 1 if resolved else 0
                resolved[new_person_id] = [removed_idx]

        return resolved


# =============================================================================
# OUTPUT GENERATION
# =============================================================================

class PoseSequenceAssembler:
    """Assemble per-dancer pose sequences from tracklet assignments."""

    def __init__(self, config: TrackerConfig):
        self.config = config

    def assemble(
        self,
        person_assignments: Dict[int, List[int]],
        tracklets: List[Tracklet],
        total_frames: int
    ) -> List[DancerPoseSequence]:
        """
        Create complete pose sequences for each identified person.

        Args:
            person_assignments: Dict mapping person_id to tracklet indices
            tracklets: List of all tracklets
            total_frames: Total number of frames in video

        Returns:
            List of DancerPoseSequence objects
        """
        sequences = []

        for person_id, tracklet_indices in sorted(person_assignments.items()):
            # Initialize arrays
            keypoints = np.zeros((total_frames, self.config.num_keypoints, 3), dtype=np.float32)
            visibility = np.zeros(total_frames, dtype=bool)
            confidence = np.zeros(total_frames, dtype=np.float32)
            bboxes = np.zeros((total_frames, 4), dtype=np.float32)
            interpolated = np.zeros(total_frames, dtype=bool)

            # Fill from tracklets
            for tid in tracklet_indices:
                tracklet = tracklets[tid]
                for det in tracklet.detections:
                    frame = det.frame_id
                    if frame >= total_frames:
                        continue

                    det_conf = det.keypoints[:, 2].mean()

                    if not visibility[frame] or det_conf > confidence[frame]:
                        keypoints[frame] = det.keypoints
                        visibility[frame] = True
                        confidence[frame] = det_conf
                        bboxes[frame] = det.bbox

            # Optionally interpolate gaps
            if self.config.interpolate_gaps:
                keypoints, visibility, confidence, interpolated = self._interpolate_gaps(
                    keypoints, visibility, confidence, interpolated
                )

            sequence = DancerPoseSequence(
                person_id=person_id,
                name=None,
                total_frames=total_frames,
                keypoints=keypoints,
                visibility=visibility,
                confidence=confidence,
                bboxes=bboxes,
                interpolated=interpolated
            )

            sequences.append(sequence)

        return sequences

    def _interpolate_gaps(
        self,
        keypoints: np.ndarray,
        visibility: np.ndarray,
        confidence: np.ndarray,
        interpolated: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Interpolate keypoints for short gaps."""
        i = 0
        while i < len(visibility):
            if not visibility[i]:
                # Found start of gap
                gap_start = i
                while i < len(visibility) and not visibility[i]:
                    i += 1
                gap_end = i
                gap_length = gap_end - gap_start

                # Check if gap is short enough and bounded
                if gap_length <= self.config.max_interpolation_gap:
                    if gap_start > 0 and gap_end < len(visibility):
                        if visibility[gap_start - 1] and visibility[gap_end]:
                            # Interpolate
                            start_kpts = keypoints[gap_start - 1]
                            end_kpts = keypoints[gap_end]

                            for j in range(gap_start, gap_end):
                                alpha = (j - gap_start + 1) / (gap_length + 1)
                                keypoints[j] = (1 - alpha) * start_kpts + alpha * end_kpts
                                # Reduce confidence for interpolated frames
                                keypoints[j, :, 2] *= self.config.interpolation_confidence_factor
                                confidence[j] = keypoints[j, :, 2].mean()
                                visibility[j] = True
                                interpolated[j] = True
            else:
                i += 1

        return keypoints, visibility, confidence, interpolated


class OutputWriter:
    """Write tracking results to various formats."""

    def __init__(self, config: TrackerConfig, output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(
        self,
        sequences: List[DancerPoseSequence],
        video_info: Dict[str, Any],
        output_name: str = "tracking_results.json"
    ):
        """Write results to JSON format."""
        output_path = self.output_dir / output_name

        result = {
            "video_info": video_info,
            "num_dancers": len(sequences),
            "dancers": []
        }

        for seq in sequences:
            dancer_data = {
                "person_id": seq.person_id,
                "name": seq.name,
                "statistics": seq.get_statistics(),
                "visible_segments": seq.get_visible_segments(),
                "pose_sequence": []
            }

            # Add per-frame data (only for visible frames to save space)
            for frame_id in range(seq.total_frames):
                if seq.visibility[frame_id]:
                    frame_data = {
                        "frame_id": frame_id,
                        "visible": True,
                        "interpolated": bool(seq.interpolated[frame_id]),
                        "bbox": seq.bboxes[frame_id].tolist(),
                        "keypoints": seq.keypoints[frame_id].tolist(),
                        "mean_confidence": float(seq.confidence[frame_id])
                    }
                    dancer_data["pose_sequence"].append(frame_data)

            result["dancers"].append(dancer_data)

        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(f"Saved JSON results to {output_path}")

    def write_numpy(
        self,
        sequences: List[DancerPoseSequence],
        output_name: str = "tracking_results.npz"
    ):
        """Write results to compressed NumPy format."""
        output_path = self.output_dir / output_name

        np.savez_compressed(
            output_path,
            person_ids=np.array([s.person_id for s in sequences]),
            names=np.array([s.name or "" for s in sequences]),
            keypoints=np.stack([s.keypoints for s in sequences]),  # [N, T, K, 3]
            visibility=np.stack([s.visibility for s in sequences]),  # [N, T]
            confidence=np.stack([s.confidence for s in sequences]),  # [N, T]
            bboxes=np.stack([s.bboxes for s in sequences]),  # [N, T, 4]
            interpolated=np.stack([s.interpolated for s in sequences])  # [N, T]
        )

        logger.info(f"Saved NumPy results to {output_path}")

    def write_per_dancer_json(self, sequences: List[DancerPoseSequence]):
        """Write individual JSON files for each dancer."""
        for seq in sequences:
            output_path = self.output_dir / f"dancer_{seq.person_id}_poses.json"

            dancer_data = {
                "person_id": seq.person_id,
                "name": seq.name,
                "total_frames": seq.total_frames,
                "statistics": seq.get_statistics(),
                "frames": []
            }

            for frame_id in range(seq.total_frames):
                frame_data = {
                    "frame_id": frame_id,
                    "visible": bool(seq.visibility[frame_id]),
                    "interpolated": bool(seq.interpolated[frame_id]),
                    "confidence": float(seq.confidence[frame_id]),
                    "bbox": seq.bboxes[frame_id].tolist() if seq.visibility[frame_id] else [0, 0, 0, 0],
                    "keypoints": seq.keypoints[frame_id].tolist()
                }
                dancer_data["frames"].append(frame_data)

            with open(output_path, "w") as f:
                json.dump(dancer_data, f)

            logger.info(f"Saved dancer {seq.person_id} poses to {output_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DancePoseTracker:
    """
    Main pipeline for dance pose tracking.

    Orchestrates:
    1. AlphaPose parsing
    2. Feature extraction
    3. Short-term tracking
    4. Global tracklet association
    5. Output generation
    """

    def __init__(
        self,
        config: TrackerConfig,
        use_deep_features: bool = True
    ):
        self.config = config

        # Initialize components
        self.parser = AlphaPoseParser(config)
        self.feature_extractor = FeatureExtractor(config, use_deep_features)
        self.tracker = ShortTermTracker(config)
        self.associator = TrackletAssociator(config)
        self.assembler = PoseSequenceAssembler(config)

    def run(
        self,
        alphapose_json: str,
        video_path: Optional[str] = None,
        output_dir: str = "./output",
        num_dancers: Optional[int] = None,
        extract_features: bool = True
    ) -> List[DancerPoseSequence]:
        """
        Run the complete tracking pipeline.

        Args:
            alphapose_json: Path to AlphaPose JSON output
            video_path: Path to source video (required for feature extraction)
            output_dir: Directory for output files
            num_dancers: Optional known number of dancers
            extract_features: Whether to extract deep features (requires video)

        Returns:
            List of DancerPoseSequence for each identified dancer
        """
        # Step 1: Parse AlphaPose results
        logger.info("Step 1: Parsing AlphaPose results...")
        frame_detections = self.parser.parse(alphapose_json)

        if not frame_detections:
            logger.error("No detections found in AlphaPose output")
            return []

        # Determine frame range
        all_frames = sorted(frame_detections.keys())
        min_frame = min(all_frames)
        max_frame = max(all_frames)
        total_frames = max_frame + 1

        logger.info(f"Frame range: {min_frame} to {max_frame} ({total_frames} total)")

        # Step 2: Extract features (if video provided)
        if extract_features and video_path:
            logger.info("Step 2: Extracting features from video...")
            self._extract_features(video_path, frame_detections)
        else:
            logger.info("Step 2: Skipping feature extraction (no video provided)")
            # Use pose descriptors only
            for frame_id, detections in frame_detections.items():
                for det in detections:
                    det.pose_descriptor = self.feature_extractor.pose_extractor.extract(det.keypoints)

        # Step 3: Short-term tracking
        logger.info("Step 3: Running short-term tracking...")
        for frame_id in range(total_frames):
            detections = frame_detections.get(frame_id, [])
            self.tracker.update(frame_id, detections)

        tracklets = self.tracker.finalize()
        logger.info(f"Generated {len(tracklets)} tracklets")

        if not tracklets:
            logger.warning("No tracklets generated")
            return []

        # Step 4: Global tracklet association
        logger.info("Step 4: Running global tracklet association...")
        person_assignments = self.associator.associate(tracklets, num_dancers)

        # Step 5: Assemble pose sequences
        logger.info("Step 5: Assembling pose sequences...")
        sequences = self.assembler.assemble(person_assignments, tracklets, total_frames)

        # Step 6: Write outputs
        logger.info("Step 6: Writing outputs...")
        writer = OutputWriter(self.config, output_dir)

        video_info = {
            "path": video_path or alphapose_json,
            "total_frames": total_frames,
            "frame_range": [min_frame, max_frame]
        }

        writer.write_json(sequences, video_info)
        writer.write_numpy(sequences)
        writer.write_per_dancer_json(sequences)

        # Print summary
        self._print_summary(sequences)

        return sequences

    def _extract_features(
        self,
        video_path: str,
        frame_detections: Dict[int, List[Detection]]
    ):
        """Extract features from video frames."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Processing {total_frames_video} video frames for feature extraction...")

        frame_idx = 0
        processed = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in frame_detections:
                for det in frame_detections[frame_idx]:
                    self.feature_extractor.extract_all(frame, det)
                processed += 1

            frame_idx += 1

            if frame_idx % 500 == 0:
                logger.info(f"  Processed {frame_idx}/{total_frames_video} frames...")

        cap.release()
        logger.info(f"Feature extraction complete. Processed {processed} frames with detections.")

    def _print_summary(self, sequences: List[DancerPoseSequence]):
        """Print tracking summary."""
        logger.info("\n" + "=" * 60)
        logger.info("TRACKING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Number of dancers identified: {len(sequences)}")

        for seq in sequences:
            stats = seq.get_statistics()
            logger.info(f"\nDancer {seq.person_id}:")
            logger.info(f"  Visible frames: {stats['visible_frames']}/{stats['total_frames']} "
                       f"({stats['visibility_ratio']:.1%})")
            logger.info(f"  Interpolated frames: {stats['interpolated_frames']}")
            logger.info(f"  Number of segments: {stats['num_segments']}")
            logger.info(f"  Mean confidence: {stats['mean_confidence']:.3f}")

        logger.info("=" * 60 + "\n")


# =============================================================================
# VISUALIZATION (Optional)
# =============================================================================

class TrackingVisualizer:
    """Generate visualization videos with tracking overlays."""

    # Halpe-136 skeleton connections for body (simplified)
    SKELETON_CONNECTIONS = [
        # Body
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
    ]

    # Color palette for different persons
    COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0), (128, 128, 0),
        (0, 128, 128), (128, 0, 0), (0, 128, 0)
    ]

    def __init__(self, config: TrackerConfig):
        self.config = config

    def visualize(
        self,
        video_path: str,
        sequences: List[DancerPoseSequence],
        output_path: str,
        fps: Optional[float] = None
    ):
        """Generate visualization video."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Draw overlays for each dancer
            for seq in sequences:
                if frame_idx < seq.total_frames and seq.visibility[frame_idx]:
                    color = self.COLORS[seq.person_id % len(self.COLORS)]

                    # Draw bounding box
                    bbox = seq.bboxes[frame_idx].astype(int)
                    thickness = 1 if seq.interpolated[frame_idx] else 2
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

                    # Draw skeleton
                    self._draw_skeleton(frame, seq.keypoints[frame_idx], color)

                    # Draw label
                    label = f"ID:{seq.person_id}"
                    if seq.interpolated[frame_idx]:
                        label += " (interp)"
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            writer.write(frame)
            frame_idx += 1

        cap.release()
        writer.release()

        logger.info(f"Visualization saved to {output_path}")

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color: Tuple[int, int, int]):
        """Draw skeleton on frame."""
        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints[:26]):  # Body keypoints only
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1)

        # Draw connections
        for i, j in self.SKELETON_CONNECTIONS:
            if i < keypoints.shape[0] and j < keypoints.shape[0]:
                if keypoints[i, 2] > 0.3 and keypoints[j, 2] > 0.3:
                    pt1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                    pt2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                    cv2.line(frame, pt1, pt2, color, 2)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Dance Pose Tracker - Track multiple dancers using AlphaPose outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with AlphaPose JSON
  python dance_pose_tracker.py --alphapose-json results.json --output-dir ./output

  # With video for feature extraction
  python dance_pose_tracker.py --alphapose-json results.json --video dance.mp4 --output-dir ./output

  # Specify number of dancers
  python dance_pose_tracker.py --alphapose-json results.json --video dance.mp4 --num-dancers 8

  # Generate visualization video
  python dance_pose_tracker.py --alphapose-json results.json --video dance.mp4 --visualize
        """
    )

    # Required arguments
    parser.add_argument(
        "--alphapose-json",
        required=True,
        help="Path to AlphaPose JSON output file"
    )

    # Optional arguments
    parser.add_argument(
        "--video",
        help="Path to source video (required for feature extraction and visualization)"
    )

    parser.add_argument(
        "--output-dir",
        default="./tracking_output",
        help="Output directory for results (default: ./tracking_output)"
    )

    parser.add_argument(
        "--num-dancers",
        type=int,
        help="Known number of dancers (optional, improves clustering)"
    )

    parser.add_argument(
        "--no-deep-features",
        action="store_true",
        help="Disable deep feature extraction (uses histogram features instead)"
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization video with tracking overlays"
    )

    parser.add_argument(
        "--vis-output",
        help="Path for visualization video (default: output_dir/tracking_visualization.mp4)"
    )

    # Tracking parameters
    parser.add_argument(
        "--max-age",
        type=int,
        default=60,
        help="Max frames to keep lost tracks (default: 60)"
    )

    parser.add_argument(
        "--appearance-weight",
        type=float,
        default=0.6,
        help="Weight for appearance vs motion in tracking (default: 0.6)"
    )

    parser.add_argument(
        "--affinity-threshold",
        type=float,
        default=0.4,
        help="Min affinity to merge tracklets (default: 0.4)"
    )

    parser.add_argument(
        "--no-interpolation",
        action="store_true",
        help="Disable gap interpolation"
    )

    parser.add_argument(
        "--max-interp-gap",
        type=int,
        default=15,
        help="Max frames to interpolate (default: 15)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create config
    config = TrackerConfig(
        max_age=args.max_age,
        appearance_weight=args.appearance_weight,
        affinity_threshold=args.affinity_threshold,
        interpolate_gaps=not args.no_interpolation,
        max_interpolation_gap=args.max_interp_gap
    )

    # Initialize pipeline
    tracker = DancePoseTracker(
        config=config,
        use_deep_features=not args.no_deep_features
    )

    # Run tracking
    sequences = tracker.run(
        alphapose_json=args.alphapose_json,
        video_path=args.video,
        output_dir=args.output_dir,
        num_dancers=args.num_dancers,
        extract_features=args.video is not None
    )

    # Generate visualization if requested
    if args.visualize and args.video:
        vis_output = args.vis_output or os.path.join(args.output_dir, "tracking_visualization.mp4")
        visualizer = TrackingVisualizer(config)
        visualizer.visualize(args.video, sequences, vis_output)
    elif args.visualize and not args.video:
        logger.warning("Cannot generate visualization without video path")

    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
