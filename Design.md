# Dance Video Multi-Person Tracking & Pose Association System

## Design Document v1.0

---

## 1. Executive Summary

This document presents a comprehensive design for tracking multiple dancers in choreographed dance videos where complex movements, frequent occlusions, and dancers entering/exiting the frame create significant challenges for traditional tracking approaches. The goal is to produce consistent per-dancer pose sequences from AlphaPose Halpe-136 keypoint detections, with appropriate confidence markers for frames where individuals are not visible.

### 1.1 Problem Statement

Given:
- A 720p dance video with N dancers performing choreographed movements
- AlphaPose Halpe-136 keypoint detections at the frame level (unassociated)
- Complex non-linear motion patterns by design
- Frequent occlusions (dancers passing in front of each other)
- Dancers entering/exiting frame boundaries
- Dancers in back rows becoming visible/invisible
- Motion blur and grain during fast movements

Produce:
- Consistent person IDs across all frames for each dancer
- Per-dancer pose sequences spanning the entire video
- Frames where a dancer is not visible marked with 0 confidence
- Robust re-identification when dancers reappear after occlusion or exit/re-entry

### 1.2 Why Current Approach Falls Short

The existing `face_indexer.py` and `human_tracker.py` implementation has fundamental limitations:

| Limitation | Impact |
|------------|--------|
| Face-centric identification | Fails when dancers turn around, look down, or faces are occluded by hair/motion blur |
| DeepSORT Kalman filter | Assumes linear motion; completely fails for spins, jumps, direction changes |
| Single-frame matching | No temporal context; cannot leverage motion history or future frames |
| No tracklet stitching | Once a track is lost, it starts a new ID—catastrophic for our use case |
| Greedy matching | No global optimization; early errors propagate |
| No body appearance features | Misses the most persistent visual cue (clothing) |

---

## 2. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BATCH PROCESSING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐    ┌─────────────────┐    ┌──────────────────┐                │
│  │   Stage 1   │    │     Stage 2     │    │     Stage 3      │                │
│  │  Detection  │───▶│  Pose Estimation│───▶│  Feature Extraction│               │
│  │   (YOLO)    │    │   (AlphaPose)   │    │  (Multi-Modal)    │                │
│  └─────────────┘    └─────────────────┘    └──────────────────┘                │
│         │                    │                      │                           │
│         ▼                    ▼                      ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Per-Frame Detection Store                           │   │
│  │  [frame_id, bbox, keypoints_136, face_emb, body_emb, pose_emb, ...]    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        Stage 4: Short-Term Tracking                      │   │
│  │              (OC-SORT / BoT-SORT with appearance features)              │   │
│  │                     → Produces Initial Tracklets                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                     Stage 5: Tracklet Analysis                           │   │
│  │           - Confidence scoring per tracklet                              │   │
│  │           - Aggregate appearance features                                 │   │
│  │           - Identify anchor tracklets (high confidence)                  │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                  Stage 6: Global Tracklet Association                    │   │
│  │         - Graph-based clustering / Hierarchical merging                  │   │
│  │         - Multi-cue affinity (appearance + spatiotemporal)              │   │
│  │         - Hungarian algorithm for global assignment                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                    Stage 7: Identity Refinement                          │   │
│  │           - Optional: Face gallery matching                              │   │
│  │           - Optional: Manual anchor specification                        │   │
│  │           - Consistency enforcement across video                         │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                      │                                          │
│                                      ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                      Stage 8: Output Generation                          │   │
│  │         - Per-dancer pose sequences (all frames)                         │   │
│  │         - Interpolation for short gaps (optional)                        │   │
│  │         - Zero-confidence for invisible frames                           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Stage Design

### 3.1 Stage 1: Person Detection

**Objective**: Detect all persons in every frame with high recall.

**Recommended Approach**: YOLOv8x or RT-DETR

**Why**:
- YOLOv8x offers excellent accuracy/speed tradeoff for batch processing
- RT-DETR (Real-Time DETR) provides end-to-end detection without NMS, reducing missed detections in crowded scenes

**Configuration**:
```yaml
detector:
  model: "yolov8x"  # or "RT-DETR-X"
  confidence_threshold: 0.25  # Low threshold for high recall
  nms_iou_threshold: 0.45
  classes: [0]  # person class only
  input_size: 1280  # Higher resolution for 720p video
  augment: true  # TTA for better detection at edges
```

**Handling Edge Cases**:
- **Partial bodies at frame edges**: Enable `agnostic_nms` and lower confidence threshold
- **Overlapping dancers**: Use lower NMS threshold; let tracking sort out associations
- **Motion blur**: Consider running detection on multiple temporal neighbors and aggregating

**Output Schema**:
```python
@dataclass
class Detection:
    frame_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    detection_id: int  # Unique within frame
```

---

### 3.2 Stage 2: Pose Estimation

**Objective**: Extract Halpe-136 keypoints for each detected person.

**Approach**: AlphaPose with Halpe Full-Body model

**Why Halpe-136**:
- 136 keypoints cover face (68), body (26), hands (42), providing rich pose representation
- Face keypoints enable face-based identity matching without separate face detector
- Hand keypoints help disambiguate dancers during close interactions

**Configuration**:
```yaml
pose_estimator:
  model: "halpe136"
  detector: "yolov8x"  # Use same detector for consistency
  pose_batch_size: 32
  pose_nms: true
  min_keypoint_confidence: 0.1  # Keep low-confidence keypoints

  # Critical for dance videos
  tracking_mode: "pose"  # Use pose-flow for better association
  pose_flow:
    enabled: true
    neighbor_frames: 2
```

**Multi-Person Pose Matching**:

When AlphaPose runs in detection mode, we need to match pose outputs back to our detections. Use IoU matching between detection bboxes and pose bboxes:

```python
def match_poses_to_detections(
    detections: List[Detection],
    poses: List[PoseResult],
    iou_threshold: float = 0.5
) -> Dict[int, PoseResult]:
    """
    Match AlphaPose outputs to detector bboxes.
    Returns mapping: detection_id -> PoseResult
    """
    cost_matrix = compute_iou_matrix(detections, poses)
    row_ind, col_ind = linear_sum_assignment(-cost_matrix)

    matches = {}
    for det_idx, pose_idx in zip(row_ind, col_ind):
        if cost_matrix[det_idx, pose_idx] >= iou_threshold:
            matches[detections[det_idx].detection_id] = poses[pose_idx]

    return matches
```

**Output Schema**:
```python
@dataclass
class PoseDetection:
    frame_id: int
    detection_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # [136, 3] - x, y, confidence
    pose_score: float
```

---

### 3.3 Stage 3: Multi-Modal Feature Extraction

This is the **critical stage** that enables robust re-identification. We extract multiple complementary features:

#### 3.3.1 Body Appearance Features (Primary Re-ID Signal)

**Model**: OSNet-AIN or TransReID

**Why**:
- Clothing is the most persistent visual cue in dance videos
- OSNet-AIN handles scale variations well
- TransReID captures fine-grained appearance details via attention

**Implementation**:
```yaml
body_reid:
  model: "osnet_ain_x1_0"  # or "transreid_vit_base"
  input_size: [256, 128]  # Height x Width
  normalize: true
  feature_dim: 512

  # Crop settings
  crop_padding: 0.1  # Expand bbox by 10%
  aspect_ratio_correction: true
```

**Robustness Enhancements**:
- Extract features from multiple frames within a tracklet and aggregate (mean + attention pooling)
- Use horizontal flip augmentation at inference for more robust features
- Consider clothing segmentation to focus on distinctive regions

#### 3.3.2 Face Embeddings (When Available)

**Model**: InsightFace (ArcFace) - same as current implementation

**Improvements Over Current Approach**:
- Extract face region from Halpe-136 face keypoints (68 points) instead of running separate face detector
- Use face visibility score (average confidence of face keypoints) to weight face embeddings
- Only use face embeddings when visibility > 0.5

```python
def extract_face_from_pose(keypoints: np.ndarray) -> Optional[FaceRegion]:
    """
    Extract face bounding box from Halpe-136 keypoints.
    Face keypoints are indices 26-93 (68 points).
    """
    face_kpts = keypoints[26:94]  # 68 face keypoints
    visible = face_kpts[:, 2] > 0.3

    if visible.sum() < 20:  # Not enough visible face keypoints
        return None

    visible_kpts = face_kpts[visible, :2]
    x1, y1 = visible_kpts.min(axis=0)
    x2, y2 = visible_kpts.max(axis=0)

    # Expand for context
    w, h = x2 - x1, y2 - y1
    x1, y1 = x1 - 0.2 * w, y1 - 0.2 * h
    x2, y2 = x2 + 0.2 * w, y2 + 0.2 * h

    confidence = face_kpts[:, 2].mean()
    return FaceRegion(bbox=[x1, y1, x2, y2], confidence=confidence)
```

**Configuration**:
```yaml
face_reid:
  model: "buffalo_l"  # ArcFace
  det_size: 640
  min_face_confidence: 0.5
  embedding_dim: 512
```

#### 3.3.3 Pose-Based Features (Motion Signature)

**Novel Component**: Use pose structure as an identity cue

Dance poses create distinctive body configurations. We can use:

1. **Static Pose Descriptor**: Normalized joint angles and limb length ratios
2. **Pose Sequence Descriptor**: Short-term pose dynamics (1-2 seconds)

**Static Pose Features**:
```python
def compute_pose_descriptor(keypoints: np.ndarray) -> np.ndarray:
    """
    Compute identity-relevant pose features:
    - Limb length ratios (body proportions)
    - Normalized joint positions (relative to torso)
    """
    # Key body points
    neck = keypoints[0, :2]
    left_shoulder = keypoints[1, :2]
    right_shoulder = keypoints[2, :2]
    left_hip = keypoints[7, :2]
    right_hip = keypoints[8, :2]

    # Torso-normalized coordinates
    torso_center = (neck + (left_hip + right_hip) / 2) / 2
    torso_scale = np.linalg.norm(neck - (left_hip + right_hip) / 2)

    normalized_kpts = (keypoints[:, :2] - torso_center) / (torso_scale + 1e-6)

    # Limb ratios (relatively stable across poses)
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    torso_length = torso_scale

    ratios = np.array([
        shoulder_width / (torso_length + 1e-6),
        hip_width / (torso_length + 1e-6),
        shoulder_width / (hip_width + 1e-6),
    ])

    return np.concatenate([normalized_kpts.flatten(), ratios])
```

#### 3.3.4 Feature Fusion

Combine all features into a unified representation:

```python
@dataclass
class PersonFeatures:
    frame_id: int
    detection_id: int
    bbox: np.ndarray
    keypoints: np.ndarray  # [136, 3]

    # Multi-modal embeddings
    body_embedding: np.ndarray  # [512]
    face_embedding: Optional[np.ndarray]  # [512] or None
    face_confidence: float  # 0.0 if no face
    pose_descriptor: np.ndarray  # [~300]

    # Derived features
    body_center: np.ndarray  # [2]
    body_scale: float
```

---

### 3.4 Stage 4: Short-Term Tracking

**Objective**: Create initial tracklets by linking detections across consecutive frames.

**Recommended Approach**: OC-SORT with Appearance (Deep OC-SORT)

**Why OC-SORT over DeepSORT**:

| Aspect | DeepSORT | OC-SORT |
|--------|----------|---------|
| Motion Model | Kalman filter (linear) | Observation-Centric, handles non-linear better |
| Occlusion Handling | Loses track quickly | Virtual trajectory for occluded objects |
| Re-identification | Appearance only | Motion + Appearance with observation recovery |
| Dance Suitability | Poor | Significantly better |

**Deep OC-SORT Implementation**:

```yaml
short_term_tracker:
  type: "deep_ocsort"

  # Motion parameters
  det_thresh: 0.3
  max_age: 60  # ~2 seconds at 30fps - longer due to dance occlusions
  min_hits: 3
  iou_threshold: 0.3

  # OC-SORT specific
  delta_t: 3  # Observation recovery window
  inertia: 0.2  # Virtual trajectory weight (lower for non-linear motion)

  # Appearance parameters
  appearance_weight: 0.6  # Higher weight on appearance for dance
  motion_weight: 0.4
  appearance_threshold: 0.4

  # Feature configuration
  use_body_embedding: true
  use_face_embedding: true  # When available
  face_weight: 0.7  # Face is more discriminative when visible
  body_weight: 0.3
```

**Association Cost Matrix**:

```python
def compute_association_cost(
    tracks: List[Track],
    detections: List[PersonFeatures],
    alpha: float = 0.6  # appearance weight
) -> np.ndarray:
    """
    Compute cost matrix for Hungarian matching.

    Cost = (1 - alpha) * motion_cost + alpha * appearance_cost
    """
    n_tracks, n_dets = len(tracks), len(detections)
    cost_matrix = np.full((n_tracks, n_dets), np.inf)

    for i, track in enumerate(tracks):
        predicted_bbox = track.predict()  # OC-SORT virtual trajectory
        track_features = track.get_aggregated_features()

        for j, det in enumerate(detections):
            # Motion cost (IoU-based)
            iou = compute_iou(predicted_bbox, det.bbox)
            motion_cost = 1 - iou

            # Appearance cost (combined body + face)
            body_sim = cosine_similarity(track_features.body, det.body_embedding)

            if det.face_embedding is not None and track_features.face is not None:
                face_sim = cosine_similarity(track_features.face, det.face_embedding)
                appearance_sim = 0.7 * face_sim + 0.3 * body_sim
            else:
                appearance_sim = body_sim

            appearance_cost = 1 - appearance_sim

            # Combined cost
            cost_matrix[i, j] = (1 - alpha) * motion_cost + alpha * appearance_cost

    return cost_matrix
```

**Tracklet Output**:

```python
@dataclass
class Tracklet:
    tracklet_id: int
    start_frame: int
    end_frame: int
    detections: List[PersonFeatures]  # Ordered by frame

    # Aggregated identity features
    mean_body_embedding: np.ndarray
    face_embeddings: List[np.ndarray]  # All valid face embeddings
    mean_face_embedding: Optional[np.ndarray]
    pose_descriptors: List[np.ndarray]

    # Quality metrics
    detection_rate: float  # Fraction of frames with detection
    avg_confidence: float
    face_visibility_rate: float
```

---

### 3.5 Stage 5: Tracklet Analysis & Quality Scoring

**Objective**: Analyze tracklet quality and identify "anchor" tracklets for reliable identity seeding.

**Anchor Tracklet Criteria**:
- Long duration (>3 seconds)
- High detection rate (>0.8)
- Multiple high-confidence face observations
- Consistent appearance features (low intra-tracklet variance)

```python
@dataclass
class TrackletQuality:
    tracklet_id: int
    duration_frames: int
    detection_rate: float
    face_visibility_rate: float
    appearance_consistency: float  # 1 - (std/mean of pairwise similarities)
    avg_detection_confidence: float

    @property
    def anchor_score(self) -> float:
        """Higher score = more reliable identity anchor."""
        return (
            0.3 * min(self.duration_frames / 150, 1.0) +  # ~5 sec at 30fps
            0.2 * self.detection_rate +
            0.2 * self.face_visibility_rate +
            0.2 * self.appearance_consistency +
            0.1 * self.avg_detection_confidence
        )

    @property
    def is_anchor(self) -> bool:
        return self.anchor_score > 0.7
```

**Aggregated Tracklet Features**:

```python
def aggregate_tracklet_features(tracklet: Tracklet) -> TrackletFeatures:
    """
    Create robust identity representation by aggregating features.
    """
    # Body embeddings - attention-weighted mean
    body_embs = np.stack([d.body_embedding for d in tracklet.detections])
    body_confidences = np.array([d.bbox_confidence for d in tracklet.detections])
    weights = softmax(body_confidences / 0.1)  # Temperature-scaled
    mean_body = (body_embs * weights[:, None]).sum(axis=0)

    # Face embeddings - select highest confidence ones
    face_embs = [
        (d.face_embedding, d.face_confidence)
        for d in tracklet.detections
        if d.face_embedding is not None
    ]
    if face_embs:
        face_embs.sort(key=lambda x: x[1], reverse=True)
        top_faces = [f[0] for f in face_embs[:5]]  # Top 5
        mean_face = np.mean(top_faces, axis=0)
    else:
        mean_face = None

    # Pose features - median to reduce noise
    pose_descs = np.stack([d.pose_descriptor for d in tracklet.detections])
    median_pose = np.median(pose_descs, axis=0)

    return TrackletFeatures(
        body_embedding=mean_body,
        face_embedding=mean_face,
        pose_descriptor=median_pose,
        temporal_span=(tracklet.start_frame, tracklet.end_frame)
    )
```

---

### 3.6 Stage 6: Global Tracklet Association

This is the **most critical stage** for handling our specific challenges. We need to merge tracklets that belong to the same person across:
- Short occlusions (dancer temporarily hidden)
- Long absences (dancer exits frame and returns)
- Back-row visibility changes

**Approach**: Hierarchical Clustering with Multi-Cue Affinity

#### 3.6.1 Affinity Matrix Construction

```python
def compute_tracklet_affinity(t1: Tracklet, t2: Tracklet) -> float:
    """
    Compute affinity between two tracklets.
    Returns value in [0, 1] where 1 = same person.
    """
    f1 = aggregate_tracklet_features(t1)
    f2 = aggregate_tracklet_features(t2)

    # 1. Temporal compatibility
    # Tracklets cannot overlap significantly
    overlap = compute_temporal_overlap(
        (t1.start_frame, t1.end_frame),
        (t2.start_frame, t2.end_frame)
    )
    if overlap > 0.3:  # >30% overlap = different people
        return 0.0

    # 2. Spatial plausibility
    # Check if transition between end of t1 and start of t2 is plausible
    if t1.end_frame < t2.start_frame:
        gap_frames = t2.start_frame - t1.end_frame
        spatial_dist = np.linalg.norm(
            t1.detections[-1].body_center - t2.detections[0].body_center
        )
        max_plausible_dist = min(gap_frames * 50, 500)  # ~50 pixels/frame max speed
        if spatial_dist > max_plausible_dist:
            spatial_score = 0.0
        else:
            spatial_score = 1 - (spatial_dist / max_plausible_dist)
    else:
        spatial_score = 0.5  # Overlapping in time, unclear

    # 3. Appearance similarity
    body_sim = cosine_similarity(f1.body_embedding, f2.body_embedding)

    if f1.face_embedding is not None and f2.face_embedding is not None:
        face_sim = cosine_similarity(f1.face_embedding, f2.face_embedding)
        appearance_sim = 0.6 * face_sim + 0.4 * body_sim
    else:
        appearance_sim = body_sim

    # 4. Pose consistency (body proportions)
    pose_sim = cosine_similarity(f1.pose_descriptor, f2.pose_descriptor)

    # Combined affinity
    affinity = (
        0.1 * spatial_score +
        0.6 * appearance_sim +
        0.3 * pose_sim
    )

    return affinity
```

#### 3.6.2 Hierarchical Agglomerative Clustering

```python
def cluster_tracklets(
    tracklets: List[Tracklet],
    affinity_threshold: float = 0.5,
    num_persons: Optional[int] = None
) -> Dict[int, List[int]]:
    """
    Cluster tracklets into person identities.

    Returns: person_id -> [tracklet_ids]
    """
    n = len(tracklets)

    # Build affinity matrix
    affinity_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            aff = compute_tracklet_affinity(tracklets[i], tracklets[j])
            affinity_matrix[i, j] = aff
            affinity_matrix[j, i] = aff

    # Convert to distance matrix
    distance_matrix = 1 - affinity_matrix

    # Hierarchical clustering
    condensed_dist = squareform(distance_matrix)
    linkage_matrix = linkage(condensed_dist, method='average')

    # Cut dendrogram
    if num_persons is not None:
        clusters = fcluster(linkage_matrix, num_persons, criterion='maxclust')
    else:
        clusters = fcluster(linkage_matrix, 1 - affinity_threshold, criterion='distance')

    # Build result
    result = {}
    for tracklet_idx, cluster_id in enumerate(clusters):
        result.setdefault(cluster_id, []).append(tracklet_idx)

    return result
```

#### 3.6.3 Graph-Based Refinement (Alternative/Complementary)

For more robust association, use a graph-based approach:

```python
def graph_based_association(
    tracklets: List[Tracklet],
    affinity_matrix: np.ndarray,
    num_persons: int
) -> Dict[int, List[int]]:
    """
    Use spectral clustering or community detection for tracklet grouping.
    """
    import networkx as nx
    from sklearn.cluster import SpectralClustering

    # Build weighted graph
    G = nx.Graph()
    for i in range(len(tracklets)):
        G.add_node(i)

    for i in range(len(tracklets)):
        for j in range(i + 1, len(tracklets)):
            if affinity_matrix[i, j] > 0.3:  # Only add significant edges
                G.add_edge(i, j, weight=affinity_matrix[i, j])

    # Spectral clustering
    if nx.is_connected(G):
        clustering = SpectralClustering(
            n_clusters=num_persons,
            affinity='precomputed',
            assign_labels='kmeans'
        )
        labels = clustering.fit_predict(affinity_matrix)
    else:
        # Handle disconnected components separately
        labels = np.zeros(len(tracklets), dtype=int)
        current_label = 0
        for component in nx.connected_components(G):
            component = list(component)
            if len(component) == 1:
                labels[component[0]] = current_label
                current_label += 1
            else:
                sub_affinity = affinity_matrix[np.ix_(component, component)]
                sub_k = min(len(component), num_persons - current_label)
                clustering = SpectralClustering(
                    n_clusters=max(1, sub_k),
                    affinity='precomputed'
                )
                sub_labels = clustering.fit_predict(sub_affinity)
                for idx, node in enumerate(component):
                    labels[node] = current_label + sub_labels[idx]
                current_label = labels.max() + 1

    # Build result
    result = {}
    for tracklet_idx, person_id in enumerate(labels):
        result.setdefault(int(person_id), []).append(tracklet_idx)

    return result
```

#### 3.6.4 Temporal Conflict Resolution

When tracklets assigned to the same person overlap in time, resolve conflicts:

```python
def resolve_temporal_conflicts(
    person_tracklets: List[Tracklet],
    affinity_to_person: List[float]
) -> List[Tracklet]:
    """
    Resolve overlapping tracklets assigned to same person.
    Keep higher affinity tracklet, reassign others.
    """
    # Sort by start time
    sorted_tracklets = sorted(
        zip(person_tracklets, affinity_to_person),
        key=lambda x: x[0].start_frame
    )

    resolved = []
    removed = []

    for i, (t1, aff1) in enumerate(sorted_tracklets):
        conflict = False
        for t2, aff2 in resolved:
            if temporal_overlap(t1, t2) > 0.1:
                conflict = True
                if aff1 > aff2:
                    resolved.remove((t2, aff2))
                    removed.append(t2)
                    resolved.append((t1, aff1))
                else:
                    removed.append(t1)
                break

        if not conflict:
            resolved.append((t1, aff1))

    return [t for t, _ in resolved], removed
```

---

### 3.7 Stage 7: Identity Refinement

**Optional but Recommended**: Use additional signals to verify and refine assignments.

#### 3.7.1 Face Gallery Matching

If reference face images are available (e.g., from a different source or manual annotation):

```python
def match_to_gallery(
    person_tracklets: List[Tracklet],
    gallery: Dict[str, np.ndarray],  # name -> face embedding
    threshold: float = 0.5
) -> Optional[str]:
    """
    Match aggregated person features to a reference gallery.
    """
    # Collect all face embeddings from tracklets
    all_faces = []
    for t in person_tracklets:
        for det in t.detections:
            if det.face_embedding is not None and det.face_confidence > 0.6:
                all_faces.append(det.face_embedding)

    if not all_faces:
        return None

    # Compare to gallery
    mean_face = np.mean(all_faces, axis=0)
    best_match = None
    best_score = -1

    for name, gallery_emb in gallery.items():
        score = cosine_similarity(mean_face, gallery_emb)
        if score > best_score:
            best_score = score
            best_match = name

    if best_score >= threshold:
        return best_match
    return None
```

#### 3.7.2 Manual Anchor Specification

Allow user to specify anchor frames for each dancer:

```yaml
anchors:
  - person_id: 1
    frame: 150
    bbox: [100, 200, 250, 600]  # Approximate bbox
    name: "Dancer A"
  - person_id: 2
    frame: 150
    bbox: [300, 180, 450, 580]
    name: "Dancer B"
```

```python
def apply_manual_anchors(
    tracklets: List[Tracklet],
    anchors: List[AnchorSpec],
    person_assignments: Dict[int, List[int]]
) -> Dict[int, List[int]]:
    """
    Reassign tracklets based on manual anchor specifications.
    """
    anchor_embeddings = {}

    for anchor in anchors:
        # Find tracklet containing anchor frame and bbox
        for tracklet in tracklets:
            if tracklet.start_frame <= anchor.frame <= tracklet.end_frame:
                det = tracklet.get_detection_at_frame(anchor.frame)
                if det and iou(det.bbox, anchor.bbox) > 0.5:
                    anchor_embeddings[anchor.person_id] = (
                        det.body_embedding,
                        det.face_embedding
                    )
                    break

    # Reassign tracklets based on anchor similarity
    new_assignments = {pid: [] for pid in anchor_embeddings.keys()}

    for tracklet in tracklets:
        best_person = None
        best_sim = -1

        t_features = aggregate_tracklet_features(tracklet)

        for person_id, (body_emb, face_emb) in anchor_embeddings.items():
            body_sim = cosine_similarity(t_features.body_embedding, body_emb)
            if face_emb is not None and t_features.face_embedding is not None:
                face_sim = cosine_similarity(t_features.face_embedding, face_emb)
                sim = 0.6 * face_sim + 0.4 * body_sim
            else:
                sim = body_sim

            if sim > best_sim:
                best_sim = sim
                best_person = person_id

        if best_person is not None and best_sim > 0.4:
            new_assignments[best_person].append(tracklet.tracklet_id)

    return new_assignments
```

---

### 3.8 Stage 8: Output Generation

**Objective**: Generate per-dancer pose sequences for the entire video.

#### 3.8.1 Pose Sequence Assembly

```python
@dataclass
class DancerPoseSequence:
    person_id: int
    name: Optional[str]
    total_frames: int
    keypoints: np.ndarray  # [total_frames, 136, 3]
    visibility: np.ndarray  # [total_frames] boolean
    confidence: np.ndarray  # [total_frames] float
    bboxes: np.ndarray  # [total_frames, 4] or zeros

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
```

```python
def assemble_pose_sequences(
    person_assignments: Dict[int, List[int]],
    tracklets: List[Tracklet],
    total_frames: int
) -> List[DancerPoseSequence]:
    """
    Create complete pose sequences for each identified person.
    """
    sequences = []

    for person_id, tracklet_ids in person_assignments.items():
        # Initialize arrays
        keypoints = np.zeros((total_frames, 136, 3), dtype=np.float32)
        visibility = np.zeros(total_frames, dtype=bool)
        confidence = np.zeros(total_frames, dtype=np.float32)
        bboxes = np.zeros((total_frames, 4), dtype=np.float32)

        # Fill in from tracklets
        for tid in tracklet_ids:
            tracklet = tracklets[tid]
            for det in tracklet.detections:
                frame = det.frame_id
                if not visibility[frame]:  # Don't overwrite
                    keypoints[frame] = det.keypoints
                    visibility[frame] = True
                    confidence[frame] = det.keypoints[:, 2].mean()
                    bboxes[frame] = det.bbox
                elif det.keypoints[:, 2].mean() > confidence[frame]:
                    # Overwrite with higher confidence
                    keypoints[frame] = det.keypoints
                    confidence[frame] = det.keypoints[:, 2].mean()
                    bboxes[frame] = det.bbox

        sequences.append(DancerPoseSequence(
            person_id=person_id,
            name=None,  # Set if matched to gallery
            total_frames=total_frames,
            keypoints=keypoints,
            visibility=visibility,
            confidence=confidence,
            bboxes=bboxes
        ))

    return sequences
```

#### 3.8.2 Gap Interpolation (Optional)

For short gaps where dancer is momentarily occluded, interpolate poses:

```python
def interpolate_gaps(
    sequence: DancerPoseSequence,
    max_gap_frames: int = 15  # ~0.5 seconds
) -> DancerPoseSequence:
    """
    Linearly interpolate keypoints for short gaps.
    Mark interpolated frames with reduced confidence.
    """
    visibility = sequence.visibility.copy()
    keypoints = sequence.keypoints.copy()
    confidence = sequence.confidence.copy()

    # Find gaps
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
            if gap_length <= max_gap_frames:
                if gap_start > 0 and gap_end < len(visibility):
                    # Interpolate
                    start_kpts = keypoints[gap_start - 1]
                    end_kpts = keypoints[gap_end]

                    for j in range(gap_start, gap_end):
                        alpha = (j - gap_start + 1) / (gap_length + 1)
                        keypoints[j] = (1 - alpha) * start_kpts + alpha * end_kpts
                        # Mark as interpolated with reduced confidence
                        keypoints[j, :, 2] *= 0.5  # Halve confidence
                        confidence[j] = keypoints[j, :, 2].mean()
                        visibility[j] = True  # Now "visible" via interpolation
        else:
            i += 1

    return DancerPoseSequence(
        person_id=sequence.person_id,
        name=sequence.name,
        total_frames=sequence.total_frames,
        keypoints=keypoints,
        visibility=visibility,
        confidence=confidence,
        bboxes=sequence.bboxes
    )
```

#### 3.8.3 Output Formats

**Primary Output**: AlphaPose-compatible JSON

```json
{
  "video_info": {
    "path": "/path/to/video.mp4",
    "total_frames": 3600,
    "fps": 30,
    "resolution": [1280, 720]
  },
  "dancers": [
    {
      "person_id": 1,
      "name": "Dancer A",
      "visible_frame_count": 3200,
      "pose_sequence": [
        {
          "frame_id": 0,
          "visible": true,
          "interpolated": false,
          "bbox": [100, 150, 280, 650],
          "keypoints": [[x, y, conf], ...],  // 136 keypoints
          "mean_confidence": 0.85
        },
        {
          "frame_id": 1,
          "visible": false,
          "interpolated": false,
          "keypoints": [[0, 0, 0], ...],  // All zeros
          "mean_confidence": 0.0
        }
        // ... all frames
      ]
    }
  ]
}
```

**NumPy Archive** (for efficient downstream processing):

```python
def save_as_numpy(sequences: List[DancerPoseSequence], output_path: str):
    """Save sequences as compressed NumPy archive."""
    np.savez_compressed(
        output_path,
        person_ids=np.array([s.person_id for s in sequences]),
        names=np.array([s.name or "" for s in sequences]),
        keypoints=np.stack([s.keypoints for s in sequences]),  # [N, T, 136, 3]
        visibility=np.stack([s.visibility for s in sequences]),  # [N, T]
        confidence=np.stack([s.confidence for s in sequences]),  # [N, T]
        bboxes=np.stack([s.bboxes for s in sequences])  # [N, T, 4]
    )
```

---

## 4. Technology Stack & Dependencies

### 4.1 Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Detection | ultralytics | >=8.0 | YOLOv8 person detection |
| Detection (Alt) | transformers | >=4.30 | RT-DETR detection |
| Pose | AlphaPose | latest | Halpe-136 keypoint estimation |
| Body Re-ID | torchreid | >=0.2 | OSNet/TransReID embeddings |
| Face | insightface | >=0.7 | ArcFace embeddings |
| Tracking | boxmot | >=10.0 | OC-SORT, Deep OC-SORT |
| Clustering | scipy | >=1.10 | Hierarchical clustering |
| Graph | networkx | >=3.0 | Graph-based association |
| Core | numpy, opencv-python, torch | - | Fundamental operations |

### 4.2 Recommended Models

```yaml
models:
  detection:
    primary: "yolov8x.pt"
    alternative: "rtdetr-x"

  pose:
    model: "halpe136/fast_421_res152_256x192"
    # Or for higher accuracy:
    # model: "halpe136/fast_dcn_421_res152_256x192"

  body_reid:
    primary: "osnet_ain_x1_0_imagenet"
    alternative: "transreid_vit_base_ics_lup"

  face:
    model: "buffalo_l"  # ArcFace from InsightFace
```

### 4.3 Hardware Requirements

| Configuration | GPU Memory | Batch Processing Speed |
|--------------|------------|------------------------|
| Minimum | 8GB VRAM | ~5 fps |
| Recommended | 16GB VRAM | ~15 fps |
| Optimal | 24GB+ VRAM | ~30 fps |

---

## 5. Configuration Schema

```yaml
# config.yaml - Full pipeline configuration

pipeline:
  name: "dance_tracker"
  version: "1.0"

input:
  video_path: "/path/to/dance_video.mp4"
  num_dancers: 8  # Optional: if known
  anchor_file: null  # Optional: manual anchors
  face_gallery: null  # Optional: reference faces

output:
  directory: "./output"
  save_json: true
  save_numpy: true
  save_visualization: false
  visualization_fps: 10  # For debug videos

detection:
  model: "yolov8x"
  confidence: 0.25
  nms_iou: 0.45
  input_size: 1280
  batch_size: 16

pose:
  model: "halpe136"
  batch_size: 32
  min_keypoint_confidence: 0.1

features:
  body_reid:
    model: "osnet_ain_x1_0"
    input_size: [256, 128]
    batch_size: 64

  face_reid:
    model: "buffalo_l"
    det_size: 640
    min_confidence: 0.5

  pose_descriptor:
    enabled: true
    include_limb_ratios: true

short_term_tracking:
  algorithm: "deep_ocsort"
  max_age: 60
  min_hits: 3
  iou_threshold: 0.3
  appearance_weight: 0.6

tracklet_analysis:
  min_tracklet_length: 10
  anchor_min_duration: 90  # 3 seconds at 30fps
  anchor_min_detection_rate: 0.8
  anchor_min_face_visibility: 0.3

global_association:
  algorithm: "hierarchical"  # or "spectral"
  affinity_threshold: 0.5
  temporal_overlap_threshold: 0.3
  max_spatial_velocity: 50  # pixels/frame

  weights:
    appearance: 0.6
    pose: 0.3
    spatial: 0.1

refinement:
  use_face_gallery: false
  use_manual_anchors: false
  temporal_conflict_resolution: true

output_generation:
  interpolate_gaps: true
  max_interpolation_gap: 15
  interpolation_confidence_factor: 0.5
```

---

## 6. Handling Specific Challenges

### 6.1 Complex Non-Linear Motion

**Challenge**: Dancers perform spins, jumps, sudden direction changes that break Kalman filter assumptions.

**Solutions**:
1. **OC-SORT's observation-centric approach**: Uses actual observations rather than predictions for association
2. **Low inertia parameter**: Reduces reliance on motion prediction
3. **High appearance weight**: Prioritizes visual similarity over motion consistency
4. **Longer max_age**: Maintains tracks through brief motion anomalies

### 6.2 Frequent Occlusions

**Challenge**: Dancers constantly pass in front of each other.

**Solutions**:
1. **Tracklet-based approach**: Allow tracks to fragment and merge later
2. **Multi-cue affinity**: Use body, face, and pose features for robust matching
3. **Temporal consistency**: Verify that merged tracklets don't overlap
4. **Detection-level handling**: Run AlphaPose with pose-flow to maintain associations through brief occlusions

### 6.3 Frame Exit and Re-Entry

**Challenge**: Dancers leave the frame completely and return later.

**Solutions**:
1. **Appearance-based re-identification**: Body and face embeddings persist across absences
2. **No temporal distance penalty**: Don't penalize long gaps in affinity computation
3. **Anchor tracklets**: Use high-confidence tracklets as identity references
4. **Face gallery matching**: If available, match returning dancers to known identities

### 6.4 Back-Row Visibility

**Challenge**: Dancers in back rows appear/disappear as front dancers move.

**Solutions**:
1. **High recall detection**: Low confidence threshold catches partial bodies
2. **Pose-based matching**: Even partial poses can match via body proportions
3. **Spatial context**: Back-row positions tend to be consistent
4. **Appearance consistency**: Clothing features remain recognizable

### 6.5 Motion Blur and Grain

**Challenge**: 720p video with motion blur during fast movements.

**Solutions**:
1. **Robust detection models**: YOLOv8x handles degraded images well
2. **Temporal aggregation**: Average features across multiple frames
3. **Lower confidence thresholds**: Accept noisier detections
4. **Feature quality weighting**: Weight high-clarity frames more heavily in aggregation

---

## 7. Evaluation & Validation

### 7.1 Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **MOTA** | Multi-Object Tracking Accuracy | >70% |
| **IDF1** | ID F1 Score (identity preservation) | >80% |
| **HOTA** | Higher Order Tracking Accuracy | >60% |
| **ID Switches** | Number of identity swaps | <5 per minute |
| **Track Fragmentation** | Tracklets per true identity | <3 |
| **Mostly Tracked (MT)** | Trajectories tracked >80% of lifespan | >80% |

### 7.2 Validation Pipeline

```python
def validate_tracking_output(
    predictions: List[DancerPoseSequence],
    ground_truth: List[DancerPoseSequence]
) -> Dict[str, float]:
    """
    Compute tracking metrics against ground truth.
    """
    # Build per-frame ID assignments
    pred_assignments = defaultdict(dict)  # frame -> {bbox: person_id}
    gt_assignments = defaultdict(dict)

    # ... build assignments ...

    # Compute metrics using motmetrics or TrackEval
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=True)

    for frame_id in range(total_frames):
        gt_ids = list(gt_assignments[frame_id].keys())
        pred_ids = list(pred_assignments[frame_id].keys())

        gt_bboxes = [gt_assignments[frame_id][i] for i in gt_ids]
        pred_bboxes = [pred_assignments[frame_id][i] for i in pred_ids]

        distances = mm.distances.iou_matrix(gt_bboxes, pred_bboxes, max_iou=0.5)
        acc.update(gt_ids, pred_ids, distances)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'idf1', 'num_switches'], name='dance')

    return summary.to_dict()
```

### 7.3 Visual Debugging

Generate debug visualizations:

```python
def generate_debug_video(
    video_path: str,
    sequences: List[DancerPoseSequence],
    output_path: str
):
    """
    Create video with:
    - Bounding boxes colored by person ID
    - Skeleton overlays
    - ID labels
    - Confidence indicators
    """
    # Color palette for consistent person colors
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 128), (255, 165, 0)
    ]

    cap = cv2.VideoCapture(video_path)
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        cap.get(cv2.CAP_PROP_FPS),
        (int(cap.get(3)), int(cap.get(4)))
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for seq in sequences:
            if seq.visibility[frame_idx]:
                color = colors[seq.person_id % len(colors)]

                # Draw bbox
                bbox = seq.bboxes[frame_idx].astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                # Draw skeleton
                kpts = seq.keypoints[frame_idx]
                draw_halpe136_skeleton(frame, kpts, color)

                # Draw label
                label = f"ID:{seq.person_id} ({seq.confidence[frame_idx]:.2f})"
                cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
```

---

## 8. Implementation Roadmap

### Phase 1: Core Infrastructure
- Implement detection and pose estimation pipeline
- Set up feature extraction (body, face, pose)
- Create data structures and storage formats

### Phase 2: Short-Term Tracking
- Integrate Deep OC-SORT with appearance features
- Implement tracklet creation and management
- Add quality scoring

### Phase 3: Global Association
- Implement affinity matrix computation
- Add hierarchical clustering
- Handle temporal conflicts

### Phase 4: Refinement & Output
- Implement face gallery matching (optional)
- Add manual anchor support
- Create output generation pipeline
- Add gap interpolation

### Phase 5: Evaluation & Tuning
- Implement evaluation metrics
- Create debug visualization tools
- Parameter tuning on sample videos

---

## 9. Alternative Approaches Considered

### 9.1 End-to-End Transformer Trackers

**Models**: TrackFormer, MOTR, GTR

**Pros**:
- Unified architecture
- Attention handles occlusions well
- No hand-crafted association logic

**Cons**:
- Require training on dance-specific data
- Less interpretable
- Harder to debug and tune
- Current models optimized for pedestrians, not dancers

**Decision**: Not recommended for initial implementation. Consider for future work if labeled dance data becomes available.

### 9.2 Pose Tracking (vs Detection + Tracking)

**Models**: PoseTrack, AlphaPose with PoseFlow

**Pros**:
- Direct pose-to-pose association
- Leverages pose structure

**Cons**:
- AlphaPose PoseFlow uses simple optical flow
- Doesn't handle long occlusions
- No robust re-identification

**Decision**: Use as complementary signal within short-term tracking, but not as primary association method.

### 9.3 Graph Neural Network Tracking

**Models**: MPNTrack, GMTracker

**Pros**:
- Models complex relationships between detections
- Can incorporate long-range temporal dependencies
- Handles occlusions via message passing

**Cons**:
- Computationally expensive
- Requires training
- Complex to implement

**Decision**: Consider as upgrade path for global association stage. Hierarchical clustering is simpler and more interpretable for initial implementation.

---

## 10. Future Enhancements

1. **Active Learning Loop**: Use model uncertainty to request human annotations for ambiguous cases

2. **Dance-Specific Motion Models**: Train motion prediction models on dance sequences to better predict complex movements

3. **Clothing Segmentation**: Use segmentation models to extract more robust appearance features

4. **3D Pose Lifting**: Extend to 3D pose estimation for richer motion analysis

5. **Multi-View Fusion**: If multiple camera angles are available, fuse information across views

6. **Online Processing**: Adapt pipeline for streaming/real-time applications

---

## 11. Conclusion

This design presents a robust, multi-stage approach to tracking dancers in complex choreographed videos. By combining:

1. High-recall person detection
2. Multi-modal feature extraction (body, face, pose)
3. Modern short-term tracking (OC-SORT with appearance)
4. Global tracklet association via hierarchical clustering
5. Optional identity refinement through galleries or manual anchors

We can achieve reliable per-dancer pose sequences even in the presence of:
- Complex non-linear motion
- Frequent occlusions
- Frame exits and re-entries
- Variable visibility

The modular design allows for incremental implementation and easy component upgrades as better models become available.

---

## Appendix A: Halpe-136 Keypoint Index Reference

```
Body (26 keypoints): 0-25
  0: Nose
  1-2: Eyes (L/R)
  3-4: Ears (L/R)
  5-6: Shoulders (L/R)
  7-8: Elbows (L/R)
  9-10: Wrists (L/R)
  11-12: Hips (L/R)
  13-14: Knees (L/R)
  15-16: Ankles (L/R)
  17: Head top
  18: Neck
  19-20: Hip centers
  21-22: Feet (L/R)
  23-24: Toe tips (L/R)
  25: Spine

Face (68 keypoints): 26-93
  26-42: Jaw line (17 points)
  43-51: Right eyebrow (5 points) + Left eyebrow (5 points)
  52-59: Nose bridge and tip (8 points)
  60-71: Right eye (6 points) + Left eye (6 points)
  72-91: Outer lip (12 points) + Inner lip (8 points)
  92-93: Pupil centers (estimated)

Left Hand (21 keypoints): 94-114
  Wrist, thumb (4), index (4), middle (4), ring (4), pinky (4)

Right Hand (21 keypoints): 115-135
  Wrist, thumb (4), index (4), middle (4), ring (4), pinky (4)
```

---

## Appendix B: Skeleton Connectivity for Visualization

```python
HALPE136_SKELETON = [
    # Body
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (12, 14), (13, 15), (14, 16),  # Legs
    (15, 21), (16, 22), (21, 23), (22, 24),  # Feet

    # Face - jaw line, eyebrows, nose, eyes, lips
    # (26-93, connect sequentially within each feature)

    # Hands - connect finger joints
    # (94-114 left, 115-135 right)
]
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-24*
