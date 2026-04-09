import os
import time
import tempfile
import warnings
from collections import deque
from threading import Lock
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# new imports
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Suppress Streamlit ScriptRunContext warnings during initialization
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="PPE Detection Suite", page_icon="🦺", layout="wide")

# DEFAULT_MAIN_MODEL = r"D:\PPE (Personal Protective Equipment) detection system\PPE-Project\models\ppe_yolo_model_v1_best.pt"
# DEFAULT_GLASSES_MODEL = r"D:\PPE (Personal Protective Equipment) detection system\PPE-Project\models\Old-Model\best.pt"

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MAIN_MODEL = str(BASE_DIR / "models" / "ppe_yolo_model_v1_best.pt")
DEFAULT_GLASSES_MODEL = str(BASE_DIR / "models" / "best.pt")

MAIN_CLASS_NAMES = {
    0: "boots",
    1: "gloves",
    2: "helmet",
    3: "person",
    4: "vest",
}

REQUIRED_PPE = ["helmet", "glasses", "vest"]
OPTIONAL_PPE = ["gloves", "boots"]

COLORS = {
    "person_safe": (0, 180, 0),
    "person_violation": (0, 0, 255),
    "helmet": (255, 0, 255),
    "glasses": (0, 255, 0),
    "vest": (0, 255, 255),
    "gloves": (0, 165, 255),
    "boots": (255, 255, 0),
}

# =========================================================
# SESSION STATE
# =========================================================
if "tracks" not in st.session_state:
    st.session_state.tracks = {}
if "next_track_id" not in st.session_state:
    st.session_state.next_track_id = 1
if "last_perf" not in st.session_state:
    st.session_state.last_perf = {}
if "live_shared_state" not in st.session_state:
    st.session_state.live_shared_state = {"perf": {}, "rows": [], "error": ""}


# =========================================================
# HELPER FUNCTIONS FOR GLASSES DETECTION
# =========================================================
def get_glasses_class_id(glasses_model: YOLO) -> Optional[int]:
    names = getattr(glasses_model, "names", {})
    if isinstance(names, list):
        for i, name in enumerate(names):
            if str(name).strip().lower() in ["glasses", "goggles", "safety glasses", "eyeglasses", "glass", "spectacles", "eye_glass", "eye glasses"]:
                return i
    elif isinstance(names, dict):
        for i, name in names.items():
            if str(name).strip().lower() in ["glasses", "goggles", "safety glasses", "eyeglasses", "glass", "spectacles", "eye_glass", "eye glasses"]:
                return int(i)
    return None


@st.cache_resource(show_spinner=False)
def load_models(main_model_path: str, glasses_model_path: str):
    main_model = YOLO(main_model_path)
    g_model = YOLO(glasses_model_path)
    return main_model, g_model


# =========================================================
# HELPERS
# =========================================================
def get_color_bgr(name: str) -> Tuple[int, int, int]:
    return COLORS[name]


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def resize_for_display(img: np.ndarray, max_width: int = 820, max_height: int = 520) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    if scale >= 1.0:
        return img
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def point_in_box(point: Tuple[float, float], box: Tuple[int, int, int, int]) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter_area
    return inter_area / union if union > 0 else 0.0


def expand_box(box: Tuple[int, int, int, int], pad_x: int = 8, pad_y: int = 6) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)


def merge_two_boxes(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    return (min(x11, x21), min(y11, y21), max(x12, x22), max(y12, y22))


def boxes_are_close(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int], x_gap: int = 40, y_gap: int = 25) -> bool:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    horizontal_close = abs(x11 - x21) <= x_gap or abs(x12 - x22) <= x_gap
    vertical_close = abs(y11 - y21) <= y_gap or abs(y12 - y22) <= y_gap
    return horizontal_close and vertical_close


def merge_glasses_detections(glasses_list: List[dict]) -> Optional[dict]:
    if len(glasses_list) == 0:
        return None
    if len(glasses_list) == 1:
        g = glasses_list[0].copy()
        g["box"] = expand_box(g["box"], pad_x=10, pad_y=8)
        return g

    glasses_list = sorted(glasses_list, key=lambda d: d["conf"], reverse=True)
    best = glasses_list[0]
    second = glasses_list[1]

    if boxes_are_close(best["box"], second["box"]):
        merged = best.copy()
        merged["box"] = expand_box(merge_two_boxes(best["box"], second["box"]), pad_x=12, pad_y=10)
        merged["conf"] = max(best["conf"], second["conf"])
        return merged

    best = best.copy()
    best["box"] = expand_box(best["box"], pad_x=10, pad_y=8)
    return best


def is_valid_person(box: Tuple[int, int, int, int], conf: float, min_conf: float, min_area: int, min_height: int) -> bool:
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    area = w * h
    return conf >= min_conf and area >= min_area and h >= min_height


def get_person_regions(person_box: Tuple[int, int, int, int]) -> Tuple[Tuple[int, int, int, int], ...]:
    x1, y1, x2, y2 = person_box
    h = y2 - y1
    w = x2 - x1

    helmet_region = (x1, y1, x2, y1 + int(0.30 * h))
    glasses_region = (x1 + int(0.08 * w), y1 + int(0.08 * h), x2 - int(0.08 * w), y1 + int(0.48 * h))
    vest_region = (x1, y1 + int(0.22 * h), x2, y1 + int(0.78 * h))
    gloves_region = (x1, y1 + int(0.30 * h), x2, y1 + int(0.90 * h))
    boots_region = (x1, y1 + int(0.80 * h), x2, y2)
    return helmet_region, glasses_region, vest_region, gloves_region, boots_region


def smooth_box(old_box: Tuple[int, int, int, int], new_box: Tuple[int, int, int, int], alpha: float = 0.75) -> Tuple[int, int, int, int]:
    ox1, oy1, ox2, oy2 = old_box
    nx1, ny1, nx2, ny2 = new_box
    return (
        int(alpha * ox1 + (1 - alpha) * nx1),
        int(alpha * oy1 + (1 - alpha) * ny1),
        int(alpha * ox2 + (1 - alpha) * nx2),
        int(alpha * oy2 + (1 - alpha) * ny2),
    )


def remove_duplicate_persons(persons: List[dict], iou_threshold: float = 0.55) -> List[dict]:
    persons = sorted(persons, key=lambda p: p["conf"], reverse=True)
    filtered = []
    for p in persons:
        keep = True
        for fp in filtered:
            if box_iou(p["box"], fp["box"]) > iou_threshold:
                keep = False
                break
        if keep:
            filtered.append(p)
    return filtered


def detect_main(frame_bgr: np.ndarray, ppe_model: YOLO, conf_threshold: float, iou_threshold: float, infer_size: int,
                min_person_conf: float, min_person_area: int, min_person_height: int):
    results = ppe_model.predict(
        source=frame_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=infer_size,
        verbose=False,
        agnostic_nms=True,
    )[0]

    persons, boots, gloves, helmets, vests = [], [], [], [], []

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id not in MAIN_CLASS_NAMES:
                continue

            label = MAIN_CLASS_NAMES[cls_id]
            det = {"label": label, "conf": conf, "box": (x1, y1, x2, y2)}

            if label == "person":
                if is_valid_person(det["box"], conf, min_person_conf, min_person_area, min_person_height):
                    persons.append(det)
            elif label == "boots":
                boots.append(det)
            elif label == "gloves":
                gloves.append(det)
            elif label == "helmet":
                helmets.append(det)
            elif label == "vest":
                vests.append(det)

    return remove_duplicate_persons(persons), boots, gloves, helmets, vests


# def detect_glasses(frame_bgr: np.ndarray, glasses_model: YOLO, infer_size: int):
#     results = glasses_model.predict(
#         source=frame_bgr,
#         conf=0.25,
#         iou=0.45,
#         imgsz=infer_size,
#         verbose=False,
#         agnostic_nms=True,
#     )[0]

#     glasses = []
#     if results.boxes is not None:
#         for box in results.boxes:
#             cls_id = int(box.cls[0].item())
#             conf = float(box.conf[0].item())
#             x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

#             if cls_id != 1:
#                 continue
#             if conf < 0.25:
#                 continue

#             bw, bh = x2 - x1, y2 - y1
#             if bw < 10 or bh < 10 or bw * bh < 300:
#                 continue

#             glasses.append({"label": "glasses", "conf": conf, "box": (x1, y1, x2, y2)})

#     return glasses

def detect_glasses(frame_bgr: np.ndarray, glasses_model: YOLO, infer_size: int, glasses_class_id: Optional[int] = None):
    if glasses_class_id is None:
        return []

    results = glasses_model.predict(
        source=frame_bgr,
        conf=0.12,
        iou=0.40,
        imgsz=infer_size,
        verbose=False,
        agnostic_nms=True,
    )[0]

    glasses = []
    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls_id != glasses_class_id:
                continue
            if conf < 0.12:
                continue

            bw, bh = x2 - x1, y2 - y1
            if bw < 5 or bh < 5 or bw * bh < 60:
                continue

            glasses.append({
                "label": "glasses",
                "conf": conf,
                "box": (x1, y1, x2, y2),
            })

    return glasses


def create_track(box: Tuple[int, int, int, int]):
    track_id = st.session_state.next_track_id
    st.session_state.next_track_id += 1

    st.session_state.tracks[track_id] = {
        "box": box,
        "missing_frames": 0,
        "violation_counter": 0,
        "alarm_active": False,
        "ppe_history": {
            "helmet": deque(maxlen=8),
            "glasses": deque(maxlen=8),
            "vest": deque(maxlen=8),
            "gloves": deque(maxlen=8),
            "boots": deque(maxlen=8),
        },
    }
    return track_id


def update_tracks(persons: List[dict], track_match_iou: float = 0.30, track_max_missing_frames: int = 20,
                  box_smooth_alpha: float = 0.75):
    tracks = st.session_state.tracks
    matched_track_ids = set()
    assigned_persons = []

    for person in persons:
        best_track_id = None
        best_iou = 0.0

        for track_id, track in tracks.items():
            if track_id in matched_track_ids:
                continue
            iou = box_iou(person["box"], track["box"])
            if iou > track_match_iou and iou > best_iou:
                best_iou = iou
                best_track_id = track_id

        if best_track_id is None:
            best_track_id = create_track(person["box"])
        else:
            old_box = tracks[best_track_id]["box"]
            tracks[best_track_id]["box"] = smooth_box(old_box, person["box"], box_smooth_alpha)
            tracks[best_track_id]["missing_frames"] = 0

        matched_track_ids.add(best_track_id)
        assigned = person.copy()
        assigned["id"] = best_track_id
        assigned["box"] = tracks[best_track_id]["box"]
        assigned_persons.append(assigned)

    stale_ids = []
    for track_id in list(tracks.keys()):
        if track_id not in matched_track_ids:
            tracks[track_id]["missing_frames"] += 1
            if tracks[track_id]["missing_frames"] > track_max_missing_frames:
                stale_ids.append(track_id)

    for track_id in stale_ids:
        del tracks[track_id]

    return assigned_persons


def update_ppe_history(track_id: int, ppe_status: dict):
    history = st.session_state.tracks[track_id]["ppe_history"]
    for key, value in ppe_status.items():
        history[key].append(1 if value else 0)


def get_stable_ppe_status(track_id: int, threshold: float = 0.45):
    history = st.session_state.tracks[track_id]["ppe_history"]
    stable = {}
    for key, q in history.items():
        stable[key] = (sum(q) / len(q)) >= threshold if len(q) > 0 else False
    return stable


def annotate_frame_live(frame_bgr: np.ndarray, persons: List[dict], boots: List[dict], gloves: List[dict], 
                       helmets: List[dict], vests: List[dict], glasses: List[dict], show_ppe_boxes: bool = True):
    """Simple annotation for live mode - no session_state required"""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    
    total_persons = len(persons)
    total_violations = 0
    
    # Draw PPE detections
    for person in persons:
        px1, py1, px2, py2 = person["box"]
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 180, 0), 2)  # Green person box
        
        # Check if person has PPE violations based on nearby detections
        person_region = (px1, py1, px2, py2)
        has_helmet = any(box_iou(h_item["box"], person_region) > 0.1 for h_item in helmets)
        has_vest = any(box_iou(v_item["box"], person_region) > 0.1 for v_item in vests)
        
        if not (has_helmet and has_vest):
            total_violations += 1
    
    if show_ppe_boxes:
        # Draw helmet boxes
        for item in helmets:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 255), 1)  # Magenta
        
        # Draw vest boxes
        for item in vests:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 1)  # Cyan
        
        # Draw gloves boxes
        for item in gloves:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 1)  # Orange
        
        # Draw boots boxes
        for item in boots:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Yellow
        
        # Draw glasses boxes
        for item in glasses:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 1)  # Green
    
    return annotated, total_persons, total_violations


def annotate_frame(frame_bgr: np.ndarray, persons: List[dict], boots: List[dict], gloves: List[dict], helmets: List[dict],
                   vests: List[dict], glasses: List[dict], show_ppe_boxes: bool, show_regions: bool,
                   persistence_frames: int, alarm_enabled: bool = False):
    tracks = st.session_state.tracks
    frame = frame_bgr.copy()
    current_ids = set()
    perf_rows = []

    for person in persons:
        person_id = person["id"]
        current_ids.add(person_id)
        px1, py1, px2, py2 = person["box"]
        person_box = person["box"]

        helmet_region, glasses_region, vest_region, gloves_region, boots_region = get_person_regions(person_box)

        best_helmet = None
        best_vest = None
        best_gloves = None
        best_boots = None
        candidate_glasses = []

        for item in helmets:
            if point_in_box(get_center(item["box"]), helmet_region):
                if best_helmet is None or item["conf"] > best_helmet["conf"]:
                    best_helmet = item

        for item in glasses:
            if point_in_box(get_center(item["box"]), glasses_region):
                candidate_glasses.append(item)
        best_glasses = merge_glasses_detections(candidate_glasses)

        for item in vests:
            if point_in_box(get_center(item["box"]), vest_region):
                if best_vest is None or item["conf"] > best_vest["conf"]:
                    best_vest = item

        for item in gloves:
            if point_in_box(get_center(item["box"]), gloves_region):
                if best_gloves is None or item["conf"] > best_gloves["conf"]:
                    best_gloves = item

        for item in boots:
            if point_in_box(get_center(item["box"]), boots_region):
                if best_boots is None or item["conf"] > best_boots["conf"]:
                    best_boots = item

        raw_ppe_status = {
            "helmet": best_helmet is not None,
            "glasses": best_glasses is not None,
            "vest": best_vest is not None,
            "gloves": best_gloves is not None,
            "boots": best_boots is not None,
        }

        update_ppe_history(person_id, raw_ppe_status)
        ppe_status = get_stable_ppe_status(person_id)
        missing_items = [item for item in REQUIRED_PPE if not ppe_status[item]]

        if missing_items:
            tracks[person_id]["violation_counter"] += 1
        else:
            tracks[person_id]["violation_counter"] = 0
            tracks[person_id]["alarm_active"] = False

        is_violation = len(missing_items) > 0
        color = get_color_bgr("person_violation") if is_violation else get_color_bgr("person_safe")
        status_text = f"ID {person_id}: Missing -> {', '.join(missing_items)}" if is_violation else f"ID {person_id}: SAFE"

        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, status_text, (px1, max(25, py1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

        detail_text = (
            f"H:{'Y' if ppe_status['helmet'] else 'N'}  "
            f"Gl:{'Y' if ppe_status['glasses'] else 'N'}  "
            f"V:{'Y' if ppe_status['vest'] else 'N'}  "
            f"Glv:{'Y' if ppe_status['gloves'] else 'N'}  "
            f"Bts:{'Y' if ppe_status['boots'] else 'N'}"
        )
        text_y = py2 + 24 if py2 + 24 < frame.shape[0] else py2 - 12
        cv2.putText(frame, detail_text, (px1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 0), 2)

        if show_regions:
            for region, col in [
                (helmet_region, (255, 0, 255)),
                (glasses_region, (0, 255, 0)),
                (vest_region, (0, 255, 255)),
                (gloves_region, (0, 165, 255)),
                (boots_region, (255, 255, 0)),
            ]:
                cv2.rectangle(frame, (region[0], region[1]), (region[2], region[3]), col, 1)

        if show_ppe_boxes:
            linked_items = [
                ("helmet", best_helmet, get_color_bgr("helmet")),
                ("glasses", best_glasses, get_color_bgr("glasses")),
                ("vest", best_vest, get_color_bgr("vest")),
                ("gloves", best_gloves, get_color_bgr("gloves")),
                ("boots", best_boots, get_color_bgr("boots")),
            ]
            for item_name, item_det, item_color in linked_items:
                if item_det is not None:
                    ix1, iy1, ix2, iy2 = item_det["box"]
                    cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), item_color, 2)
                    cv2.putText(frame, f"{item_name} {item_det['conf']:.2f}", (ix1, max(20, iy1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, item_color, 2)

        if alarm_enabled and missing_items and tracks[person_id]["violation_counter"] >= persistence_frames:
            if not tracks[person_id]["alarm_active"]:
                play_alarm()
                snapshot_path = save_snapshot(frame, person_id, missing_items)
                log_violation(person_id, missing_items, snapshot_path)
                tracks[person_id]["alarm_active"] = True

        perf_rows.append({
            "person_id": person_id,
            "helmet": ppe_status["helmet"],
            "glasses": ppe_status["glasses"],
            "vest": ppe_status["vest"],
            "gloves": ppe_status["gloves"],
            "boots": ppe_status["boots"],
            "missing": ", ".join(missing_items) if missing_items else "None",
            "violation_counter": tracks[person_id]["violation_counter"],
        })

    total_violations = sum(1 for pid in current_ids if tracks[pid]["violation_counter"] >= persistence_frames)
    return frame, len(current_ids), total_violations, perf_rows


def infer_one_frame(frame_bgr: np.ndarray, ppe_model: YOLO, glasses_model: YOLO,
                    conf_threshold: float, iou_threshold: float, infer_size: int,
                    min_person_conf: float, min_person_area: int, min_person_height: int,
                    show_ppe_boxes: bool, show_regions: bool, persistence_frames: int,
                    alarm_enabled: bool = False):
    t0 = time.perf_counter()
    persons, boots, gloves, helmets, vests = detect_main(
        frame_bgr, ppe_model, conf_threshold, iou_threshold, infer_size,
        min_person_conf, min_person_area, min_person_height
    )
    t1 = time.perf_counter()
    glasses = detect_glasses(
        frame_bgr,
        glasses_model,
        infer_size,
        st.session_state.get("glasses_class_id")
    )
    t2 = time.perf_counter()
    persons = update_tracks(persons)
    annotated, total_persons, total_violations, perf_rows = annotate_frame(
        frame_bgr, persons, boots, gloves, helmets, vests, glasses,
        show_ppe_boxes, show_regions, persistence_frames, alarm_enabled
    )
    t3 = time.perf_counter()

    perf = {
        "main_model_ms": round((t1 - t0) * 1000, 1),
        "glasses_model_ms": round((t2 - t1) * 1000, 1),
        "annotation_ms": round((t3 - t2) * 1000, 1),
        "total_ms": round((t3 - t0) * 1000, 1),
        "est_fps": round(1000 / max(((t3 - t0) * 1000), 1), 2),
        "persons": total_persons,
        "violations": total_violations,
    }

    status_bar = f"Persons: {total_persons} | Violations: {total_violations} | FPS: {perf['est_fps']:.1f}"
    cv2.rectangle(annotated, (5, 5), (annotated.shape[1] - 5, 45), (0, 0, 0), -1)
    cv2.putText(annotated, status_bar, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    st.session_state.last_perf = perf
    return annotated, perf_rows
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self, main_model, glasses_model, glasses_class_id, settings, shared_state):
        self.main_model = main_model
        self.glasses_model = glasses_model
        self.glasses_class_id = glasses_class_id
        self.settings = settings
        self.shared_state = shared_state
        self.lock = Lock()
        self.frame_count = 0
        self.last_annotated = None
        self.cached_glasses = []
        
        # Video recording for download
        self.frames_buffer = []
        self.recording = True
        self.max_frames = 500  # ~30 seconds at 16 FPS

    def _update_shared_state(self, perf=None, rows=None, error=None):
        with self.lock:
            if perf is not None:
                self.shared_state["perf"] = perf
            if rows is not None:
                self.shared_state["rows"] = rows
            if error is not None:
                self.shared_state["error"] = error

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        try:
            self.frame_count += 1
            frame_skip = max(1, int(self.settings.get("live_frame_skip", 1)))
            if self.frame_count % frame_skip != 0 and self.last_annotated is not None:
                return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")

            t0 = time.perf_counter()
            persons, boots, gloves, helmets, vests = detect_main(
                img,
                self.main_model,
                self.settings["conf_threshold"],
                self.settings["iou_threshold"],
                self.settings.get("infer_size_live", 320),
                self.settings["min_person_conf"],
                self.settings["min_person_area"],
                self.settings["min_person_height"],
            )
            t1 = time.perf_counter()

            use_glasses = bool(self.settings.get("live_detect_glasses", True)) and self.glasses_class_id is not None
            if use_glasses and (self.frame_count % max(1, int(self.settings.get("glasses_every_n", 3))) == 0):
                self.cached_glasses = detect_glasses(
                    img,
                    self.glasses_model,
                    self.settings.get("infer_size_live", 320),
                    self.glasses_class_id,
                )
            glasses = self.cached_glasses if use_glasses else []
            t2 = time.perf_counter()

            # Use simple live annotation (no session_state access)
            annotated, total_persons, total_violations = annotate_frame_live(
                img, persons, boots, gloves, helmets, vests, glasses,
                self.settings["show_ppe_boxes"],
            )
            t3 = time.perf_counter()

            perf = {
                "main_model_ms": round((t1 - t0) * 1000, 1),
                "glasses_model_ms": round((t2 - t1) * 1000, 1),
                "annotation_ms": round((t3 - t2) * 1000, 1),
                "total_ms": round((t3 - t0) * 1000, 1),
                "est_fps": round(1000 / max(((t3 - t0) * 1000), 1), 2),
                "persons": total_persons,
                "violations": total_violations,
            }

            status_bar = f"LIVE | Persons: {total_persons} | Violations: {total_violations} | FPS: {perf['est_fps']:.1f}"
            cv2.rectangle(annotated, (5, 5), (annotated.shape[1] - 5, 45), (0, 0, 0), -1)
            cv2.putText(annotated, status_bar, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

            self.last_annotated = annotated
            
            # Record frame for download (keep last 500 frames)
            if self.recording and len(self.frames_buffer) < self.max_frames:
                self.frames_buffer.append(annotated.copy())
            elif self.recording and len(self.frames_buffer) == self.max_frames:
                self.frames_buffer.pop(0)
                self.frames_buffer.append(annotated.copy())
            
            self._update_shared_state(perf=perf, rows=[], error="")
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

        except Exception as e:
            fallback = img.copy()
            cv2.rectangle(fallback, (0, 0), (fallback.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(fallback, f"Live detection error: {str(e)[:80]}", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            self.last_annotated = fallback
            self._update_shared_state(error=str(e))
            return av.VideoFrame.from_ndarray(fallback, format="bgr24")
    
    def save_video(self, output_path: str, fps: int = 16):
        """Save recorded frames to video file"""
        if not self.frames_buffer:
            return None
        
        h, w = self.frames_buffer[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        for frame in self.frames_buffer:
            out.write(frame)
        
        out.release()
        return output_path

def save_uploaded_file(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


def open_video_source(source):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return cap


def list_local_camera_candidates(max_index: int = 5) -> List[Tuple[str, str]]:
    candidates = []
    for idx in range(max_index + 1):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            ret, _ = cap.read()
            if ret:
                candidates.append((f"Local Camera {idx}", str(idx)))
        cap.release()
    return candidates


# =========================================================
# UI
# =========================================================
print("\n")
st.title("🦺 PPE Detection Suite")
st.caption("Image, video, and live-source PPE detection with person-wise compliance and performance insights.")
# st.markdown("""
# <style>
#     .block-container {padding-top: 1.2rem; padding-bottom: 1.2rem; max-width: 1400px;}
#     [data-testid="stSidebar"] {min-width: 320px; max-width: 320px;}
# </style>
# """, unsafe_allow_html=True)
st.markdown("""
    <style>
    .block-container {
        padding-top: 5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100% !important;
    }
    h1 {
        font-size: 2.5rem !important;
    }
    </style>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.header("Navigation")
    mode = st.radio("Select mode", ["Image Detection", "Video Detection", "Live Detection"], index=0)

    st.header("Model Settings")
    main_model_path = st.text_input("Main model path", value=DEFAULT_MAIN_MODEL)
    glasses_model_path = st.text_input("Glasses model path", value=DEFAULT_GLASSES_MODEL)

    st.header("Detection Settings")
    conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.01)
    iou_threshold = st.slider("IoU threshold", 0.10, 0.90, 0.45, 0.01)
    infer_size = st.selectbox("Inference size", [416, 512, 640, 768], index=2)
    min_person_conf = st.slider("Min person confidence", 0.10, 0.90, 0.40, 0.01)
    min_person_area = st.number_input("Min person area", min_value=1000, max_value=100000, value=12000, step=1000)
    min_person_height = st.number_input("Min person height", min_value=50, max_value=1000, value=140, step=10)
    persistence_frames = st.slider("Persistence frames", 1, 30, 10, 1)
    show_ppe_boxes = st.toggle("Show PPE boxes", True)
    show_regions = st.toggle("Show person regions", False)
    alarm_enabled = st.toggle("Enable alarm/logging", False)

    st.divider()
    st.subheader("Status")
    status_placeholder_sidebar = st.empty()

try:
    ppe_model, glasses_model = load_models(main_model_path, glasses_model_path)
    glasses_class_id = get_glasses_class_id(glasses_model)
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.session_state["ppe_model_obj"] = ppe_model
st.session_state["glasses_model_obj"] = glasses_model
st.session_state["glasses_class_id"] = glasses_class_id
st.session_state["conf_threshold"] = conf_threshold
st.session_state["iou_threshold"] = iou_threshold
st.session_state["infer_size"] = infer_size
st.session_state["min_person_conf"] = min_person_conf
st.session_state["min_person_area"] = min_person_area
st.session_state["min_person_height"] = min_person_height
st.session_state["show_ppe_boxes"] = show_ppe_boxes
st.session_state["show_regions"] = show_regions
st.session_state["persistence_frames"] = persistence_frames

with st.sidebar:
    status_placeholder_sidebar.caption(f"✓ Glasses class id: {glasses_class_id}")
    status_placeholder_sidebar.caption(f"✓ Glasses model labels: {getattr(glasses_model, 'names', {})}")

perf_placeholder = st.empty()
status_placeholder = st.empty()

if mode == "Image Detection":
    st.subheader("1) Image Detection")
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"], key="img")

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated, perf_rows = infer_one_frame(
            image_bgr, ppe_model, glasses_model,
            conf_threshold, iou_threshold, infer_size,
            min_person_conf, min_person_area, min_person_height,
            show_ppe_boxes, show_regions, persistence_frames,
            alarm_enabled=False
        )

        c1, c2 = st.columns(2)
        with c1:
            st.image(bgr_to_rgb(resize_for_display(image_bgr)), caption="Original", width=820)
        with c2:
            st.image(bgr_to_rgb(resize_for_display(annotated)), caption="Detection Output", width=820)

        if perf_rows:
            st.dataframe(pd.DataFrame(perf_rows), width="stretch")
        perf_placeholder.json(st.session_state.last_perf)

elif mode == "Video Detection":
    st.subheader("2) Video Detection")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"], key="vid")
    frame_skip = st.slider("Frame skip", 1, 10, 2, 1)

    if uploaded_video is not None:
        input_path = save_uploaded_file(uploaded_video)
        output_path = os.path.join(tempfile.gettempdir(), f"annotated_{Path(uploaded_video.name).stem}.mp4")

        cap = open_video_source(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        progress = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        current = 0
        preview = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            current += 1

            if current % frame_skip != 0:
                writer.write(frame)
                progress.progress(min(current / frame_count, 1.0))
                continue

            annotated, _ = infer_one_frame(
                frame, ppe_model, glasses_model,
                conf_threshold, iou_threshold, infer_size,
                min_person_conf, min_person_area, min_person_height,
                show_ppe_boxes, show_regions, persistence_frames,
                alarm_enabled=False
            )
            writer.write(annotated)
            if current % max(frame_skip * 5, 5) == 0:
                preview.image(bgr_to_rgb(resize_for_display(annotated)), caption="Processing preview", width=820)
            progress.progress(min(current / frame_count, 1.0))

        cap.release()
        writer.release()

        st.success("Video processed successfully.")
        with open(output_path, "rb") as f:
            st.download_button("Download processed video", f, file_name="ppe_detected_video.mp4", mime="video/mp4")
        perf_placeholder.json(st.session_state.last_perf)

else:
    st.subheader("3) Live Detection")
    
    # Ensure live_shared_state is initialized
    if "live_shared_state" not in st.session_state:
        st.session_state.live_shared_state = {"perf": {}, "rows": [], "error": ""}
    
    st.info(
        """
        ### Cloud Environment Note
        Browser webcam is the **recommended option** for cloud deployment.  
        Local USB cameras don't work on cloud servers.  
        Use IP/RTSP for remote camera streams.
        """
    )

    live_source_type = st.radio(
        "Choose live source",
        [
            "Browser Webcam (Recommended)",
            "IP / RTSP / HTTP Camera",
        ],
        index=0,
    )

    frame_skip_live = st.slider("Live frame skip", 1, 8, 1, 1)
    live_infer_size = st.selectbox("Live inference size (FOR SPEED: use 320)", [320, 416, 512, 640], index=0)
    live_detect_glasses = st.toggle("Live glasses detection", True, help="Turn off for maximum speed.")
    glasses_every_n = st.slider("Run glasses model every Nth processed frame", 1, 5, 3, 1)
    run_seconds = st.slider("Run duration per session (seconds)", 5, 120, 20, 5)

    st.session_state["live_frame_skip"] = frame_skip_live
    st.session_state["infer_size_live"] = live_infer_size
    st.session_state["live_detect_glasses"] = live_detect_glasses
    st.session_state["glasses_every_n"] = glasses_every_n

    st.info("⚡ **Live mode optimized for smooth webcam detection:** thread-safe processing + lower webcam resolution + optional glasses throttling.")

    if live_source_type == "Browser Webcam (Recommended)":
        live_settings = {
            "conf_threshold": conf_threshold,
            "iou_threshold": iou_threshold,
            "infer_size_live": live_infer_size,
            "min_person_conf": min_person_conf,
            "min_person_area": min_person_area,
            "min_person_height": min_person_height,
            "show_ppe_boxes": show_ppe_boxes,
            "show_regions": show_regions,
            "persistence_frames": persistence_frames,
            "live_frame_skip": frame_skip_live,
            "live_detect_glasses": live_detect_glasses,
            "glasses_every_n": glasses_every_n,
        }

        # Capture shared_state before lambda to avoid session_state access in thread
        shared_state = st.session_state.live_shared_state
        
        processor_factory = lambda: PPEVideoProcessor(
            ppe_model,
            glasses_model,
            glasses_class_id,
            live_settings,
            shared_state,
        )

        webrtc_ctx = webrtc_streamer(
            key="ppe-live-browser",
            mode=WebRtcMode.SENDRECV,
            desired_playing_state=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                ]
            },
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 960},
                    "height": {"ideal": 540},
                    "frameRate": {"ideal": 20, "max": 24},
                },
                "audio": False,
            },
            video_processor_factory=processor_factory,
            async_processing=True,
            video_html_attrs={
                "autoPlay": True,
                "playsInline": True,
                "controls": False,
                "muted": True,
            },
        )

        live_state = st.session_state.live_shared_state
        if webrtc_ctx.state.playing:
            st.markdown("---")
            st.subheader("📊 Live Detection Stats")
            live_perf = live_state.get("perf", {})
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("FPS", live_perf.get("est_fps", 0))
            with col2:
                st.metric("Persons", live_perf.get("persons", 0))
            with col3:
                st.metric("Violations", live_perf.get("violations", 0))
            with col4:
                st.metric("Total Time (ms)", live_perf.get("total_ms", 0))

            if live_state.get("rows"):
                st.dataframe(pd.DataFrame(live_state["rows"]), width="stretch")
            if live_state.get("error"):
                st.error(f"Live pipeline error: {live_state['error']}")
            
            # Download recorded video
            st.markdown("---")
            st.subheader("💾 Download Recorded Video")
            col_down1, col_down2 = st.columns(2)
            
            with col_down1:
                if st.button("Save & Download Video", key="save_live_video"):
                    if webrtc_ctx.video_processor:
                        output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        webrtc_ctx.video_processor.save_video(output_file, fps=16)
                        with open(output_file, "rb") as f:
                            st.download_button(
                                label="📥 Click to Download",
                                data=f.read(),
                                file_name="ppe_live_detection.mp4",
                                mime="video/mp4"
                            )
                        st.success(f"✅ Video saved! ({len(webrtc_ctx.video_processor.frames_buffer)} frames)")
                    else:
                        st.warning("Video processor not available yet")
            
            with col_down2:
                st.info(f"📹 Recording: {len(webrtc_ctx.video_processor.frames_buffer if webrtc_ctx.video_processor else [])} frames (max 500)")
        else:
            st.info("⏳ **Waiting for video connection...**\n\n1. Click START\n2. Allow camera permission\n3. If the feed stays black, refresh once and start again")

    elif live_source_type == "IP / RTSP / HTTP Camera":
        st.markdown("Examples: `rtsp://...`, `http://.../video`, `https://...`")
        cam_url = st.text_input("Camera stream URL")
        start_stream = st.button("Start IP camera detection")

        if start_stream and cam_url:
            try:
                cap = open_video_source(cam_url)
                frame_holder = st.empty()
                table_holder = st.empty()
                start_t = time.time()
                frame_idx = 0

                while time.time() - start_t < run_seconds:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx % frame_skip_live != 0:
                        continue

                    annotated, perf_rows = infer_one_frame(
                        frame, ppe_model, glasses_model,
                        conf_threshold, iou_threshold, infer_size,
                        min_person_conf, min_person_area, min_person_height,
                        show_ppe_boxes, show_regions, persistence_frames,
                        alarm_enabled=alarm_enabled
                    )
                    frame_holder.image(bgr_to_rgb(resize_for_display(annotated)), channels="RGB", width=820)
                    if perf_rows:
                        table_holder.dataframe(pd.DataFrame(perf_rows), width="stretch")
                    perf_placeholder.json(st.session_state.last_perf)

                cap.release()
                status_placeholder.success("IP camera session finished.")
            except Exception as e:
                st.error(f"Could not open IP camera stream: {e}")

st.markdown("---")
st.subheader("Performance Summary")
last_perf = st.session_state.get("live_shared_state", {}).get("perf") or st.session_state.get("last_perf", {})
if last_perf:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Main model (ms)", last_perf.get("main_model_ms", 0))
    c2.metric("Glasses model (ms)", last_perf.get("glasses_model_ms", 0))
    c3.metric("Annotation (ms)", last_perf.get("annotation_ms", 0))
    c4.metric("Total (ms)", last_perf.get("total_ms", 0))
    c5.metric("Estimated FPS", last_perf.get("est_fps", 0))
else:
    st.caption("Run any mode to see performance metrics.")

st.markdown("### Deployment note")
st.write(
    "For a public Streamlit link, browser webcam and public/reachable IP camera streams work best. "
    "Auto-discovering USB/COM cameras is only practical when the app runs on the same local machine that owns those devices."
)
