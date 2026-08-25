#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import time
import csv

import numpy as np
import cv2
try:
    import cv2.objdetect
except Exception:
    pass

import threading

MODEL_LOCK = threading.Lock()

def safe_predict(model, roi_resized):
    with MODEL_LOCK:
        return model.predict(roi_resized, verbose=0)


from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import (
    VarianceScaling as TFVarianceScaling,
    Zeros as TFZeros,
    Ones as TFOnes,
)
from tensorflow.keras.layers import SeparableConv2D as TFSeparableConv2D

# ------------------------------------------------------------------
# Paths (relative to 04-WebApp/library)
# ------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(THIS_DIR, "..", "Models")

VIDEO_MODEL_PATH = os.path.join(MODELS_DIR, "video.h5")
CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")

DB_DIR = os.path.join(THIS_DIR, "..", "static", "js", "db")
os.makedirs(DB_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Compatibility wrappers for old Keras configs
# ------------------------------------------------------------------
class VarianceScalingCompat(TFVarianceScaling):
    def __init__(
        self,
        scale=1.0,
        mode="fan_avg",
        distribution="uniform",
        seed=None,
        dtype=None,
        **kwargs,
    ):
        super().__init__(
            scale=scale,
            mode=mode,
            distribution=distribution,
            seed=seed,
        )


class ZerosCompat(TFZeros):
    def __init__(self, dtype=None, **kwargs):
        super().__init__()


class OnesCompat(TFOnes):
    def __init__(self, dtype=None, **kwargs):
        super().__init__()


class SeparableConv2DCompat(TFSeparableConv2D):
    def __init__(
        self,
        *args,
        kernel_initializer=None,
        kernel_regularizer=None,
        kernel_constraint=None,
        **kwargs,
    ):
        if kernel_initializer is not None:
            kwargs.setdefault("depthwise_initializer", kernel_initializer)
            kwargs.setdefault("pointwise_initializer", kernel_initializer)
        if kernel_regularizer is not None:
            kwargs.setdefault("depthwise_regularizer", kernel_regularizer)
            kwargs.setdefault("pointwise_regularizer", kernel_regularizer)
        if kernel_constraint is not None:
            kwargs.setdefault("depthwise_constraint", kernel_constraint)
            kwargs.setdefault("pointwise_constraint", kernel_constraint)

        super().__init__(*args, **kwargs)


_GLOBAL_VIDEO_MODEL = None

def load_video_model(model_path: str = VIDEO_MODEL_PATH):
    global _GLOBAL_VIDEO_MODEL
    with MODEL_LOCK:
        if _GLOBAL_VIDEO_MODEL is None:
            print(f"[video_emotion_recognition] Loading video model from: {model_path}")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"video model not found at {model_path}")

            _GLOBAL_VIDEO_MODEL = load_model(
                model_path,
                custom_objects={
                    "VarianceScaling": VarianceScalingCompat,
                    "Zeros": ZerosCompat,
                    "Ones": OnesCompat,
                    "SeparableConv2D": SeparableConv2DCompat,
                },
                compile=False,
            )
def load_cascade_classifier(cascade_path: str):
    """Safely load OpenCV Haar CascadeClassifier with namespace fallback."""
    if not os.path.exists(cascade_path):
        return None

    classifier_cls = getattr(cv2, "CascadeClassifier", None)
    if classifier_cls is None:
        objdetect_mod = getattr(cv2, "objdetect", None)
        if objdetect_mod is not None:
            classifier_cls = getattr(objdetect_mod, "CascadeClassifier", None)

    if classifier_cls is not None:
        try:
            cascade = classifier_cls(cascade_path)
            if cascade is not None and not cascade.empty():
                return cascade
        except Exception as e:
            print("[video_emotion_recognition] Error instantiating CascadeClassifier:", e)

    return None


def _yield_error_frame(text: str):
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(
        img,
        text,
        (40, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
    )
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return
    jpg_bytes = buf.tobytes()
    yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
    )


EMOTION_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral",
]


# ------------------------------------------------------------------
# Helpers for saving stats
# ------------------------------------------------------------------
def _save_stats(predictions, angry, disgust, fear, happy, sad, surprise, neutral):
    if not predictions:
        return

    histo_perso_path = os.path.join(DB_DIR, "histo_perso.txt")
    with open(histo_perso_path, "w") as d:
        d.write("density\n")
        for val in predictions:
            d.write(val + "\n")

    histo_global_path = os.path.join(DB_DIR, "histo.txt")
    with open(histo_global_path, "a") as d:
        for val in predictions:
            d.write(val + "\n")

    prob_csv_path = os.path.join(DB_DIR, "prob.csv")
    rows = zip(angry, disgust, fear, happy, sad, surprise, neutral)
    with open(prob_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

    prob_tot_path = os.path.join(DB_DIR, "prob_tot.csv")
    rows = zip(angry, disgust, fear, happy, sad, surprise, neutral)
    with open(prob_tot_path, "a", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)


def _dominant_emotion_idx(predictions):
    if not predictions:
        return 6
    arr = np.array(list(map(int, predictions)), dtype=np.int64)
    values, counts = np.unique(arr, return_counts=True)
    return int(values[np.argmax(counts)])


# ==================================================================
# MAIN GENERATOR – WEBCAM (NO TIME LIMIT)
# ==================================================================
def gen():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[video_emotion_recognition] Webcam not available.")
        for chunk in _yield_error_frame("Webcam not available"):
            yield chunk
        return

    try:
        model = load_video_model(VIDEO_MODEL_PATH)
    except Exception as e:
        print("[video_emotion_recognition] Could not load model:", e)
        for chunk in _yield_error_frame("Model load error"):
            yield chunk
        cap.release()
        return

    if not os.path.exists(CASCADE_PATH):
        print("[video_emotion_recognition] Cascade file missing:", CASCADE_PATH)
        for chunk in _yield_error_frame("Cascade file missing"):
            yield chunk
        cap.release()
        return

    face_cascade = load_cascade_classifier(CASCADE_PATH)
    if face_cascade is None or face_cascade.empty():
        print("[video_emotion_recognition] Failed to load Haar cascade.")
        for chunk in _yield_error_frame("Face detector load error"):
            yield chunk
        cap.release()
        return

    shape_x, shape_y = 48, 48

    predictions = []
    angry_0, disgust_1, fear_2 = [], [], []
    happy_3, sad_4, surprise_5, neutral_6 = [], [], [], []

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                print("[video_emotion_recognition] Cannot read from webcam.")
                for chunk in _yield_error_frame("Cannot read from webcam"):
                    yield chunk
                break

            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)

            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                print("[video_emotion_recognition] cvtColor error:", e)
                for chunk in _yield_error_frame("Camera image error"):
                    yield chunk
                break

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(shape_x, shape_y),
            )

            for i, (x, y, w, h) in enumerate(faces):
                roi = gray[y:y + h, x:x + w]
                if roi.size == 0:
                    continue

                try:
                    roi_resized = cv2.resize(roi, (shape_x, shape_y))
                except Exception:
                    continue

                roi_resized = roi_resized.astype(np.float32)
                if roi_resized.max() > 0:
                    roi_resized /= roi_resized.max()
                roi_resized = np.reshape(roi_resized, (1, shape_x, shape_y, 1))

                preds = safe_predict(model, roi_resized)

                angry_0.append(float(preds[0][0]))
                disgust_1.append(float(preds[0][1]))
                fear_2.append(float(preds[0][2]))
                happy_3.append(float(preds[0][3]))
                sad_4.append(float(preds[0][4]))
                surprise_5.append(float(preds[0][5]))
                neutral_6.append(float(preds[0][6]))

                pred_idx = int(np.argmax(preds))
                predictions.append(str(pred_idx))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    EMOTION_LABELS[pred_idx],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

                base_y = 100 + 140 * i
                cv2.putText(
                    frame,
                    f"Face #{i+1}",
                    (40, base_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                for idx, lbl in enumerate(EMOTION_LABELS):
                    cv2.putText(
                        frame,
                        f"{lbl}: {preds[0][idx]:.3f}",
                        (40, base_y + 20 + 20 * idx),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                    )

            cv2.putText(
                frame,
                f"Number of Faces : {len(faces)}",
                (40, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                1,
            )

            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg_bytes = buf.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    except GeneratorExit:
        pass
    finally:
        cap.release()
        _save_stats(
            predictions,
            angry_0,
            disgust_1,
            fear_2,
            happy_3,
            sad_4,
            surprise_5,
            neutral_6,
        )
        K.clear_session()


# ==================================================================
# OFFLINE ANALYSIS – UPLOADED VIDEO FILE
# ==================================================================
def analyze_video_file(video_path: str):
    """
    Analyze an uploaded video file frame by frame (with frame sampling).

    Returns:
        emo_idx (int): dominant emotion index 0..6
        metrics (dict): keys must match DB columns:
            'avg_faces', 'angry', 'happy', 'fear',
            'sad', 'surprise', 'disgust', 'neutral'
    """

    def empty_metrics():
        return {
            "avg_faces": 0.0,
            "angry": 0.0,
            "happy": 0.0,
            "fear": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "disgust": 0.0,
            "neutral": 0.0,
        }

    if not os.path.exists(video_path):
        print("[analyze_video_file] File not found:", video_path)
        return 6, empty_metrics()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[analyze_video_file] Cannot open video:", video_path)
        return 6, empty_metrics()

    try:
        model = load_video_model(VIDEO_MODEL_PATH)
    except Exception as e:
        print("[analyze_video_file] Could not load model:", e)
        return 6, empty_metrics()

    if not os.path.exists(CASCADE_PATH):
        print("[analyze_video_file] Cascade file missing:", CASCADE_PATH)
        return 6, empty_metrics()

    face_cascade = load_cascade_classifier(CASCADE_PATH)
    if face_cascade is None:
        print("[analyze_video_file] Cascade classifier not loaded; will rely on center crop.")

    shape_x, shape_y = 48, 48

    predictions = []
    angry_0, disgust_1, fear_2 = [], [], []
    happy_3, sad_4, surprise_5, neutral_6 = [], [], [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Frame sampling: process max ~40 frames to keep execution fast (<2s) on Render CPU
    max_samples = 40
    step = max(1, total_frames // max_samples) if total_frames > 0 else 1

    curr_frame_idx = 0
    sampled_frame_count = 0
    total_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        if curr_frame_idx % step != 0:
            curr_frame_idx += 1
            continue

        curr_frame_idx += 1
        sampled_frame_count += 1

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        fh, fw = frame.shape[:2]
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            continue

        # Multi-pass face detection
        faces = []
        if face_cascade is not None:
            try:
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=3,
                    minSize=(30, 30),
                )
                if len(faces) == 0:
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=2,
                        minSize=(20, 20),
                    )
            except Exception:
                faces = []

        total_faces += len(faces)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            roi = gray[y:y + h, x:x + w]
        else:
            # Fallback center crop if face detector is missed
            ch, cw = int(fh * 0.6), int(fw * 0.6)
            cy, cx = (fh - ch) // 2, (fw - cw) // 2
            roi = gray[cy:cy + ch, cx:cx + cw]

        if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
            roi = gray

        try:
            roi_resized = cv2.resize(roi, (shape_x, shape_y))
        except Exception:
            continue

        roi_resized = roi_resized.astype(np.float32)
        if roi_resized.max() > 0:
            roi_resized /= roi_resized.max()
        roi_resized = np.reshape(roi_resized, (1, shape_x, shape_y, 1))

        preds = safe_predict(model, roi_resized)

        angry_0.append(float(preds[0][0]))
        disgust_1.append(float(preds[0][1]))
        fear_2.append(float(preds[0][2]))
        happy_3.append(float(preds[0][3]))
        sad_4.append(float(preds[0][4]))
        surprise_5.append(float(preds[0][5]))
        neutral_6.append(float(preds[0][6]))

        pred_idx = int(np.argmax(preds))
        predictions.append(str(pred_idx))

    cap.release()

    _save_stats(
        predictions,
        angry_0,
        disgust_1,
        fear_2,
        happy_3,
        sad_4,
        surprise_5,
        neutral_6,
    )

    emo_idx = _dominant_emotion_idx(predictions)

    def avg(lst):
        return float(np.mean(lst)) if lst else 0.0

    avg_faces = float(total_faces / sampled_frame_count) if sampled_frame_count > 0 else 0.0

    metrics = {
        "avg_faces": avg_faces,
        "angry": avg(angry_0),
        "happiness": avg(happy_3),
        "fear": avg(fear_2),
        "sadness": avg(sad_4),
        "surprise": avg(surprise_5),
        "disgust": avg(disgust_1),
        "neutral": avg(neutral_6),
    }

    last_rec_path = os.path.join(DB_DIR, "last_recording.txt")
    try:
        with open(last_rec_path, "w", encoding="utf-8") as f:
            rel_path = os.path.relpath(video_path, os.path.join(THIS_DIR, ".."))
            f.write(rel_path + "\n")
            f.write(str(emo_idx) + "\n")
    except Exception as e:
        print("[analyze_video_file] Failed to write last_recording.txt:", e)

    return emo_idx, metrics


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
_FRAME_MODEL = None
_FRAME_CASCADE = None

VAD_MAPPING = {
    "Angry":    {"V": 0.15, "A": 0.85, "D": 0.75},
    "Disgust":  {"V": 0.18, "A": 0.55, "D": 0.45},
    "Fear":     {"V": 0.20, "A": 0.82, "D": 0.20},
    "Happy":    {"V": 0.88, "A": 0.65, "D": 0.72},
    "Sad":      {"V": 0.22, "A": 0.28, "D": 0.25},
    "Surprise": {"V": 0.62, "A": 0.80, "D": 0.52},
    "Neutral":  {"V": 0.50, "A": 0.25, "D": 0.50},
}


def compute_vad_scores(preds_vec):
    """Computes continuous Valence, Arousal, and Dominance (VAD) scores from emotion probabilities."""
    v_score = 0.0
    a_score = 0.0
    d_score = 0.0

    for idx, prob in enumerate(preds_vec):
        label = EMOTION_LABELS[idx]
        mapping = VAD_MAPPING.get(label, {"V": 0.5, "A": 0.5, "D": 0.5})
        v_score += prob * mapping["V"]
        a_score += prob * mapping["A"]
        d_score += prob * mapping["D"]

    v_val = round(float(v_score), 2)
    a_val = round(float(a_score), 2)
    d_val = round(float(d_score), 2)

    v_lbl = "Positive" if v_val >= 0.60 else ("Negative" if v_val <= 0.40 else "Neutral")
    a_lbl = "Active" if a_val >= 0.60 else ("Calm" if a_val <= 0.35 else "Moderate")
    d_lbl = "Confident" if d_val >= 0.60 else ("Passive" if d_val <= 0.35 else "Balanced")

    return {
        "valence": v_val,
        "arousal": a_val,
        "dominance": d_val,
        "valence_label": v_lbl,
        "arousal_label": a_lbl,
        "dominance_label": d_lbl,
    }


def analyze_frame_bytes(image_bytes: bytes):
    global _FRAME_MODEL, _FRAME_CASCADE
    if _FRAME_MODEL is None:
        _FRAME_MODEL = load_video_model(VIDEO_MODEL_PATH)
    if _FRAME_CASCADE is None:
        _FRAME_CASCADE = load_cascade_classifier(CASCADE_PATH)

    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    fh, fw = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Multi-scale face detection
    faces = []
    if _FRAME_CASCADE is not None:
        try:
            faces = _FRAME_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces) == 0:
                faces = _FRAME_CASCADE.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=2, minSize=(20, 20))
        except Exception:
            faces = []

    if len(faces) > 0:
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        bbox = [int(x), int(y), int(w), int(h)]
    else:
        # Center crop fallback if face detector is missed
        ch, cw = int(fh * 0.6), int(fw * 0.6)
        cy, cx = (fh - ch) // 2, (fw - cw) // 2
        roi = gray[cy:cy+ch, cx:cx+cw]
        bbox = [cx, cy, cw, ch]

    if roi.size == 0 or roi.shape[0] == 0 or roi.shape[1] == 0:
        roi = gray

    roi_resized = cv2.resize(roi, (48, 48)).astype(np.float32)
    if roi_resized.max() > 0:
        roi_resized /= roi_resized.max()
    roi_resized = np.reshape(roi_resized, (1, 48, 48, 1))

    preds = safe_predict(_FRAME_MODEL, roi_resized)[0]
    probs = {EMOTION_LABELS[i]: round(float(preds[i]) * 100, 1) for i in range(7)}
    dominant_idx = int(np.argmax(preds))
    dominant_label = EMOTION_LABELS[dominant_idx]

    vad_data = compute_vad_scores(preds)

    return {
        "faces": len(faces),
        "dominant_emotion": dominant_label,
        "probabilities": probs,
        "vad": vad_data,
        "bbox": bbox,
        "frame_size": [fw, fh]
    }



