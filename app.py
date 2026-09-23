
import os
import math
import zipfile
import tempfile
import subprocess
import urllib.request

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="ParkVision AI",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 ParkVision AI")
st.caption("Fall and activity detection using MediaPipe Pose + TensorFlow")


# ============================================================
# CONSTANTS
# ============================================================

CLASSES = ["fall", "walking", "sitting", "standing", "normal"]
POSE_CLASSES = ["fall", "walking", "sitting", "standing"]

IMG_SIZE = (224, 224)

MODEL_DIR = "models"
POSE_MODEL_PATH = os.path.join(MODEL_DIR, "pose_model.keras")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "activity_cnn.keras")
POSE_TASK_PATH = os.path.join(MODEL_DIR, "pose_landmarker.task")

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DOWNLOAD MEDIAPIPE MODEL
# ============================================================

@st.cache_resource
def load_pose_detector():
    if not os.path.exists(POSE_TASK_PATH):
        url = (
            "https://storage.googleapis.com/"
            "mediapipe-models/pose_landmarker/"
            "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
        )
        urllib.request.urlretrieve(url, POSE_TASK_PATH)

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=POSE_TASK_PATH
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )

    return vision.PoseLandmarker.create_from_options(options)


pose = load_pose_detector()


# ============================================================
# POSE FUNCTIONS
# ============================================================

POSE_LINKS = [
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
]


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = []

    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    for a, b in POSE_LINKS:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)

    return frame


def knee_bend(hip, knee, ankle):
    ax, ay = hip.x - knee.x, hip.y - knee.y
    bx, by = ankle.x - knee.x, ankle.y - knee.y

    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)

    if na * nb == 0:
        return 180.0

    cosine = max(
        -1.0,
        min(1.0, (ax * bx + ay * by) / (na * nb))
    )

    return math.degrees(math.acos(cosine))


def body_features(landmarks):
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2

    knee_y = (landmarks[25].y + landmarks[26].y) / 2

    ankle_gap = abs(
        landmarks[27].x - landmarks[28].x
    )

    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y

    tilt = abs(
        math.degrees(math.atan2(dx, dy))
    ) if (dx or dy) else 0.0

    bend = (
        knee_bend(
            landmarks[23],
            landmarks[25],
            landmarks[27]
        )
        +
        knee_bend(
            landmarks[24],
            landmarks[26],
            landmarks[28]
        )
    ) / 2

    torso = hip_y - shoulder_y
    thigh = knee_y - hip_y

    return [tilt, bend, ankle_gap, torso, thigh]


def body_activity(landmarks):
    tilt, bend, ankle_gap, torso, thigh = body_features(
        landmarks
    )

    if tilt > 55 and bend > 110:
        return "fall"

    if bend < 130:
        return "sitting"

    if ankle_gap > 0.14 and bend > 150:
        return "walking"

    return "standing"


# ============================================================
# FALL ANNOTATION FUNCTIONS
# ============================================================

def read_fall_window(annotation_path):
    try:
        with open(annotation_path) as f:
            lines = [line.strip() for line in f if line.strip()]

        if (
            len(lines) >= 2
            and lines[0].isdigit()
            and lines[1].isdigit()
        ):
            return int(lines[0]), int(lines[1])

    except Exception:
        pass

    return 0, 0


def find_pairs(root):
    videos = {}
    notes = {}

    for dirpath, _, files in os.walk(root):
        if "sample_data" in dirpath:
            continue

        for name in files:
            path = os.path.join(dirpath, name)
            base, ext = os.path.splitext(name)
            ext = ext.lower()

            if ext in (".avi", ".mp4", ".mov", ".mkv"):
                videos[base] = path

            elif (
                ext == ".txt"
                and "annotation" in dirpath.lower()
            ):
                notes[base] = path

    return [
        (videos[base], notes[base])
        for base in videos
        if base in notes
    ]


# ============================================================
# FFMPEG FRAME READING
# ============================================================

def frame_count(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=nb_frames",
                "-of", "csv=p=0",
                video_path
            ],
            capture_output=True,
            text=True
        )

        text = result.stdout.strip().split("\n")[0]

        if text.isdigit() and int(text) > 1:
            return int(text)

    except Exception:
        pass

    return 0


def read_frame(video_path, frame_number):
    output_path = os.path.join(
        tempfile.gettempdir(),
        "parkvision_frame.jpg"
    )

    if os.path.exists(output_path):
        os.remove(output_path)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-i", video_path,
                "-vf",
                f"select=eq(n\\,{frame_number - 1})",
                "-vframes", "1",
                output_path
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        if os.path.exists(output_path):
            return cv2.imread(output_path)

    except Exception:
        return None

    return None


# ============================================================
# IMAGE PREDICTION
# ============================================================

def analyze_image(frame_bgr):
    rgb = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb)
    )

    result = pose.detect(mp_image)

    if not result.pose_landmarks:
        return frame_bgr, "normal", 0.0

    landmarks = result.pose_landmarks[0]

    frame_bgr = draw_pose(
        frame_bgr,
        landmarks
    )

    label = body_activity(landmarks)

    # If a trained pose model exists, use it as an additional signal.
    if os.path.exists(POSE_MODEL_PATH):
        try:
            pose_model = load_model(
                POSE_MODEL_PATH,
                compile=False
            )

            features = np.array(
                [body_features(landmarks)],
                dtype="float32"
            )

            probabilities = pose_model.predict(
                features,
                verbose=0
            )[0]

            model_label = POSE_CLASSES[
                int(np.argmax(probabilities))
            ]

            confidence = float(
                probabilities.max()
            )

            # Keep the original rule-based fallback
            # when the model and rules disagree.
            if model_label == label:
                return frame_bgr, model_label, confidence

            rule_index = POSE_CLASSES.index(label)

            return (
                frame_bgr,
                label,
                float(probabilities[rule_index])
            )

        except Exception:
            pass

    # Rule-based confidence when no trained model exists.
    confidence = 0.90 if label == "fall" else 0.75

    return frame_bgr, label, confidence


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def analyze_video(video_path, max_frames=20):
    total = frame_count(video_path)

    if total <= 1:
        return None, "Could not read video."

    frame_numbers = np.linspace(
        1,
        total,
        min(max_frames, total),
        dtype=int
    )

    results = []

    for number in frame_numbers:
        frame = read_frame(
            video_path,
            int(number)
        )

        if frame is None:
            continue

        processed, label, confidence = analyze_image(
            frame
        )

        results.append({
            "frame": int(number),
            "activity": label,
            "confidence": round(
                confidence * 100,
                2
            ),
            "image": processed
        })

    if not results:
        return None, "No readable frames found."

    return results, None


# ============================================================
# DATASET EXTRACTION
# ============================================================

def extract_uploaded_dataset(uploaded_file):
    dataset_dir = os.path.join(
        tempfile.gettempdir(),
        "parkvision_dataset"
    )

    if os.path.exists(dataset_dir):
        import shutil
        shutil.rmtree(dataset_dir)

    os.makedirs(dataset_dir)

    zip_path = os.path.join(
        dataset_dir,
        "dataset.zip"
    )

    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dataset_dir)

    return dataset_dir


# ============================================================
# TRAIN POSE DATASET
# ============================================================

def create_pose_dataset(root, max_videos=40, max_per_class=30):
    pairs = find_pairs(root)

    features = []
    labels = []

    counts = {
        name: 0
        for name in POSE_CLASSES
    }

    for video_path, ann_path in pairs[:max_videos]:

        if all(
            counts[name] >= max_per_class
            for name in POSE_CLASSES
        ):
            break

        fall_start, fall_end = read_fall_window(
            ann_path
        )

        total = frame_count(video_path)

        if total <= 1:
            total = max(
                fall_end + 5,
                30
            )

        picks = [
            max(1, total // 10),
            max(1, total // 4),
            max(1, total // 2)
        ]

        if fall_start > 0 and fall_end >= fall_start:
            picks.append(
                max(
                    1,
                    (fall_start + fall_end) // 2
                )
            )

        for frame_number in picks:

            frame = read_frame(
                video_path,
                frame_number
            )

            if frame is None:
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            result = pose.detect(mp_image)

            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks[0]

            label = body_activity(
                landmarks
            )

            if counts[label] >= max_per_class:
                continue

            features.append(
                body_features(landmarks)
            )

            labels.append(
                POSE_CLASSES.index(label)
            )

            counts[label] += 1

    return (
        np.array(features, dtype="float32"),
        np.array(labels),
        counts
    )


# ============================================================
# TRAIN POSE MODEL
# ============================================================

def train_pose_model(pose_X, pose_y, epochs=40):

    if len(pose_X) < 4:
        raise ValueError(
            "Not enough pose samples to train."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        pose_X,
        pose_y,
        test_size=0.30,
        random_state=42
    )

    model = Sequential([
        Dense(
            16,
            activation="relu",
            input_shape=(5,)
        ),
        Dense(
            len(POSE_CLASSES),
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=16,
        verbose=0,
        validation_split=0.2
    )

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    model.save(POSE_MODEL_PATH)

    return model, history, accuracy


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.header("ParkVision Controls")

page = st.sidebar.radio(
    "Choose a function",
    [
        "🏠 Home",
        "📷 Image Detection",
        "🎥 Video Detection",
        "🧠 Train Pose Model"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.subheader("AI Activity & Fall Detection")

    st.write(
        """
        This Streamlit version converts the original Colab
        ParkVision workflow into an interactive application.

        **Supported activities**
        """
    )

    cols = st.columns(5)

    for col, name in zip(cols, CLASSES):
        col.metric(
            name.capitalize(),
            "✓"
        )

    st.info(
        "Upload an image or video to analyse body posture. "
        "The system uses MediaPipe Pose and the trained "
        "TensorFlow pose classifier when available."
    )

    st.markdown(
        """
        ### Original workflow converted

        1. Dataset / annotations
        2. MediaPipe pose detection
        3. Body-feature extraction
        4. Pose model training
        5. Image detection
        6. Video detection
        """
    )


# ============================================================
# IMAGE DETECTION
# ============================================================

elif page == "📷 Image Detection":

    st.subheader("📷 Upload an Image")

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:

        file_bytes = np.asarray(
            bytearray(uploaded_image.read()),
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            file_bytes,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            st.error("Could not read the image.")
        else:

            processed, label, confidence = analyze_image(
                frame
            )

            processed_rgb = cv2.cvtColor(
                processed,
                cv2.COLOR_BGR2RGB
            )

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Original",
                    use_container_width=True
                )

            with col2:
                st.image(
                    processed_rgb,
                    caption="Pose Detection",
                    use_container_width=True
                )

            st.divider()

            if label == "fall":
                st.error(
                    f"🚨 FALL DETECTED — "
                    f"{confidence * 100:.1f}% confidence"
                )
            else:
                st.success(
                    f"Activity: **{label.upper()}**  \n"
                    f"Confidence: **{confidence * 100:.1f}%**"
                )


# ============================================================
# VIDEO DETECTION
# ============================================================

elif page == "🎥 Video Detection":

    st.subheader("🎥 Upload a Video")

    uploaded_video = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    max_frames = st.slider(
        "Frames to analyse",
        min_value=5,
        max_value=30,
        value=12
    )

    if uploaded_video:

        video_path = os.path.join(
            tempfile.gettempdir(),
            uploaded_video.name
        )

        with open(video_path, "wb") as f:
            f.write(
                uploaded_video.getbuffer()
            )

        st.video(
            uploaded_video
        )

        if st.button(
            "🔍 Analyse Video",
            type="primary"
        ):

            with st.spinner(
                "Analysing video..."
            ):

                results, error = analyze_video(
                    video_path,
                    max_frames=max_frames
                )

            if error:
                st.error(error)

            else:

                df = pd.DataFrame([
                    {
                        "Frame": item["frame"],
                        "Activity": item["activity"],
                        "Confidence (%)": item["confidence"]
                    }
                    for item in results
                ])

                st.dataframe(
                    df,
                    use_container_width=True
                )

                falls = df[
                    df["Activity"] == "fall"
                ]

                if len(falls) > 0:
                    st.error(
                        f"🚨 Fall detected in "
                        f"{len(falls)} analysed frame(s)."
                    )
                else:
                    st.success(
                        "No fall detected in the analysed frames."
                    )

                st.subheader(
                    "Sample analysed frames"
                )

                cols = st.columns(
                    min(3, len(results))
                )

                for i, item in enumerate(results):
                    with cols[i % len(cols)]:
                        image_rgb = cv2.cvtColor(
                            item["image"],
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            image_rgb,
                            caption=(
                                f"Frame {item['frame']} — "
                                f"{item['activity']}"
                            ),
                            use_container_width=True
                        )


# ============================================================
# TRAINING
# ============================================================

elif page == "🧠 Train Pose Model":

    st.subheader("🧠 Train the Pose Model")

    st.write(
        """
        Upload the ZIP version of your dataset.

        The ZIP should contain the videos and their annotation
        `.txt` files. The application searches recursively for
        matching video/annotation pairs.
        """
    )

    dataset_zip = st.file_uploader(
        "Upload dataset ZIP",
        type=["zip"]
    )

    max_videos = st.slider(
        "Maximum videos",
        5,
        100,
        40
    )

    max_per_class = st.slider(
        "Maximum samples per class",
        5,
        100,
        30
    )

    epochs = st.slider(
        "Training epochs",
        5,
        100,
        40
    )

    if dataset_zip:

        if st.button(
            "🚀 Prepare Dataset & Train",
            type="primary"
        ):

            with st.spinner(
                "Extracting dataset..."
            ):
                dataset_root = extract_uploaded_dataset(
                    dataset_zip
                )

            st.success(
                "Dataset extracted successfully."
            )

            with st.spinner(
                "Creating pose features..."
            ):

                pose_X, pose_y, counts = (
                    create_pose_dataset(
                        dataset_root,
                        max_videos=max_videos,
                        max_per_class=max_per_class
                    )
                )

            st.write(
                "Samples per class:"
            )

            st.json(counts)

            if len(pose_X) < 4:
                st.error(
                    "Not enough valid pose samples were found."
                )
            else:

                st.write(
                    f"Feature matrix shape: "
                    f"`{pose_X.shape}`"
                )

                with st.spinner(
                    "Training TensorFlow model..."
                ):

                    try:
                        model, history, accuracy = (
                            train_pose_model(
                                pose_X,
                                pose_y,
                                epochs=epochs
                            )
                        )

                        st.success(
                            f"Training completed. "
                            f"Test accuracy: "
                            f"{accuracy * 100:.2f}%"
                        )

                        chart = pd.DataFrame({
                            "Epoch": range(
                                1,
                                len(
                                    history.history[
                                        "accuracy"
                                    ]
                                ) + 1
                            ),
                            "Training Accuracy":
                                history.history[
                                    "accuracy"
                                ],
                            "Validation Accuracy":
                                history.history[
                                    "val_accuracy"
                                ]
                        })

                        st.line_chart(
                            chart.set_index("Epoch")
                        )

                        st.info(
                            "The trained model was saved as "
                            "`models/pose_model.keras`."
                        )

                    except Exception as e:
                        st.exception(e)
