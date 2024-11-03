import streamlit as st
import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from PIL import Image
from deepface import DeepFace
import tempfile
from ultralytics.utils.plotting import colors

# Load the YOLO model
model = YOLO("yolov8n.pt").to("cpu")
names = model.model.names

# Directory to save faces
FACE_DB = "face_db"
os.makedirs(FACE_DB, exist_ok=True)

# Initialize tracking history
track_history = {}
last_seen = {}
id_to_object = {}

def get_person_name(face_image):
    try:
        result = DeepFace.find(img_path=face_image, db_path=FACE_DB, model_name="DeepID", silent=True, enforce_detection=False, threshold=0.5, distance_metric="cosine")
        if len(result) > 0:
            person_name = str(result[0]).split("/")[-2]
            return person_name
        return "unknown"
    except Exception:
        return "unknown"

def add_face(image, name):
    person_dir = os.path.join(FACE_DB, name)
    os.makedirs(person_dir, exist_ok=True)
    image_path = os.path.join(person_dir, f"{name}_{int(time.time())}.jpg")
    cv2.imwrite(image_path, image)

def track_image(image):
    current_time = time.time()
    results = model.track(image, persist=True, verbose=False, tracker="bytetrack.yaml")
    boxes = results[0].boxes.xyxy.cpu() if hasattr(results[0], 'boxes') else []
    tracked_objects = []

    person_count = {}

    if results[0].boxes.id is not None:
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, cls, track_id in zip(boxes, clss, track_ids):
            if track_id not in track_history:
                track_history[track_id] = []
                last_seen[track_id] = current_time

            track = track_history[track_id]
            track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if len(track) > 30:
                track.pop(0)

            object_name = names[int(cls)]
            id_to_object[track_id] = object_name

            if object_name == "person":
                face_region = image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                person_name = get_person_name(face_region)
                id_to_object[track_id] = person_name or "unknown"

                if person_name not in person_count:
                    person_count[person_name] = 0
                person_count[person_name] += 1

            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(image, (track[-1]), 7, colors(int(cls), True), -1)
            cv2.putText(image, f"{id_to_object[track_id]} {track_id}", (track[-1][0] + 10, track[-1][1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colors(int(cls), True), 2)
            cv2.polylines(image, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

            last_seen[track_id] = current_time

    for track_id, last_time in list(last_seen.items()):
        if current_time - last_time > 10:
            del last_seen[track_id]
            del track_history[track_id]
            del id_to_object[track_id]

    most_frequent_person = max(person_count, key=person_count.get, default="unknown")
    final_tracked_objects = [most_frequent_person] + [id_to_object[tid] for tid in id_to_object if id_to_object[tid] != most_frequent_person]

    return image, final_tracked_objects

def process_frame(frame):
    tracked_image, tracked_objects = track_image(frame)
    return tracked_image, tracked_objects

def video_survey(video_file):
    st.title("Video Surveillance with YOLOv8 and Face Recognition")

    temp_video_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(temp_video_path)
    
    stframe = st.empty()
    ret = True

    while ret:
        ret, frame = cap.read()
        if ret:
            processed_frame, tracked_objects = process_frame(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(processed_frame, channels="RGB", use_column_width=True)

    cap.release()
    
    st.success(f"Tracked  objects: {tracked_objects}")

# Streamlit UI
st.title("Surveillance System")

menu = st.sidebar.selectbox("Choose Module", ["Add Face", "Track Object", "Video Surveillance"])

if menu == "Add Face":
    st.header("Add Face to Database")
    option = st.selectbox("Choose Option", ["Upload Image", "Take Snapshot"])

    if option == "Upload Image":
        name = st.text_input("Enter Name:")
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None and st.button("Add Face"):
            image = Image.open(uploaded_file)
            image = np.array(image)
            add_face(image, name)
            st.success(f"Face added for {name}")

    elif option == "Take Snapshot":
        name = st.text_input("Enter Name:")
        snapshot = st.camera_input("Capture Image")

        if snapshot is not None and st.button("Add Face"):
            image = Image.open(snapshot)
            image = np.array(image)
            add_face(image, name)
            st.success(f"Face added for {name}")

elif menu == "Track Object":
    st.header("Track Object in Image")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)

        tracked_image, tracked_objects = track_image(image)
        st.image(tracked_image, caption="Tracked Image", use_column_width=True)
        st.write("Detected and Tracked Objects:", tracked_objects)

elif menu == "Video Surveillance":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        if st.button("Track Objects"):
            video_survey(uploaded_video)
