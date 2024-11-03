import streamlit as st
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import time, datetime
from ultralytics import YOLO
from PIL import Image
from deepface import DeepFace
import tempfile
from ultralytics.utils.plotting import colors

# Import Utils
from utils.log_data import *
from utils.db_query import  *

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
        # Change the threshold value for the Better Accuracy of the Face Recognition
        result = DeepFace.find(img_path=face_image, db_path=FACE_DB, model_name="DeepID", silent=True, enforce_detection=False, threshold=0.01, distance_metric="cosine")
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

def video_survey(video_file, location):
    temp_video_path = tempfile.mktemp(suffix=".mp4")
    with open(temp_video_path, "wb") as f:
        f.write(video_file.getbuffer())

    cap = cv2.VideoCapture(temp_video_path)
    
    # Create a temp file for saving the result video
    result_path = "result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(result_path, fourcc, 30, (width, height))

    stframe = st.empty()
    ret = True
    
    start_time = time.time()
    person_found = None  # Initialize to track if a person is found

    while ret:
        ret, frame = cap.read()
        if ret:
            processed_frame, tracked_objects = process_frame(frame)
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(processed_frame, channels="RGB", use_column_width=True)

            # Save the frame to the result video
            out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))

            # Check if a person is tracked
            if tracked_objects:
                person_found = tracked_objects[0]  # Assume the first tracked object is the person

    cap.release()
    out.release()  # Finalize the video file
    
    end_time = time.time()
    
    if person_found:
        log_to_csv(person_found, start_time, end_time, location)
    
    # # Provide a download link for the result video
    # with open(result_path, "rb") as video_file:
    #     btn = st.download_button(label="Download Result Video", data=video_file, file_name="result.mp4", mime="video/mp4")
    
    st.success(f"Tracked objects: {tracked_objects}")

    return person_found, location



# Streamlit UI
st.title("Surveillance System")

menu = st.sidebar.selectbox("Choose Module", ["Add Face", "Track Object", "Video Surveillance", "Surveillance Query"])

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
    st.header("Upload Videos for Surveillance")

    uploaded_videos = []
    results = []  # List to store results

    location_labels = ["Outside", "Office Front Side", "Cabin Ceiling View"]

    for i, label in enumerate(location_labels):
        uploaded_video = st.file_uploader(f"Upload Video - {label}", type=["mp4", "avi", "mov"], key=f"video_{i}")
        if uploaded_video is not None:
            uploaded_videos.append((uploaded_video, label))

    if st.button("Track Objects"):
        if len(uploaded_videos) == 3:
            for video, location in uploaded_videos:
                person_found, loc = video_survey(video, location)
                results.append((person_found, loc))  # Store the result

            # Now plot the results
            if results:
                names = []
                locations = []
                for person, loc in results:
                    if person:  # If a person was found
                        names.append(person)
                        locations.append(loc)

                # Create a DataFrame for plotting
                plot_df = pd.DataFrame({
                    "Location": locations,
                    "Person": names
                })

                # # Plotting
                # plt.figure(figsize=(10, 6))
                # plt.scatter(plot_df["Location"], plot_df["Person"], color='blue', s=100)
                # plt.title("Person Found at Locations")
                # plt.xlabel("Locations")
                # plt.ylabel("Persons Found")

                # # Draw lines
                # if len(locations) > 1:
                #     plt.plot(plot_df["Location"], plot_df["Person"], color='orange', linewidth=2)

                # st.pyplot(plt)  # Display the plot

elif menu == "Surveillance Query":
    st.header("Query Surveillance Data")
    
    df = load_and_display_database()

    query_type = st.selectbox("Select Query Type", ["Date Query (Day/Month)", "Person Location Query", "Path Generation"])
    
    if query_type == "Date Query (Day/Month)":
        query_date = st.date_input("Select Date")
        if st.button("Search by Date"):
            result = query_database(query_type="date_query", query_date=query_date)
            if isinstance(result, pd.DataFrame):
                st.write(result)
            else:
                st.write(result)

    elif query_type == "Person Location Query":
        person_name = st.text_input("Enter Person Name")
        location = st.text_input("Enter Location")
        if st.button("Search by Person and Location"):
            result = query_database(query_type="person_location_query", person_name=person_name, location=location)
            if isinstance(result, pd.DataFrame):
                st.write(result)
            else:
                st.write(result)
                
    elif query_type == "Path Generation":
        person_name = st.text_input("Enter Person Name")
        query_date = st.date_input("Select Date")
        if st.button("Generate Path"):
            generate_person_path(person_name, query_date)
