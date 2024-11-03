# Real-Time Surveillance System

## Overview
The Real-Time Surveillance System is a web application designed to enhance security and monitoring through advanced video processing, face recognition, and object tracking capabilities. This application utilizes state-of-the-art AI models to provide a comprehensive solution for surveillance needs.

![FlowChart](https://github.com/user-attachments/assets/04f12101-86aa-4cbe-a6f3-936f622ecc2a)


## Features
- **Face Database Management**: Easily add and manage faces for future recognition.
- **Object Tracking**: Real-time tracking of individuals using YOLO (You Only Look Once) model.
- **Video Surveillance**: Upload videos to detect and track individuals with detailed logging.
- **Surveillance Query**: Query historical data to generate reports on detected individuals and their activities.

## Technologies Used
- **Streamlit**: Web interface framework for easy interaction.
- **YOLOv8**: Model for real-time object detection.
- **DeepFace**: Library for facial recognition.
- **OpenCV**: For image and video processing.
- **Pandas**: For data manipulation and storage.
- **NumPy**: For numerical operations.
- **Matplotlib**: For plotting results.

## Installation

### Prerequisites
- Python 3.7 or later
- pip (Python package installer)

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/surveillance-system.git
   cd surveillance-system
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
3. Run the application:
   ```bash
   streamlit run app.py
