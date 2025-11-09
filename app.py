import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import deque
import time

# Set page config
st.set_page_config(
    page_title="Vehicle Tracking System",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)


class VehicleTracker:
    def __init__(self, max_disappeared=50, max_distance=100):
        self.next_vehicle_id = 0
        self.vehicles = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self.positions_history = {}
        self.timestamps = {}
        self.counting_line_y = None
        self.counted_vehicles = set()
        self.vehicle_count = 0
        self.brand_count = {}
        self.direction_history = {}

    def update(self, detections, brands, frame):
        if self.counting_line_y is None:
            height = frame.shape[0]
            self.counting_line_y = height // 2

        if len(self.vehicles) == 0:
            for bbox, brand in zip(detections, brands):
                self.register_vehicle(bbox, brand)
            return self.vehicles

        updated_vehicles = {}
        used_indices = set()

        for vehicle_id, vehicle_data in self.vehicles.items():
            if len(detections) == 0:
                self.disappeared[vehicle_id] += 1
                if self.disappeared[vehicle_id] > self.max_disappeared:
                    continue
                updated_vehicles[vehicle_id] = vehicle_data
                continue

            min_distance = float('inf')
            best_idx = -1
            current_center = vehicle_data['center']

            for i, bbox in enumerate(detections):
                if i in used_indices:
                    continue

                x1, y1, x2, y2 = bbox
                detection_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                d = np.sqrt((current_center[0] - detection_center[0]) ** 2 +
                            (current_center[1] - detection_center[1]) ** 2)

                if d < min_distance and d < self.max_distance:
                    min_distance = d
                    best_idx = i

            if best_idx != -1:
                x1, y1, x2, y2 = detections[best_idx]
                new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                brand = brands[best_idx]

                if vehicle_id in self.positions_history:
                    self.positions_history[vehicle_id].append(new_center)
                    if len(self.positions_history[vehicle_id]) > 10:
                        self.positions_history[vehicle_id].pop(0)
                else:
                    self.positions_history[vehicle_id] = [new_center]

                self.timestamps[vehicle_id] = time.time()
                self.calculate_direction(vehicle_id, new_center)
                self.check_counting_line(vehicle_id, vehicle_data['center'], new_center)

                updated_vehicles[vehicle_id] = {
                    'bbox': detections[best_idx],
                    'center': new_center,
                    'brand': brand,
                    'counted': vehicle_data.get('counted', False),
                    'direction': self.direction_history.get(vehicle_id, 'Unknown'),
                    'speed': self.calculate_speed(vehicle_id)
                }

                used_indices.add(best_idx)
                self.disappeared[vehicle_id] = 0
            else:
                self.disappeared[vehicle_id] += 1
                if self.disappeared[vehicle_id] <= self.max_disappeared:
                    updated_vehicles[vehicle_id] = vehicle_data

        for i, (bbox, brand) in enumerate(zip(detections, brands)):
            if i not in used_indices:
                self.register_vehicle(bbox, brand, updated_vehicles)

        self.vehicles = updated_vehicles
        return self.vehicles

    def register_vehicle(self, bbox, brand, vehicle_dict=None):
        x1, y1, x2, y2 = bbox
        center = ((x1 + x2) / 2, (y1 + y2) / 2)

        vehicle_id = self.next_vehicle_id
        self.next_vehicle_id += 1

        vehicle_data = {
            'bbox': bbox,
            'center': center,
            'brand': brand,
            'counted': False,
            'direction': 'Unknown',
            'speed': 0
        }

        self.positions_history[vehicle_id] = [center]
        self.timestamps[vehicle_id] = time.time()
        self.disappeared[vehicle_id] = 0
        self.direction_history[vehicle_id] = 'Unknown'

        if vehicle_dict is not None:
            vehicle_dict[vehicle_id] = vehicle_data
        else:
            self.vehicles[vehicle_id] = vehicle_data

    def calculate_direction(self, vehicle_id, new_center):
        if vehicle_id not in self.positions_history:
            return 'Unknown'

        positions = self.positions_history[vehicle_id]
        if len(positions) < 2:
            return 'Unknown'

        old_center = positions[0]
        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]

        if abs(dx) > abs(dy):
            direction = "Right" if dx > 0 else "Left"
        else:
            direction = "Down" if dy > 0 else "Up"

        self.direction_history[vehicle_id] = direction
        return direction

    def calculate_speed(self, vehicle_id):
        if vehicle_id not in self.positions_history:
            return 0

        positions = self.positions_history[vehicle_id]
        if len(positions) < 2:
            return 0

        total_distance = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i - 1][0]
            dy = positions[i][1] - positions[i - 1][1]
            total_distance += np.sqrt(dx ** 2 + dy ** 2)

        if vehicle_id not in self.timestamps:
            return 0

        current_time = time.time()
        time_elapsed = current_time - self.timestamps[vehicle_id]

        if time_elapsed == 0:
            return 0

        speed_pps = total_distance / time_elapsed
        pixels_per_meter = 10
        speed_kmh = (speed_pps / pixels_per_meter) * 3.6

        return speed_kmh

    def check_counting_line(self, vehicle_id, old_center, new_center):
        if vehicle_id in self.counted_vehicles:
            return

        old_y = old_center[1]
        new_y = new_center[1]

        if old_y <= self.counting_line_y and new_y > self.counting_line_y:
            self.vehicle_count += 1
            self.counted_vehicles.add(vehicle_id)

            brand = self.vehicles[vehicle_id]['brand']
            if brand in self.brand_count:
                self.brand_count[brand] += 1
            else:
                self.brand_count[brand] = 1


class CarBrandClassifier:
    def __init__(self):
        self.color_based_brands = {
            'red': ['Toyota', 'Honda', 'Ford'],
            'blue': ['BMW', 'Hyundai', 'Ford'],
            'white': ['Toyota', 'Honda', 'Mercedes'],
            'black': ['BMW', 'Audi', 'Mercedes'],
            'silver': ['Toyota', 'Honda', 'Nissan'],
            'gray': ['BMW', 'Audi', 'Volkswagen']
        }

    def predict_brand(self, car_roi):
        if car_roi.size == 0:
            return "Unknown"

        try:
            hsv = cv2.cvtColor(car_roi, cv2.COLOR_BGR2HSV)
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])

            if avg_saturation < 50 or avg_value < 50:
                if avg_value > 150:
                    color = 'white'
                elif avg_value > 100:
                    color = 'silver'
                else:
                    color = 'black'
            elif avg_hue < 10 or avg_hue > 170:
                color = 'red'
            elif 100 <= avg_hue <= 130:
                color = 'blue'
            else:
                color = 'gray'

            return np.random.choice(self.color_based_brands[color])
        except:
            return "Unknown"


def detect_vehicles_motion(frame, background_subtractor, min_area):
    """Detect vehicles using motion detection"""
    try:
        fgmask = background_subtractor.apply(frame)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, kernel, iterations=2)

        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                detections.append((x, y, x + w, y + h))

        return detections, fgmask
    except Exception as e:
        st.error(f"Detection error: {e}")
        return [], None


def process_video(video_file, tracker, brand_classifier, min_area):
    """Process uploaded video file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

        stframe = st.empty()
        progress_bar = st.progress(0)
        stats_placeholder = st.empty()

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 600))
            height, width = frame.shape[:2]

            detections, motion_mask = detect_vehicles_motion(frame, fgbg, min_area)
            brands = [brand_classifier.predict_brand(frame) for _ in detections]

            vehicles = tracker.update(detections, brands, frame)

            counting_line_y = height // 2
            cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (255, 0, 0), 2)

            for vehicle_id, vehicle_data in vehicles.items():
                x1, y1, x2, y2 = vehicle_data['bbox']
                center = vehicle_data['center']
                speed = vehicle_data['speed']
                direction = vehicle_data['direction']
                brand = vehicle_data['brand']

                color = (0, 255, 0) if not vehicle_data['counted'] else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (int(center[0]), int(center[1])), 4, (255, 0, 0), -1)

                info_text = f"ID:{vehicle_id} {brand}"
                speed_text = f"Speed: {speed:.1f} km/h"
                direction_text = f"Dir: {direction}"
                status = "COUNTED" if vehicle_data['counted'] else "TRACKING"

                cv2.putText(frame, info_text, (x1, y1 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, speed_text, (x1, y1 - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, direction_text, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, status, (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_count += 1
            progress = min(frame_count / total_frames, 1.0)
            progress_bar.progress(progress)

            if frame_count % 10 == 0:
                with stats_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Vehicles", tracker.vehicle_count)
                    with col2:
                        st.metric("Currently Tracking", len(vehicles))
                    with col3:
                        st.metric("Progress", f"{progress:.1%}")

            if frame_count % 5 == 0:
                stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        os.unlink(video_path)
        return tracker

    except Exception as e:
        st.error(f"Error processing video: {e}")
        return tracker


def main():
    st.markdown('<h1 class="main-header">🚗 Vehicle Tracking System</h1>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Settings")
        video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
        min_area = st.slider("Minimum Vehicle Area", 500, 3000, 1500)
        max_distance = st.slider("Tracking Distance", 50, 200, 100)

        st.info("""
        **Features:**
        - Vehicle counting
        - Speed estimation
        - Direction detection
        - Brand recognition
        """)

    tracker = VehicleTracker(max_distance=max_distance)
    brand_classifier = CarBrandClassifier()

    if video_file is not None:
        st.success("✅ Video uploaded successfully!")

        with st.spinner("🔄 Processing video... This may take a while."):
            tracker = process_video(video_file, tracker, brand_classifier, min_area)

        st.success("✅ Processing completed!")

        st.header("📊 Results")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Vehicle Statistics")
            st.metric("Total Vehicles Counted", tracker.vehicle_count)
            st.metric("Unique Vehicles Tracked", tracker.next_vehicle_id)

        with col2:
            st.subheader("Brand Distribution")
            if tracker.brand_count:
                for brand, count in tracker.brand_count.items():
                    st.write(f"**{brand}**: {count} vehicles")
            else:
                st.info("No brand data available")

    else:
        st.info("👆 Please upload a video file to start vehicle tracking")
        st.write("""
        ### 🎯 How to use:
        1. Upload a traffic video (MP4, AVI, MOV, MKV)
        2. Adjust detection settings in sidebar
        3. Watch real-time vehicle tracking
        4. View analytics and results
        """)


if __name__ == "__main__":
    main()