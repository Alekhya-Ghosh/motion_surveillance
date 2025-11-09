import streamlit as st
import tempfile
import os
import time

# Set page config first
st.set_page_config(
    page_title="Vehicle Tracking System",
    page_icon="🚗",
    layout="wide"
)

# Try to import dependencies with error handling
try:
    import cv2
    import numpy as np
    from collections import deque

    DEPENDENCIES_LOADED = True
except ImportError as e:
    st.error(f"❌ Missing dependency: {e}")
    DEPENDENCIES_LOADED = False

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

if DEPENDENCIES_LOADED:
    # Your existing VehicleTracker and other classes here
    # (Copy all your class definitions from previous app.py)

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

        # ... (include all your existing methods)


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
                # ... (rest of your brand prediction code)
                return "Toyota"  # Simplified for example
            except:
                return "Unknown"


    def detect_vehicles_motion(frame, background_subtractor, min_area):
        """Simple motion detection"""
        try:
            fgmask = background_subtractor.apply(frame)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

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


    def main():
        st.markdown('<h1 class="main-header">🚗 Vehicle Tracking System</h1>', unsafe_allow_html=True)

        # Sidebar
        with st.sidebar:
            st.header("⚙️ Settings")
            video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            min_area = st.slider("Minimum Vehicle Area", 500, 3000, 1500)

        if video_file is not None:
            st.success("✅ Video uploaded successfully!")

            # Process video
            with st.spinner("🔄 Processing video..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(video_file.read())
                        video_path = tmp_file.name

                    cap = cv2.VideoCapture(video_path)
                    tracker = VehicleTracker()
                    brand_classifier = CarBrandClassifier()
                    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

                    stframe = st.empty()
                    progress_bar = st.progress(0)

                    frame_count = 0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame = cv2.resize(frame, (800, 600))
                        detections, _ = detect_vehicles_motion(frame, fgbg, min_area)
                        brands = [brand_classifier.predict_brand(frame) for _ in detections]
                        vehicles = tracker.update(detections, brands, frame)

                        # Draw counting line
                        height, width = frame.shape[:2]
                        counting_line_y = height // 2
                        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (255, 0, 0), 2)

                        # Draw vehicles
                        for vehicle_id, vehicle_data in vehicles.items():
                            x1, y1, x2, y2 = vehicle_data['bbox']
                            color = (0, 255, 0) if not vehicle_data['counted'] else (0, 0, 255)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(frame, f"ID:{vehicle_id}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                        # Display frame
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if frame_count % 5 == 0:
                            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

                        # Update progress
                        frame_count += 1
                        progress_bar.progress(min(frame_count / total_frames, 1.0))

                    cap.release()
                    os.unlink(video_path)

                    # Show results
                    st.success(f"✅ Processing completed! Counted {tracker.vehicle_count} vehicles")

                except Exception as e:
                    st.error(f"❌ Error processing video: {e}")

        else:
            st.info("👆 Please upload a video file to start tracking")
            st.write("""
            ### 🎯 Features:
            - Vehicle counting
            - Motion detection
            - Real-time tracking
            - Speed estimation
            """)


    if __name__ == "__main__":
        main()

else:
    st.error("""
    ❌ Dependencies not loaded!

    Please check your requirements.txt file and make sure it contains:
    ```
    streamlit==1.28.0
    opencv-python-headless==4.8.1.78
    numpy==1.24.3
    pillow==10.0.0
    ```

    Then push the changes to GitHub.
    """)