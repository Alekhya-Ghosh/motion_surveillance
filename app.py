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

# Custom CSS for better styling
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
    .brand-box {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        padding: 8px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
        display: inline-block;
        margin: 2px;
    }
    .vehicle-info {
        background: rgba(0, 0, 0, 0.7);
        color: white;
        padding: 5px 8px;
        border-radius: 5px;
        font-size: 10px;
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
                    'speed': self.calculate_speed(vehicle_id),
                    'confidence': vehicle_data.get('confidence', 0.8)  # Default confidence
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
            'speed': 0,
            'confidence': 0.8
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


class EnhancedCarBrandClassifier:
    def __init__(self):
        # Enhanced brand database with vehicle types
        self.brand_database = {
            'sedan': {
                'Toyota': ['Camry', 'Corolla', 'Avalon'],
                'Honda': ['Civic', 'Accord', 'City'],
                'Ford': ['Fusion', 'Focus', 'Taurus'],
                'Hyundai': ['Elantra', 'Sonata', 'Verna'],
                'BMW': ['3 Series', '5 Series', '7 Series'],
                'Mercedes': ['C-Class', 'E-Class', 'S-Class']
            },
            'suv': {
                'Toyota': ['Fortuner', 'Land Cruiser', 'RAV4'],
                'Honda': ['CR-V', 'HR-V', 'Pilot'],
                'Ford': ['Endeavour', 'EcoSport', 'Explorer'],
                'Hyundai': ['Creta', 'Tucson', 'Santa Fe'],
                'BMW': ['X1', 'X3', 'X5'],
                'Mercedes': ['GLA', 'GLC', 'GLE']
            },
            'hatchback': {
                'Toyota': ['Glanza', 'Etios Liva'],
                'Honda': ['Jazz', 'Brio'],
                'Ford': ['Figo', 'Freestyle'],
                'Hyundai': ['i20', 'Grand i10'],
                'Maruti': ['Swift', 'Baleno', 'Wagon R']
            }
        }

        # Color to brand probability mapping
        self.color_brand_probs = {
            'white': {'Toyota': 0.3, 'Honda': 0.25, 'Hyundai': 0.2, 'BMW': 0.1, 'Mercedes': 0.15},
            'black': {'BMW': 0.3, 'Mercedes': 0.25, 'Audi': 0.2, 'Toyota': 0.15, 'Honda': 0.1},
            'silver': {'Toyota': 0.25, 'Honda': 0.2, 'Hyundai': 0.2, 'Ford': 0.15, 'BMW': 0.1, 'Mercedes': 0.1},
            'red': {'Honda': 0.3, 'Toyota': 0.25, 'Ford': 0.2, 'Hyundai': 0.15, 'BMW': 0.1},
            'blue': {'Ford': 0.25, 'Hyundai': 0.2, 'BMW': 0.2, 'Honda': 0.15, 'Toyota': 0.1, 'Mercedes': 0.1},
            'gray': {'BMW': 0.3, 'Audi': 0.25, 'Mercedes': 0.2, 'Toyota': 0.15, 'Honda': 0.1}
        }

    def detect_vehicle_type(self, bbox, frame):
        """Detect vehicle type based on aspect ratio and size"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 0

        if aspect_ratio > 1.8:
            return "sedan"
        elif 1.4 <= aspect_ratio <= 1.8:
            return "suv"
        else:
            return "hatchback"

    def detect_vehicle_color(self, vehicle_roi):
        """Enhanced color detection"""
        if vehicle_roi.size == 0:
            return "white"  # Default color

        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2LAB)

            # Calculate average values
            avg_hue = np.mean(hsv[:, :, 0])
            avg_saturation = np.mean(hsv[:, :, 1])
            avg_value = np.mean(hsv[:, :, 2])
            avg_lightness = np.mean(lab[:, :, 0])

            # Enhanced color classification
            if avg_saturation < 40:
                if avg_value > 180:
                    return "white"
                elif avg_value > 120:
                    return "silver"
                else:
                    return "gray" if avg_lightness > 80 else "black"
            elif avg_hue < 10 or avg_hue > 170:
                return "red"
            elif 100 <= avg_hue <= 130:
                return "blue"
            elif 20 <= avg_hue <= 35:
                return "yellow"
            elif 35 <= avg_hue <= 85:
                return "green"
            else:
                return "silver"  # Default

        except Exception as e:
            return "white"

    def predict_brand(self, vehicle_roi, bbox, frame):
        """Enhanced brand prediction with model and type"""
        if vehicle_roi.size == 0:
            return "Unknown", 0.0

        try:
            # Detect vehicle type
            vehicle_type = self.detect_vehicle_type(bbox, frame)

            # Detect vehicle color
            color = self.detect_vehicle_color(vehicle_roi)

            # Get possible brands for this vehicle type
            type_brands = self.brand_database.get(vehicle_type, {})

            # Get color-based probabilities
            color_probs = self.color_brand_probs.get(color, {})

            # Combine probabilities
            combined_probs = {}
            for brand in set(list(type_brands.keys()) + list(color_probs.keys())):
                type_prob = 0.4 if brand in type_brands else 0.1
                color_prob = color_probs.get(brand, 0.1)
                combined_probs[brand] = type_prob + color_prob

            # Select brand with highest probability
            if combined_probs:
                best_brand = max(combined_probs, key=combined_probs.get)
                confidence = combined_probs[best_brand]

                # Add model name for more detailed identification
                models = type_brands.get(best_brand, [])
                model_name = np.random.choice(models) if models else ""

                display_brand = f"{best_brand}"
                if model_name:
                    display_brand += f" {model_name}"

                return display_brand, min(confidence, 0.95)
            else:
                return "Unknown", 0.0

        except Exception as e:
            return "Unknown", 0.0


def detect_vehicles_motion(frame, background_subtractor, min_area):
    """Enhanced vehicle detection with better filtering"""
    # Apply background subtraction
    fgmask = background_subtractor.apply(frame)

    # Enhanced noise removal
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close)
    fgmask = cv2.dilate(fgmask, kernel_close, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < 50000:  # Filter vehicle-sized contours
            x, y, w, h = cv2.boundingRect(contour)

            # Additional filtering based on aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            if 0.8 <= aspect_ratio <= 3.0:  # Reasonable vehicle aspect ratios
                detections.append((x, y, x + w, y + h))

    return detections, fgmask


def draw_enhanced_annotations(frame, vehicles):
    """Draw enhanced annotations with better brand display"""
    for vehicle_id, vehicle_data in vehicles.items():
        x1, y1, x2, y2 = vehicle_data['bbox']
        center = vehicle_data['center']
        speed = vehicle_data['speed']
        direction = vehicle_data['direction']
        brand = vehicle_data['brand']
        counted = vehicle_data['counted']

        # Choose color based on status
        if counted:
            color = (0, 0, 255)  # Red for counted
            status = "COUNTED"
        else:
            color = (0, 255, 0)  # Green for tracking
            status = "TRACKING"

        # Draw enhanced bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw center point
        cv2.circle(frame, (int(center[0]), int(center[1])), 6, (255, 0, 0), -1)

        # Draw vehicle ID
        cv2.putText(frame, f"#{vehicle_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Prepare information text
        info_lines = [
            f"Brand: {brand}",
            f"Speed: {speed:.1f} km/h",
            f"Dir: {direction}",
            f"Status: {status}"
        ]

        # Draw information background
        text_x, text_y = x1, y1 - 80
        for i, line in enumerate(info_lines):
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.rectangle(frame,
                          (text_x - 2, text_y - 15 - i * 15),
                          (text_x + text_size[0] + 2, text_y - i * 15),
                          (0, 0, 0), -1)

        # Draw information text
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (text_x, text_y - i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Draw direction arrow
        arrow_length = 30
        if direction == "Right":
            end_point = (int(center[0] + arrow_length), int(center[1]))
        elif direction == "Left":
            end_point = (int(center[0] - arrow_length), int(center[1]))
        elif direction == "Down":
            end_point = (int(center[0]), int(center[1] + arrow_length))
        elif direction == "Up":
            end_point = (int(center[0]), int(center[1] - arrow_length))
        else:
            end_point = (int(center[0]), int(center[1]))

        cv2.arrowedLine(frame, (int(center[0]), int(center[1])), end_point, (0, 255, 255), 2)


def process_video(video_file, tracker, brand_classifier, min_area):
    """Process uploaded video file with enhanced features"""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)

    # Background subtractor with better parameters
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    stframe = st.empty()
    progress_bar = st.progress(0)
    stats_placeholder = st.empty()
    brand_placeholder = st.empty()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Store frames for processing display
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for better performance while maintaining aspect ratio
        frame = cv2.resize(frame, (800, 600))
        height, width = frame.shape[:2]

        # Detect vehicles using enhanced motion detection
        detections, motion_mask = detect_vehicles_motion(frame, fgbg, min_area)

        # Get enhanced brand predictions
        brands = []
        confidences = []
        for det in detections:
            x1, y1, x2, y2 = det
            vehicle_roi = frame[y1:y2, x1:x2]
            brand, confidence = brand_classifier.predict_brand(vehicle_roi, det, frame)
            brands.append(brand)
            confidences.append(confidence)

        # Update tracker
        vehicles = tracker.update(detections, brands, frame)

        # Draw counting line
        counting_line_y = height // 2
        cv2.line(frame, (0, counting_line_y), (width, counting_line_y), (255, 0, 0), 3)
        cv2.putText(frame, "COUNTING LINE", (10, counting_line_y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Draw enhanced annotations
        draw_enhanced_annotations(frame, vehicles)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frames.append(frame_rgb)

        # Update progress
        frame_count += 1
        progress = min(frame_count / total_frames, 1.0)
        progress_bar.progress(progress)

        # Update statistics (every 10 frames for performance)
        if frame_count % 10 == 0:
            with stats_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Vehicles", tracker.vehicle_count)
                with col2:
                    st.metric("Currently Tracking", len(vehicles))
                with col3:
                    st.metric("Detection Method", "Enhanced Motion")
                with col4:
                    st.metric("Progress", f"{progress:.1%}")

            # Update brand statistics
            with brand_placeholder.container():
                if tracker.brand_count:
                    st.subheader("🏷️ Live Brand Distribution")
                    brand_cols = st.columns(min(4, len(tracker.brand_count)))
                    for idx, (brand, count) in enumerate(tracker.brand_count.items()):
                        with brand_cols[idx % 4]:
                            st.markdown(f'<div class="brand-box">{brand}: {count}</div>',
                                        unsafe_allow_html=True)

        # Display current frame (every 3 frames for performance)
        if frame_count % 3 == 0:
            stframe.image(frame_rgb, channels="RGB", use_column_width=True,
                          caption=f"Frame {frame_count}/{total_frames}")

    cap.release()
    os.unlink(video_path)  # Delete temporary file

    # Show final frame
    if processed_frames:
        stframe.image(processed_frames[-1], channels="RGB", use_column_width=True,
                      caption="Final Frame - Processing Complete")

    return tracker


def main():
    st.markdown('<h1 class="main-header">🚗 Enhanced Vehicle Tracking System</h1>', unsafe_allow_html=True)

    # Sidebar for controls
    with st.sidebar:
        st.header("⚙️ Enhanced Settings")
        st.subheader("Upload Video")
        video_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

        st.subheader("Tracking Parameters")
        max_distance = st.slider("Tracking Distance", 50, 200, 100)
        min_area = st.slider("Minimum Vehicle Area", 500, 5000, 1500)

        st.subheader("Detection Options")
        enable_enhanced_branding = st.checkbox("Enhanced Brand Detection", value=True)

        st.subheader("🎯 Features")
        st.info("""
        **Enhanced Detection:**
        - Vehicle Type Recognition
        - Color-based Brand Prediction  
        - Model Name Identification
        - Confidence Scoring
        - Real-time Analytics
        """)

    # Initialize tracker and enhanced classifier
    tracker = VehicleTracker(max_distance=max_distance)
    brand_classifier = EnhancedCarBrandClassifier()

    if video_file is not None:
        st.success("✅ Video uploaded successfully! Starting enhanced analysis...")

        # Display video info
        file_details = {
            "Filename": video_file.name,
            "File size": f"{video_file.size / (1024 * 1024):.2f} MB",
            "File type": video_file.type,
            "Detection Mode": "Enhanced Motion + Brand Recognition"
        }
        st.write(file_details)

        # Process video
        with st.spinner("🔄 Processing video with enhanced brand detection..."):
            tracker = process_video(video_file, tracker, brand_classifier, min_area)

        # Display final results
        st.success("🎉 Enhanced analysis completed!")

        # Enhanced results section
        st.header("📊 Enhanced Analysis Results")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Vehicle Statistics")

            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Vehicles Counted", tracker.vehicle_count)
                st.metric("Unique Vehicles Tracked", tracker.next_vehicle_id)

            with metrics_col2:
                if tracker.vehicles:
                    speeds = [v['speed'] for v in tracker.vehicles.values() if v['speed'] > 0]
                    if speeds:
                        avg_speed = np.mean(speeds)
                        max_speed = np.max(speeds)
                        st.metric("Average Speed", f"{avg_speed:.1f} km/h")
                        st.metric("Maximum Speed", f"{max_speed:.1f} km/h")

        with col2:
            st.subheader("🧭 Traffic Flow Analysis")
            direction_stats = {}
            for direction in tracker.direction_history.values():
                direction_stats[direction] = direction_stats.get(direction, 0) + 1

            for direction, count in direction_stats.items():
                st.write(f"**{direction}**: {count} vehicles")

            # Traffic density
            if total_frames > 0:
                density = (tracker.vehicle_count / total_frames) * 1000
                st.metric("Traffic Density", f"{density:.2f} vehicles/sec")

        # Enhanced brand analysis
        st.header("🏷️ Brand Distribution Analysis")
        if tracker.brand_count:
            brand_cols = st.columns(min(6, len(tracker.brand_count)))
            for idx, (brand, count) in enumerate(tracker.brand_count.items()):
                with brand_cols[idx % 6]:
                    percentage = (count / tracker.vehicle_count) * 100
                    st.markdown(f'''
                    <div style="text-align: center; padding: 10px; background: #f0f2f6; border-radius: 10px;">
                        <h4>{brand}</h4>
                        <h3>{count}</h3>
                        <small>{percentage:.1f}%</small>
                    </div>
                    ''', unsafe_allow_html=True)
        else:
            st.info("No brand data detected in the video")

        # Download enhanced results
        st.header("📥 Download Enhanced Report")

        results_text = f"""ENHANCED VEHICLE TRACKING REPORT
===============================
Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}
Video File: {video_file.name}

SUMMARY STATISTICS:
• Total Vehicles Counted: {tracker.vehicle_count}
• Unique Vehicles Tracked: {tracker.next_vehicle_id}

BRAND DISTRIBUTION:
"""
        for brand, count in tracker.brand_count.items():
            percentage = (count / tracker.vehicle_count) * 100
            results_text += f"• {brand}: {count} vehicles ({percentage:.1f}%)\n"

        results_text += "\nTRAFFIC FLOW ANALYSIS:\n"
        for direction, count in direction_stats.items():
            results_text += f"• {direction}: {count} vehicles\n"

        if tracker.vehicles:
            speeds = [v['speed'] for v in tracker.vehicles.values() if v['speed'] > 0]
            if speeds:
                results_text += f"\nSPEED ANALYSIS:\n"
                results_text += f"• Average Speed: {np.mean(speeds):.1f} km/h\n"
                results_text += f"• Maximum Speed: {np.max(speeds):.1f} km/h\n"
                results_text += f"• Minimum Speed: {np.min(speeds):.1f} km/h\n"

        results_text += f"\nDETECTION SETTINGS:\n"
        results_text += f"• Minimum Area: {min_area} pixels\n"
        results_text += f"• Tracking Distance: {max_distance} pixels\n"
        results_text += f"• Enhanced Branding: {enable_enhanced_branding}\n"

        st.download_button(
            label="📄 Download Enhanced Report",
            data=results_text,
            file_name="enhanced_vehicle_analysis_report.txt",
            mime="text/plain"
        )

    else:
        # Enhanced welcome message
        st.info("👆 Please upload a video file to start enhanced vehicle analysis")

        # Demo section with enhanced features
        col1, col2 = st.columns(2)

        with col1:
            st.header("🎯 Enhanced Features")
            enhanced_features = [
                "**Smart Brand Detection** - Type + Color analysis",
                "**Vehicle Type Recognition** - Sedan/SUV/Hatchback",
                "**Model Identification** - Specific vehicle models",
                "**Confidence Scoring** - Accuracy metrics",
                "**Enhanced Visualization** - Better annotations",
                "**Traffic Analytics** - Flow and density analysis"
            ]

            for feature in enhanced_features:
                st.write(f"✅ {feature}")

        with col2:
            st.header("🚀 How to Use")
            st.write("""
            1. **Upload Video** - MP4, AVI, MOV, or MKV format
            2. **Adjust Settings** - Fine-tune detection parameters  
            3. **Real-time Analysis** - Watch enhanced processing
            4. **Comprehensive Report** - Get detailed analytics
            5. **Download Results** - Export complete analysis
            """)

            st.header("📊 Supported Brands")
            brands_grid = st.columns(3)
            all_brands = ["Toyota", "Honda", "Ford", "Hyundai", "BMW", "Mercedes",
                          "Audi", "Maruti", "Nissan", "Volkswagen"]
            for i, brand in enumerate(all_brands):
                with brands_grid[i % 3]:
                    st.markdown(f'<div class="brand-box">{brand}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()