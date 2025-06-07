#!/usr/bin/env python3
"""
Enhanced Streamlit Web Interface for Real-time Hand Gesture Detection with Video Streaming
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time
from collections import deque
import os
from PIL import Image
import threading
import queue
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.detector = None
        self.setup_detector()
        
    def setup_detector(self):
        """Initialize the hand gesture detector"""
        try:
            # Initialize MediaPipe
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.mp_draw = mp.solutions.drawing_utils
            
            # Load model and preprocessing objects
            self.load_model_and_preprocessing()
            
            # Prediction parameters
            self.confidence_threshold = getattr(st.session_state, 'confidence_threshold', 0.6)
            self.smoothing_window = getattr(st.session_state, 'smoothing_window', 5)
            self.prediction_buffer = deque(maxlen=self.smoothing_window)
            
            # Display variables
            self.current_prediction = None
            self.current_confidence = 0.0
            self.last_prediction_time = 0
            self.prediction_cooldown = 0.1  # Faster updates for real-time
            
        except Exception as e:
            st.error(f"Failed to initialize detector: {e}")
            
    def load_model_and_preprocessing(self):
        """Load the trained model and preprocessing objects"""
        try:
            # Load model
            self.model = tf.keras.models.load_model('hand_gesture_model.h5')
            
            # Load label encoder
            with open('label_encoder.pkl', 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load scaler
            with open('feature_scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load feature names
            with open('feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            self.model_loaded = True
            
        except Exception as e:
            st.error(f"Error loading model/preprocessing: {e}")
            self.model_loaded = False
    
    def extract_hand_keypoints(self, frame):
        """Extract hand keypoints from frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        keypoints = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    hand_keypoints.extend([landmark.x, landmark.y])
                
                # Check if we need confidence values
                if hasattr(self, 'feature_names') and any('confidence' in name for name in self.feature_names):
                    hand_keypoints_with_conf = []
                    for i in range(0, len(hand_keypoints), 2):
                        hand_keypoints_with_conf.extend([
                            hand_keypoints[i],     # x (normalized)
                            hand_keypoints[i+1],   # y (normalized)
                            1.0                    # confidence
                        ])
                    hand_keypoints = hand_keypoints_with_conf
                
                keypoints.extend(hand_keypoints)
                
                # Draw landmarks on frame
                self.mp_draw.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return keypoints if keypoints else None
    
    def prepare_features(self, keypoints):
        """Prepare keypoints for model prediction"""
        if keypoints is None or not hasattr(self, 'feature_names'):
            return None
        
        # Create feature vector matching training data
        feature_vector = [0.0] * len(self.feature_names)
        
        # Determine expected features per hand
        features_per_hand = len([name for name in self.feature_names if 'left_hand' in name])
        coords_per_landmark = 3 if any('confidence' in name for name in self.feature_names) else 2
        landmarks_per_hand = features_per_hand // coords_per_landmark if features_per_hand > 0 else 21
        
        # Fill keypoints into appropriate positions
        if len(keypoints) >= landmarks_per_hand * coords_per_landmark:
            right_hand_start = len([name for name in self.feature_names if 'left_hand' in name])
            
            for i, value in enumerate(keypoints[:landmarks_per_hand * coords_per_landmark]):
                if right_hand_start + i < len(feature_vector):
                    feature_vector[right_hand_start + i] = value
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_gesture(self, keypoints):
        """Predict gesture from keypoints"""
        if keypoints is None or not hasattr(self, 'model_loaded') or not self.model_loaded:
            return None
        
        try:
            # Prepare features
            features = self.prepare_features(keypoints)
            if features is None:
                return None
            
            # Normalize features
            features_normalized = self.scaler.transform(features)
            
            # Make prediction
            predictions = self.model.predict(features_normalized, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Get gesture name
            gesture_name = self.label_encoder.inverse_transform([predicted_class])[0]
            
            return {
                'gesture': gesture_name,
                'confidence': float(confidence),
                'predictions': predictions[0]
            }
            
        except Exception as e:
            return None

    def smooth_predictions(self, prediction):
        """Apply smoothing to predictions"""
        if prediction is None:
            return None
        
        # Add to buffer
        self.prediction_buffer.append(prediction)
        
        if len(self.prediction_buffer) == 0:
            return None
        
        # Count occurrences of each gesture in buffer
        gesture_counts = {}
        confidence_sums = {}
        
        for pred in self.prediction_buffer:
            gesture = pred['gesture']
            confidence = pred['confidence']
            
            if gesture not in gesture_counts:
                gesture_counts[gesture] = 0
                confidence_sums[gesture] = 0
            
            gesture_counts[gesture] += 1
            confidence_sums[gesture] += confidence
        
        # Find most frequent gesture
        most_frequent_gesture = max(gesture_counts.keys(), key=lambda x: gesture_counts[x])
        avg_confidence = confidence_sums[most_frequent_gesture] / gesture_counts[most_frequent_gesture]
        
        return {
            'gesture': most_frequent_gesture,
            'confidence': avg_confidence,
            'stability': gesture_counts[most_frequent_gesture] / len(self.prediction_buffer)
        }

    def transform(self, frame):
        """Transform each frame of the video stream"""
        # Update parameters from session state
        self.confidence_threshold = getattr(st.session_state, 'confidence_threshold', 0.6)
        if hasattr(st.session_state, 'smoothing_window') and st.session_state.smoothing_window != self.smoothing_window:
            self.smoothing_window = st.session_state.smoothing_window
            self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Extract keypoints
        keypoints = self.extract_hand_keypoints(frame)
        
        # Make prediction
        prediction = self.predict_gesture(keypoints)
        
        # Apply smoothing
        smoothed_prediction = self.smooth_predictions(prediction)
        
        # Update session state with predictions
        if smoothed_prediction and smoothed_prediction['confidence'] > self.confidence_threshold:
            current_time = time.time()
            if current_time - self.last_prediction_time > self.prediction_cooldown:
                st.session_state.current_prediction = smoothed_prediction['gesture']
                st.session_state.current_confidence = smoothed_prediction['confidence']
                st.session_state.current_stability = smoothed_prediction.get('stability', 0)
                self.last_prediction_time = current_time
        
        # Draw overlays on frame
        self.draw_overlays(frame, smoothed_prediction)
        
        return frame
    
    def draw_overlays(self, frame, prediction):
        """Draw prediction overlays on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Title
        cv2.putText(frame, "Hand Gesture Detection", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current prediction
        if prediction and prediction['confidence'] > self.confidence_threshold:
            gesture_text = f"Gesture: {prediction['gesture']}"
            confidence_text = f"Conf: {prediction['confidence']:.3f}"
            
            # Color based on confidence
            color = (0, 255, 0) if prediction['confidence'] > 0.8 else (0, 255, 255)
            
            cv2.putText(frame, gesture_text, (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, confidence_text, (20, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        else:
            cv2.putText(frame, "No gesture detected", (20, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Threshold display
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.1f}", (width - 200, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

def main():
    st.set_page_config(
        page_title="Real-time Hand Gesture Detection",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("👋 Real-time Hand Gesture Detection")
    st.markdown("*Powered by MediaPipe, TensorFlow, and WebRTC*")
    
    # Check if model files exist
    required_files = [
        'hand_gesture_model.h5',
        'label_encoder.pkl',
        'feature_scaler.pkl',
        'feature_names.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        st.error("Missing required model files:")
        for file in missing_files:
            st.write(f"• {file}")
        st.info("Please run the training script first: `python hand_gesture_trainer.py`")
        return
    
    # Initialize session state
    if 'confidence_threshold' not in st.session_state:
        st.session_state.confidence_threshold = 0.6
    if 'smoothing_window' not in st.session_state:
        st.session_state.smoothing_window = 5
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = None
    if 'current_confidence' not in st.session_state:
        st.session_state.current_confidence = 0.0
    if 'current_stability' not in st.session_state:
        st.session_state.current_stability = 0.0
    
    # Sidebar controls
    st.sidebar.header("🎛️ Controls")
    
    # Confidence threshold
    st.session_state.confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="Minimum confidence required for gesture detection"
    )
    
    # Smoothing window
    st.session_state.smoothing_window = st.sidebar.slider(
        "Smoothing Window",
        min_value=1,
        max_value=10,
        value=st.session_state.smoothing_window,
        help="Number of frames to smooth predictions over"
    )
    
    # Model info (load basic info)
    st.sidebar.header("📋 Model Information")
    try:
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        st.sidebar.write(f"**Available Gestures:**")
        for gesture in label_encoder.classes_:
            st.sidebar.write(f"• {gesture}")
    except:
        st.sidebar.write("Model info not available")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📹 Live Video Stream")
        
        # WebRTC configuration
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        # Video streamer
        webrtc_ctx = webrtc_streamer(
            key="hand-gesture-detection",
            video_transformer_factory=VideoTransformer,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": True,
                "audio": False
            },
            async_processing=True,
        )
        
        if webrtc_ctx.video_transformer:
            st.info("🎥 Camera is active - show your hand to detect gestures!")
        else:
            st.info("📷 Click 'START' to begin gesture detection")
    
    with col2:
        st.header("🎯 Detection Results")
        
        # Real-time results display
        if st.session_state.current_prediction:
            st.success("✅ Gesture Detected!")
            
            # Display current prediction
            st.metric(
                "Detected Gesture",
                st.session_state.current_prediction
            )
            
            st.metric(
                "Confidence",
                f"{st.session_state.current_confidence:.3f}",
                delta=f"{st.session_state.current_confidence - st.session_state.confidence_threshold:.3f}"
            )
            
            if st.session_state.current_stability > 0:
                st.metric(
                    "Stability",
                    f"{st.session_state.current_stability:.2f}"
                )
            
            # Confidence indicator
            if st.session_state.current_confidence > 0.8:
                st.success("🟢 High Confidence")
            elif st.session_state.current_confidence > 0.6:
                st.warning("🟡 Medium Confidence")
            else:
                st.info("🔵 Low Confidence")
        else:
            st.info("👋 Show your hand to detect gestures")
            st.metric("Detected Gesture", "None")
            st.metric("Confidence", "0.000")
        
        # Real-time status
        st.header("📊 Status")
        if webrtc_ctx.video_transformer:
            st.success("🔴 Live")
        else:
            st.info("⚫ Stopped")
        
        # Settings summary
        st.header("⚙️ Current Settings")
        st.write(f"**Confidence Threshold:** {st.session_state.confidence_threshold:.1f}")
        st.write(f"**Smoothing Window:** {st.session_state.smoothing_window}")
    
    # Instructions
    st.markdown("---")
    st.markdown(
        """
        ### 📖 Instructions:
        1. **Click START** to begin real-time gesture detection
        2. **Position your hand** in front of the camera
        3. **Make gestures** - the system will detect and classify them in real-time
        4. **Adjust settings** in the sidebar to fine-tune detection sensitivity
        5. **View results** in the right panel with confidence scores and stability metrics
        
        **Tips:**
        - Ensure good lighting for better detection
        - Keep your hand clearly visible in the camera frame
        - Use steady movements for more stable predictions
        """
    )

if __name__ == "__main__":
    main() 