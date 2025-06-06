#!/usr/bin/env python3
"""
Real-time Hand Gesture Detection using MediaPipe and trained TensorFlow model
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle
import time
from collections import deque

class RealTimeHandGestureDetector:
    def __init__(self, model_path='hand_gesture_model.h5', 
                 encoder_path='label_encoder.pkl',
                 scaler_path='feature_scaler.pkl',
                 features_path='feature_names.pkl'):
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load trained model and preprocessing objects
        self.load_model_and_preprocessing(model_path, encoder_path, scaler_path, features_path)
        
        # Prediction parameters
        self.confidence_threshold = 0.6
        self.smoothing_window = 5
        self.prediction_buffer = deque(maxlen=self.smoothing_window)
        
        # Display variables
        self.current_prediction = None
        self.current_confidence = 0.0
        self.last_prediction_time = 0
        self.prediction_cooldown = 0.5  # seconds
        
    def load_model_and_preprocessing(self, model_path, encoder_path, scaler_path, features_path):
        """Load the trained model and preprocessing objects"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
            
            # Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded: {encoder_path}")
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Feature scaler loaded: {scaler_path}")
            
            # Load feature names
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Feature names loaded: {features_path}")
            
            print(f"Model expects {len(self.feature_names)} features")
            print(f"Can predict {len(self.label_encoder.classes_)} gestures")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model/preprocessing: {e}")
            return False
    
    def extract_hand_keypoints(self, frame):
        """Extract hand keypoints from frame using MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        keypoints = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract 21 landmarks - use normalized coordinates (0-1) directly from MediaPipe
                # This matches the training data format after coordinate normalization
                hand_keypoints = []
                for landmark in hand_landmarks.landmark:
                    # MediaPipe returns normalized coordinates (0-1), keep them as-is
                    hand_keypoints.extend([landmark.x, landmark.y])
                
                # Check if we need confidence values (for older models)
                if any('confidence' in name for name in self.feature_names):
                    # Add confidence values (set to 1.0 for live detection)
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
        if keypoints is None:
            return None
        
        # Create feature vector matching training data
        feature_vector = [0.0] * len(self.feature_names)
        
        # Determine expected features per hand
        features_per_hand = len([name for name in self.feature_names if 'left_hand' in name])
        coords_per_landmark = 3 if any('confidence' in name for name in self.feature_names) else 2
        landmarks_per_hand = features_per_hand // coords_per_landmark
        
        # Fill keypoints into appropriate positions
        if len(keypoints) >= landmarks_per_hand * coords_per_landmark:
            # Assume it's right hand data (adjust based on your feature names)
            right_hand_start = len([name for name in self.feature_names if 'left_hand' in name])
            
            for i, value in enumerate(keypoints[:landmarks_per_hand * coords_per_landmark]):
                if right_hand_start + i < len(feature_vector):
                    feature_vector[right_hand_start + i] = value
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict_gesture(self, keypoints):
        """Predict gesture from keypoints"""
        if keypoints is None:
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
            print(f"Prediction error: {e}")
            return None
    
    def smooth_predictions(self, prediction):
        """Apply smoothing to predictions"""
        if prediction is None:
            return None
        
        # Add to buffer
        self.prediction_buffer.append(prediction)
        
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
    
    def draw_predictions(self, frame, prediction):
        """Draw prediction information on frame"""
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
        
        # Title
        cv2.putText(frame, "Hand Gesture Detection (Normalized)", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current prediction
        if prediction and prediction['confidence'] > self.confidence_threshold:
            gesture_text = f"Gesture: {prediction['gesture']}"
            confidence_text = f"Confidence: {prediction['confidence']:.3f}"
            
            # Color based on confidence
            color = (0, 255, 0) if prediction['confidence'] > 0.8 else (0, 255, 255)
            
            cv2.putText(frame, gesture_text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, confidence_text, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Stability indicator if available
            if 'stability' in prediction:
                stability_text = f"Stability: {prediction['stability']:.2f}"
                cv2.putText(frame, stability_text, (20, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(frame, "No gesture detected", (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Press 'q' to quit, 't/g' to adjust threshold", (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Threshold display
        cv2.putText(frame, f"Threshold: {self.confidence_threshold:.1f}", (width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        return frame
    
    def run(self):
        """Run real-time gesture detection"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🚀 Starting real-time gesture detection...")
        print("Controls:")
        print("  'q' - Quit")
        print("  't' - Increase confidence threshold")
        print("  'g' - Decrease confidence threshold")
        print("  'r' - Reset prediction buffer")
        
        fps_counter = 0
        fps_start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Extract keypoints
            keypoints = self.extract_hand_keypoints(frame)
            
            # Make prediction
            prediction = self.predict_gesture(keypoints)
            
            # Apply smoothing
            smoothed_prediction = self.smooth_predictions(prediction)
            
            # Update current prediction with cooldown
            current_time = time.time()
            if (smoothed_prediction and 
                smoothed_prediction['confidence'] > self.confidence_threshold and
                current_time - self.last_prediction_time > self.prediction_cooldown):
                
                self.current_prediction = smoothed_prediction['gesture']
                self.current_confidence = smoothed_prediction['confidence']
                self.last_prediction_time = current_time
                
                print(f"Detected: {self.current_prediction} "
                      f"(Confidence: {self.current_confidence:.3f})")
            
            # Draw predictions
            frame = self.draw_predictions(frame, smoothed_prediction)
            
            # Calculate and display FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 100, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Display frame
            cv2.imshow('Hand Gesture Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                print(f"Threshold increased to: {self.confidence_threshold:.2f}")
            elif key == ord('g'):
                self.confidence_threshold = max(0.1, self.confidence_threshold - 0.05)
                print(f"Threshold decreased to: {self.confidence_threshold:.2f}")
            elif key == ord('r'):
                self.prediction_buffer.clear()
                print("Prediction buffer reset")
        
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Gesture detection stopped")

def main():
    print("Real-time Hand Gesture Detection")
    print("=" * 40)
    
    # Check if model files exist
    required_files = [
        'hand_gesture_model.h5',
        'label_encoder.pkl',
        'feature_scaler.pkl',
        'feature_names.pkl'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease run the training script first:")
        print("python hand_gesture_trainer.py")
        return
    
    # Initialize detector
    detector = RealTimeHandGestureDetector()
    
    if detector.model is None:
        print("❌ Failed to load model")
        return
    
    # Run detection
    try:
        detector.run()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    import os
    main() 