import cv2
import mediapipe as mp
import numpy as np
import sys

class HandGestureDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect_gestures(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Get gesture information
                gesture = self._classify_gesture(hand_landmarks)
                if gesture:
                    # Add text to show the detected gesture
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
        
        return frame

    def _classify_gesture(self, landmarks):
        # Get the coordinates of key points
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        ring_tip = landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        pinky_tip = landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        
        # Simple gesture classification based on finger positions
        # This is a basic example - you can expand this with more complex gestures
        if (thumb_tip.y < index_tip.y and
            index_tip.y < middle_tip.y and
            middle_tip.y < ring_tip.y and
            ring_tip.y < pinky_tip.y):
            return "Fist"
        elif (thumb_tip.y > index_tip.y and
              index_tip.y < middle_tip.y and
              middle_tip.y < ring_tip.y and
              ring_tip.y < pinky_tip.y):
            return "Thumbs Up"
        else:
            return "Open Hand"

def main():
    # Initialize the webcam
    print("Attempting to open webcam...")
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Please check if:")
        print("1. Your webcam is properly connected")
        print("2. No other application is using the webcam")
        print("3. You have the necessary permissions to access the webcam")
        sys.exit(1)
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit the application")
    
    detector = HandGestureDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            break

        # Flip the frame horizontally for a more natural view
        frame = cv2.flip(frame, 1)
        
        # Detect and draw hand gestures
        frame = detector.detect_gestures(frame)
        
        # Display the frame
        cv2.imshow('Hand Gesture Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 