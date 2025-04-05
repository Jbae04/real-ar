import cv2
import numpy as np
import threading
import time
import requests
import json
import mediapipe as mp
from display.ar_display import ARDisplay

class GestureRecognition:
    def __init__(self, ar_display, video_stream, use_gemini=True):
        """
        Initialize gesture recognition with Gemini integration
        
        Args:
            ar_display (ARDisplay): Shared AR display system
            video_stream: Existing VideoStream instance
            use_gemini (bool): Whether to use Gemini API for enhanced responses
        """
        self.ar_display = ar_display
        self.vs = video_stream
        self.running = False
        self.thread = None
        self.use_gemini = use_gemini
        
        print("Initializing MediaPipe Hands...")
        # MediaPipe Hands configuration
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        print("MediaPipe Hands initialized successfully")
        
        # Gemini API configuration
        self.gemini_api_key = "AIzaSyBV03V0SHlmXXZKfFKuPnlWgGA94NN0gHE"
        self.gemini_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        
        # Gesture context for Gemini
        self.context = """
        You're an AR assistant interpreting hand gestures. Respond conversationally.
        Known gestures:
        - Wave: Greeting
        - Thumbs up: Positive feedback  
        - Thumbs down: Negative feedback
        - Point: Requesting information
        - Fist: Stop/command hold
        - Open palm: Ready state
        """
        
        # Rate limiting
        self.last_api_call = 0
        self.api_cooldown = 2.0  # seconds

    def query_gemini(self, gesture):
        """Query Gemini API with gesture context"""
        if time.time() - self.last_api_call < self.api_cooldown:
            return None
            
        try:
            print(f"Attempting Gemini API call for gesture: {gesture}")
            headers = {
                "Content-Type": "application/json"
            }
            params = {"key": self.gemini_api_key}
            
            prompt = {
                "contents": [{
                    "parts": [{
                        "text": f"{self.context}\nDetected gesture: {gesture}. Generate helpful AR response."
                    }]
                }]
            }
            
            response = requests.post(
                self.gemini_url,
                headers=headers,
                params=params,
                json=prompt
            )
            
            self.last_api_call = time.time()
            
            if response.status_code == 200:
                print("Gemini API call successful")
                result = response.json()
                return result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
            else:
                print(f"Gemini API error: Status code {response.status_code}, Response: {response.text}")
                self.ar_display.show_notification("Gemini API error", duration=2)
                return None
                
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None

    def detect_gesture(self, frame):
        """Enhanced gesture detection with MediaPipe"""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
                
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Finger state detection
            finger_states = {
                'thumb': hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y,
                'index': hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,
                'middle': hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,
                'ring': hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,
                'pinky': hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y
            }
            
            # Gesture classification
            if not any(finger_states.values()):
                return "fist"
            elif all(finger_states.values()):
                return "open_palm"
            elif finger_states['thumb'] and not any([finger_states['index'], finger_states['middle'], finger_states['ring']]):
                return "thumbs_up"
            elif not finger_states['thumb'] and finger_states['index'] and not any([finger_states['middle'], finger_states['ring']]):
                return "point"
            elif sum(finger_states.values()) >= 3:
                return "wave"
                
            return None
        except Exception as e:
            print(f"Error detecting gesture: {e}")
            return None

    def process_gesture(self, gesture):
        """Handle gesture with Gemini integration"""
        try:
            # Basic response
            basic_responses = {
                "wave": "Hello! How can I help?",
                "thumbs_up": "Thank you for the feedback!",
                "thumbs_down": "I'll try to do better next time.",
                "point": "You're pointing at something interesting.",
                "fist": "Command acknowledged.",
                "open_palm": "I'm ready for your next request."
            }
            
            print(f"Detected gesture: {gesture}")
            self.ar_display.show_notification(basic_responses.get(gesture, "Gesture detected"), duration=2)
            
            # Enhanced response from Gemini only if enabled
            if self.use_gemini:
                gemini_response = self.query_gemini(gesture)
                if gemini_response:
                    self.ar_display.show_notification(gemini_response, duration=4)
        except Exception as e:
            print(f"Error processing gesture: {e}")

    def run_gesture_loop(self):
        """Main recognition thread"""
        self.running = True
        last_gesture_time = 0
        gesture_cooldown = 1.5  # seconds
        
        try:
            print("Gesture recognition thread starting...")
            while self.running:
                try:
                    frame = self.vs.read()
                    if frame is None:
                        time.sleep(0.1)
                        continue
                        
                    gesture = self.detect_gesture(frame)
                    current_time = time.time()
                    
                    if gesture and (current_time - last_gesture_time) > gesture_cooldown:
                        last_gesture_time = current_time
                        self.process_gesture(gesture)
                        
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Error in gesture loop iteration: {e}")
                    time.sleep(0.5)  # Prevent tight error loop
        except Exception as e:
            print(f"Critical error in gesture recognition thread: {e}")
        finally:
            print("Gesture recognition thread exiting")

    def start(self):
        """Start gesture system"""
        if not self.thread or not self.thread.is_alive():
            print("Starting gesture recognition thread...")
            self.thread = threading.Thread(target=self.run_gesture_loop, daemon=True)
            self.thread.start()
            self.ar_display.show_notification("Gesture control activated", duration=2)
            print("Gesture recognition thread started")

    def stop(self):
        """Stop system"""
        print("Stopping gesture recognition...")
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        try:
            self.hands.close()
        except Exception as e:
            print(f"Error closing MediaPipe Hands: {e}")
        self.ar_display.show_notification("Gesture control offline", duration=2)
        print("Gesture recognition stopped")

    def __del__(self):
        self.stop()