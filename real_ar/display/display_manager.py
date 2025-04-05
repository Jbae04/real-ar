# display_manager.py (修正後)
import threading
import time
import cv2
import numpy as np
import logging
import os
import sys
# from queue import Queue # No longer needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from display.ar_display import ARDisplay
from face_recognitionai.detect_and_recognition import FaceDatabase # Keep if needed for other methods
# from voice_processing.database.database import Database # Keep if needed
# from voice_processing.voice_recognition import VoiceRecognition # Keep if needed
# from voice_processing.voice_activation import VoiceActivation # Keep if needed
# from utils.text_to_speech import TextToSpeech # Keep if needed

# Set up logging (Keep existing logging setup)
# Prevent duplicate handlers if script is reloaded
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
root_logger = logging.getLogger()
# Check if handlers already exist to prevent duplication
if not root_logger.handlers:
    root_logger.setLevel(logging.DEBUG)
    # File Handler
    from logging.handlers import RotatingFileHandler # Ensure import is available
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, "ar_system.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    console_handler.setLevel(logging.INFO)  # Keep console output at INFO level
    root_logger.addHandler(console_handler)
    logging.info("Logging system initialized with DEBUG level to file and INFO level to console")
else:
     logging.debug("Logging handlers already configured.") # Use debug level if re-initializing


class DisplayManager:
    """
    Manages the integration between the AR display and the face recognition system.
    Receives face recognition results and updates the AR display state accordingly.
    (Voice listening, registration triggering, and event processing are handled externally).
    """

    def __init__(self, ar_display, server_callback=None, face_db=None, sql_db=None):
        """
        Initialize the display manager and its components.

        Args:
            ar_display: An instance of ARDisplay (required).
            server_callback: Optional callback function to send face data to server.
            face_db: Optional shared FaceDatabase instance.
            sql_db: Optional shared Database instance.
        """
        if ar_display is None:
             raise ValueError("ARDisplay instance is required.")
        self.display = ar_display # Use the passed ARDisplay instance

        # Keep references to shared DBs if needed
        self.face_db = face_db # Assume passed instance or None
        self.sql_db = sql_db   # Assume passed instance or None

        self.running = True # Flag to indicate if manager should operate (might be redundant)
        self.current_face_data = None # Store last processed face data
        self.unknown_faces = [] # List of currently detected unknown faces
        self.server_callback = server_callback # Callback for server sync
        self.lock = threading.Lock() # Lock for thread-safe access to unknown_faces

        logging.info("DisplayManager initialized.")
        # Initial notification/render should be handled by the caller

    # Removed _wake_word_listener_worker
    # Removed voice_listener_thread
    # Removed start_registration
    # Removed _process_messages
    # Removed process_frame

    def update_display_with_recognition_results(self, recognition_results):
        """
        Update the AR display state based on face recognition results.
        This method updates the internal state and calls ARDisplay methods
        which might trigger rendering internally based on ARDisplay's design.

        :param recognition_results: List of dicts containing face recognition data.
        """
        if not recognition_results:
            # No faces detected
            # Update ARDisplay state directly
            self.display.update_status(face_detection=False) # Update relevant status
            # Set mode to normal to ensure "No faces detected" is shown
            self.display.current_mode = "normal" # Ensure mode is reset
            self.current_face_data = None
            # self.display.render() # Let ARDisplay methods or main loop handle rendering

            with self.lock:
                had_unknown_faces = bool(self.unknown_faces)
                if had_unknown_faces: # Only clear and log if there were unknown faces before
                    self.unknown_faces = [] # Clear unknown faces
                    logging.info("No faces detected, cleared unknown_faces list")
            return # Exit early

        # Faces were detected
        self.display.update_status(face_detection=True)

        new_unknown_faces = []
        displayed_face = False # Track if any face (known or unknown) was processed for display

        # Process each detected face
        for face_data in recognition_results:
            self.current_face_data = face_data # Store last processed face
            if face_data["name"] == "Unknown":
                new_unknown_faces.append(face_data)
                # Update ARDisplay for unknown face
                self.display.display_unknown_face(face_data["box"]) # Assumes this updates state/renders
                displayed_face = True
                logging.debug("Updated display for unknown face.")
            else:
                # Update ARDisplay for recognized face
                self.display.display_recognized_face(face_data) # Assumes this updates state/renders
                displayed_face = True
                logging.debug(f"Updated display for recognized face: {face_data['name']}")

        # Log if results were present but nothing seemed to be displayed
        if not displayed_face and recognition_results:
            logging.warning(f"Processed {len(recognition_results)} results, but no face display method seemed applicable.")

        # Update the shared list of unknown faces safely
        with self.lock:
            if self.unknown_faces != new_unknown_faces:
                 logging.info(f"Unknown faces updated: {len(new_unknown_faces)} currently tracked.")
                 self.unknown_faces = new_unknown_faces


    def close(self):
        """Clean up resources managed specifically by DisplayManager (if any)."""
        logging.debug("DisplayManager.close() called")
        self.running = False # Signal to stop processing if checked elsewhere

        # No threads managed by DisplayManager anymore.
        # No owned resources like ARDisplay to close here.

        logging.info("Display manager shutdown actions complete.")
