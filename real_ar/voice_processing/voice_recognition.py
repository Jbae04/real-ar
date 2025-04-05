import pyaudio
import wave
import os
import time
import pyttsx3
import numpy as np
from google.cloud import speech
from face_recognitionai.detect_and_recognition import FaceDatabase
from voice_processing.database.database import Database

class VoiceRecognition:
    def __init__(self, device_index=None, database=None, face_database=None):
        """Initialize the voice recognition system with text-to-speech, database, and face database."""
        self.engine = pyttsx3.init('espeak')
        self.database = database if database is not None else Database()
        self.face_db = face_database if face_database is not None else FaceDatabase()
        self.recordings_dir = "recordings"
        os.makedirs(self.recordings_dir, exist_ok=True)
        self.fs = 16000
        self.chunk_size = 1024
        self.device_index = device_index if device_index is not None else self.get_airpods_device_index()
        print(f"Using device index: {self.device_index}")

    def get_airpods_device_index(self):
        """Find and return the index of AirPods microphone."""
        p = pyaudio.PyAudio()
        print("Listing all available audio devices:")
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            print(f"Device {i}: {dev['name']} (Channels: {dev['maxInputChannels']}, host API: {dev['hostApi']})")
            # Updated condition to be more specific if needed, or keep 'pulse' if that's correct for the setup
            if "pulse" in dev['name'].lower(): # Or a more specific name like "AirPods"
                p.terminate()
                return i
        p.terminate()
        print("Target microphone not found. Using default input device.")
        # Fallback to default if specific device not found
        try:
            # Need a new PyAudio instance as the previous one was terminated
            p_fallback = pyaudio.PyAudio()
            default_device_index = p_fallback.get_default_input_device_info()['index']
            p_fallback.terminate()
            return default_device_index
        except Exception as e:
             print(f"Could not get default device index: {e}")
             # Ensure termination even if getting default fails
             if 'p_fallback' in locals() and p_fallback._is_open:
                 p_fallback.terminate()
             return None # Indicate failure to find any suitable device


    def record_audio(self, duration=3, timeout=5):
        """Record audio for a specified duration with timeout."""
        if self.device_index is None:
            print("Error: No input device found.")
            return None

        print("Recording started. Speak now...")
        p = pyaudio.PyAudio() # Instantiate PyAudio
        stream = None         # Initialize stream variable
        recording = None      # Initialize recording variable
        try:
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.fs,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            frames = []
            start_time = time.time()
            print(f"Recording for {duration} seconds (timeout: {timeout}s)...")
            # Calculate number of chunks needed for the duration
            num_chunks = int(self.fs / self.chunk_size * duration)
            for i in range(0, num_chunks):
                # Check for timeout
                if time.time() - start_time > timeout:
                    print("Recording timed out.")
                    break
                try:
                    # Read data from stream
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                except IOError as e:
                    # Handle potential input overflow or other IO errors
                    print(f"IOError during recording chunk {i+1}/{num_chunks}: {e}")
                    # Optionally break or continue based on error handling strategy
                    break # Example: stop recording on error
            # Combine recorded frames if any were captured
            if frames:
                recording = b''.join(frames)
                print("Recording stream read complete.")
            else:
                print("No frames recorded.")
        except Exception as e:
            # Catch any other exceptions during stream opening or setup
            print(f"Error during recording setup or process: {e}")
            # recording remains None
        finally:
            # Ensure stream is stopped and closed if it was opened
            if stream is not None:
                try:
                    if stream.is_active(): # Check if stream is active before stopping
                         stream.stop_stream()
                         print("Audio stream stopped.")
                except Exception as e:
                    print(f"Error stopping stream: {e}")
                try:
                    stream.close()
                    print("Audio stream closed.")
                except Exception as e:
                    print(f"Error closing stream: {e}")
            # Always terminate PyAudio instance
            p.terminate()
            print("PyAudio instance terminated.")

        print("Recording completed.")
        return recording # Return the recorded data (or None if failed)


    def save_audio(self, recording, filename):
        """Save recorded audio to a WAV file."""
        full_path = os.path.join(self.recordings_dir, filename)
        if recording is None or len(recording) == 0: # Check if recording is empty
            print("Error: No audio data recorded to save.")
            return None
        p_temp = None # Initialize p_temp
        try:
            wf = wave.open(full_path, 'wb')
            wf.setnchannels(1)  # Mono audio
            # Use PyAudio instance to get sample size safely
            p_temp = pyaudio.PyAudio()
            sample_width = p_temp.get_sample_size(pyaudio.paInt16)
            wf.setsampwidth(sample_width)
            wf.setframerate(self.fs)
            wf.writeframes(recording)
            wf.close()
            print(f"Audio saved to {full_path}")
            return full_path
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None
        finally:
             # Ensure p_temp is terminated if it was created
             if p_temp is not None:
                  p_temp.terminate()


    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Google Speech-to-Text."""
        if not os.path.exists(audio_path):
             print(f"Error: Audio file not found at {audio_path}")
             return None
        try:
            client = speech.SpeechClient()
            with open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            if not content:
                 print("Error: Audio file is empty.")
                 return None
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.fs,
                language_code='en-US'
            )
            print("Sending audio for transcription...")
            response = client.recognize(config=config, audio=audio)
            print("Transcription response received.")
            if response.results:
                 transcript = response.results[0].alternatives[0].transcript
                 print(f"Transcription result: '{transcript}'")
                 return transcript
            else:
                 print("Transcription returned no results.")
                 return "" # Return empty string for no results, None for errors
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None # Indicate error

    def store_voice_notes(self, name, notes, category):
        """Store voice notes in the database."""
        try:
            self.database.store_notes(name, notes, category)
            print("Notes stored successfully.")
        except Exception as e:
            print(f"Error storing notes: {e}")

    def prompt_for_name(self,show_input_callback):
        """Prompt user to say a name and confirm."""
        while True:
            print("Please say the name of the person: ")
            name = self.get_voice_input()
            # Check if transcription was successful (not None)
            if name is not None and name != "":
                # Proceed even if name is an empty string, let user confirm
                print(f"Name: Did you say '{name}'? Please confirm with yes or no.")
                show_input_callback("name", name)
                if self.get_yes_no_response():
                    return name # Return the transcribed name (could be empty)
            else:
                 # Handle transcription failure
                 print("Transcription failed. Cannot proceed with name prompt.")
                 # Optionally ask to retry or return an indicator of failure
                 # return None # Example: indicate failure
            print("Let's try again.") # Loop back if confirmation is 'no' or transcription failed


    def prompt_for_notes(self, show_input_callback):
        """Prompt user to say notes about a person."""
        while True:
            print("Please say some notes about the person: ")
            notes = self.get_voice_input()
            if notes is not None and notes != "": # Check if transcription was successful
                print(f"Notes: Did you say '{notes}'? Please confirm with yes or no.")
                show_input_callback("notes", notes)
                if self.get_yes_no_response():
                    return notes # Return transcribed notes (could be empty)
            else:
                 print("Transcription failed. Cannot proceed with notes prompt.")
            print("Let's try again.")


    def prompt_for_category(self, show_input_callback):
        """Prompt user to say a category for a person."""
        while True:
            print("Please say the category of the person (e.g., friend, family): ")
            category = self.get_voice_input()
            if category is not None and category != "": # Check if transcription was successful
                 # Proceed even if category is empty string
                 print(f"Category: Did you say '{category}'? Please confirm with yes or no.")
                 show_input_callback("category", category)
                 if self.get_yes_no_response():
                     # Return category or default if empty and confirmed? Decide logic.
                     return category if category else "Other" # Example: default if empty
            else:
                 print("Transcription failed. Cannot proceed with category prompt.")
            print("Let's try again.")


    def get_yes_no_response(self):
        """Get a yes/no response from user."""
        while True:
            print("Listening for yes or no...")
            response = self.get_voice_input()
            if response is not None: # Check if transcription succeeded
                 response = response.lower()
                 if "yes" in response:
                     return True
                 elif "no" in response:
                     return False
                 else:
                      print("Could not understand 'yes' or 'no'. Please try again.")
            else:
                 # Handle transcription failure during yes/no
                 print("Transcription failed while listening for yes/no. Please try again.")
            # Loop continues if response is not yes/no or if transcription failed


    def get_voice_input(self):
        """Records audio and transcribes it, removing any wake word mentions."""
        recording = self.record_audio()
        if not recording:
            print("Voice input recording failed.")
            return None # Indicate failure: No recording data

        # Use a unique filename to avoid conflicts
        timestamp = int(time.time())
        audio_filename = f"input_{timestamp}.wav"
        audio_full_path = self.save_audio(recording, audio_filename)

        if audio_full_path:
            result = self.transcribe_audio(audio_full_path)
            # Keep the audio file for debugging if transcription fails
            if result is None:
                 print(f"Transcription failed for {audio_full_path}. Keeping file for debugging.")
                 return None # Indicate failure: Transcription error
            else:
                 # Optionally remove the file after successful transcription
                 try:
                     os.remove(audio_full_path)
                     print(f"Removed temporary audio file: {audio_full_path}")
                 except OSError as e:
                     print(f"Error removing temporary audio file {audio_full_path}: {e}")

                 # Clean the result (e.g., remove wake word)
                 # Consider making wake word removal optional or configurable
                 wake_word = "register" # Define wake word if needed for cleaning
                 cleaned_result = result
                 # Use regex for more robust wake word removal if needed
                 if wake_word in result.lower():
                     print(f"Wake word detected in speech: '{result}'. Removing...")
                     # Simple replacement, might need refinement
                     cleaned_result = result.lower().replace(wake_word, "").strip()
                     print(f"Cleaned result: '{cleaned_result}'")
                 return cleaned_result # Return potentially empty string if only wake word was said
        else:
            print("Failed to save audio for transcription")
            return None # Indicate failure: Saving audio failed