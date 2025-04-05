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
            if "pulse" in dev['name'].lower():
                p.terminate()
                return i
        p.terminate()
        print("AirPods microphone not found. Using default input device.")
        return None

    def record_audio(self, duration=3, timeout=5):
        """Record audio for a specified duration with timeout."""
        if self.device_index is None:
            print("Error: No input device found.")
            return None

        print("Recording started. Speak now...")
        with pyaudio.PyAudio() as p:
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
                for _ in range(0, int(self.fs / self.chunk_size * duration)):
                    if time.time() - start_time > timeout:
                        break
                    data = stream.read(self.chunk_size, exception_on_overflow=False)
                    frames.append(data)
                recording = b''.join(frames)
            except Exception as e:
                print(f"Error while recording: {e}")
                recording = None
            finally:
                if 'stream' in locals():
                    stream.stop_stream()
                    stream.close()
        print("Recording completed.")
        return recording

    def save_audio(self, recording, filename):
        """Save recorded audio to a WAV file."""
        full_path = os.path.join(self.recordings_dir, filename)
        if recording is None:
            print("Error: No audio recorded.")
            return None
        try:
            wf = wave.open(full_path, 'wb')
            wf.setnchannels(1)  # Mono audio
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.fs)
            wf.writeframes(recording)
            wf.close()
            print(f"Audio saved to {full_path}")
            return full_path
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using Google Speech-to-Text."""
        try:
            client = speech.SpeechClient()
            with open(audio_path, 'rb') as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.fs,
                language_code='en-US'
            )
            response = client.recognize(config=config, audio=audio)
            for result in response.results:
                print(f"Transcription result: {result.alternatives[0].transcript}")
                return result.alternatives[0].transcript
            return ""
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def store_voice_notes(self, name, notes, category):
        """Store voice notes in the database."""
        try:
            self.database.store_notes(name, notes, category)
            print("Notes stored successfully.")
        except Exception as e:
            print(f"Error storing notes: {e}")

    def prompt_for_name(self):
        """Prompt user to say a name and confirm."""
        while True:
            print("Please say the name of the person: ")
            name = self.get_voice_input()
            if name:
                print(f"Did you say '{name}'? Please confirm with yes or no.")
                if self.get_yes_no_response():
                    return name
            print("Let's try again.")

    def prompt_for_notes(self):
        """Prompt user to say notes about a person."""
        while True:
            print("Please say some notes about the person: ")
            notes = self.get_voice_input()
            if notes:
                print(f"Did you say '{notes}'? Please confirm with yes or no.")
                if self.get_yes_no_response():
                    return notes
            print("Let's try again.")

    def prompt_for_category(self):
        """Prompt user to say a category for a person."""
        while True:
            print("Please say the category of the person (e.g., friend, family): ")
            category = self.get_voice_input()
            if category:
                print(f"Did you say '{category}'? Please confirm with yes or no.")
                if self.get_yes_no_response():
                    return category
            print("Let's try again.")

    def get_yes_no_response(self):
        """Get a yes/no response from user."""
        while True:
            print("Listening for yes or no...")
            response = self.get_voice_input().lower()
            if "yes" in response:
                return True
            elif "no" in response:
                return False
            print("Could not understand. Please say yes or no.")

    def get_voice_input(self):
        """Records audio and transcribes it, removing any wake word mentions."""
        recording = self.record_audio()
        if not recording:
            return ""
        audio_path = f"input_{int(time.time())}.wav"
        audio_full_path = self.save_audio(recording, audio_path)
        if audio_full_path:
            result = self.transcribe_audio(audio_full_path)
            if result:
                # Remove the wake word from the result
                wake_word = "register"  # Assuming this is the wake word
                if wake_word in result.lower():
                    print(f"Wake word detected in speech: '{result}'. Removing...")
                    # Remove the wake word and surrounding whitespace
                    result = result.lower().replace(wake_word, "").strip()
                    print(f"Cleaned result: '{result}")
                return result
            else:
                print("Transcription returned empty result")
                return ""
        else:
            print("Failed to save audio for transcription")
            return ""