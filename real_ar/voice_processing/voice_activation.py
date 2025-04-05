import pyaudio
import wave
import time
import os
from google.cloud import speech
import webrtcvad
import threading

class VoiceActivation:
    def __init__(self, wake_word="register"):
        self.wake_word = wake_word.lower()
        self.frame_duration = 30  # ms
        self.sample_rate = 16000
        self.chunk_size = int(self.sample_rate * self.frame_duration / 1000)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = True
        self.device_index = self.find_airpods_device()
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(3)  # Aggressive mode
        self.speech_detected = False
        self.audio_buffer = []
        
        # Threading lock to ensure safe access to shared resources
        self.lock = threading.Lock()

        # Get credentials path from environment variable or use default
        self.credentials_path = os.environ.get("GOOGLE_CLOUD_CREDENTIALS")
        if not self.credentials_path:
            self.credentials_path = "/home/sfhacks/Downloads/fiery-plate-453204-m2-05ed6b4f1274.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        self.client = speech.SpeechClient()
        self.last_detection_time = 0
        self.cooldown_duration = 5 

    def find_airpods_device(self):
        target_name = "Lynette's AirPods Pro - Find My"
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            if target_name in dev['name']:
                print(f"Found AirPods at index {i}")
                return i

        print("AirPods not found. Using default device.")
        return self.audio.get_default_input_device_info()['index']

    def initialize_audio_stream(self):
        try:
            with self.lock:
                self.stream = self.audio.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=self.device_index,
                    frames_per_buffer=self.chunk_size
                )
            print(f"Audio stream initialized successfully with chunk size {self.chunk_size}")
            return True
        except Exception as e:
            print(f"Error initializing audio stream: {e}")
            print("Available audio devices:")
            for i in range(self.audio.get_device_count()):
                print(self.audio.get_device_info_by_index(i))
            return False

    def process_audio_chunk(self, data):
        try:
            with self.lock:
                if len(data) != self.chunk_size * 2:  # 16-bit samples
                    print(f"Warning: Received {len(data)} bytes, expected {self.chunk_size * 2} bytes")
                    return None

                if self.vad.is_speech(data, self.sample_rate):
                    self.speech_detected = True
                    self.audio_buffer.append(data)
                    return None  # Continue recording
                else:
                    if self.speech_detected:
                        # End of speech detected
                        self.speech_detected = False
                        if len(self.audio_buffer) > 0:
                            return self.audio_buffer.copy()  # Return a copy of the buffer
                        self.audio_buffer = []

            return None
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return None

    def listen_for_wake_word(self):
        try:
            # Initialize audio stream if needed
            if self.stream is None:
                print("Initializing audio stream...")
                if not self.initialize_audio_stream():
                    return False

            # Set the duration for which to listen for the wake word
            listen_duration = self.cooldown_duration  # Use the cooldown_duration for consistency
            start_time = time.time()

            # Clear the audio buffer before starting to listen
            self.audio_buffer = []
            self.speech_detected = False  # Reset speech detection state

            # Read and process audio chunks until the specified duration is reached
            while time.time() - start_time < listen_duration:
                try:
                    # Read a single chunk of audio data
                    with self.lock:
                        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                    
                    # Process the audio chunk and check if it contains speech
                    audio_chunks = self.process_audio_chunk(data)
                    if audio_chunks:
                        # If speech is detected and completed, process it
                        try:
                            # Save the buffered audio to a file
                            audio_file = self.save_audio(audio_chunks)
                            if audio_file:
                                # Transcribe the saved audio file
                                text = self.transcribe_audio(audio_file)
                                if text:
                                    print(f"Recognized text: '{text}'")
                                    if self.wake_word in text.lower():
                                        self.last_detection_time = time.time()
                                        print("Wake word detected!")
                                        self.reset_state()
                                        return True
                                else:
                                    print("Transcription returned empty result")
                            else:
                                print("Failed to save audio for transcription")
                        except Exception as e:
                            print(f"Error during transcription process: {e}")
                        
                        # Clear the buffer after processing
                        self.audio_buffer = []
                except Exception as e:
                    print(f"Error processing audio chunk: {e}")

            # If we get here with speech detected but not fully processed, try to process it
            if self.speech_detected and self.audio_buffer:
                try:
                    audio_file = self.save_audio(self.audio_buffer)
                    if audio_file:
                        text = self.transcribe_audio(audio_file)
                        if text and self.wake_word in text.lower():
                            self.last_detection_time = time.time()
                            print("Wake word detected!")
                            self.reset_state()
                            return True
                except Exception as e:
                    print(f"Error during final transcription: {e}")

        except KeyboardInterrupt:
            print("Stopped listening for wake word")
        except Exception as e:
            print(f"Error while listening: {e}")
            self.reset_state()
        
        # Function will return False if wake word not detected
        return False

    def save_audio(self, frames, filename="recorded_audio.wav"):
        """Saves recorded audio to a WAV file."""
        if not frames or len(frames) == 0:
            print("Error: No audio data to save")
            return None
        filepath = os.path.join("recordings", filename)
        try:
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            wf.close()
            print(f"Audio saved to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving audio: {e}")
            return None

    def transcribe_audio(self, audio_file):
        """Transcribes the audio file using Google Speech-to-Text."""
        try:
            with open(audio_file, 'rb') as audio_file:
                content = audio_file.read()
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.sample_rate,
                language_code='en-US'
            )
            response = self.client.recognize(config=config, audio=audio)
            for result in response.results:
                return result.alternatives[0].transcript.lower()
            return ""
        except Exception as e:
            print(f"Error during transcription: {e}")
            return ""

    def close(self):
        """Cleans up resources."""
        self.running = False
        if self.stream:
            try:
                with self.lock:
                    self.stream.stop_stream()
                    self.stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")
            self.audio.terminate()

    def reset_state(self):
        """Reset voice activation state."""
        with self.lock:
            self.speech_detected = False
            self.audio_buffer = []
            self.stream = None