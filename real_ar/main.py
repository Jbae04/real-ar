# main.py
import threading
import time
import cv2
import os
import sys
import requests
import json
from voice_processing.database.database import Database
from face_recognitionai.detect_and_recognition import VideoStream, FaceDatabase, FaceRecognitionApp
from voice_processing.voice_recognition import VoiceRecognition
from voice_processing.voice_activation import VoiceActivation
from display.display_manager import DisplayManager
from display.ar_display import ARDisplay # Import ARDisplay


class IntegratedFaceVoiceSystem:
  def __init__(self):
       print("Initializing system components...")
       self.face_db = FaceDatabase()
       self.sql_db = Database()
       self.face_app = FaceRecognitionApp(self.face_db, self.sql_db)
       self.voice_recognition = VoiceRecognition(database=self.sql_db, face_database=self.face_db)
       self.voice_activation = VoiceActivation()


       # Initialize ARDisplay first
       print("Initializing AR Display...")
       self.ar_display = ARDisplay() # Create ARDisplay instance here


       # Initialize DisplayManager, passing the ARDisplay instance
       print("Initializing display manager...")
       # Pass other necessary shared components if DisplayManager still needs them
       self.display_manager = DisplayManager(ar_display=self.ar_display) # Pass ar_display


       self.running = True  # Flag to control loops
       # 同期スレッドを追加
       print("Starting database sync thread...") # Corrected indentation
       self.sync_threads = threading.Thread(target=self.sync_thread, daemon=True) # Corrected indentation
       self.sync_threads.start() # Corrected indentation


       self.listening_for_wake_word = True # Corrected indentation
       self.registration_mode = False # Corrected indentation
       self.voice_active = False # Corrected indentation


       self.unknown_faces = []  # 未知の顔を追跡 # Corrected indentation


       # Lock for thread safety when updating shared resources # Corrected indentation
       self.lock = threading.Lock() # Corrected indentation
       self.start_registration_requested = False # Flag to request registration from main thread # Corrected indentation


       print("System initialized.") # Corrected indentation


  def voice_listener_thread(self):
       print("Voice listener thread started - listening for wake word")
       while self.running:
           try:
               # Get current display state (assuming ar_display is thread-safe for reading attributes)
               # If not, this read might need protection or a different approach
               current_display_listening_state = self.ar_display.wake_word_listening


               # Determine if we should be listening based on system state
               should_listen = False
               with self.lock:
                   should_listen = self.listening_for_wake_word and not self.registration_mode and len(self.unknown_faces) > 0


                   # Update display status only if the listening state *should* change
                   if should_listen != current_display_listening_state:
                       print(f"Voice listener: Updating display listening state to {should_listen}")
                       # Call update_status from the main thread via a queue or flag if not thread-safe
                       # For now, assuming it might be safe or needs review later. Calling directly:
                       try:
                           self.ar_display.update_status(wake_word_listening=should_listen)
                       except Exception as display_e:
                           print(f"Error updating display status from voice thread: {display_e}")




                   # If we should be listening, attempt to detect wake word
                   if self.listening_for_wake_word and not self.registration_mode and self.unknown_faces:
                       print("Listening for wake word...")
                       # This call blocks, consider alternatives if responsiveness is critical
                       if self.voice_activation.listen_for_wake_word():
                           print("Wake word detected! Requesting registration mode...")
                           print("registration_mode:", self.registration_mode)
                           print("unknown_faces:", self.unknown_faces)
                           # Double-check state before requesting registration
                           if not self.registration_mode and self.unknown_faces:
                               self.listening_for_wake_word = False # Stop internal listening flag
                               self.registration_mode = True
                               self.start_registration_requested = True # Signal main thread
                               print("Registration requested.")
                               # Display status (listening=False) will be updated by main thread later
                           else:
                               print("Wake word detected, but state changed before registration request.")
                       # else: Wake word not detected, loop continues


                   # Prevent busy-waiting
               time.sleep(0.2) # Adjust sleep time as needed


           except Exception as e:
               print(f"Error in voice listener thread: {e}")
               # Attempt to reset display state on error? Risky from non-main thread.
               # Consider signaling main thread about the error.
               time.sleep(1) # Wait longer after an error


  def sync_thread(self):
       """
       サーバーとのデータ同期を行うスレッド関数
       1分おきにサーバーに変更がないか確認し、あれば反映する
       """
       print("Database sync thread started")
       sql_db = Database()
       while self.running:
           try:
               # サーバーに変更データがあるか確認
               response = requests.get("https://autohomework.io/api/sync/check")


               if response.status_code == 200:
                   data = response.json()
                   print("Yuuuuuuuuuto")


                   # 変更があればローカルDBに反映
                   if data and len(data) > 0:
                       print(f"Received {len(data)} updates from server")
                       for entry in data:
                           # DBエントリを更新
                           success = sql_db.edit(
                               entry["id"],
                               entry["name"],
                               entry["notes"],
                               entry["category"]
                           )


                           if success:
                               print(f"Updated local entry: {entry['name']} (ID: {entry['id']})")


                               # 顔認識DBも更新する必要がある場合、ここで処理
                               # 名前が変わった場合は顔認識DBも更新が必要かも


               # 60秒待機
               time.sleep(60)


           except Exception as e:
               print(f"Error in sync thread: {e}")
               time.sleep(10)  # エラー時は短い間隔で再試行


  def send_new_face_to_server(self, name, notes, category):
       """
       新しい顔データをサーバーに送信する関数


       Args:
           name (str): 人物の名前
           notes (str): メモ
           category (str): カテゴリ


       Returns:
           bool: 送信が成功したかどうか
       """
       try:
           # POSTデータの準備
           data = {
               "name": name,
               "notes": notes,
               "category": category
           }


           # サーバーにデータを送信
           response = requests.post(
               "https://autohomework.io/api/sync/add",
               json=data,
               headers={"Content-Type": "application/json"}
           )


           if response.status_code == 200:
               result = response.json()
               print(f"Successfully sent new face data to server, ID: {result.get('id')}")
               return True
           else:
               print(f"Failed to send data to server. Status code: {response.status_code}")
               return False


       except Exception as e:
           print(f"Error sending data to server: {e}")
           return False


  def register_new_face(self, face_data):
       print("Registering a new face...")


       # Validate face data before proceeding
       if not face_data or "box" not in face_data or "encoding" not in face_data:
           print("Error: Invalid face data received. Aborting registration.")
           return


       try:
           self.ar_display.update_status(wake_word_listening=False, voice_active=True)


           # 名前入力
           self.ar_display.update_voice_feedback("Listening for name...", is_final=False)
           name = self.voice_recognition.prompt_for_name(self.ar_display.show_input_confirmation)


           if not name: # Handle empty string or None
               self.ar_display.update_voice_feedback("No name detected", is_final=True) # Update feedback
               self.ar_display.show_notification("Registration failed: No name provided")
               print("No valid name provided. Registration aborted.")
               # Reset state as we are returning early, bypassing finally
               with self.lock:
                    self.registration_mode = False
                    self.listening_for_wake_word = True # Allow listening again
               self.ar_display.update_status(voice_active=False) # Update status
               return


           # Show confirmation for the name
           self.ar_display.update_voice_feedback(name, is_final=True)
           # Proceed to next step in UI (e.g., highlight notes input)
           self.ar_display.next_registration_step(name)


           # メモ入力
           self.ar_display.update_voice_feedback("Listening for notes...", is_final=False)
           notes = self.voice_recognition.prompt_for_notes(self.ar_display.show_input_confirmation) or "" # Keep default empty string
           # Show confirmation for notes (even if empty)
           self.ar_display.update_voice_feedback(notes if notes else "No notes provided", is_final=True)
           # Proceed to next step in UI
           self.ar_display.next_registration_step(notes)


           # カテゴリ入力
           self.ar_display.update_voice_feedback("Listening for category...", is_final=False)
           category = self.voice_recognition.prompt_for_category(self.ar_display.show_input_confirmation) or "Other" # Keep default "Other"
           self.ar_display.confirmation_mode = False
           self.ar_display.render()
          
           # Show confirmation for category
           self.ar_display.update_voice_feedback(category, is_final=True)
           # Proceed to next step (final step/summary)
           self.ar_display.next_registration_step(category)

           # データを保存
           self.voice_recognition.store_voice_notes(name, notes, category)
           id = self.sql_db.get_id(name)

           self.face_db.add_face(face_data["encoding"], id)
           # サーバーに新しい顔データを送信
           cloud_thread = threading.Thread(target=self.send_new_face_to_server, args=(name, notes, category))
           cloud_thread.start()
          
           # 登録完了
           self.ar_display.show_notification(f"Face registered as {name}")
           print(f"Face registered as {name} with notes: {notes} and category: {category}")


       except Exception as e:
           print(f"Error during face registration: {e}")
           # Notify user about the error
           try:
                # Attempt to show notification, but this might also fail if display is the issue
                self.ar_display.show_notification(f"Registration Error: {e}", duration=5)
           except Exception as display_e:
                print(f"Also failed to show error notification: {display_e}")
       finally:
           # Ensure registration mode is always reset
           with self.lock: # Use lock for consistency, though main thread access might make it less critical here
                self.registration_mode = False
                # Ensure listening can restart even if registration failed mid-way
                self.listening_for_wake_word = True
           self.ar_display.update_status(voice_active=False)


  def run(self):
       # 音声リスナースレッドを開始
       voice_thread = threading.Thread(target=self.voice_listener_thread, daemon=True)
       voice_thread.start()


       print("Starting face recognition system...")
       self.ar_display.show_notification("AR Face Recognition System Started", duration=3)


       try:
           while self.running:
               print("loop is running")
               # カメラフレームを取得
               frame = self.face_app.vs.read()
               if frame is None:
                   time.sleep(0.1)
                   continue


               # 顔検出と認識
               boxes = self.face_app.detect_faces_dnn(frame)
               recognition_results = [] # Initialize before the check
               if boxes:
                   # Only perform recognition if faces are detected
                   recognition_results = self.face_app.recognize_faces(frame, boxes)
                   self.face_app.recognition_results = recognition_results # Keep this if face_app uses it elsewhere
                   print(recognition_results) # Debug print


               with self.lock:
                   self.unknown_faces = [f for f in recognition_results if f["name"] == "Unknown"]


               self.display_manager.update_display_with_recognition_results(recognition_results)


               # Check for display events directly using ar_display
               if self.ar_display.process_events():
                   break


               # Check if registration was requested by the voice listener thread
               face_to_register = None
               with self.lock:
                   if self.start_registration_requested and self.unknown_faces:
                       print("Main loop: Registration requested and unknown face available.")
                       self.start_registration_requested = False # Reset the flag
                       # Get the face data to register (use a copy if needed)
                       face_to_register = self.unknown_faces[0].copy() # Use copy to avoid modification issues
                   elif self.start_registration_requested:
                       # Requested but no unknown faces, reset flag
                       print("Main loop: Registration requested but no unknown faces.")
                       self.start_registration_requested = False
                       self.registration_mode = False # Exit registration mode
                       self.listening_for_wake_word = True # Allow listening again


               # Call registration from the main thread if a face is available
               if face_to_register:
                   print("Main loop: Starting registration UI and calling register_new_face...")
                   # Start the registration UI from the main thread
                   self.ar_display.start_registration(face_to_register["box"])
                   # Update status before starting registration (now safe in main thread)
                   self.ar_display.update_status(wake_word_listening=False, voice_active=True)
                   # Call the registration logic
                   self.register_new_face(face_to_register)
                   # Reset flags after registration attempt (register_new_face handles its own finally block)
                   # Flags might be reset inside register_new_face's finally, but ensure listening restarts
                   with self.lock:
                        self.listening_for_wake_word = True
                        # self.voice_active should be False after registration, handled in register_new_face finally
                   print("Main loop: Returned from register_new_face.")


               # Render the final display state for this frame
               self.ar_display.render() # Ensure display is updated every loop iteration


               # CPUの過剰使用を防止
               time.sleep(0.01)


       except Exception as e:
           print(f"Error in main loop: {e}")
       finally:
           self.running = False  # Stop the loop
           # Clean up threads and resources
           print("Shutting down threads and resources...")
           if voice_thread.is_alive():
                print("Joining voice thread...")
                voice_thread.join(timeout=1.0)
           if self.sync_threads.is_alive():
                print("Joining sync thread...")
                self.sync_threads.join(timeout=1.0)
           if hasattr(self, 'voice_activation'):
                print("Closing voice activation...")
                self.voice_activation.close()
           if hasattr(self, 'face_app') and hasattr(self.face_app, 'vs'):
                print("Stopping video stream...")
                self.face_app.vs.stop()
           # Close DisplayManager first if it holds resources like ARDisplay (if not passed)
           # If ARDisplay is owned by IntegratedFaceVoiceSystem, close it here.
           if hasattr(self, 'display_manager'):
                print("Closing display manager...")
                self.display_manager.close() # DisplayManager's close should handle its own resources (like ARDisplay if owned)
           # If ARDisplay is owned here, close it explicitly
           if hasattr(self, 'ar_display'):
                print("Closing AR display...")
                self.ar_display.close() # Ensure ARDisplay is closed


           print("System shutdown complete.")


if __name__ == "__main__":
  system = IntegratedFaceVoiceSystem()
  system.run()
