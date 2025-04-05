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
from utils.text_to_speech import TextToSpeech

class IntegratedFaceVoiceSystem:
    def __init__(self):
        print("Initializing system components...")
        self.face_db = FaceDatabase()
        self.sql_db = Database()
        self.face_app = FaceRecognitionApp(self.face_db, self.sql_db)
        self.voice_recognition = VoiceRecognition(database=self.sql_db, face_database=self.face_db)
        self.voice_activation = VoiceActivation()
        self.tts = TextToSpeech()
        
        # Lock for thread safety when updating shared resources
        self.lock = threading.Lock()
        
        # Initialize DisplayManager with server callback AND shared components
        print("Initializing display manager...")
        self.display_manager = DisplayManager(
            server_callback=self.send_new_face_to_server,
            voice_activation=self.voice_activation,     # 共有インスタンスを渡す
            voice_recognition=self.voice_recognition,   # 共有インスタンスを渡す
            tts=self.tts,                               # 共有インスタンスを渡す
            face_db=self.face_db,                       # 共有インスタンスを渡す
            sql_db=self.sql_db                          # 共有インスタンスを渡す
        )
        
        self.running = True  # Flag to control loops
        # 同期スレッドを追加
        print("Starting database sync thread...")
        self.sync_thread = threading.Thread(target=self.sync_thread, daemon=True)
        self.sync_thread.start()

        # These flags will be managed by DisplayManager now
        self.unknown_faces = []  # 未知の顔を追跡
        
        print("System initialized.")

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
                    
                    # 変更があればローカルDBに反映
                    if data and len(data) > 0:
                        print(f"Received {len(data)} updates from server")
                        for entry in data:
                            # DBエントリを更新
                            success = sql_db.edit(
                                entry["id"], 
                                entry["name"], 
                                entry["note"], 
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
                "note": notes,
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

    def run(self):
        print("Starting face recognition system...")
        self.display_manager.display.show_notification("AR Face Recognition System Started", duration=3)
        
        try:
            while self.running:
                print("loop is running")
                # カメラフレームを取得
                frame = self.face_app.vs.read()
                if frame is None:
                    time.sleep(0.1)  # Wait before retrying (avoid high CPU usage)
                    continue
                
                # 顔検出と認識
                boxes = self.face_app.detect_faces_dnn(frame)
                recognition_results = []
                if boxes:
                    recognition_results = self.face_app.recognize_faces(frame, boxes)
                    self.face_app.recognition_results = recognition_results
                    
                    # 未知の顔を追跡 (for other parts of the system)
                    with self.lock:
                        self.unknown_faces = [f for f in recognition_results if f["name"] == "Unknown"]
                    
                    print(recognition_results)
                
                # DisplayManagerを使用して表示を更新
                if not self.display_manager.process_frame(frame, recognition_results):
                    break
                
                # CPUの過剰使用を防止
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.running = False  # Stop the loop
            self.sync_thread.join(timeout=1.0) 
            # No need to close voice_activation here, DisplayManager will do it
            self.face_app.vs.stop()
            self.display_manager.close()  # DisplayManagerのクリーンアップ
            print("System shutdown complete.")

if __name__ == "__main__":
    system = IntegratedFaceVoiceSystem()
    system.run()
