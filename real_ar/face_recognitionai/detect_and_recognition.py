import numpy as np
import face_recognition
import threading
import time
import pickle
import os
import cv2
from datetime import datetime
from picamera2 import Picamera2
from voice_processing.database.database import Database

# =============================== マルチスレッドによるピカメラキャプチャクラス ================================
class VideoStream:
    def __init__(self, src=0):
        # Picamera2の初期化
        self.camera = Picamera2()
        # カメラの設定（必要に応じて調整してください）
        config = self.camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.camera.configure(config)
        self.camera.start()
        # カメラの起動時間を確保
        time.sleep(2)  # Allow the camera to warm up
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            # Picamera2からフレームを取得
            with self.lock:
                self.frame = self.camera.capture_array()
            # CPUの過剰使用を避けるための短い遅延
            time.sleep(0.01)

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.camera.stop()

    def read(self):
        with self.lock:
            # フレームが存在する場合、コピーを返す
            return self.frame.copy() if self.frame is not None else None

# 顔データベース（登録済みの顔情報）クラス
class FaceDatabase:
    def __init__(self, db_path="face_db.pkl"):
        self.db_path = db_path
        self.encodings = []  # 登録済みの顔エンコーディングリスト
        self.names = []  # 対応する名前リスト
        self.load()

    def load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    data = pickle.load(f)
                    self.encodings = data.get("encodings", [])
                    self.names = data.get("names", [])
                    print(f"Loaded {len(self.names)} faces from database.")
            except Exception as e:
                print(f"Error loading face database: {e}")

    def save(self):
        data = {"encodings": self.encodings, "names": self.names}
        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved {len(self.names)} faces to database.")

    def add_face(self, encoding, id):
        self.encodings.append(encoding)
        self.names.append(id)
        self.save()

# 顔認識アプリケーションクラス
class FaceRecognitionApp:
    def __init__(self, face_db=None, sql_db=None):
        # データベースのロード
        self.db = face_db if face_db else FaceDatabase()
        self.sql_database = sql_db if sql_db else Database()
        # cv2.dnn による顔検出モデルのロード
        # ※ 以下のファイル (deploy.prototxt & res10_300x300_ssd_iter_140000.caffemodel) が必要です。
        prototxt_path = "face_recognitionai/deploy.prototxt"
        model_path = "face_recognitionai/res10_300x300_ssd_iter_140000.caffemodel"
        self.face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
        # マルチスレッドビデオストリームの開始 (picamera2を使用)
        self.vs = VideoStream()
        # 顔認識の頻度 (10フレームに1回)
        self.frame_interval = 10
        self.frame_count = 0
        # 表示更新のインターバル (秒単位)
        self.display_update_interval = 3.0
        self.last_update_time = 0
        # 最新の認識結果を保持 (各要素は {"box": (startX, startY, endX, endY), "name": str, 'encoding': ndarray)}
        self.recognition_results = []

    def detect_faces_dnn(self, frame, conf_threshold=0.5):
        """cv2.dnn を使って顔検出を行い、検出した顔の領域 (startX, startY, endX, endY) を返す"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                boxes.append((startX, startY, endX, endY))
        return boxes

    def recognize_faces(self, frame, boxes):
        if frame is None or not boxes:
            return []
        face_locations = [(y1, x2, y2, x1) for (x1, y1, x2, y2) in boxes]
        encodings = face_recognition.face_encodings(frame, face_locations)
        results = []
        for (box, encoding) in zip(boxes, encodings):
            name = "Unknown"
            notes = ""
            category = ""
            if self.db.encodings:
                matches = face_recognition.compare_faces(self.db.encodings, encoding, tolerance=0.5)
                face_distances = face_recognition.face_distance(self.db.encodings, encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
                if best_match_index is not None and matches[best_match_index]:
                    id = self.db.names[best_match_index]
                    name, notes, category = self.sql_database.get_notes(id) or ("", "")
            results.append({"box": box, "name": name, "encoding": encoding, "notes": notes, "category": category})
        return results

    def register_face(self, face_data):
        """'r' キー押下時に呼ばれ、最新の認識結果から Unknown な顔を登録する（名前入力によりデータベースに追加）処理を行います。"""
        for face in face_data:
            if face["name"] == "Unknown":
                print("=== Face Registration ===")
                name = input("Enter name for the detected face: ")
                if name:
                    self.db.add_face(face["encoding"], name)
                    print(f"Face registered as {name}")
                    return
        print("No unknown face available for registration.")

    def run(self):
        print("Starting Face Recognition App.")
        print("Press 'r' to register an unknown face. Press 'q' to quit.")
        while True:
            frame = self.vs.read()
            if frame is None:
                continue
            self.frame_count += 1
            # cv2.dnn による顕検出（每フレーム実施）
            boxes = self.detect_faces_dnn(frame)
            current_time = time.time()
            # 認識は指定したフレーム間隔または表示更新インターバル毎に実施
            if (current_time - self.last_update_time > self.display_update_interval or
                    self.frame_count % self.frame_interval == 0):
                if self.frame_count % self.frame_interval == 0 and len(boxes) > 0:
                    self.recognition_results = self.recognize_faces(frame, boxes)
                    self.last_update_time = current_time
                # 検出結果の描画（顕枠と名前）
                for res in self.recognition_results:
                    (startX, startY, endX, endY) = res["box"]
                    name = res["name"]
                    notes = res["notes"]
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name}: {notes}", (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("Face Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    # 登録処理：最新の認識結果から Unknown な顕を対象にする
                    self.register_face(self.recognition_results)
        self.vs.stop()
        cv2.destroyAllWindows()

# メイン処理
if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.run()