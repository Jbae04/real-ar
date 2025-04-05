**AR Face Recognition System - Comprehensive Documentation**  

## **Development Team & Roles**  

### **Asahi Kato**  
- **Primary Responsibilities**:  
  - Developed `detect_and_recognition.py`, the core face detection and recognition module.  
  - Implemented the **API** for remote database synchronization.  
  - Managed **data flow** between the local `.pkl` face database and the SQL database.  
  - Built the **web interface** for manually editing registered users.  
  - Integrated **OpenCV DNN** with Caffe models for real-time face detection.  

### **Yuto Shibuya**  
- **Primary Responsibilities**:  
  - Designed and implemented the **AR Display System** (`ar_display.py`, `display_manager.py`).  
  - Developed **Pygame-based overlays** for real-time feedback on Xreal glasses.  
  - Handled **threading** in `main.py` for concurrent face detection, voice processing, and display updates.  
  - Created **UI states** (normal mode, unknown face prompt, registration workflow).  
  - Optimized **rendering performance** for smooth operation on Raspberry Pi.  

### **Joseph Bae**  
- **Primary Responsibilities**:  
  - Developed **voice integration** (`voice_recognition.py`, `voice_activation.py`).  
  - Implemented **wake word detection** ("register") using **WebRTC VAD**.  
  - Integrated **Google Speech-to-Text** for voice input processing.  
  - Built the **SQLite database** (`database.py`) for storing face metadata.  
  - Designed **voice confirmation prompts** (Yes/No responses).  

### **Collaborative Work**  
- All members contributed to **system architecture** in `main.py`.  
- Jointly worked on **debugging and optimization** for Raspberry Pi deployment.  

---

## **Hardware Requirements & Setup**  

### **Essential Components**  
| **Component**               | **Purpose** | **Notes** |
|-----------------------------|------------|-----------|
| **Raspberry Pi 4/5 (4GB+ RAM)** | Main processing unit | Pi 5 recommended for better performance |
| **Pi Camera Module 3** | Face capture | Connected via CSI port |
| **Xreal AR Glasses** | Display output | Requires HDMI-to-USB-C adapter |
| **AirPods Pro (or alternative mic)** | Voice input/output | Bluetooth or USB mic works |
| **Heatsink + Fan** | Prevents thermal throttling | Critical for sustained operation |
| **MicroSD Card (32GB+)** | OS and storage | Use a high-speed card (A2-rated) |
| **USB-C Power Supply (5V/3A)** | Powers the Pi | Must provide stable power |

### **Connection Setup**  
1. **Display Chain (for Xreal Glasses)**  
   - **Micro HDMI → HDMI coupler → HDMI-to-USB-C → Xreal glasses**  
   - *Alternative*: Use **Pi Connect** if developing from a MacBook.  

2. **Camera Setup**  
   - Connect the **Pi Camera Module 3** to the **CSI port**.  
   - Enable camera in `raspi-config`:  
     ```bash
     sudo raspi-config
     # Navigate to Interface Options → Camera → Enable
     ```  

3. **Audio Setup**  
   - Pair AirPods via Bluetooth:  
     ```bash
     bluetoothctl
     scan on
     pair <MAC_ADDRESS>
     trust <MAC_ADDRESS>
     connect <MAC_ADDRESS>
     ```  
   - Verify audio input:  
     ```bash
     arecord -l  # List recording devices
     ```  

4. **Remote Development (if direct SSH is blocked)**  
   - Use **Tailscale** for secure remote access:  
     ```bash
     curl -fsSL https://tailscale.com/install.sh | sh
     sudo tailscale up
     ```  
   - Now SSH via Tailscale IP:  
     ```bash
     ssh pi@<tailscale-ip>
     ```  

---

## **Software Setup & Installation**  

### **1. Raspberry Pi OS Setup**  
- Flash **Raspberry Pi OS (64-bit)** using **Raspberry Pi Imager**.  
- Enable **SSH** and configure Wi-Fi before booting.  

### **2. Python Environment**  
```bash
# Create a virtual environment
python -m venv ar-env
source ar-env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Additional system dependencies
sudo apt-get install libatlas-base-dev libjasper-dev libqtgui4 libqt4-test
```  

### **3. Google Cloud Speech API Setup**  
1. **Download service account JSON** from Google Cloud Console.  
2. Set environment variable:  
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
   ```  

---

## **System Architecture Deep Dive**  

### **1. Face Detection & Recognition (`detect_and_recognition.py`)**  
- **Face Detection**: Uses **OpenCV DNN** with a **Caffe model** (`deploy.prototxt` + `res10_300x300_ssd_iter_140000.caffemodel`).  
- **Face Encoding**: `face_recognition` library generates **128D embeddings**.  
- **Database Handling**:  
  - Local `.pkl` file for fast face matching.  
  - SQLite (`database.py`) for metadata (name, notes, category).  

### **2. AR Display System (`ar_display.py`)**  
- **Pygame-based UI** with multiple states:  
  - **Normal Mode**: Shows recognized faces.  
  - **Unknown Face Mode**: Prompts user to register.  
  - **Registration Mode**: Guides through voice input.  
- **Overlay System**:  
  - `draw_overlay()` for single-line text.  
  - `draw_multiline_overlay()` for structured info.  

### **3. Voice Processing (`voice_activation.py`, `voice_recognition.py`)**  
- **Wake Word Detection**:  
  - Uses **WebRTC VAD** (Voice Activity Detection).  
  - Listens for "register" before activating voice input.  
- **Speech-to-Text**:  
  - Google Cloud Speech API transcribes voice input.  
  - Confirmation system ("Did you say X? Yes/No").  

### **4. Database & Sync (`database.py`, WebApp API)**  
- **SQLite Structure**:  
  ```sql
  CREATE TABLE face_notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    notes TEXT,
    category TEXT
  )
  ```  
- **WebApp API Sync**:  
  - Checks for updates every **60 seconds**.  
  - Pushes new registrations to the cloud.  

### **5. Main System (`main.py`)**  
- **Threading Model**:  
  - **Face Detection Thread**: Continuously processes camera frames.  
  - **Voice Listener Thread**: Listens for wake word.  
  - **Sync Thread**: Handles database updates.  
- **State Management**:  
  - Controls transitions between **normal**, **unknown face**, and **registration** modes.  

---

## **Running the System**  
```bash
# Activate virtual environment
source ar-env/bin/activate

# Run the main application
python main.py
```  

### **Expected Workflow**  
1. **Face Detected?** → Display name + notes.  
2. **Unknown Face?** → "Say 'Register' to add to database."  
3. **Wake Word Heard?** → Start registration process.  
4. **Voice Input**:  
   - "What is their name?"  
   - "Add notes about this person."  
   - "What category (friend/family)?"  
5. **Confirmation**: "Did you say [X]? Yes/No."  

---

## **Troubleshooting & Optimization**  

### **Common Issues**  
| **Issue** | **Solution** |
|-----------|-------------|
| **Overheating** | Use heatsink + fan, disable unused services. |
| **Display Not Working** | Check HDMI chain, verify Xreal glasses input mode. |
| **Audio Not Detected** | Confirm Bluetooth pairing, check `arecord -l`. |
| **High CPU Usage** | Reduce camera resolution, optimize OpenCV. |

### **Performance Tweaks**  
- **Lower camera resolution** (e.g., 480p instead of 720p).  
- **Disable desktop GUI** (boot to CLI for lower overhead).  
- **Use `sudo raspi-config` to overclock** (if cooling allows).  

---

## **Future Improvements**  
1. **Multi-Camera Support** (USB webcam + Pi Camera).  
2. **Offline Speech Recognition** (e.g., Vosk for privacy).  
3. **Real-Time Face Database Sync** (WebSocket updates).  
4. **Mobile App Companion** (for remote management).  

---

## **Conclusion**  
This system integrates **face recognition**, **voice control**, and **AR display** into a seamless experience. The modular design allows for easy expansion, making it suitable for **security, assistive tech, or smart glasses applications**.  

For questions, contact:  
- **Asahi Kato**: Face detection & API  
- **Yuto Shibuya**: AR display & threading  
- **Joseph Bae**: Voice integration & database
