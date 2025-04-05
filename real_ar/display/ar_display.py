import pygame
import time
import os
import sys
import logging
import math

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Global constants for UI layout
OVERLAY_PADDING = 10
OVERLAY_ALPHA = 150
OVERLAY_Y_OFFSET = 50
BACKGROUND_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
STATUS_FONT_SIZE = 24
LINE_SPACING = 5  # 行間のスペース
HIGHLIGHT_COLOR = (0, 255, 255)  # シアン色のハイライト
SUCCESS_COLOR = (0, 255, 0)  # 緑色の成功表示
WARNING_COLOR = (255, 165, 0)  # オレンジ色の警告表示

def draw_overlay(screen, text, font, screen_width, screen_height, y_offset=OVERLAY_Y_OFFSET, padding=OVERLAY_PADDING, bg_color=(0, 0, 0), text_color=TEXT_COLOR):
    """
    Helper function to create and draw a single-line semi-transparent overlay with the given text.
    Positions the overlay at the bottom center of the screen.
    """
    text_surface = font.render(text, True, text_color)
    text_width, text_height = text_surface.get_size()
    overlay_width = text_width + padding * 2
    overlay_height = text_height + padding * 2
    x = (screen_width - overlay_width) // 2
    y = screen_height - overlay_height - y_offset
    overlay = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
    overlay.fill((*bg_color, OVERLAY_ALPHA))
    screen.blit(overlay, (x, y))
    screen.blit(text_surface, (x + padding, y + padding))
    return (x, y, overlay_width, overlay_height)

def draw_multiline_overlay(screen, text, font, screen_width, screen_height, y_offset=OVERLAY_Y_OFFSET, padding=OVERLAY_PADDING, bg_color=(0, 0, 0), text_color=TEXT_COLOR):
    """
    Helper function to create and draw a semi-transparent overlay with multiline text.
    Splits text by newline and draws each line with a specified line spacing.
    Positions the overall overlay at the bottom center of the screen.
    """
    lines = text.split("\n")
    # Render each line and calculate dimensions
    rendered_lines = [font.render(line, True, text_color) for line in lines]
    line_heights = [surface.get_height() for surface in rendered_lines]
    line_widths = [surface.get_width() for surface in rendered_lines]
    overlay_width = max(line_widths) + padding * 2
    overlay_height = sum(line_heights) + padding * 2 + (len(lines)-1)*LINE_SPACING
    x = (screen_width - overlay_width) // 2
    y = screen_height - overlay_height - y_offset
    overlay = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
    overlay.fill((*bg_color, OVERLAY_ALPHA))
    screen.blit(overlay, (x, y))
    
    current_y = y + padding
    for surface in rendered_lines:
        line_width = surface.get_width()
        # Center each line horizontally within overlay
        line_x = x + (overlay_width - line_width) // 2
        screen.blit(surface, (line_x, current_y))
        current_y += surface.get_height() + LINE_SPACING
    return (x, y, overlay_width, overlay_height)

class ARDisplay:
    """Main display manager for Xreal AR glasses."""
    
    def __init__(self):
        """Initialize the AR display system."""
        pygame.init()
        
        # Get screen info for fullscreen display
        self.info_object = pygame.display.Info()
        self.screen_width = self.info_object.current_w
        self.screen_height = self.info_object.current_h
        
        # 常に全画面表示モードを使用
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.NOFRAME | pygame.FULLSCREEN
        )
        
        pygame.display.set_caption("Xreal Face Recognition")
        
        # Initialize UI components
        self.face_display = FaceRecognitionDisplay(self.screen, self.screen_width, self.screen_height)
        self.unknown_face_display = UnknownFaceDisplay(self.screen, self.screen_width, self.screen_height)
        self.registration_display = RegistrationDisplay(self.screen, self.screen_width, self.screen_height)
        self.status_indicators = StatusIndicators(self.screen, self.screen_width, self.screen_height)
        self.notification_area = NotificationArea(self.screen, self.screen_width, self.screen_height)
        self.no_faces_display = NoFacesDisplay(self.screen, self.screen_width, self.screen_height)
        
        # 新しいUI要素を追加
        self.voice_feedback = VoiceRecognitionFeedback(self.screen, self.screen_width, self.screen_height)
        self.registration_summary = RegistrationSummary(self.screen, self.screen_width, self.screen_height)
        
        # Display state variables
        self.current_mode = "normal"  # normal, unknown, registration
        self.face_detection_active = False
        self.voice_active = False
        self.wake_word_listening = False  # ウェイクワードリスニング状態
        self.wake_word_detected = False   # ウェイクワード検出状態
        self.current_step = 0  # For registration process
        self.notification_text = ""
        self.notification_time = 0
        self.notification_duration = 3  # seconds
        self.registration_data = {  # 登録データを保持
            "name": "",
            "notes": "",
            "category": ""
        }
        
        # 確認プロンプト用の変数
        self.confirmation_mode = False
        self.confirmation_text = ""
        self.confirmation_type = None  # "name", "notes", "category"
        
        # Clear screen on startup (black background)
        self.clear_display()
    
    def clear_display(self):
        """Clear the display (black screen)."""
        self.screen.fill(BACKGROUND_COLOR)
        pygame.display.flip()
    
    def update_status(self, face_detection=None, voice_active=None, wake_word_listening=None):
        """Update system status indicators."""
        if face_detection is not None:
            self.face_detection_active = face_detection
        if voice_active is not None:
            self.voice_active = voice_active
        if wake_word_listening is not None:
            self.wake_word_listening = wake_word_listening
        self.status_indicators.update(self.face_detection_active, self.voice_active, self.wake_word_listening)
    
    def show_notification(self, text, duration=3):
        """Show a temporary notification."""
        self.notification_text = text
        self.notification_time = time.time()
        self.notification_duration = duration
        self.notification_area.show(text)
        
    def show_wake_word_detected(self):
        """ウェイクワード検出時の表示"""
        self.wake_word_detected = True
        self.show_notification("Wake word detected! Now you can register", duration=5)
        self.render()
        
    def show_input_confirmation(self, input_type, text):
        """入力確認プロンプトの表示"""
        self.confirmation_mode = True
        self.confirmation_type = input_type
        
        if input_type == "name":
            self.confirmation_text = f"This person's name is '{text}'? Confirm with Yes or No"
        elif input_type == "notes":
            self.confirmation_text = f"Notes about this person: '{text}'? Confirm with Yes or No"
        elif input_type == "category":
            self.confirmation_text = f"Category for this person: '{text}'? Confirm with Yes or No"
            
        self.render()
    
    def display_recognized_face(self, face_data):
        """Display information for a recognized face."""
        self.current_mode = "normal"
        self.face_display.update(face_data)
        self.update_status(face_detection=True)
        self.render()
    
    def display_unknown_face(self, face_box):
        """Display UI for an unknown face."""
        self.current_mode = "unknown"
        self.unknown_face_display.update(face_box)
        self.update_status(face_detection=True)
        self.render()
    
    def update_voice_feedback(self, text, is_final=False):
        """音声認識のフィードバックを更新"""
        self.voice_feedback.update(text, is_final)
        self.render()
    
    def start_registration(self, face_box):
        """Start the face registration process."""
        self.current_mode = "registration"
        self.current_step = 1
        # 登録データをリセット
        self.registration_data = {
            "name": "",
            "notes": "",
            "category": ""
        }
        self.registration_display.update(face_box, step=self.current_step)
        self.update_status(face_detection=True, voice_active=True)
        self.render()
    
    def next_registration_step(self, input_text=""):
        """Move to the next step in registration."""
        # 現在のステップの入力を保存
        if self.current_step == 1 and input_text:
            self.registration_data["name"] = input_text
        elif self.current_step == 2 and input_text:
            self.registration_data["notes"] = input_text
        elif self.current_step == 3 and input_text:
            self.registration_data["category"] = input_text
        
        self.current_step += 1
        if self.current_step > 3:  # Registration complete
            # 登録サマリーを表示
            self.registration_summary.update(self.registration_data)
            self.show_notification("Face registered successfully!")
            self.current_mode = "normal"
            self.current_step = 0
            self.update_status(voice_active=False)
        else:
            self.registration_display.update(None, step=self.current_step,
                                            current_data=self.registration_data)
        self.render()
    
    def render(self):
        """Render the current UI state on a black background."""
        self.screen.fill(BACKGROUND_COLOR)
        
        if self.current_mode == "normal":
            if self.face_detection_active:
                self.face_display.render()
            else:
                # No faces detected
                self.no_faces_display.render()
        elif self.current_mode == "unknown":
            self.unknown_face_display.render()
        elif self.current_mode == "registration":
            self.registration_display.render() # Keep registration steps display
            # Removed voice feedback display during registration
            # self.voice_feedback.render()

            # 登録完了時はサマリーを表示
            if self.current_step == 0 and any(self.registration_data.values()):
                self.registration_summary.render()
        
        # 音声認識中は常に音声フィードバックを表示 (Removed)
        # if self.voice_active and self.current_mode != "registration":
        #     self.voice_feedback.render()

        # ウェイクワード検出表示
        if self.wake_word_detected:
            self._draw_wake_word_notification()
            
        # 確認プロンプト表示
        if self.confirmation_mode:
            self._draw_confirmation_prompt()
        
        self.status_indicators.render()
        if self.notification_text and time.time() - self.notification_time < self.notification_duration:
            self.notification_area.render()
        
        pygame.display.flip()
        
    def _draw_wake_word_notification(self):
        """ウェイクワード検出通知の描画"""
        text = "Wake word detected! Now you can register"
        font = pygame.font.Font(None, 36)
        draw_overlay(self.screen, text, font, self.screen_width, self.screen_height,
                    y_offset=self.screen_height // 2, bg_color=(0, 100, 0))
    
    def _draw_confirmation_prompt(self):
        """確認プロンプトの描画"""
        font = pygame.font.Font(None, 36)
        draw_overlay(self.screen, self.confirmation_text, font, self.screen_width, self.screen_height,
                    y_offset=self.screen_height // 4, bg_color=(0, 0, 100))
    
    def process_events(self):
        """Process pygame events and return True if quit event occurs."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False
    
    def close(self):
        """Clean up resources."""
        pygame.quit()

class FaceRecognitionDisplay:
    """UI component for displaying recognized face information as a non-intrusive overlay."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.name = ""
        self.category = ""
        self.notes = ""
        self.font = pygame.font.Font(None, 48)
    
    def update(self, face_data):
        """Update the display with new face data."""
        if face_data:
            self.name = face_data.get("name", "")
            self.category = face_data.get("category", "")
            self.notes = face_data.get("notes", "")
    
    def render(self):
        """Render recognized face information with each piece on a separate line."""
        info_lines = []
        info_lines.append(f"Name: {self.name}")
        if self.category:
            info_lines.append(f"Category: {self.category}")
        if self.notes:
            info_lines.append(f"Notes: {self.notes}")
        info_text = "\n".join(info_lines)
        draw_multiline_overlay(self.screen, info_text, self.font, self.screen_width, self.screen_height)

class UnknownFaceDisplay:
    """UI component for displaying unknown face information as a non-intrusive overlay."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font = pygame.font.Font(None, 48)
    
    def update(self, face_box):
        # In this mode, face_box is not used.
        pass
    
    def render(self):
        """Render unknown face information with a prompt to register."""
        info_text = "Unknown Person\nSay 'Register' to add to database"
        draw_multiline_overlay(self.screen, info_text, self.font, self.screen_width, self.screen_height)

class RegistrationDisplay:
    """UI component for displaying registration instructions as a non-intrusive overlay."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.step = 0
        self.font = pygame.font.Font(None, 36)
        self.instruction_font = pygame.font.Font(None, 36)
        self.data_font = pygame.font.Font(None, 30)
        self.current_data = {"name": "", "notes": "", "category": ""}
    
    def update(self, face_box, step=1, current_data=None):
        """Update registration step and current data."""
        self.step = step
        if current_data:
            self.current_data = current_data
    
    def render(self):
        """Render registration instructions and current input data."""
        # ステップ指示テキスト
        if self.step == 1:
            step_text = "Step 1: Say the name"
        elif self.step == 2:
            step_text = "Step 2: Say notes about the person"
        elif self.step == 3:
            step_text = "Step 3: Say the category (friend, family, etc.)"
        else:
            step_text = "Registration complete"
        
        # メインの指示テキストを表示
        draw_overlay(self.screen, step_text, self.instruction_font,
                    self.screen_width, self.screen_height,
                    y_offset=OVERLAY_Y_OFFSET,
                    bg_color=(0, 0, 100))  # 青みがかった背景
        
        # 現在までの入力データを表示（上部に配置）
        data_lines = []
        if self.current_data["name"]:
            data_lines.append(f"Name: {self.current_data['name']}")
        if self.current_data["notes"]:
            data_lines.append(f"Notes: {self.current_data['notes']}")
        if self.current_data["category"]:
            data_lines.append(f"Category: {self.current_data['category']}")
        
        if data_lines:
            data_text = "\n".join(data_lines)
            draw_multiline_overlay(self.screen, data_text, self.data_font,
                                  self.screen_width, self.screen_height,
                                  y_offset=self.screen_height - 150,  # 画面上部に配置
                                  bg_color=(50, 50, 50))  # 暗めの背景

class StatusIndicators:
    """UI component for displaying system status indicators (small icons)."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.face_detection_active = False
        self.voice_active = False
        self.wake_word_listening = False
        self.font = pygame.font.Font(None, STATUS_FONT_SIZE)
        self.animation_time = 0
    
    def update(self, face_detection_active, voice_active, wake_word_listening=False):
        self.face_detection_active = face_detection_active
        self.voice_active = voice_active
        self.wake_word_listening = wake_word_listening
    
    def render(self):
        x = self.screen_width - 200
        y = 20
        
        # 顔検出状態
        face_color = (0, 255, 0) if self.face_detection_active else (255, 0, 0)
        pygame.draw.circle(self.screen, face_color, (x, y), 8)
        face_text = self.font.render("Face Detection", True, TEXT_COLOR)
        self.screen.blit(face_text, (x + 20, y - 8))
        
        # 音声認識状態
        voice_color = (0, 255, 0) if self.voice_active else (255, 0, 0)
        pygame.draw.circle(self.screen, voice_color, (x, y + 30), 8)
        voice_text = self.font.render("Voice Active", True, TEXT_COLOR)
        self.screen.blit(voice_text, (x + 20, y + 22))
        
        # ウェイクワードリスニング状態 - Always show this indicator
        # Change color based on listening state (green when listening, red/orange when not)
        if self.wake_word_listening:
            # アニメーション効果（点滅するサークル）
            self.animation_time += 0.05
            pulse_size = 8 + math.sin(self.animation_time * 5) * 3
            wake_color = SUCCESS_COLOR  # 緑色 (Green)
            wake_text_color = SUCCESS_COLOR
        else:
            # Static circle when not listening
            pulse_size = 8
            wake_color = (255, 0, 0)  # 赤色 (Red)
            wake_text_color = WARNING_COLOR  # オレンジ色 (Orange)
        
        pygame.draw.circle(self.screen, wake_color, (x, y + 60), pulse_size)
        wake_text = self.font.render("Listening for 'Register'", True, wake_text_color)
        self.screen.blit(wake_text, (x + 20, y + 52))

class NotificationArea:
    """UI component for displaying temporary notifications."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.text = ""
        self.font = pygame.font.Font(None, 36)
    
    def show(self, text):
        self.text = text
    
    def render(self):
        if not self.text:
            return
        draw_overlay(self.screen, self.text, self.font, self.screen_width, self.screen_height)

class VoiceRecognitionFeedback:
    """音声認識のリアルタイムフィードバックを表示するUI要素"""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.text = ""
        self.is_final = False
        self.font = pygame.font.Font(None, 32)
        self.animation_time = 0
        self.dots = ""
    
    def update(self, text, is_final=False):
        """音声認識テキストを更新"""
        self.text = text
        self.is_final = is_final
    
    def render(self):
        """音声認識フィードバックを表示"""
        if not self.text and not self.is_final:
            # テキストがない場合は「リスニング中」のみ表示
            self.animation_time += 0.1
            if self.animation_time > 1:
                self.animation_time = 0
                self.dots = "." * (len(self.dots) % 3 + 1)
            
            listening_text = f"Listening{self.dots}"
            draw_overlay(self.screen, listening_text, self.font,
                        self.screen_width, self.screen_height,
                        y_offset=150, bg_color=(50, 50, 50))
        elif self.text:
            # 認識テキストを表示
            text_color = SUCCESS_COLOR if self.is_final else TEXT_COLOR
            bg_color = (0, 100, 0) if self.is_final else (50, 50, 50)
            
            # 「認識中」または「認識完了」のプレフィックス
            prefix = "Recognized: " if self.is_final else "Recognizing: "
            display_text = f"{prefix}{self.text}"
            
            draw_overlay(self.screen, display_text, self.font,
                        self.screen_width, self.screen_height,
                        y_offset=150, bg_color=bg_color, text_color=text_color)

class NoFacesDisplay:
    """UI component for displaying a message when no faces are detected."""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.font = pygame.font.Font(None, 48)
    
    def render(self):
        """Render a message indicating no faces are detected."""
        info_text = "No faces detected"
        draw_multiline_overlay(self.screen, info_text, self.font, self.screen_width, self.screen_height)


class RegistrationSummary:
    """登録情報のサマリーを表示するUI要素"""
    
    def __init__(self, screen, screen_width, screen_height):
        self.screen = screen
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.data = {"name": "", "notes": "", "category": ""}
        self.font = pygame.font.Font(None, 36)
        self.title_font = pygame.font.Font(None, 42)
    
    def update(self, data):
        """登録データを更新"""
        self.data = data
    
    def render(self):
        """登録サマリーを表示"""
        if not any(self.data.values()):
            return
        
        # タイトル行
        title_text = "Registration Complete"
        title_surface = self.title_font.render(title_text, True, SUCCESS_COLOR)
        title_width, title_height = title_surface.get_size()
        
        # データ行
        data_lines = []
        if self.data["name"]:
            data_lines.append(f"Name: {self.data['name']}")
        if self.data["category"]:
            data_lines.append(f"Category: {self.data['category']}")
        if self.data["notes"]:
            data_lines.append(f"Notes: {self.data['notes']}")
        
        data_text = "\n".join(data_lines)
        
        # 全体のテキスト
        full_text = f"{title_text}\n\n{data_text}"
        
        # サマリーを中央に表示
        draw_multiline_overlay(self.screen, full_text, self.font,
                              self.screen_width, self.screen_height,
                              y_offset=self.screen_height // 3,  # 画面中央付近
                              bg_color=(0, 100, 0),  # 緑色の背景
                              text_color=TEXT_COLOR)