import customtkinter as ctk
import cv2
import time

from components.theme import Theme
from components.text import BodyText
from collections import deque
from PIL import Image

class VideoDisplay(ctk.CTkFrame):
    def __init__(self, parent, title_text, video_path):
        super().__init__(parent, fg_color=Theme.WHITE)

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.grid_columnconfigure((0,1), weight=1)

        self.title = BodyText(self, text=title_text)
        self.title.grid(row=0, column=0, padx=30, sticky="w")

        self.display_label = ctk.CTkLabel(self, text="")
        self.display_label.grid(row=1, column=0, padx=10, pady=20)

        self.stats_label = None

        #Data list
        self.fps_list = deque(maxlen=30)
        self.latency_list = deque(maxlen=30)

        # self.update_view()
    
    def update_view(self):
        time_start = time.time()
        real_h, real_w, _ = frame.shape #height, width, channels
        ret, frame = self.cap.read() #Tries to read frame

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #Replay video when ending
            self.after(30, self.update_frame)
            return
    
        frame_resized = cv2.resize(frame, (500, 300)) #Shrinking for better CPU performance
        cv2_img = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2_img)
        ctk_img = ctk.CTkImage(light_image=pil_img, size=(500, 300)) #UI size

        self.display_label.configure(image=ctk_img)
        self.display_label.image = ctk_img
        
        #Calculation
        time_end = time.time()
        time_diff = time_end - time_start

        #FPS Calculation
        self.fps_list.append(time_diff)

        if len(self.fps_list) > 0:
            avg_fps = sum(self.fps_list) / len(self.fps_list)
            if avg_fps > 0:
                mean_fps = 1 / avg_fps
            else:
                mean_fps = 0

        #Latency Calculation
        latency_ms = time_diff * 1000
        self.latency_list.append(latency_ms)

        if len(self.latency_list) > 0:
            avg_latency = sum(self.latency_list) / len(self.latency_list)
        
        stats_text = (
            f"Resolution: {real_w} x {real_h}\n"
            f"FPS: {mean_fps:.1f}\n"
            f"Latency: {avg_latency:.1f} ms"
        )

        #Update UI stats
        if self.stats_label is None:
            self.stats_label = BodyText(self, text=stats_text)
            self.stats_label.grid(row = 2, column=0, padx=(110,0), pady=(2,10), sticky="w")
        else:
            self.stats_label.configure(text=stats_text)