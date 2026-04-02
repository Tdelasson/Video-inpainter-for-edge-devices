import customtkinter as ctk
import cv2
import os
import time

from components.header_content import Header
from components.text import BodyText
from components.theme import Theme
from PIL import Image
from collections import deque


class MainPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.WHITE)
        self.is_active = True

        #Placeholder using downloaded video instead of live video footage from camera
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(base_path, "assets", "video_placeholder.mp4")

        self.cap = cv2.VideoCapture(video_path)

        self.grid_columnconfigure((0, 1), weight=1)
       
        self.left_frame = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.left_frame.grid_columnconfigure((0,1), weight=1)
        self.left_frame.grid(row=2, column=0, padx = 80, pady=(90,0), sticky="ew")

        self.title_left = BodyText(self.left_frame, text="Input")
        self.title_left.grid(row=2, column=0, padx=30, sticky="w")

        self.btn = ctk.CTkButton(self.left_frame, text="Start      \u25B6", font=(Theme.FONT_T,18), text_color=Theme.WHITE, fg_color=Theme.BLUE, width=120, height=40)
        self.btn.grid(row=2, column=1, padx=(70,30), sticky="e")

        #Right Column
        self.right_frame = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.right_frame.grid_columnconfigure((0,1), weight=1) 
        self.right_frame.grid(row=2, column=1, padx = 80, pady=(90,0), sticky="ew")

        self.title_right = BodyText(self.right_frame, text="Output")
        self.title_right.grid(row=2, column=0, padx=30, sticky="w")

        #Display
        self.display_left = ctk.CTkLabel(self, text="")
        self.display_left.grid(row=4, column=0, padx=10, pady=20)

        self.display_right = ctk.CTkLabel(self, text="")
        self.display_right.grid(row=4, column=1, padx=10, pady=20)

        self.desc_left = BodyText(self, text="")
        self.desc_left.grid(row=5, column=0, padx=(110,0), pady=(2,10), sticky="w")

        self.desc_right = BodyText(self, text="")
        self.desc_right.grid(row=5, column=1, padx=(110,0), pady=(2,10), sticky="w")
        
        self.fps_list = deque(maxlen=30)
        self.latency_list = deque(maxlen=30)
        self.update_frame()
    
    def update_frame(self):
        if not self.winfo_ismapped():
            self.after(500, self.update_frame)
            return
        
        time_start = time.time()
        ret, frame = self.cap.read() #Tries to read frame

        if ret:
            real_h, real_w, _ = frame.shape #height, width, channels

            frame = cv2.resize(frame, (500, 300)) #Shrinking for better CPU performance
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img)
            ctk_img = ctk.CTkImage(light_image=pil_img, size=(500, 300)) #UI size

            self.display_left.configure(image=ctk_img)
            self.display_right.configure(image=ctk_img)
        

            time_end = time.time()
            time_diff = time_end - time_start
            self.fps_list.append(time_diff)

            #FPS
            if len(self.fps_list) > 0:
                avg_fps = sum(self.fps_list) / len(self.fps_list)
                mean_fps = 1 / avg_fps if avg_fps > 0 else 0

            #Latency
            latency_ms = time_diff * 1000
            self.latency_list.append(latency_ms)
            if len(self.latency_list) > 0:
                avg_latency = sum(self.latency_list) / len(self.latency_list)
            
            stats_text = (
                f"Resolution: {real_w} x {real_h}\n"
                f"FPS: {mean_fps:.1f}\n"
                f"Latency: {avg_latency:.1f} ms"
            )

            self.desc_left.configure(text=stats_text)
            self.desc_right.configure(text=stats_text)

        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #Replay video when ending

        self.after(30, self.update_frame)