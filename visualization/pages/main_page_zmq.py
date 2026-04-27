import customtkinter as ctk
import cv2
import zmq
import numpy as np
import queue
import threading
import json


from components.header_content import Header
from components.text import BodyText
from components.theme import Theme
from PIL import Image

JETSON_IP = "192.168.137.108"
DIRECT_PORT = 5000
AI_PORT = 5001
STATS_PORT = 5002

class MainPage_zmq(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.WHITE)
        self.running = True

        #Queue setup
        self.frame_queue_l = queue.Queue(maxsize=1)
        self.frame_queue_r = queue.Queue(maxsize=1)
        self.stats_queue = queue.Queue(maxsize=1)

        #Zmq sockets setup
        self.context = zmq.Context()
        self.sub_direct = self.setup_socket(DIRECT_PORT)
        self.sub_ai = self.setup_socket(AI_PORT)
        self.sub_stats = self.setup_socket(STATS_PORT)

        self.grid_columnconfigure((0, 1), weight=1)


        self.left_side = VideoSection(self, title="Input")
        self.left_side.grid(row=2, column=0, padx = 80, pady=(90,0), sticky="ew")
        self.right_side = VideoSection(self, title="Output")
        self.right_side.grid(row=2, column=1, padx = 80, pady=(90,0), sticky="ew")


        self.display_left = self.left_side.display
        self.display_right = self.right_side.display
        self.desc_left = self.left_side.desc
        self.desc_right = self.right_side.desc

        # Start background worker threads independently so a slow AI stream
        # does not hold back the live camera preview.
        self.left_thread = threading.Thread(
            target=self.frame_worker,
            args=(self.sub_direct, self.frame_queue_l),
            daemon=True,
        )
        self.right_thread = threading.Thread(
            target=self.frame_worker,
            args=(self.sub_ai, self.frame_queue_r),
            daemon=True,
        )
        self.stats_thread = threading.Thread(target=self.stats_worker, daemon=True)
        self.left_thread.start()
        self.right_thread.start()
        self.stats_thread.start()

        self.update_frame()

    def setup_socket(self, port):
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://{JETSON_IP}:{port}")
        socket.setsockopt(zmq.SUBSCRIBE, b"") #Let all message through
        socket.setsockopt(zmq.CONFLATE, 1) #Keep only the lastest message in the buffer
        return socket

    def push_latest(self, target_queue, value):
        if not target_queue.empty():
            try:
                target_queue.get_nowait()
            except queue.Empty:
                pass
        target_queue.put(value)

    def frame_worker(self, socket, target_queue):
        while self.running:
            try:
                msg = socket.recv()
                img = self.process_image(msg)
                if img:
                    self.push_latest(target_queue, img)
            except zmq.error.Again:
                continue
            except Exception as e:
                print(f"Video Worker Error {e}")

    def stats_worker(self):
        while self.running:
            try:
                stats_msg = self.sub_stats.recv()
                stats = json.loads(stats_msg.decode("utf-8"))
                self.push_latest(self.stats_queue, stats)
            except zmq.error.Again:
                continue
            except Exception as e:
                print(f"Stats Worker Error {e}")

    def update_frame(self):
        if not self.winfo_ismapped():
            self.after(500, self.update_frame)
            return

        try:
            img_l = self.frame_queue_l.get_nowait()
            if self.display_left.cget("text") != "":
                self.display_left.configure(text="")
            self.display_left.configure(image=img_l)
        except queue.Empty:
            pass

        try:
            img_r = self.frame_queue_r.get_nowait()
            if self.display_right.cget("text") != "":
                self.display_right.configure(text="")
            self.display_right.configure(image=img_r)
        except queue.Empty:
            pass

        try:
            stats = self.stats_queue.get_nowait()
            mem_val = stats.get("memory_mb", -1)
            if isinstance(mem_val, (int, float)) and mem_val >= 0:
                mem_text = f"{mem_val:.2f} MB"
            else:
                mem_text = "N/A"

            # Left panel: raw camera input stats
            left_stats_text = (
                f"Resolution: {stats.get('resolution', '--')}\n"
                f"FPS: {stats.get('cam_fps', '--')}\n"
            )
            # Right panel: AI pipeline stats
            right_stats_text = (
                f"Resolution: {stats.get('resolution', '--')}\n"
                f"FPS: {stats.get('fps', '--')}\n"
                f"Latency: {stats.get('latency_ms', '--')} ms\n"
                f"Memory: {mem_text}\n"
            )
            self.desc_left.configure(text=left_stats_text)
            self.desc_right.configure(text=right_stats_text)
        except queue.Empty:
            pass
    
        self.after(15, self.update_frame)

    #Convert ZMQ data to a CTkImgae for display
    def process_image(self, message):
            data = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.resize(frame, (550, 350)) #Shrinking for better CPU performance
                cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(cv2_img)
                return ctk.CTkImage(light_image=pil_img, size=(550, 350)) #UI size
            return None

class VideoSection(ctk.CTkFrame):
    def __init__(self, parent, title):
        super().__init__(parent, fg_color=Theme.TP)

        #Title
        self.title_label = BodyText(self, text=title)
        self.title_label.grid(row=0, column=0, padx=10, sticky="w")

        #Display
        self.display = ctk.CTkLabel(self, text="Connecting to Jetson...", width=550, height=350)
        self.display.grid(row=1, column=0, padx=10, pady=20)
        
        #Describtion stats text
        self.desc = BodyText(self, text="")
        self.desc.grid(row=2, column=0, padx=10, pady=(2,10), sticky="w")