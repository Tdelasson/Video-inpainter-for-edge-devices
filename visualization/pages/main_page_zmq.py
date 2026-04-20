import customtkinter as ctk
import cv2
#print(cv2.getBuildInformation())
import zmq
import numpy as np


from components.header_content import Header
from components.text import BodyText
from components.theme import Theme
from PIL import Image
from collections import deque

JETSON_IP = "192.168.137.108"
DIRECT_PORT = 5000
AI_PORT = 5001



# def gstreamer_pipelin_on_pc(port):
#     return (
#         f"udpsrc port={port} ! "
#         f"application/x-rtp, encoding-name=JPEG, payload=26 !"
#         f"rtpjpegdepay ! "
#         f"jpegdec ! "
#         f"videoconvert ! "
#         f"appsink"
#     )



class MainPage_zmq(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, fg_color=Theme.WHITE)
        #self.is_active = True

        #setup zmq sockets
        self.context = zmq.Context()

        self.sub_direct = self.context.socket(zmq.SUB)
        self.sub_direct.connect(f"tcp://{JETSON_IP}:{DIRECT_PORT}")
        self.sub_direct.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_direct.setsockopt(zmq.CONFLATE, 1)
        

        self.sub_ai = self.context.socket(zmq.SUB)
        self.sub_ai.connect(f"tcp://{JETSON_IP}:{AI_PORT}")
        self.sub_ai.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub_ai.setsockopt(zmq.CONFLATE, 1)


        self.grid_columnconfigure((0, 1), weight=1)
        self.display_left = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.display_left.grid(row=4, column=0, padx=10, pady=20)
        self.display_left = ctk.CTkLabel(self.display_left, text="")
        self.display_left.grid(row=4, column=0, padx=10, pady=20)


        self.display_right = ctk.CTkFrame(self, fg_color=Theme.TP)
        self.display_right = ctk.CTkLabel(self.display_right, text="")
        self.display_right.grid(row=4, column=1, padx=10, pady=20)


        self.update_frame()
    

    def get_frame_from_zmq(self, socket):
        try:
            message = socket.recv()
            # print("Pakke modtaget!")
            data = np.frombuffer(message, dtype=np.uint8)
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return True, frame
        except zmq.Again:
            return False, None



    def update_frame(self):
        if not self.winfo_ismapped():
            self.after(500, self.update_frame)
            return
        
        #time_start = time.time()
        ret_direct, frame_direct = self.get_frame_from_zmq(self.sub_direct) #Tries to read frame
        ret_ai, frame_ai = self.get_frame_from_zmq(self.sub_ai)

        if ret_direct:
            img_l = self.process_image(frame_direct)
            self.display_left.configure(image=img_l)
        #else:
        #    self.cap_direct.set(cv2.CAP_PROP_POS_FRAMES, 0) #Replay video when ending

        
        if ret_ai:
            img_r = self.process_image(frame_ai)
            self.display_right.configure(image=img_r)
        #else:
        #    self.cap_ai.set(cv2.CAP_PROP_POS_FRAMES, 0) #Replay video when ending
        

        self.after(1, self.update_frame)



    def process_image(self, frame):
            real_h, real_w, _ = frame.shape #height, width, channels
            frame = cv2.resize(frame, (500, 300)) #Shrinking for better CPU performance
            cv2_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv2_img)
            return ctk.CTkImage(light_image=pil_img, size=(500, 300)) #UI size
            
            # self.display_left.configure(image=ctk_img)
            # self.display_right.configure(image=ctk_img)

    #def calculations():
        # time_end = time.time()
        # time_diff = time_end - time_start
        # self.fps_list.append(time_diff)

            
        # #FPS
        # if len(self.fps_list) > 0:
        #     avg_fps = sum(self.fps_list) / len(self.fps_list)
        #     mean_fps = 1 / avg_fps if avg_fps > 0 else 0

        # #Latency
        # latency_ms = time_diff * 1000
        # self.latency_list.append(latency_ms)
        # if len(self.latency_list) > 0:
        #     avg_latency = sum(self.latency_list) / len(self.latency_list)
        
        # stats_text = (
        #     f"Resolution: {real_w} x {real_h}\n"
        #     f"FPS: {mean_fps:.1f}\n"
        #     f"Latency: {avg_latency:.1f} ms"
        # )

        # self.desc_left.configure(text=stats_text)
        # self.desc_right.configure(text=stats_text)




# self.cap_direct = cv2.VideoCapture(gstreamer_pipelin_on_pc(DIRECT_PORT), cv2.CAP_GSTREAMER)
#         self.cap_ai = cv2.VideoCapture(gstreamer_pipelin_on_pc(AI_PORT), cv2.CAP_GSTREAMER)

#         self.grid_columnconfigure((0, 1), weight=1)
       
#         self.left_frame = ctk.CTkFrame(self, fg_color=Theme.TP)
#         self.left_frame.grid_columnconfigure((0,1), weight=1)
#         self.left_frame.grid(row=2, column=0, padx = 80, pady=(90,0), sticky="ew")

#         self.title_left = BodyText(self.left_frame, text="Input")
#         self.title_left.grid(row=2, column=0, padx=30, sticky="w")

#         self.btn = ctk.CTkButton(self.left_frame, text="Start      \u25B6", font=(Theme.FONT_T,18), text_color=Theme.WHITE, fg_color=Theme.BLUE, width=120, height=40)
#         self.btn.grid(row=2, column=1, padx=(70,30), sticky="e")

#         #Right Column
#         self.right_frame = ctk.CTkFrame(self, fg_color=Theme.TP)
#         self.right_frame.grid_columnconfigure((0,1), weight=1) 
#         self.right_frame.grid(row=2, column=1, padx = 80, pady=(90,0), sticky="ew")

#         self.title_right = BodyText(self.right_frame, text="Output")
#         self.title_right.grid(row=2, column=0, padx=30, sticky="w")

#         #Display
#         self.display_left = ctk.CTkLabel(self, text="")
#         self.display_left.grid(row=4, column=0, padx=10, pady=20)

#         self.display_right = ctk.CTkLabel(self, text="")
#         self.display_right.grid(row=4, column=1, padx=10, pady=20)

#         self.desc_left = BodyText(self, text="")
#         self.desc_left.grid(row=5, column=0, padx=(110,0), pady=(2,10), sticky="w")

#         self.desc_right = BodyText(self, text="")
#         self.desc_right.grid(row=5, column=1, padx=(110,0), pady=(2,10), sticky="w")
        
#         # self.fps_list = deque(maxlen=30)
        # self.latency_list = deque(maxlen=30)