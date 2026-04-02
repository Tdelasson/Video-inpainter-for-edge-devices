import cv2
import sys
import threading
import queue

PC_IP = "xxx.xxx.xxx.xxx"
DIRECT_PORT = 5000
AI_PORT = 5001
WIDTH = 1280
HEIGHT = 720
FPS = 30


#Generating a GStream-pipeline -> collects the video from the CSI-camera and converts it to a format OpenCV can read.
#Image processing happens on the GPU to spare the CPU's power
def gstreamer_pipeline_in(sensor_id=0, w=WIDTH, h=HEIGHT, fps=FPS):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1, format=NV12 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


#Takes the images and convert it to jpeg such that i can be sent via UDP to the PC host.
#using jpegenc saves CPU
def gstreamer_pipeline_out(port):
    return (
        f"appsrc ! "
        f"video/x-raw, format=BGR, width{WIDTH}, height={HEIGHT}, framerate={FPS}/1 ! "
        f"videoconvert ! "
        f"video/x-raw, format=I420 !"
        f"jpegenc quality=80 ! "
        f"rtpjpegpay ! "
        f"udpsink host={PC_IP} port={port} sync=false"
    )    



ai_queue = queue.Queue(maxsize=1)

out_ai = None


def ai_thread():
    global out_ai

    out_ai = cv2.VideoWriter(gstreamer_pipeline_out(AI_PORT), cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)

    while True:
        frame = ai_queue.get()

        if frame is None:
            break

        #modellen kaldes på frame her

        if out_ai.isOpened():
            out_ai.write(frame)
        
        ai_queue.task_done()

    if out_ai is not None:
        out_ai.release()




#Opens the camera using the GStream string.
cap = cv2.VideoCapture(gstreamer_pipeline_in(sensor_id=1), cv2.CAP_GSTREAMER)

#OpenCV uses GStreamer to "write" the video onto the network.
out_direct = cv2.VideoWriter(gstreamer_pipeline_out(DIRECT_PORT), cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)


#checks if there is connection to the camera and if GStreamer could start udpsink correct
#if not the program is exited.
if not cap.isOpened() or not out_direct.isOpened():
    print("Fejl: Kunne ikke åbne kamera eller netværks-pipeline.")
    sys.exit()


#makes threading after we know the camera is accessible 
t = threading.Thread(target=ai_thread, daemon=True)
t.start()

print(f"Streamer nu CAM1 direkte til {PC_IP}:{DIRECT_PORT}...")
print(f"Streamer AI til {PC_IP}:{AI_PORT}")


#starts the main loop which retrieves images from the camera and sendes them
#through the network until the user interrupts the program.
try: 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out_direct.write(frame)

        try:
            ai_queue.put_nowait(frame)
        except queue.Full:
            pass

except KeyboardInterrupt:
    print("\n Stopper stream...")


#Releases the camera (CSI-port)
#closes the network pipeline and clears memoery
#closes all windows that OpenCV has opened.
finally:

    ai_queue.put(None)

    cap.release()
    out_direct.release()

    #venter på at AI-tråden er lukket
    t.join(timeout=1.0)

    print("Alt er released correct.")