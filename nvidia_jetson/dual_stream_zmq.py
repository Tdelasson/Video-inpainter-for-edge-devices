import cv2
import sys
import threading
import queue
import zmq

#PC_IP = "xxx.xxx.xxx.xxx"
DIRECT_PORT = 5000
AI_PORT = 5001
WIDTH = 1280
HEIGHT = 720
FPS = 30


#ZMQ setup
context = zmq.Context()

#Direct stream socket
socket_direct = context.socket(zmq.PUB)
socket_direct.setsockopt(zmq.SNDHWM, 1)
socket_direct.bind(f"tcp://*:{DIRECT_PORT}")

#AI stream socket
socket_ai = context.socket(zmq.PUB)
socket_ai.setsockopt(zmq.SNDHWM, 1)
socket_ai.bind(f"tcp://*:{AI_PORT}")


#Generating a GStream-pipeline -> collects the video from the CSI-camera and converts it to a format OpenCV can read.
#Image processing happens on the GPU to spare the CPU's power
def gstreamer_pipeline_in(sensor_id=0, w=WIDTH, h=HEIGHT, fps=FPS):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h}, framerate={fps}/1, format=NV12 ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

ai_queue = queue.Queue(maxsize=1)

def ai_thread():
    while True:
        frame = ai_queue.get()

        if frame is None:
            break

        #modellen kaldes på frame her

        #komprimer og send via ZMQ
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        socket_ai.send(buffer)

        ai_queue.task_done()


#Opens the camera using the GStream string.
cap = cv2.VideoCapture(gstreamer_pipeline_in(sensor_id=0), cv2.CAP_GSTREAMER)

#checks if there is connection to the camera and if GStreamer could start udpsink correct
#if not the program is exited.
if not cap.isOpened():
    print("Fejl: Kunne ikke åbne kamera.")
    sys.exit()


#makes threading after we know the camera is accessible 
t = threading.Thread(target=ai_thread, daemon=True)
t.start()

print(f"Streamer nu CAM1 direkte til port:{DIRECT_PORT} og port:{AI_PORT}\n")


#starts the main loop which retrieves images from the camera and sendes them
#through the network until the user interrupts the program.
try: 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #out_direct.write(frame)
        _, buffer_direct = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        socket_direct.send(buffer_direct)

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

    #venter på at AI-tråden er lukket
    t.join(timeout=1.0)
    print("Alt er released correct.")