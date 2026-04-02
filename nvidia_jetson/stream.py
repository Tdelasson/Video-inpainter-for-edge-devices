import cv2
import sys

PC_IP = "xxx.xxx.xxx.xxx"
PORT = 5000
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
gst_pipeline_out = (
    f"appsrc ! "
    f"video/x-raw, format=BGR, width{WIDTH}, height={HEIGHT}, framerate={FPS}/1 ! "
    f"videoconvert ! "
    f"video/x-raw, format=I420 !"
    f"jpegenc quality=80 ! "
    f"rtpjpegpay ! "
    f"udpsink host={PC_IP} port={PORT} sync=false"
)

#Opens the camera using the GStream string.
cap = cv2.VideoCapture(gstreamer_pipeline_in(sensor_id=1), cv2.CAP_GSTREAMER)

#OpenCV uses GStreamer to "write" the video onto the network.
out = cv2.VideoWriter(gst_pipeline_out, cv2.CAP_GSTREAMER, 0, FPS, (WIDTH, HEIGHT), True)


#checks if there is connection to the camera and if GStreamer could start udpsink correct
#if not the program is exited.
if not cap.isOpened() or not out.isOpened():
    print("Fejl: Kunne ikke åbne kamera eller netværks-pipeline.")
    sys.exit()

print(f"Streamer nu CAM1 direkte til {PC_IP}:{PORT}...")


#starts the main loop which retrieves images from the camera and sendes them
#through the network until the user interrupts the program.
try: 
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        out.write(frame)

except KeyboardInterrupt:
    print("\n Stopper stream...")


#Releases the camera (CSI-port)
#closes the network pipeline and clears memoery
#closes all windows that OpenCV has opened.
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
