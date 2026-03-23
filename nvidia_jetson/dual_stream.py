import cv2
import jetson_utils

PC_IP = ""      #her indsættes ip-addresse på pc'en
WIDTH = 520
HEIGHT = 520
BITRATE = 4000000


#pipline med "tee" altså splittet.
pipeline = (
    f"nvarguscamerasrc ! "      #starter sensoren på CSI-kameraet.
    f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, framerate=30/1 !"       #videon skal blive i jetsons GPU i stedet for RAM
    f"tee name=t "      #splitter kablet i to. splittet hedder t.
    f"t. ! queue ! nvv4l2h264enc bitrate={BITRATE} ! h264parse ! rtph264pay ! udpsink host={PC_IP} port=5000 "      #første udgang af t. hardware pakker sendes til PC via UDP
    f"t. ! queue ! nvvidconv ! video/x-raw, format=BGR ! appsink drop=False"        #overvej at skrive drop=True så den smider billederne ud og undgår for meget kø.
    #f"t. ! queue max-size-buffers=1 leaky=downstram ! nvvidconv ! ... ! appsink "        <- denne sørger for der kun er ét billede som venter og at ældste billede skubes ud af køen.
)

#modtagelse af kanal 2 (OpenCV til AI-delen)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

#opsætning af sending af kanal 2
output_ai_ready = jestson_utils_videoOutput(f"rtp://{PC_IP}:5001 --output-codec=h264")

print(f"Forbindelsestest kører")
print(f"kanal 1: tjek rtp://@:5000")
print(f"kanal 2: tjek rtp://@:5001")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fejl: Kunne ikke hente frame.")
            break

        cv2.putText(frame, "Kanal 2 - klar til modellen"), (20, 50), cv2.FONT_HERSHEY_SIMPLEX(0.7, (0, 255, 0), 2)

        #konvertering fra OpenCV til CUDA for hurtig sending.
        cuda_mem = jetson_utils.cudaFromNumpy(frame)

        output_ai_ready.Render(cuda_mem)

except KeyboardInterrupt:
    print("Stopper...")


#ryd op
cap.release()