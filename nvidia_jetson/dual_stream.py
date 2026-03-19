import cv2
import jetson_utils

PC_IP = ""      #her indsættes ip-addresse på pc'en
WIDTH = 520
HEIGHT = 520
BITRATE = 4000000


pipeline = (
    f"nvarguscamerasrc ! "      #starter sensoren på CSI-kameraet.
    f"video/x-raw(memory:NVMM), width={WIDTH}, height={HEIGHT}, framerate=30/1 !"       #videon skal blive i jetsons GPU i stedet for RAM
    f"tee name=t "      #splitter kablet i to. splittet hedder t.
    f"t. ! queue ! nvv4l2h264enc bitrate={BITRATE} ! h264parse ! rtph264pay ! udpsink host={PC_IP} port=5000 "      #første udgang af t. hardware pakker sendes til PC via UDP
    f"t. ! queue ! nvvidconv ! video/x-raw, format=BGR ! appsink drop=False"        #overvej at skrive drop=True så den smider billederne ud og undgår for meget kø.
    #f"t. ! queue max-size-buffers=1 leaky=downstram ! nvvidconv ! ... ! appsink "        <- denne sørger for der kun er ét billede som venter og at ældste billede skubes ud af køen.
)