import cv2
import sys
import threading
import queue
import zmq
import argparse
import time
import json
from collections import deque
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from Baselines.fuseformer_om_adapter import FuseFormerOMAdapter
from Baselines.propainter_adapter import ProPainterAdapter
from Baselines.vinet_adapter import ViNETAdapter
from viper_adapter import ViperAdapter
from Masking.yolo_segmenter import YOLOSegmenter

#PC_IP = "xxx.xxx.xxx.xxx"
DIRECT_PORT = 5000
AI_PORT = 5001
STATS_PORT = 5002
WIDTH = 256
HEIGHT = 256
FPS = 30
SENSOR_ID = 0

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FUSEFORMER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/OnlineInpainting/fuseformer.pth").resolve()
DEFAULT_PROPAINTER_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/ProPainter.pth").resolve()
DEFAULT_PROPAINTER_RAFT_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/raft-things.pth").resolve()
DEFAULT_PROPAINTER_FLOW_WEIGHTS_PATH = (
    REPO_ROOT / "../Baselines_Repos/pthFiles/ProPainter/recurrent_flow_completion.pth"
).resolve()
DEFAULT_VINET_WEIGHTS_PATH = (REPO_ROOT / "../Baselines_Repos/pthFiles/ViNETsave_agg_rec_512.pth").resolve()
DEFAULT_VIPER_WEIGHTS_PATH = (REPO_ROOT / "final_model.pth").resolve()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual ZMQ stream with segmentation and optional inpainting")
    parser.add_argument("--seg-model", type=str, default="yolo26n-seg")
    parser.add_argument("--seg-model-path", type=str, default=None)
    parser.add_argument(
        "--inpaint-model",
        type=str,
        default="none",
        choices=["none", "fuseformer_om", "propainter", "vinet", "viper"],
        help="Optional baseline inpainting model",
    )
    parser.add_argument(
        "--right-view",
        type=str,
        default="auto",
        choices=["auto", "mask", "inpaint"],
        help="Right stream mode: mask, inpaint, or auto",
    )
    parser.add_argument("--infer-every", type=int, default=2, help="Run inpainting every N frames")
    parser.add_argument("--fp16", action="store_true", help="Use fp16 where supported")
    parser.add_argument("--display", action="store_true", help="Optional local preview")
    return parser.parse_args()


args = parse_args()


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

# Stats stream socket (JSON payload)
socket_stats = context.socket(zmq.PUB)
socket_stats.setsockopt(zmq.SNDHWM, 1)
socket_stats.bind(f"tcp://*:{STATS_PORT}")


#Generating a GStream-pipeline -> collects the video from the CSI-camera and converts it to a format OpenCV can read.
#Image processing happens on the GPU to spare the CPU's power
def gstreamer_pipeline_in(sensor_id=0, w=WIDTH, h=HEIGHT, fps=FPS):
    # nvarguscamerasrc caps select the sensor mode (native resolution).
    # A second nvvidconv stage performs the actual scaling to w x h.
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), framerate={fps}/1, format=NV12 ! "
        f"nvvidconv ! "
        f"video/x-raw(memory:NVMM), width={w}, height={h} ! "
        f"nvvidconv ! video/x-raw, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )

def open_camera():
    return cv2.VideoCapture(gstreamer_pipeline_in(sensor_id=SENSOR_ID), cv2.CAP_GSTREAMER)


def build_inpainter(model_name: str, device: str):
    if model_name == "none":
        return None, 0
    if model_name == "fuseformer_om":
        return FuseFormerOMAdapter(str(DEFAULT_FUSEFORMER_WEIGHTS_PATH), device=device, fp16=args.fp16), 8
    if model_name == "propainter":
        return ProPainterAdapter(
            weights_path=str(DEFAULT_PROPAINTER_WEIGHTS_PATH),
            raft_weights_path=str(DEFAULT_PROPAINTER_RAFT_WEIGHTS_PATH),
            flow_weights_path=str(DEFAULT_PROPAINTER_FLOW_WEIGHTS_PATH),
            device=device,
            fp16=args.fp16,
        ), 12
    if model_name == "vinet":
        return ViNETAdapter(str(DEFAULT_VINET_WEIGHTS_PATH), device=device, fp16=args.fp16), 10
    if model_name == "viper":
        return ViperAdapter(str(DEFAULT_VIPER_WEIGHTS_PATH), device=device, seq_len=5, fp16=args.fp16), 5
    raise ValueError(f"Unsupported inpaint model: {model_name}")


def make_mask_overlay(frame, mask):
    color_mask = frame.copy()
    color_mask[:, :, 0] = 0
    color_mask[:, :, 2] = 0
    color_mask[:, :, 1] = cv2.max(color_mask[:, :, 1], mask)
    return color_mask

ai_queue = queue.Queue(maxsize=1)
cam_stats = {"fps": 0.0}   # updated by main loop, read by ai_thread for stats payload
_cam_fps_counter = {"n": 0, "t0": time.time()}
device = "cuda" if torch.cuda.is_available() else "cpu"
segmenter = YOLOSegmenter(model_name=args.seg_model, model_path=args.seg_model_path, target_classes=[0])
inpainter, window_size = build_inpainter(args.inpaint_model, device)
frame_buffer = deque(maxlen=max(1, window_size))
mask_buffer = deque(maxlen=max(1, window_size))
last_inpaint = None
fps_counter = {"n": 0, "t0": time.time()}
stats_counter = {"n": 0, "t0": time.time(), "sum_stage_ms": 0.0, "sum_total_ms": 0.0}

def ai_thread():
    global last_inpaint

    while True:
        item = ai_queue.get()

        if item is None:
            break

        frame_id, frame, ts_capture = item

        ts_stage_start = time.perf_counter()

        mask = segmenter.segment(frame, return_annotated=False)
        frame_buffer.append(frame)
        mask_buffer.append(mask)

        if inpainter is not None and (frame_id % args.infer_every) == 0 and len(frame_buffer) == window_size:
            try:
                pred = inpainter.inpaint(list(frame_buffer), list(mask_buffer), resize_to_original=True)
                if pred:
                    last_inpaint = pred[-1]
            except Exception as exc:
                print(f"Inpainting warning on frame {frame_id}: {exc}")

        if args.right_view == "mask":
            out_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif args.right_view == "inpaint":
            out_frame = last_inpaint if last_inpaint is not None else make_mask_overlay(frame, mask)
        else:
            out_frame = last_inpaint if last_inpaint is not None else make_mask_overlay(frame, mask)

        #komprimer og send via ZMQ
        _, buffer = cv2.imencode('.jpg', out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        socket_ai.send(buffer)

        stage_ms = (time.perf_counter() - ts_stage_start) * 1000.0
        total_ms = (time.time() - ts_capture) * 1000.0

        stats_counter["n"] += 1
        stats_counter["sum_stage_ms"] += stage_ms
        stats_counter["sum_total_ms"] += total_ms
        stats_elapsed = time.time() - stats_counter["t0"]
        if stats_elapsed >= 0.5:
            n = max(1, stats_counter["n"])
            if torch.cuda.is_available():
                mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            else:
                mem_mb = -1.0
                peak_mem_mb = -1.0

            stats_payload = {
                "resolution": f"{WIDTH}x{HEIGHT}",
                "cam_fps": round(cam_stats["fps"], 2),
                "fps": round(stats_counter["n"] / stats_elapsed, 2),
                "latency_ms": round(stats_counter["sum_total_ms"] / n, 2),
                "stage_ms": round(stats_counter["sum_stage_ms"] / n, 2),
                "memory_mb": round(mem_mb, 2),
                "peak_memory_mb": round(peak_mem_mb, 2),
                "inpaint_model": args.inpaint_model,
            }
            socket_stats.send_json(stats_payload)

            stats_counter["n"] = 0
            stats_counter["sum_stage_ms"] = 0.0
            stats_counter["sum_total_ms"] = 0.0
            stats_counter["t0"] = time.time()

        if args.display:
            cv2.imshow("Dual Stream | Right Output", out_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        fps_counter["n"] += 1
        elapsed = time.time() - fps_counter["t0"]
        if elapsed >= 2.0:
            print(f"Pipeline FPS: {fps_counter['n'] / elapsed:.1f}")
            fps_counter["n"] = 0
            fps_counter["t0"] = time.time()

        ai_queue.task_done()


#Opens the camera using the GStream string.
cap = open_camera()

#checks if there is connection to the camera and if GStreamer could start udpsink correct
#if not the program is exited.
if not cap.isOpened():
    print("Fejl: Kunne ikke åbne kamera.")
    sys.exit()


#makes threading after we know the camera is accessible
t = threading.Thread(target=ai_thread, daemon=True)
t.start()

print(f"Streamer nu CAM1 direkte til port:{DIRECT_PORT} og port:{AI_PORT}\n")
print(f"Capture config: {WIDTH}x{HEIGHT}@{FPS} sensor-id={SENSOR_ID}")
print(f"Segmentation model: {args.seg_model}")
if args.seg_model_path:
    print(f"Segmentation weights override: {args.seg_model_path}")
print(f"Inpainting model: {args.inpaint_model}")


#starts the main loop which retrieves images from the camera and sendes them
#through the network until the user interrupts the program.
try:
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #out_direct.write(frame)
        _, buffer_direct = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        socket_direct.send(buffer_direct)

        try:
            ai_queue.put_nowait((frame_id, frame, time.time()))
        except queue.Full:
            pass

        # Track camera capture FPS for the stats payload
        _cam_fps_counter["n"] += 1
        _elapsed = time.time() - _cam_fps_counter["t0"]
        if _elapsed >= 2.0:
            cam_stats["fps"] = _cam_fps_counter["n"] / _elapsed
            _cam_fps_counter["n"] = 0
            _cam_fps_counter["t0"] = time.time()

        frame_id += 1

except KeyboardInterrupt:
    print("\n Stopper stream...")


#Releases the camera (CSI-port)
#closes the network pipeline and clears memoery
#closes all windows that OpenCV has opened.
finally:

    try:
        ai_queue.put(None)
    except queue.Full:
        pass
    cap.release()

    #venter på at AI-tråden er lukket
    t.join(timeout=1.0)
    socket_direct.close(0)
    socket_ai.close(0)
    socket_stats.close(0)
    context.term()
    if args.display:
        cv2.destroyAllWindows()
    print("Alt er released correct.")
