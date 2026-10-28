import time
import cv2
import numpy as np
import depthai as dai

W, H = 640, 360
pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 3)
cam.setFps(30)

monoL = pipeline.create(dai.node.MonoCamera)
monoR = pipeline.create(dai.node.MonoCamera)
monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
for m in (monoL, monoR):
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    m.setFps(30)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)   # NOWE
stereo.setOutputSize(W, H)                          # NOWE
monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

xr = pipeline.create(dai.node.XLinkOut); xr.setStreamName("rgb");   cam.isp.link(xr.input)
xd = pipeline.create(dai.node.XLinkOut); xd.setStreamName("depth"); stereo.depth.link(xd.input)

mouse = [W // 2, H // 2]
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

PALM = [0, 5, 9, 13, 17]
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]   # (czubek, PIP)

def depth_at(depth, u, v, r=6):
    patch = depth[max(0, v - r):v + r, max(0, u - r):u + r]
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size > 10 else None

def gesture(pts):
    wrist = pts[0]
    ext = sum(np.linalg.norm(pts[t] - wrist) > 1.1 * np.linalg.norm(pts[p] - wrist)
              for t, p in FINGERS)
    return "OPEN" if ext == 4 else "FIST" if ext == 0 else f"OTHER({ext})"

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                       min_detection_confidence=0.6, min_tracking_confidence=0.5)

with dai.Device(pipeline) as dev:
    K = np.array(dev.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    qr = dev.getOutputQueue("rgb", maxSize=2, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=2, blocking=False)

    while True:
        rgb = qr.get().getCvFrame()
        depth = qd.get().getFrame()

        t0 = time.time()
        res = hands.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))   # MediaPipe chce RGB, OpenCV daje BGR
        mp_ms = (time.time() - t0) * 1000

        y_txt = 25
        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                pts = np.array([[p.x * W, p.y * H] for p in lm.landmark])   # 21 x 2, piksele
                u, v = pts[PALM].mean(axis=0).astype(int)
                Z = depth_at(depth, u, v)
                mp_draw.draw_landmarks(rgb, lm, mp_hands.HAND_CONNECTIONS)
                cv2.circle(rgb, (u, v), 7, (0, 255, 0), -1)
                if Z is not None:
                    X, Y = (u - cx) * Z / fx, (v - cy) * Z / fy
                    txt = f"{gesture(pts):10s} X {X:+.2f} Y {Y:+.2f} Z {Z:.2f} m"
                else:
                    txt = f"{gesture(pts):10s} brak glebi na dloni"
                cv2.putText(rgb, txt, (10, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                y_txt += 25

        cv2.putText(rgb, f"mediapipe {mp_ms:.0f} ms", (10, H - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("hand", rgb)
        if cv2.waitKey(1) == ord("q"):
            break

hands.close()