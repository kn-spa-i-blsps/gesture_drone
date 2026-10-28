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
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse[0], mouse[1] = min(x, W - 1), min(y, H - 1)

def depth_at(depth, u, v, r=5):
    """Mediana z okna (2r)x(2r) - pojedynczy piksel czesto trafia w dziure albo krawedz."""
    patch = depth[max(0, v - r):v + r, max(0, u - r):u + r]
    valid = patch[patch > 0]
    return float(np.median(valid)) / 1000.0 if valid.size > 10 else None

with dai.Device(pipeline) as dev:
    # Intrinsyki kamery RGB dla rozdzielczosci, w ktorej pracujemy
    K = np.array(dev.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    qr = dev.getOutputQueue("rgb", maxSize=2, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=2, blocking=False)
    cv2.namedWindow("aligned")
    cv2.setMouseCallback("aligned", on_mouse)

    while True:
        mr = qr.get()
        md = qd.get()
        rgb = mr.getCvFrame()
        depth = md.getFrame()
        dt_ms = (mr.getTimestamp() - md.getTimestamp()).total_seconds() * 1000

        # nakladka: glebia polprzezroczysta na RGB
        vis = np.clip(depth, 0, 2000).astype(np.float32) / 2000.0
        vis = cv2.applyColorMap((255 - vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[depth == 0] = 0
        out = cv2.addWeighted(rgb, 0.6, vis, 0.4, 0)

        u, v = mouse
        Z = depth_at(depth, u, v)
        cv2.drawMarker(out, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        if Z is not None:
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            txt = f"X {X:+.2f}  Y {Y:+.2f}  Z {Z:.2f} m"
        else:
            txt = "brak glebi"
        cv2.putText(out, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(out, f"rgb-depth dt {dt_ms:+.0f} ms", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("aligned", out)
        if cv2.waitKey(1) == ord("q"):
            break