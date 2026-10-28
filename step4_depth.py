import time
import cv2
import numpy as np
import depthai as dai

pipeline = dai.Pipeline()

# RGB jak w kroku 3
cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 3)
cam.setFps(30)

# Dwie kamery mono (global shutter, 640x400)
monoL = pipeline.create(dai.node.MonoCamera)
monoR = pipeline.create(dai.node.MonoCamera)
monoL.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoR.setBoardSocket(dai.CameraBoardSocket.CAM_C)
for m in (monoL, monoR):
    m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    m.setFps(30)

# Wezel stereo
stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setLeftRightCheck(True)   # odrzuca piksele, gdzie L->R i R->L sie nie zgadzaja (okluzje)
stereo.setSubpixel(False)
monoL.out.link(stereo.left)
monoR.out.link(stereo.right)

# Wyjscia do hosta
xr = pipeline.create(dai.node.XLinkOut); xr.setStreamName("rgb");   cam.isp.link(xr.input)
xd = pipeline.create(dai.node.XLinkOut); xd.setStreamName("depth"); stereo.depth.link(xd.input)

mouse = [320, 200]
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mouse[0], mouse[1] = x, y

with dai.Device(pipeline) as dev:
    qr = dev.getOutputQueue("rgb", maxSize=2, blocking=False)
    qd = dev.getOutputQueue("depth", maxSize=2, blocking=False)
    cv2.namedWindow("depth")
    cv2.setMouseCallback("depth", on_mouse)

    t0, n = time.time(), 0
    while True:
        rgb = qr.get().getCvFrame()
        msg = qd.get()
        depth = msg.getFrame()                      # uint16, milimetry, 0 = brak danych
        lat = (dai.Clock.now() - msg.getTimestamp()).total_seconds() * 1000

        n += 1
        fps = n / (time.time() - t0)

        # wizualizacja: 0..2 m na palete kolorow, brak danych na czarno
        vis = np.clip(depth, 0, 2000).astype(np.float32) / 2000.0
        vis = cv2.applyColorMap((255 - vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        vis[depth == 0] = 0

        u, v = min(mouse[0], depth.shape[1] - 1), min(mouse[1], depth.shape[0] - 1)
        z = depth[v, u]
        cv2.drawMarker(vis, (u, v), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(vis, f"{z/1000:.2f} m" if z else "brak", (u + 10, v - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, f"{depth.shape[1]}x{depth.shape[0]}  {fps:.1f} fps  {lat:.0f} ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("rgb", rgb)
        cv2.imshow("depth", vis)
        if cv2.waitKey(1) == ord("q"):
            break