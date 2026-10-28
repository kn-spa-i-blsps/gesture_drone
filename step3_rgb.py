import time 
import cv2 
import depthai as dai

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setIspScale(1, 3)
cam.setFps(30)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.isp.link(xout.input)     


with dai.Device(pipeline) as dev:
    print("USB:", dev.getUsbSpeed())
    q = dev.getOutputQueue("rgb", maxSize=2, blocking=False)  # trzymaj tylko najnowsze klatki

    t0, n = time.time(), 0
    while True:
        msg = q.get()                                # czeka na nowa klatke
        frame = msg.getCvFrame()                     # YUV z urzadzenia -> BGR dla OpenCV
        latency_ms = (dai.Clock.now() - msg.getTimestamp()).total_seconds() * 1000

        n += 1
        fps = n / (time.time() - t0)
        cv2.putText(frame, f"{frame.shape[1]}x{frame.shape[0]}  {fps:.1f} fps  {latency_ms:.0f} ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("rgb", frame)
        if cv2.waitKey(1) == ord("q"):
            break