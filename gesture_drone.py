#!/usr/bin/env python3
"""
gesture_drone.py - sterowanie dronem ArduCopter (tryb GUIDED) gestami dloni.

Zrodlo obrazu: Luxonis OAK-D (RGB + glebia stereo) albo zwykla kamerka (--source webcam).
Detekcja dloni: MediaPipe Hands na hoscie.
Wyjscie: SET_POSITION_TARGET_LOCAL_NED (predkosci w ukladzie heading drona) przez pymavlink.

Gesty (odwiedzajacy):
  OTWARTA DLON przytrzymana ~0.5 s  -> "zakotwicz" pozycje dloni, potem przesuniecie dloni = predkosc drona
  PIESC                             -> stop / hover, kotwica kasowana
  brak dloni > 0.3 s                -> hover; > 1 s -> kotwica kasowana

Klawisze (operator, w oknie podgladu):
  t        -> GUIDED + arm + takeoff (CFG takeoff_alt)
  l / SPACJA -> LAND
  g        -> wlacz/wylacz sterowanie gestami (wylaczone = hover)
  f        -> pelny ekran
  q / ESC  -> wyjscie (jesli uzbrojony: najpierw LAND)

Bezpieczenstwo: pilot z aparatura (ELRS) jest zawsze nadrzedny - przelacznik trybu na
ALT_HOLD/LAND i motor kill. Ten skrypt nigdy nie uzbraja drona sam z siebie.

Uzycie:
  python3 gesture_drone.py --no-drone                       # tylko gesty + wirtualny dron na mapce
  python3 gesture_drone.py --conn udpin:0.0.0.0:14551       # SITL albo MAVProxy z ELRS
  python3 gesture_drone.py --source webcam --no-drone       # test bez OAK-D
"""
import argparse
import math
import threading
import time
from collections import deque

import cv2
import numpy as np

# ----------------------------------------------------------------------------------------
# KONFIGURACJA - wszystko do strojenia w srode jest tutaj
# ----------------------------------------------------------------------------------------
CFG = dict(
    # przesuniecie dloni [m] -> predkosc [m/s]
    deadzone_m=0.05,
    gain=1.2,              # [m/s na m] przesuniecia ponad deadzone
    v_max_xy=0.30,         # [m/s]
    v_max_z=0.25,          # [m/s]
    a_max=0.6,             # [m/s^2] limit zmiany komendy (wygladzenie)
    ema_alpha=0.4,         # filtr pozycji dloni (0..1, wieksze = szybsze)

    # geofence wzgledem punktu startu, w ukladzie heading drona
    box_fwd=0.75,          # polowa dlugosci boxu przod/tyl [m]
    box_lat=0.75,          # polowa szerokosci boxu lewo/prawo [m]
    alt_min=0.5,           # [m]
    alt_max=1.6,           # [m]
    fence_margin=0.15,     # [m] poza boxem o tyle -> aktywne cofanie
    push_back=0.15,        # [m/s] predkosc cofania do boxu
    takeoff_alt=1.0,       # [m]
    telem_timeout=0.5,     # [s] starsza pozycja z drona -> komenda 0

    # wybor dloni (ignorowanie ludzi w tle)
    roi_frac=0.7,          # dlon musi byc w srodkowych 70% kadru
    z_min=0.35, z_max=1.6, # [m] dopuszczalna odleglosc dloni od kamery
    palm_len_m=0.09,       # nadgarstek->MCP srodkowego palca, do estymaty Z bez glebi

    # czasy
    gesture_frames=6,      # tyle klatek z rzedu, zeby zmienic gest
    engage_hold_s=0.5,
    lost_stop_s=0.3,
    lost_reset_s=1.0,
    send_hz=10.0,

    # mapowanie osi kamera -> dron (domyslnie: kamera naprzeciw odwiedzajacego,
    # dron ustawiony NOSEM W STRONE KAMERY, tylem do odwiedzajacego)
    #   dlon w prawo odwiedzajacego  -> dron w prawo
    #   dlon do kamery               -> dron do przodu (w strone kamery)
    #   dlon w gore                  -> dron w gore
    sign_lateral=-1,       # obraz nie jest lustrzany: prawo odwiedzajacego = lewo w obrazie
    sign_depth=-1,
    sign_vertical=+1,
    swap_fwd_lat=False,    # True jesli dron stoi bokiem do osi kamera-odwiedzajacy
)

COPTER_MODES = {"STABILIZE": 0, "ALT_HOLD": 2, "GUIDED": 4, "LOITER": 5, "LAND": 9, "FLOWHOLD": 22}
EKF_ORIGIN = (52.2112, 20.9822, 100.0)  # dowolny punkt; potrzebny tylko zeby EKF mial origin

# ----------------------------------------------------------------------------------------
# Zrodla obrazu
# ----------------------------------------------------------------------------------------
class OakSource:
    """OAK-D (depthai 2.x): RGB 640x360 + glebia wyrownana do RGB."""

    def __init__(self, w=640, h=360, fps=30):
        import depthai as dai
        self.w, self.h = w, h
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setIspScale(1, 3)  # 1920x1080 -> 640x360
        cam.setFps(fps)
        cam.setInterleaved(False)

        ml = p.create(dai.node.MonoCamera)
        mr = p.create(dai.node.MonoCamera)
        ml.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mr.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        for m in (ml, mr):
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            m.setFps(fps)

        st = p.create(dai.node.StereoDepth)
        st.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        st.setLeftRightCheck(True)
        st.setSubpixel(False)
        st.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        st.setOutputSize(w, h)
        ml.out.link(st.left)
        mr.out.link(st.right)

        xr = p.create(dai.node.XLinkOut)
        xr.setStreamName("rgb")
        cam.isp.link(xr.input)
        xd = p.create(dai.node.XLinkOut)
        xd.setStreamName("depth")
        st.depth.link(xd.input)

        self.dev = dai.Device(p)
        calib = self.dev.readCalibration()
        K = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, w, h))
        self.intr = (K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.qr = self.dev.getOutputQueue("rgb", maxSize=2, blocking=False)
        self.qd = self.dev.getOutputQueue("depth", maxSize=2, blocking=False)
        self.depth = None

    def read(self):
        f = self.qr.get()  # blokuje do nowej klatki RGB
        d = self.qd.tryGet()
        if d is not None:
            self.depth = d.getFrame()  # uint16 [mm]
        return f.getCvFrame(), self.depth, self.intr

    def close(self):
        self.dev.close()


class WebcamSource:
    """Zwykla kamerka - Z estymowane z rozmiaru dloni."""

    def __init__(self, idx=0, w=640, h=360):
        self.cap = cv2.VideoCapture(idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.intr = None

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("Brak klatki z kamerki")
        if self.intr is None:
            h, w = f.shape[:2]
            fx = 0.9 * w  # ~60 deg HFOV, wystarczy do demo
            self.intr = (fx, fx, w / 2, h / 2)
        return f, None, self.intr

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------------------------------
# Dlon: gest + pozycja 3D
# ----------------------------------------------------------------------------------------
FINGERS = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (czubek, PIP)
PALM_IDS = [0, 5, 9, 13, 17]


def landmarks_px(lm, w, h):
    return np.array([[p.x * w, p.y * h] for p in lm.landmark])


def classify_gesture(pts):
    """pts: (21,2) w pikselach. Zwraca 'OPEN' / 'FIST' / 'OTHER'."""
    wrist = pts[0]
    ext = 0
    for tip, pip in FINGERS:
        if np.linalg.norm(pts[tip] - wrist) > 1.1 * np.linalg.norm(pts[pip] - wrist):
            ext += 1
    if ext >= 4:
        return "OPEN"
    if ext == 0:
        return "FIST"
    return "OTHER"


def hand_point_3d(pts, depth, intr, cfg):
    """Srodek dloni w ukladzie kamery [m]: X w prawo (obraz), Y w dol, Z od kamery."""
    fx, fy, cx, cy = intr
    u, v = pts[PALM_IDS].mean(axis=0)
    Z = None
    if depth is not None:
        r = 6
        ui, vi = int(u), int(v)
        patch = depth[max(0, vi - r):vi + r, max(0, ui - r):ui + r]
        valid = patch[patch > 0]
        if valid.size > 10:
            Z = float(np.median(valid)) / 1000.0
    if Z is None:
        size = np.linalg.norm(pts[0] - pts[9])
        Z = fx * cfg["palm_len_m"] / max(size, 1.0)
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.array([X, Y, Z]), (u, v)


def select_hand(results, w, h, depth, intr, cfg):
    """Najwieksza dlon w ROI i w zakresie odleglosci. Zwraca (pts, pos3d, uv, lm) albo None."""
    if not results.multi_hand_landmarks:
        return None
    best, best_area = None, 0.0
    m = (1 - cfg["roi_frac"]) / 2
    for lm in results.multi_hand_landmarks:
        pts = landmarks_px(lm, w, h)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        ucen, vcen = (x0 + x1) / 2, (y0 + y1) / 2
        if not (m * w < ucen < (1 - m) * w and m * h < vcen < (1 - m) * h):
            continue
        pos, uv = hand_point_3d(pts, depth, intr, cfg)
        if not (cfg["z_min"] <= pos[2] <= cfg["z_max"]):
            continue
        area = (x1 - x0) * (y1 - y0)
        if area > best_area:
            best, best_area = (pts, pos, uv, lm), area
    return best


class Debounce:
    def __init__(self, n):
        self.n, self.cand, self.cnt, self.state = n, None, 0, "NONE"

    def update(self, g):
        if g == self.cand:
            self.cnt += 1
        else:
            self.cand, self.cnt = g, 1
        if self.cnt >= self.n:
            self.state = g
        return self.state

    def reset(self):
        self.cand, self.cnt, self.state = None, 0, "NONE"


# ----------------------------------------------------------------------------------------
# Kontroler: pozycja dloni -> predkosc (fwd, right, up) w ukladzie heading drona
# ----------------------------------------------------------------------------------------
def deadzone_gain(d, dz, gain, vmax):
    s = np.sign(d)
    return float(np.clip(s * max(abs(d) - dz, 0.0) * gain, -vmax, vmax))


class GestureController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.anchor = None
        self.pos_f = None
        self.open_since = None
        self.last_seen = -1e9
        self.v_target = np.zeros(3)
        self.v_out = np.zeros(3)
        self.t_prev = None
        self.status = "CZEKAM NA DLON"

    def _cmd_from_disp(self, d):
        c = self.cfg
        lat = c["sign_lateral"] * d[0]
        up = -c["sign_vertical"] * d[1]   # Y kamery w dol
        fwd = c["sign_depth"] * d[2]
        if c["swap_fwd_lat"]:
            fwd, lat = lat, fwd
        return np.array([
            deadzone_gain(fwd, c["deadzone_m"], c["gain"], c["v_max_xy"]),
            deadzone_gain(lat, c["deadzone_m"], c["gain"], c["v_max_xy"]),
            deadzone_gain(up, c["deadzone_m"], c["gain"], c["v_max_z"]),
        ])

    def update(self, t, pos, gesture):
        c = self.cfg
        dt = 0.0 if self.t_prev is None else min(t - self.t_prev, 0.2)
        self.t_prev = t

        if pos is None:
            lost = t - self.last_seen
            if lost > c["lost_reset_s"]:
                self.anchor, self.pos_f, self.open_since = None, None, None
                self.status = "CZEKAM NA DLON"
            if lost > c["lost_stop_s"]:
                self.v_target = np.zeros(3)
                if self.anchor is not None:
                    self.status = "ZGUBIONA DLON - HOVER"
        else:
            self.last_seen = t
            a = c["ema_alpha"]
            self.pos_f = pos.copy() if self.pos_f is None else a * pos + (1 - a) * self.pos_f

            if gesture == "FIST":
                self.anchor, self.open_since = None, None
                self.v_target = np.zeros(3)
                self.status = "PIESC - STOP"
            elif gesture == "OPEN":
                if self.anchor is None:
                    if self.open_since is None:
                        self.open_since = t
                    self.v_target = np.zeros(3)
                    if t - self.open_since >= c["engage_hold_s"]:
                        self.anchor = self.pos_f.copy()
                        self.status = "STERUJESZ"
                    else:
                        self.status = "TRZYMAJ OTWARTA DLON..."
                else:
                    self.v_target = self._cmd_from_disp(self.pos_f - self.anchor)
                    self.status = "STERUJESZ"
            else:  # OTHER / NONE - pauza, kotwica zostaje
                self.open_since = None if self.anchor is None else self.open_since
                self.v_target = np.zeros(3)
                if self.anchor is not None:
                    self.status = "PAUZA - OTWORZ DLON"

        # limit przyspieszenia
        step = c["a_max"] * dt
        self.v_out = self.v_out + np.clip(self.v_target - self.v_out, -step, step)
        return self.v_out.copy()


def apply_fence(v, p_rel, alt, cfg):
    """v=[fwd,right,up], p_rel=[fwd,right] wzgledem startu [m], alt [m]."""
    v = v.copy()
    lims = (cfg["box_fwd"], cfg["box_lat"])
    for i in (0, 1):
        p, L = p_rel[i], lims[i]
        if p > L and v[i] > 0:
            v[i] = 0.0
        if p < -L and v[i] < 0:
            v[i] = 0.0
        if p > L + cfg["fence_margin"]:
            v[i] = -cfg["push_back"]
        if p < -L - cfg["fence_margin"]:
            v[i] = cfg["push_back"]
    if alt > cfg["alt_max"] and v[2] > 0:
        v[2] = 0.0
    if alt < cfg["alt_min"] and v[2] < 0:
        v[2] = 0.0
    if alt > cfg["alt_max"] + cfg["fence_margin"]:
        v[2] = -cfg["push_back"]
    if alt < cfg["alt_min"] - cfg["fence_margin"]:
        v[2] = cfg["push_back"]
    return v


# ----------------------------------------------------------------------------------------
# Dron (pymavlink) i dron wirtualny
# ----------------------------------------------------------------------------------------
class Drone:
    def __init__(self, conn, baud, cfg):
        from pymavlink import mavutil
        self.mu = mavutil
        self.cfg = cfg
        self.m = mavutil.mavlink_connection(conn, baud=baud, source_system=250)
        self.lock = threading.Lock()
        self.log = deque(maxlen=6)
        self.mode, self.armed = "?", False
        self.pos_ned, self.pos_t = None, 0.0
        self.yaw = 0.0
        self.rng = None
        self.flow_q = None
        self.ref = None  # (x, y, yaw) punktu startu
        self.v_cmd = np.zeros(3)
        self.control_enabled = False
        self.takeoff_active = False  # _tx milczy, zeby nie przerwac NAV_TAKEOFF
        self.origin_ok, self.origin_t = False, 0.0  # potwierdzenie GPS_GLOBAL_ORIGIN od autopilota
        self.running = True

        print("Czekam na heartbeat autopilota...")
        while True:
            hb = self.m.recv_match(type="HEARTBEAT", blocking=True, timeout=30)
            if hb is None:
                raise RuntimeError("Brak heartbeatu - sprawdz polaczenie")
            if hb.autopilot != mavutil.mavlink.MAV_AUTOPILOT_INVALID:
                self.m.target_system = hb.get_srcSystem()
                self.m.target_component = hb.get_srcComponent()
                break
        print(f"Polaczono: sys {self.m.target_system} comp {self.m.target_component}")

        threading.Thread(target=self._rx, daemon=True).start()
        L = mavutil.mavlink
        for msg_id, hz in ((L.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 10),
                           (L.MAVLINK_MSG_ID_ATTITUDE, 10),
                           (L.MAVLINK_MSG_ID_DISTANCE_SENSOR, 5),
                           (L.MAVLINK_MSG_ID_OPTICAL_FLOW, 5)):
            self.cmd(L.MAV_CMD_SET_MESSAGE_INTERVAL, msg_id, 1e6 / hz)
        self.set_origin()
        threading.Thread(target=self._tx, daemon=True).start()

    # --- niskopoziomowe ---
    def cmd(self, command, *params):
        p = list(params) + [0] * (7 - len(params))
        with self.lock:
            self.m.mav.command_long_send(self.m.target_system, self.m.target_component,
                                         command, 0, *p)

    def set_origin(self):
        lat, lon, alt = EKF_ORIGIN
        self.origin_t = time.time()
        with self.lock:
            self.m.mav.set_gps_global_origin_send(self.m.target_system,
                                                  int(lat * 1e7), int(lon * 1e7), int(alt * 1000))
        # odpowiedz przychodzi tez, gdy origin ustawil juz ktos inny
        self.cmd(self.mu.mavlink.MAV_CMD_REQUEST_MESSAGE, self.mu.mavlink.MAVLINK_MSG_ID_GPS_GLOBAL_ORIGIN)
        if not self.origin_ok:
            self.log.append("Wyslano EKF origin")

    def set_mode(self, name):
        with self.lock:
            self.m.mav.set_mode_send(self.m.target_system,
                                     self.mu.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                     COPTER_MODES[name])

    def _rx(self):
        L = self.mu.mavlink
        while self.running:
            msg = self.m.recv_match(blocking=True, timeout=0.5)
            if msg is None:
                continue
            if msg.get_srcSystem() != self.m.target_system:
                continue
            t = msg.get_type()
            if t == "HEARTBEAT" and msg.autopilot != L.MAV_AUTOPILOT_INVALID:
                self.mode = self.mu.mode_string_v10(msg)
                self.armed = bool(msg.base_mode & L.MAV_MODE_FLAG_SAFETY_ARMED)
            elif t == "LOCAL_POSITION_NED":
                self.pos_ned = np.array([msg.x, msg.y, msg.z])
                self.pos_t = time.time()
            elif t == "ATTITUDE":
                self.yaw = msg.yaw
            elif t == "DISTANCE_SENSOR":
                self.rng = msg.current_distance / 100.0
            elif t == "OPTICAL_FLOW":
                self.flow_q = msg.quality
            elif t == "GPS_GLOBAL_ORIGIN":
                if not self.origin_ok:
                    self.log.append("EKF origin OK")
                self.origin_ok = True
            elif t == "STATUSTEXT":
                self.log.append(msg.text)

    def _tx(self):
        L = self.mu.mavlink
        mask = (L.POSITION_TARGET_TYPEMASK_X_IGNORE | L.POSITION_TARGET_TYPEMASK_Y_IGNORE |
                L.POSITION_TARGET_TYPEMASK_Z_IGNORE | L.POSITION_TARGET_TYPEMASK_AX_IGNORE |
                L.POSITION_TARGET_TYPEMASK_AY_IGNORE | L.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
                L.POSITION_TARGET_TYPEMASK_YAW_IGNORE)
        period = 1.0 / self.cfg["send_hz"]
        while self.running:
            time.sleep(period)
            if not self.origin_ok and time.time() - self.origin_t > 2.0:
                self.set_origin()  # EKF odrzuca origin wyslany przed swoja inicjalizacja (kilka s po boocie)
            if not self.armed:
                self.ref = None  # nowy lot = nowy punkt odniesienia
            if self.takeoff_active or not (self.armed and self.mode == "GUIDED"):
                continue
            if self.ref is None:
                # GUIDED wlaczony z aparatury/GCS (nie klawiszem t): odniesienie = biezaca pozycja
                if not self.telem_ok():
                    continue
                self.ref = (self.pos_ned[0], self.pos_ned[1], self.yaw)
                self.log.append("Punkt odniesienia = biezaca pozycja")
            v = self.v_cmd if self.control_enabled else np.zeros(3)
            if not self.telem_ok():
                v = np.zeros(3)
            with self.lock:
                # MAV_FRAME_BODY_NED: predkosc wzgledem aktualnego heading, vz w dol
                self.m.mav.set_position_target_local_ned_send(
                    0, self.m.target_system, self.m.target_component,
                    L.MAV_FRAME_BODY_NED, mask,
                    0, 0, 0, float(v[0]), float(v[1]), float(-v[2]),
                    0, 0, 0, 0, 0.0)

    # --- stan ---
    def telem_ok(self):
        return self.pos_ned is not None and time.time() - self.pos_t < self.cfg["telem_timeout"]

    def rel_state(self):
        """(p_rel [fwd,right] wzgledem startu, wysokosc) albo None."""
        if self.pos_ned is None or self.ref is None:
            return None
        dx, dy = self.pos_ned[0] - self.ref[0], self.pos_ned[1] - self.ref[1]
        cy, sy = math.cos(self.ref[2]), math.sin(self.ref[2])
        fwd = cy * dx + sy * dy
        right = -sy * dx + cy * dy
        return np.array([fwd, right]), -self.pos_ned[2]

    # --- akcje operatora ---
    def takeoff(self):
        threading.Thread(target=self._takeoff, daemon=True).start()

    def _takeoff(self):
        self.takeoff_active = True
        try:
            self._takeoff_seq()
        finally:
            self.takeoff_active = False

    def _takeoff_seq(self):
        L = self.mu.mavlink
        self.control_enabled = False
        self.set_origin()
        self.set_mode("GUIDED")
        if not self._wait(lambda: self.mode == "GUIDED", 3):
            self.log.append("NIE WESZLO W GUIDED (EKF/origin?)")
            return
        self.cmd(L.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if not self._wait(lambda: self.armed, 5):
            self.log.append("ARM ODRZUCONY - patrz STATUSTEXT")
            return
        if self.pos_ned is not None:
            self.ref = (self.pos_ned[0], self.pos_ned[1], self.yaw)
        self.cmd(L.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, self.cfg["takeoff_alt"])
        ok = self._wait(lambda: self.pos_ned is not None
                        and -self.pos_ned[2] > 0.8 * self.cfg["takeoff_alt"], 10)
        self.log.append("TAKEOFF OK - g wlacza gesty" if ok else "TAKEOFF: brak wysokosci")

    def land(self):
        self.control_enabled = False
        self.set_mode("LAND")
        self.log.append("LAND")

    def _wait(self, cond, timeout):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.05)
        return False

    def close(self):
        self.running = False


class VirtualDrone:
    """Kinematyczny dron do trybu --no-drone (test gestow i plan B na stoisko)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.p = np.zeros(2)
        self.alt = cfg["takeoff_alt"]
        self.v_cmd = np.zeros(3)
        self.control_enabled = True
        self.mode, self.armed = "SIM", True
        self.log = deque(["Tryb bez drona (symulacja)"], maxlen=6)
        self.rng, self.flow_q = None, None
        self.t = time.time()

    def step(self):
        now = time.time()
        dt = min(now - self.t, 0.2)
        self.t = now
        v = self.v_cmd if self.control_enabled else np.zeros(3)
        self.p += v[:2] * dt
        self.alt += v[2] * dt

    def telem_ok(self):
        return True

    def rel_state(self):
        return self.p.copy(), self.alt

    def takeoff(self):
        self.p, self.alt = np.zeros(2), self.cfg["takeoff_alt"]

    def land(self):
        self.control_enabled = False

    def close(self):
        pass


# ----------------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------------
def draw_minimap(img, p_rel, alt, v, cfg, x0, y0, size=220):
    cv2.rectangle(img, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
    span = max(cfg["box_fwd"], cfg["box_lat"]) + 0.4
    sc = size / (2 * span)
    c = (x0 + size // 2, y0 + size // 2)

    def to_px(fwd, right):  # widok z gory: przod do gory obrazka
        return int(c[0] + right * sc), int(c[1] - fwd * sc)

    cv2.rectangle(img, to_px(cfg["box_fwd"], -cfg["box_lat"]), to_px(-cfg["box_fwd"], cfg["box_lat"]),
                  (0, 200, 255), 1)
    if p_rel is not None:
        dp = to_px(p_rel[0], p_rel[1])
        cv2.circle(img, dp, 9, (255, 255, 255), 2)
        tip = to_px(p_rel[0] + v[0] * 2.0, p_rel[1] + v[1] * 2.0)
        cv2.arrowedLine(img, dp, tip, (0, 255, 0), 2, tipLength=0.3)
    # pasek wysokosci
    bx = x0 + size + 10
    cv2.rectangle(img, (bx, y0), (bx + 18, y0 + size), (30, 30, 30), -1)
    amax = cfg["alt_max"] + 0.4
    for a, col in ((cfg["alt_min"], (0, 200, 255)), (cfg["alt_max"], (0, 200, 255))):
        yy = int(y0 + size - a / amax * size)
        cv2.line(img, (bx, yy), (bx + 18, yy), col, 1)
    if alt is not None:
        yy = int(y0 + size - np.clip(alt / amax, 0, 1) * size)
        cv2.rectangle(img, (bx + 3, yy - 3), (bx + 15, yy + 3), (255, 255, 255), -1)
    cv2.putText(img, "WIDOK Z GORY", (x0 + 5, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def put(img, text, y, color=(255, 255, 255), scale=0.6, x=12):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["oak", "webcam"], default="oak")
    ap.add_argument("--cam", type=int, default=0, help="indeks kamerki dla --source webcam")
    ap.add_argument("--conn", default="udpin:0.0.0.0:14551", help="polaczenie MAVLink")
    ap.add_argument("--baud", type=int, default=460800)
    ap.add_argument("--no-drone", action="store_true", help="bez drona: wirtualny dron na mapce")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    src = OakSource() if args.source == "oak" else WebcamSource(args.cam)
    drone = VirtualDrone(CFG) if args.no_drone else Drone(args.conn, args.baud, CFG)
    ctrl = GestureController(CFG)
    deb = Debounce(CFG["gesture_frames"])

    win = "KN SPA - sterowanie gestami"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    full = args.fullscreen
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)

    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, model_complexity=0,
                           min_detection_confidence=0.6, min_tracking_confidence=0.5)
    fps_t, fps = time.time(), 0.0
    try:
        while True:
            frame, depth, intr = src.read()
            h, w = frame.shape[:2]
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            sel = select_hand(res, w, h, depth, intr, CFG)

            t = time.time()
            if sel is None:
                gesture = deb.update("NONE")
                pos = None
            else:
                pts, pos, uv, lm = sel
                gesture = deb.update(classify_gesture(pts))
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (int(uv[0]), int(uv[1])), 6, (0, 255, 0), -1)

            v = ctrl.update(t, pos, gesture)
            st = drone.rel_state()
            if st is not None:
                v = apply_fence(v, st[0], st[1], CFG)
            elif not args.no_drone:
                v = np.zeros(3)
            drone.v_cmd = v
            if args.no_drone:
                drone.step()

            # --- rysowanie ---
            disp = cv2.resize(frame, (1280, 720))
            m = (1 - CFG["roi_frac"]) / 2
            cv2.rectangle(disp, (int(m * 1280), int(m * 720)), (int((1 - m) * 1280), int((1 - m) * 720)),
                          (90, 90, 90), 1)
            col = {"STERUJESZ": (0, 255, 0), "PIESC - STOP": (0, 0, 255)}.get(ctrl.status, (0, 200, 255))
            if not drone.control_enabled:
                col, status = (128, 128, 128), "GESTY WYLACZONE (operator: g)"
            else:
                status = ctrl.status
            cv2.rectangle(disp, (0, 0), (1279, 719), col, 8)
            put(disp, status, 50, col, 1.2, x=20)
            put(disp, "Otwarta dlon = steruj | Piesc = stop", 90, (230, 230, 230), 0.7, x=20)

            dt = time.time() - fps_t
            fps_t = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-3))
            y = 140
            rng = "-" if drone.rng is None else f"{drone.rng:.2f}"
            fq = "-" if drone.flow_q is None else str(drone.flow_q)
            alt = None if st is None else st[1]
            for line in (f"MODE {drone.mode}  ARMED {drone.armed}",
                         f"ALT {'-' if alt is None else f'{alt:.2f}'} m  RNG {rng} m  FLOWQ {fq}",
                         f"CMD fwd {v[0]:+.2f} right {v[1]:+.2f} up {v[2]:+.2f} m/s",
                         f"gest {gesture}  fps {fps:.0f}  telem {'OK' if drone.telem_ok() else 'BRAK'}"):
                put(disp, line, y, (255, 255, 255), 0.55, x=20)
                y += 24
            for i, line in enumerate(list(drone.log)[-4:]):
                put(disp, line, 690 - 22 * (3 - i), (180, 220, 255), 0.5, x=20)
            draw_minimap(disp, None if st is None else st[0], alt, v, CFG, 1280 - 270, 720 - 250)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("t"):
                drone.takeoff()
                ctrl.anchor = None
            elif k in (ord("l"), ord(" ")):
                drone.land()
            elif k == ord("g"):
                drone.control_enabled = not drone.control_enabled
                ctrl.anchor = None
                drone.log.append("Gesty " + ("ON" if drone.control_enabled else "OFF"))
            elif k == ord("f"):
                full = not full
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
            elif k in (ord("q"), 27):
                if drone.armed and not args.no_drone:
                    drone.land()
                    time.sleep(0.5)
                break
    finally:
        hands.close()
        src.close()
        drone.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()