#!/usr/bin/env python3
"""
sitl_headless_test.py - test lancucha Drone._tx -> ArduCopter GUIDED bez kamery i bez czlowieka.

Uzywa klasy Drone i apply_fence z gesture_drone.py (logika wysylania NIE jest duplikowana).
Obejscie startu przez RC_CHANNELS_OVERRIDE istnieje TYLKO tutaj (SITL), nigdy w gesture_drone.py.

  python3 sitl_headless_test.py --conn tcp:127.0.0.1:5760
  python3 sitl_headless_test.py --conn udpin:0.0.0.0:14551     # przez MAVProxy
"""
import argparse
import csv
import math
import os
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil

from gesture_drone import CFG, Drone, apply_fence

L = mavutil.mavlink
GCS_SYSID = 255  # SYSID_MYGCS: RC_CHANNELS_OVERRIDE jest przyjmowane tylko od tego sysid


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Probe:
    """Podsluch wiadomosci odbieranych przez Drone._rx (message_hooks pymavlink) + logger CSV."""

    def __init__(self, drone, path):
        self.d = drone
        self.tgt = None          # ostatni POSITION_TARGET_LOCAL_NED
        self.tgt_t = 0.0
        self.vel = None          # (vx, vy, vz) z LOCAL_POSITION_NED
        self.boot_ms = 0         # time_boot_ms autopilota (z ATTITUDE)
        self.ekf_flags = 0       # EKF_STATUS_REPORT.flags
        self.ekf_noise_t = time.time()  # ostatni STATUSTEXT o przelaczaniu aidingu / EKF variance
        self.statustext = []
        self.phase = "init"
        self.cmd = np.zeros(3)   # komenda po fence (jak drone.v_cmd)
        self.rows = []
        self.t0 = time.time()
        self.tx_count = 0        # licznik pakietow wyslanych przez Drone._tx
        send = drone.m.mav.set_position_target_local_ned_send

        def counted(*a, **k):
            self.tx_count += 1
            return send(*a, **k)
        drone.m.mav.set_position_target_local_ned_send = counted
        drone.m.message_hooks.append(self._hook)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["t", "phase", "mode", "armed", "telem_ok", "has_ref", "ctrl_en", "tx_count",
                         "cmd_fwd", "cmd_right", "cmd_up", "x", "y", "z", "vx", "vy", "vz",
                         "v_fwd", "v_right", "yaw_deg", "rng",
                         "tgt_frame", "tgt_mask", "tgt_vx", "tgt_vy", "tgt_vz", "tgt_age"])
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _hook(self, _conn, msg):
        if msg.get_srcSystem() != self.d.m.target_system:
            return
        t = msg.get_type()
        if t == "POSITION_TARGET_LOCAL_NED":
            self.tgt, self.tgt_t = msg, time.time()
        elif t == "LOCAL_POSITION_NED":
            self.vel = (msg.vx, msg.vy, msg.vz)
        elif t == "ATTITUDE":
            self.boot_ms = msg.time_boot_ms
        elif t == "EKF_STATUS_REPORT":
            self.ekf_flags = msg.flags
        elif t == "STATUSTEXT":
            self.statustext.append(msg.text)
            if "aiding" in msg.text or "variance" in msg.text:
                self.ekf_noise_t = time.time()
            say(f"STATUSTEXT: {msg.text}")

    def _loop(self):
        while self.running:
            d = self.d
            p = d.pos_ned if d.pos_ned is not None else [math.nan] * 3
            v = self.vel if self.vel is not None else (math.nan,) * 3
            cy, sy = math.cos(d.yaw), math.sin(d.yaw)
            vf, vr = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]
            tg = self.tgt
            row = [round(time.time() - self.t0, 2), self.phase, d.mode, int(d.armed), int(d.telem_ok()),
                   int(d.ref is not None), int(d.control_enabled), self.tx_count,
                   *[round(float(c), 3) for c in d.v_cmd],
                   *[round(float(c), 3) for c in p], *[round(float(c), 3) for c in v],
                   round(vf, 3), round(vr, 3), round(math.degrees(d.yaw), 1),
                   "" if d.rng is None else d.rng]
            if tg is None:
                row += ["", "", "", "", "", ""]
            else:
                row += [tg.coordinate_frame, tg.type_mask, round(tg.vx, 3), round(tg.vy, 3),
                        round(tg.vz, 3), round(time.time() - self.tgt_t, 2)]
            self.rows.append(dict(zip(["t", "phase", "mode", "armed", "telem_ok", "has_ref", "ctrl_en",
                                       "tx_count", "cmd_fwd", "cmd_right", "cmd_up", "x", "y", "z",
                                       "vx", "vy", "vz", "v_fwd", "v_right", "yaw_deg", "rng",
                                       "tgt_frame", "tgt_mask", "tgt_vx", "tgt_vy", "tgt_vz",
                                       "tgt_age"], row)))
            self.w.writerow(row)
            self.f.flush()
            time.sleep(0.1)

    def close(self):
        self.running = False
        time.sleep(0.15)
        self.f.close()


def alt(d):
    return math.nan if d.pos_ned is None else -d.pos_ned[2]


def command(d, v_raw, fence=True):
    """Dokladnie jak petla main(): fence gdy jest rel_state, inaczej komenda zerowana."""
    st = d.rel_state()
    if st is None:
        v = np.zeros(3)
    elif fence:
        v = apply_fence(np.asarray(v_raw, float), st[0], st[1], CFG)
    else:
        v = np.asarray(v_raw, float)
    d.v_cmd = v


def hold(d, pr, phase, v_raw, dur, fence=True, until=None):
    pr.phase = phase
    say(f"etap {phase}: v_raw={v_raw} przez {dur}s")
    t0 = time.time()
    while time.time() - t0 < dur:
        command(d, v_raw, fence)
        if until is not None and until():
            break
        time.sleep(0.05)


class RcOverride:
    """RC_CHANNELS_OVERRIDE z sysid GCS (255) przez to samo polaczenie, pod lockiem Drone."""

    def __init__(self, d):
        self.d = d
        self.enc = L.MAVLink(None, srcSystem=GCS_SYSID, srcComponent=190)

    def thr(self, pwm):  # pwm=0 -> zwolnienie kanalu
        ch = [65535] * 8
        ch[2] = pwm
        msg = self.enc.rc_channels_override_encode(self.d.m.target_system, self.d.m.target_component, *ch)
        with self.d.lock:
            self.d.m.write(msg.pack(self.enc))


def statustexts_since(pr, n0):
    return pr.statustext[n0:]


def takeoff(d, pr, force_workaround=False):
    """Najpierw sciezka produkcyjna Drone._takeoff; jesli arming odrzucony - obejscie RC (tylko SITL)."""
    pr.phase = "takeoff_prod"
    n0 = len(pr.statustext)
    if not force_workaround:
        say("Proba Drone._takeoff() z ziemi (GUIDED, arm, NAV_TAKEOFF)")
        d._takeoff()
        say(f"Drone.log: {list(d.log)}")
    if d.armed:
        ok = d._wait(lambda: alt(d) > 0.8 * CFG["takeoff_alt"], 15)
        say(f"Sciezka produkcyjna: armed, alt={alt(d):.2f} ok={ok}")
        hold(d, pr, "takeoff_settle", [0, 0, 0], 3)
        return "production"

    # dokladne komunikaty PreArm
    d.cmd(L.MAV_CMD_RUN_PREARM_CHECKS)
    time.sleep(2)
    say("--workaround: pomijam Drone._takeoff" if force_workaround else "Arming z ziemi ODRZUCONY. Komunikaty od proby:")
    for s in statustexts_since(pr, n0):
        say(f"    {s}")

    say("OBEJSCIE (tylko test): ALT_HOLD + arm + RC3 override")
    pr.phase = "takeoff_workaround"
    rc = RcOverride(d)
    rc.thr(1000)
    d.set_mode("ALT_HOLD")
    if not d._wait(lambda: d.mode == "ALT_HOLD", 5):
        raise RuntimeError("Nie weszlo w ALT_HOLD")
    for _ in range(3):
        rc.thr(1000)
        d.cmd(L.MAV_CMD_COMPONENT_ARM_DISARM, 1)
        if d._wait(lambda: d.armed, 3):
            break
    if not d.armed:
        raise RuntimeError("Arm w ALT_HOLD odrzucony")
    t0 = time.time()
    while alt(d) < 0.9 * CFG["takeoff_alt"] and time.time() - t0 < 15:
        rc.thr(1700)
        time.sleep(0.1)
    say(f"Wznoszenie zakonczone, alt={alt(d):.2f}")
    t0 = time.time()
    while d.mode != "GUIDED" and time.time() - t0 < 15:
        rc.thr(1500)  # odswiezanie (RC_OVERRIDE_TIME 3 s), inaczej ALT_HOLD spada na gazie 1000
        d.set_mode("GUIDED")
        d._wait(lambda: d.mode == "GUIDED", 1)
    if d.mode != "GUIDED":
        raise RuntimeError("Nie weszlo w GUIDED w powietrzu")
    rc.thr(0)
    say("GUIDED w powietrzu")
    return "workaround"


def phase_rows(pr, phase):
    return [r for r in pr.rows if r["phase"] == phase]


def evaluate(pr, how):
    res = []

    def check(name, ok, info):
        res.append((name, ok, info))
        say(f"{'PASS' if ok else 'FAIL'}  {name}: {info}")

    if how == "production":
        rows = phase_rows(pr, "takeoff_prod")
        settle = phase_rows(pr, "takeoff_settle")
        top = max(-r["z"] for r in rows + settle if r["z"] == r["z"])
        tx = rows[-1]["tx_count"] - rows[0]["tx_count"]
        check("takeoff_prod", top >= 0.8 * CFG["takeoff_alt"] and tx == 0,
              f"Drone._takeoff z ziemi, alt po starcie {-settle[-1]['z']:.2f} m (max {top:.2f}), "
              f"tx w trakcie NAV_TAKEOFF +{tx}, start trwal {rows[-1]['t'] - rows[0]['t']:.1f} s")
    else:
        rows = phase_rows(pr, "guided_external")
        tx = rows[-1]["tx_count"] - rows[0]["tx_count"]
        vz = max(r["vz"] for r in rows)
        check("guided_external", tx > 0 and vz > 0.1,
              f"GUIDED z zewnatrz: ref={'jest' if rows[-1]['has_ref'] else 'BRAK'}, tx +{tx}, vz w dol max {vz:.3f}, "
              f"alt {-rows[0]['z']:.2f} -> {-rows[-1]['z']:.2f}")

    for phase, key, other in (("fwd_0.2", "v_fwd", "v_right"), ("right_0.2", "v_right", "v_fwd")):
        rows = phase_rows(pr, phase)
        t0 = rows[0]["t"]
        early = [r[key] for r in rows if r["t"] - t0 <= 2.0]
        allv = [r[key] for r in rows]
        cross = max(abs(r[other]) for r in rows)
        tx = rows[-1]["tx_count"] - rows[0]["tx_count"]
        tg = [r for r in rows if r["tgt_vx"] != ""]
        tgt_sp = max((math.hypot(r["tgt_vx"], r["tgt_vy"]) for r in tg), default=0.0)
        t_reach = next((r["t"] - t0 for r in rows if r[key] >= 0.15), math.nan)
        ok = max(early) >= 0.15 and max(allv) <= 0.25
        check(phase, ok, f"t(v>=0.15)={t_reach:.1f}s max={max(allv):.3f} m/s, "
                         f"os poprzeczna max {cross:.3f}, tx +{tx}, target |v| max {tgt_sp:.3f}")

    rows = phase_rows(pr, "up_0.15")
    vz_up = [-r["vz"] for r in rows]
    check("up_0.15", max(vz_up) >= 0.1,
          f"vz w gore max {max(vz_up):.3f} m/s, alt {-rows[0]['z']:.2f} -> {-rows[-1]['z']:.2f}")

    rows = phase_rows(pr, "zero")
    t0 = rows[0]["t"]
    tail = [r for r in rows if r["t"] - t0 >= 1.5]
    vh = max(math.hypot(r["vx"], r["vy"]) for r in tail)
    vz = max(abs(r["vz"]) for r in tail)
    check("zero (hover)", vh < 0.05 and vz < 0.05,
          f"po 1.5 s: |v_xy| max {vh:.3f}, |vz| max {vz:.3f} m/s")

    rows = phase_rows(pr, "fence")
    above_max = [r for r in rows if -r["z"] > CFG["alt_max"]]
    pos_up = [r for r in above_max if r["cmd_up"] > 0]
    above_m = [r for r in rows if -r["z"] > CFG["alt_max"] + CFG["fence_margin"] + 0.02]
    pushes = all(r["cmd_up"] <= -CFG["push_back"] + 1e-6 for r in above_m)
    a0, a1 = -rows[0]["z"], -rows[-1]["z"]
    vz_dn = max(r["vz"] for r in rows)
    ok = not pos_up and pushes and a1 < a0 - 0.2 and a1 <= CFG["alt_max"] + CFG["fence_margin"] + 0.05
    check("fence alt_max", ok,
          f"komenda raw +0.2 w gore; alt {a0:.2f} -> {a1:.2f} m, cmd_up>0 nad alt_max: {len(pos_up)} probek, "
          f"push_back aktywny: {pushes}, vz w dol max {vz_dn:.3f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="tcp:127.0.0.1:5760")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workaround", action="store_true",
                    help="pomin Drone._takeoff: start ALT_HOLD+RC, potem GUIDED z zewnatrz (scenariusz z pokazu)")
    ap.add_argument("--set-ref", action="store_true",
                    help="ustaw drone.ref recznie, jesli Drone go nie zlapal (diagnostyka starej wersji)")
    args = ap.parse_args()

    tag = args.tag or args.conn.split(":")[0]
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"), exist_ok=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs",
                        f"sitl_headless_{tag}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    d = Drone(args.conn, 115200, CFG)          # wysyla tez EKF origin
    pr = Probe(d, path)
    d.cmd(L.MAV_CMD_SET_MESSAGE_INTERVAL, L.MAVLINK_MSG_ID_POSITION_TARGET_LOCAL_NED, 1e5)
    d.cmd(L.MAV_CMD_SET_MESSAGE_INTERVAL, L.MAVLINK_MSG_ID_EKF_STATUS_REPORT, 5e5)
    say(f"CSV: {path}")
    if not d._wait(lambda: d.telem_ok(), 20):
        say("UWAGA: brak LOCAL_POSITION_NED przed startem")
    while pr.boot_ms < 30000:  # po starcie SITL: zbieznosc IMU/EKF (Gyros inconsistent)
        time.sleep(0.5)
    time.sleep(3)  # EKF po origin
    # jak operator przed 't': czekaj, az EKF przewiduje pozycje wzgledna (bez tego: Need Position Estimate)
    t0 = time.time()
    d._wait(lambda: pr.ekf_flags & L.EKF_PRED_POS_HORIZ_REL and time.time() - pr.ekf_noise_t > 5, 90)
    say(f"EKF flags {pr.ekf_flags} po {time.time() - t0:.1f} s czekania (boot {pr.boot_ms / 1000:.0f} s)")

    result = 1
    try:
        how = takeoff(d, pr, args.workaround)
        say(f"Start: {how}; mode={d.mode} armed={d.armed} alt={alt(d):.2f}")

        if how == "workaround":
            # objaw z pokazu: GUIDED wlaczony z zewnatrz (nie przez _takeoff), ref nie ustawiony
            d.control_enabled = True
            n = pr.tx_count
            hold(d, pr, "guided_external", [0, 0, -0.15], 3)
            say(f"guided_external: ref={d.ref}, tx +{pr.tx_count - n}, alt={alt(d):.2f}")
            if d.ref is None and args.set_ref:
                d.ref = (d.pos_ned[0], d.pos_ned[1], d.yaw)
                say(f"--set-ref: test ustawia drone.ref = {tuple(round(float(c), 3) for c in d.ref)}")
            if d.ref is None:
                raise RuntimeError("drone.ref nadal None w GUIDED - _tx nic nie wysyla")

        d.control_enabled = True
        hold(d, pr, "pre_zero", [0, 0, 0], 2)
        hold(d, pr, "fwd_0.2", [0.2, 0, 0], 4)
        hold(d, pr, "right_0.2", [0, 0.2, 0], 4)
        hold(d, pr, "up_0.15", [0, 0, 0.15], 3)
        hold(d, pr, "zero", [0, 0, 0], 3)
        # wyprowadzenie ponad alt_max+margin z ominieciem fence (symulacja sytuacji z 2.5 m)
        hold(d, pr, "climb_nofence", [0, 0, 0.25], 15, fence=False,
             until=lambda: alt(d) > CFG["alt_max"] + 0.6)
        hold(d, pr, "fence", [0, 0, 0.2], 10)
        hold(d, pr, "final_zero", [0, 0, 0], 1)
        res = evaluate(pr, how)
        result = 0 if all(ok for _, ok, _ in res) else 1
    except Exception as e:
        say(f"BLAD: {e!r}")
    finally:
        pr.phase = "land"
        d.land()
        say("LAND")
        if not d._wait(lambda: not d.armed, 45):
            say("Nie rozbroil sie sam - wymuszam disarm")
            d.cmd(L.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)
            d._wait(lambda: not d.armed, 5)
        say(f"armed={d.armed}, alt={alt(d):.2f}, tx_count={pr.tx_count}")
        pr.close()
        d.close()
    say(f"WYNIK: {'PASS' if result == 0 else 'FAIL'}  CSV: {path}")
    sys.exit(result)


if __name__ == "__main__":
    main()
