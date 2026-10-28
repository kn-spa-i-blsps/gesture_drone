#!/usr/bin/env python3
"""Weryfikacja parametrow z sitl_flow.parm na zywym SITL (PARAM_REQUEST_READ) + wersja firmware."""
import sys
import time

from pymavlink import mavutil

conn = sys.argv[1] if len(sys.argv) > 1 else "tcp:127.0.0.1:5760"
# nazwy dla ArduCopter 4.7.0 (zakresy dalmierza w metrach)
EXPECT = dict(SIM_FLOW_ENABLE=1, FLOW_TYPE=10, RNGFND1_TYPE=100, RNGFND1_MIN=0.05, RNGFND1_MAX=5.0,
              RNGFND1_GNDCLR=0.07, SIM_SONAR_POS_Z=-0.08, SIM_TERRAIN=0,
              GPS1_TYPE=0, COMPASS_ENABLE=0, COMPASS_USE=0, EK3_SRC1_POSXY=0, EK3_SRC1_VELXY=5,
              EK3_SRC1_POSZ=2, EK3_SRC1_YAW=0, GUID_TIMEOUT=1)
EXPECT_EXTRA = sys.argv[2:]  # dodatkowe NAZWA=WARTOSC

for kv in EXPECT_EXTRA:
    k, v = kv.split("=")
    EXPECT[k] = float(v)

m = mavutil.mavlink_connection(conn, source_system=250)
hb = m.wait_heartbeat(timeout=30)
print(f"heartbeat sys {m.target_system} comp {m.target_component}")

m.mav.command_long_send(m.target_system, m.target_component,
                        mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                        mavutil.mavlink.MAVLINK_MSG_ID_AUTOPILOT_VERSION, 0, 0, 0, 0, 0, 0)
av = m.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=5)
if av:
    v = av.flight_sw_version
    print(f"firmware V{v >> 24 & 0xff}.{v >> 16 & 0xff}.{v >> 8 & 0xff} (type {v & 0xff})")

bad = 0
for name, want in EXPECT.items():
    got = None
    for _ in range(3):
        m.mav.param_request_read_send(m.target_system, m.target_component, name.encode(), -1)
        t0 = time.time()
        while time.time() - t0 < 1.5:
            p = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if p and p.param_id == name:
                got = p.param_value
                break
        if got is not None:
            break
    if got is None:
        print(f"  {name:16s} BRAK PARAMETRU (oczekiwano {want})")
        bad += 1
    elif abs(got - want) > 1e-4:
        print(f"  {name:16s} = {got:g}  NIEZGODNE (oczekiwano {want})")
        bad += 1
    else:
        print(f"  {name:16s} = {got:g}  OK")
print("WYNIK:", "OK" if bad == 0 else f"{bad} problemow")
