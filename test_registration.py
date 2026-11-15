"""
test_registration.py — ground-truth tests for the registration module.

For each required test, the tool model (Fig. 2) is placed in a known pose in
the CT frame. The CT marker positions and the EXPECTED tip and axis were
worked out on paper (shown in the comments) and typed in as numbers, so the
test does not rely on the code it is checking. The registration must
reproduce them.

Tool model (tool frame, mm; Fig. 2 is in cm, so its values x 10):
    M1 = (20, 180, 0)   M2 = (80, 180, 0)   M3 = (20, 240, 0)
    P  = (0, 0, 0)    v  = (0, -1, 0)

Run:  python test_registration.py
"""

import numpy as np
from registration import register_tool, M_TOOL, P_TOOL, V_TOOL
from geometry import rot_x, rot_z, make_F, transform_point

TOL = 1e-9  # numerical-precision tolerance (mm, or unitless for directions)
np.set_printoptions(suppress=True)


def check(name, markers_ct, p_expected, v_expected, F_expected):
    p_ct, v_ct, F = register_tool(M_TOOL, P_TOOL, V_TOOL, markers_ct)
    mapped = np.array([transform_point(F, m) for m in M_TOOL])
    errs = {
        "tip error (mm)":  np.linalg.norm(p_ct - p_expected),
        "axis error":      np.linalg.norm(v_ct - v_expected),
        "F error":         np.linalg.norm(F - F_expected),
        "marker RMS (mm)": np.sqrt(np.mean(np.sum((mapped - markers_ct) ** 2, axis=1))),
    }
    ok = all(e < TOL for e in errs.values())
    print(f"\n=== {name} ===")
    print(f"  M_ct = {markers_ct.tolist()}")
    print(f"  expected Pct = {p_expected},  computed Pct = {np.round(p_ct, 9) + 0.0}")
    print(f"  expected vct = {v_expected},  computed vct = {np.round(v_ct, 9) + 0.0}")
    for k, e in errs.items():
        print(f"  {k:<16} {e:.1e}")
    print("  PASS" if ok else "  FAIL")
    return ok


# --------------------------------------------------------------------------
# Test 1: PURE TRANSLATION by t = (10, 20, 30), R = I
#
#   On paper: every point shifts by t; directions are unchanged.
#     M1ct = (20,180,0)+(10,20,30) = (30, 200, 30)
#     M2ct = (80,180,0)+(10,20,30) = (90, 200, 30)
#     M3ct = (20,240,0)+(10,20,30) = (30, 260, 30)
#     Pct  = (0,0,0) +(10,20,30) = (10, 20, 30)
#     vct  = (0,-1,0)    translation does not affect a direction (w = 0)
# --------------------------------------------------------------------------
def test_pure_translation():
    markers_ct = np.array([[30, 200, 30], [90, 200, 30], [30, 260, 30]], float)
    return check("Test 1: pure translation, t = (10, 20, 30)", markers_ct,
                 p_expected=np.array([10, 20, 30.0]),
                 v_expected=np.array([0, -1, 0.0]),
                 F_expected=make_F(np.eye(3), [10, 20, 30]))


# --------------------------------------------------------------------------
# Test 2: PURE ROTATION of 90 deg about z, t = 0
#
#   Rz(90) = [[0,-1,0],[1,0,0],[0,0,1]]   maps (x, y, z) -> (-y, x, z)
#     M1ct = (-180, 20, 0)
#     M2ct = (-180, 80, 0)
#     M3ct = (-240, 20, 0)
#     Pct  = (0, 0, 0)     the tip is at the origin, which a rotation about
#                          the origin leaves in place
#     vct  = (1, 0, 0)     (0,-1,0) -> (-(-1), 0, 0)
# --------------------------------------------------------------------------
def test_pure_rotation():
    markers_ct = np.array([[-180, 20, 0], [-180, 80, 0], [-240, 20, 0]], float)
    return check("Test 2: pure rotation, 90 deg about z", markers_ct,
                 p_expected=np.array([0, 0, 0.0]),
                 v_expected=np.array([1, 0, 0.0]),
                 F_expected=make_F(rot_z(90), [0, 0, 0]))


# --------------------------------------------------------------------------
# Test 3: ROTATION + TRANSLATION: 90 deg about x, then t = (100, -50, 200)
#   (F = T R: rotate first, then translate)
#
#   Rx(90) = [[1,0,0],[0,0,-1],[0,1,0]]   maps (x, y, z) -> (x, -z, y)
#     M1: (20,180,0) -> (20,0,180) -> +t -> (120, -50, 380)
#     M2: (80,180,0) -> (80,0,180) -> +t -> (180, -50, 380)
#     M3: (20,240,0) -> (20,0,240) -> +t -> (120, -50, 440)
#     Pct = (0,0,0) -> (0,0,0) -> +t -> (100, -50, 200)
#     vct = (0,-1,0) -> (0, 0, -1)    rotation only, no +t
# --------------------------------------------------------------------------
def test_rotation_and_translation():
    markers_ct = np.array([[120, -50, 380], [180, -50, 380], [120, -50, 440]], float)
    return check("Test 3: 90 deg about x, then t = (100, -50, 200)", markers_ct,
                 p_expected=np.array([100, -50, 200.0]),
                 v_expected=np.array([0, 0, -1.0]),
                 F_expected=make_F(rot_x(90), [100, -50, 200]))


if __name__ == "__main__":
    results = [test_pure_translation(), test_pure_rotation(),
               test_rotation_and_translation()]
    print(f"\n{sum(results)}/{len(results)} tests passed")
