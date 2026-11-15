"""
biopsy_plan.py — compute the salient parameters of a biopsy plan.

Inputs (from the assignment):
  C_tum, R_tum : centre and radius of the spherical tumour model, CT frame (mm)
  P_ct, v_ct   : needle tip position and needle direction, CT frame
Outputs:
  targeting error  : distance of the needle axis from the tumour centre (mm)
  tissue core      : length of the needle trajectory inside the tumour (mm)
  insertion depth  : how far the needle must advance from its current tip
                     position to harvest that whole core (mm)

Plan: the needle is inserted forward along its current trajectory,
    X(s) = P_ct + s * v_ct,   s >= 0   (s = distance travelled, in mm)

Method:
  1. Targeting error = point_line_distance(C_tum, P_ct, v_ct)
  2. intersect_line_sphere gives the entry s1 and exit s2 of the line.
  3. Core length = s2 - s1, and insertion depth = s2: the needle has to reach
     the far side of the tumour to take the full core.

Edge cases:
  * Touching (tangent) trajectory: one contact point, core length 0. The depth
    to the contact point is reported, but there is nothing to harvest.
  * Miss: core length 0 and no insertion depth (None).
  * Tumour behind the needle (s2 < 0): moving forward never reaches it, so it
    is treated as a miss.
  * Tip already inside the tumour (s1 < 0): the core starts at the tip, so s1
    is clamped to 0.
"""

import numpy as np
from geometry import point_line_distance, intersect_line_sphere


def plan_biopsy(c_tum, r_tum, p_ct, v_ct):
    """
    Returns a dict with:
      'status'          : 'intersects', 'touches' or 'misses'
      'targeting_error' : mm
      'core_length'     : mm (0 if the needle touches or misses)
      'insertion_depth' : mm, or None if the needle misses
    """
    v = np.asarray(v_ct, dtype=float)
    v = v / np.linalg.norm(v)

    targeting_error = point_line_distance(c_tum, p_ct, v)
    s = intersect_line_sphere(p_ct, v, c_tum, r_tum)

    if len(s) == 2 and s[1] >= 0:                 # passes through, ahead of the tip
        s1, s2 = max(s[0], 0.0), s[1]
        return {"status": "intersects", "targeting_error": targeting_error,
                "core_length": s2 - s1, "insertion_depth": s2}

    if len(s) == 1 and s[0] >= 0:                 # just touches, ahead of the tip
        return {"status": "touches", "targeting_error": targeting_error,
                "core_length": 0.0, "insertion_depth": s[0]}

    return {"status": "misses", "targeting_error": targeting_error,
            "core_length": 0.0, "insertion_depth": None}
