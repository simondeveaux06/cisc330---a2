
"""
registration.py — register the tracked biopsy needle to the CT frame.
 
Inputs (from the assignment):
  M1tool, M2tool, M3tool : the three markers in the tool model frame
  Ptool                  : tool tip in the tool model frame
  vtool                  : tool axis direction in the tool model frame
  M1ct, M2ct, M3ct       : the same three markers, located in the CT images
Outputs:
  Pct : tool tip in the CT frame
  vct : tool axis direction in the CT frame

  1. Build the marker frame in each:  F_he from M_tool,  F_hv from M_ct
  2. Registration matrix:             F_ct_tool = F_hv @ inv(F_he)
  3. Tip (a point, w = 1):            Pct = F_ct_tool @ [Ptool, 1]
  4. Axis (a direction, w = 0):       vct = F_ct_tool @ [vtool, 0]
     A direction is not moved by translation, only rotated.
"""
 
import numpy as np
from geometry import register, transform_point, transform_vector
 
# Convert once so tool coordinates, tracker noise, and CT use mm.
CM_TO_MM = 10.0
M_TOOL = CM_TO_MM * np.array([
    [2.0, 18.0, 0.0],    # M1
    [8.0, 18.0, 0.0],    # M2
    [2.0, 24.0, 0.0],    # M3
])
P_TOOL = np.array([0.0, 0.0, 0.0])      # tool tip (mm)
V_TOOL = np.array([0.0, -1.0, 0.0])     # unit axis (dimensionless)


def register_tool(markers_tool, p_tool, v_tool, markers_ct):
    """
    Register the tool to the CT frame; return the tip and axis in CT.
 
    Parameters
    ----------
    markers_tool : (3, 3) M1tool, M2tool, M3tool as rows
    p_tool       : (3,)   tool tip in tool frame
    v_tool       : (3,)   tool axis in tool frame
    markers_ct   : (3, 3) M1ct, M2ct, M3ct as rows, in the SAME order
 
    Returns
    -------
    p_ct : (3,)   tool tip in CT frame
    v_ct : (3,)   unit tool axis in CT frame
    F    : (4,4)  tool-to-CT registration matrix (useful for checks and for
                  the Targeting Error Simulation)
    """
    F = register(markers_tool, markers_ct)
    p_ct = transform_point(F, p_tool)
    v_ct = transform_vector(F, v_tool)
    return p_ct, v_ct, F
 
