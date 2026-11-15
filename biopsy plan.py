# Line utilities (Primer Part 1)
#   A line is given by a point on it and a unit direction:  X(s) = L0 + s * u
# --------------------------------------------------------------------------

def point_line_distance(point, line_pt, line_dir):
    """
    Shortest (perpendicular) distance from a point to a line.

    With w = point - line_pt and u the unit direction, |w x u| = |w| sin(theta),
    which is exactly the perpendicular distance. The direction is normalised
    here so the formula is valid even if the caller passes a non-unit vector.
    """
    u = np.asarray(line_dir, dtype=float)
    u = u / np.linalg.norm(u)
    w = np.asarray(point, dtype=float) - np.asarray(line_pt, dtype=float)
    return np.linalg.norm(np.cross(w, u))


def intersect_line_sphere(line_pt, line_dir, center, radius, tol=1e-9):
    """
    Intersect the infinite line X(s) = line_pt + s * u with a sphere.

    Method:
      s_c = (center - line_pt) . u        parameter of the point closest to center
      d   = distance from center to line
      If d < R:  half-chord h = sqrt(R^2 - d^2)   (Pythagoras: R is the
                 hypotenuse, d and h are the two sides) -> s = s_c - h, s_c + h
      If d = R:  the line just touches the sphere      -> s = s_c
      If d > R:  no intersection                       -> []
    "d = R" is tested with a tolerance, because floating-point values are
    rarely exactly equal.

    Returns a sorted list of 0, 1 or 2 line parameters s (distances along u).
    """
    u = np.asarray(line_dir, dtype=float)
    u = u / np.linalg.norm(u)
    w = np.asarray(center, dtype=float) - np.asarray(line_pt, dtype=float)
    s_c = float(w @ u)
    d = point_line_distance(center, line_pt, u)

    if abs(d - radius) <= tol:
        return [s_c]
    if d > radius:
        return []
    h = np.sqrt(radius**2 - d**2)
    return [s_c - h, s_c + h]
