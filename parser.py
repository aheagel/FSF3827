from pathlib import Path
from typing import Union, Dict, Any
import numpy as np

def parse_tsplib(content_or_path: Union[str, Path], build_dist: bool = False) -> Dict[str, Any]: # GPT-5 GENERATED
    """
    Parse a TSPLIB TSP file with EUC_2D coordinates.
    Raises ValueError on any format inconsistency.
    
    Args:
        content_or_path: Path to file or raw TSPLIB content string.
        build_dist: If True, returns a NumPy EUC_2D distance matrix in 'dist'.

    Returns:
        dict with keys:
            name, type, comment, dimension, edge_weight_type, coords (np.ndarray shape [n,2]),
            and optionally 'dist' (np.ndarray shape [n,n], dtype=int).
    """
    # Load text
    if isinstance(content_or_path, (str, Path)) and Path(str(content_or_path)).exists():
        text = Path(str(content_or_path)).read_text(encoding="utf-8")
    else:
        text = str(content_or_path)

    if not text.strip():
        raise ValueError("Empty TSPLIB content.")

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    n_lines = len(lines)

    meta = {
        "name": None,
        "type": None,
        "comment": None,
        "dimension": None,
        "edge_weight_type": None,
    }

    # Parse header
    while i < n_lines:
        ln = lines[i]
        u = ln.upper()
        if u == "NODE_COORD_SECTION":
            i += 1
            break
        if u == "EOF":
            raise ValueError("Unexpected EOF before NODE_COORD_SECTION.")
        if u.startswith("NAME:"):
            meta["name"] = ln.split(":", 1)[1].strip()
        elif u.startswith("TYPE:"):
            meta["type"] = ln.split(":", 1)[1].strip()
        elif u.startswith("COMMENT:"):
            meta["comment"] = ln.split(":", 1)[1].strip()
        elif u.startswith("DIMENSION:"):
            try:
                meta["dimension"] = int(ln.split(":", 1)[1].strip())
            except Exception as e:
                raise ValueError(f"Invalid DIMENSION line: {ln}") from e
        elif u.startswith("EDGE_WEIGHT_TYPE:"):
            meta["edge_weight_type"] = ln.split(":", 1)[1].strip()
        i += 1

    # Basic header checks
    if meta["dimension"] is None:
        raise ValueError("Missing DIMENSION in header.")
    if meta["edge_weight_type"] is None:
        raise ValueError("Missing EDGE_WEIGHT_TYPE in header.")
    if meta["edge_weight_type"].upper() != "EUC_2D":
        raise ValueError(f"Only EDGE_WEIGHT_TYPE=EUC_2D is supported, got {meta['edge_weight_type']}.")

    # Parse coordinates (expect exactly DIMENSION lines before EOF)
    coords = []
    count = 0
    while i < n_lines and lines[i].upper() != "EOF":
        parts = lines[i].split()
        if len(parts) != 3:
            raise ValueError(f"Invalid coordinate line (expected 3 tokens): '{lines[i]}'")
        city_id_str, x_str, y_str = parts
        try:
            # TSPLIB indices are 1-based; we only validate they are integers.
            _ = int(float(city_id_str))  # allow "1" or "1.0" forms
            x = float(x_str)
            y = float(y_str)
        except Exception as e:
            raise ValueError(f"Invalid numeric values in line: '{lines[i]}'") from e
        coords.append((x, y))
        count += 1
        i += 1

    if i == n_lines or lines[i].upper() != "EOF":
        raise ValueError("Missing EOF marker.")
    if count != meta["dimension"]:
        raise ValueError(
            f"Dimension mismatch: header {meta['dimension']} vs parsed {count} coordinates."
        )

    coords = np.asarray(coords, dtype=float)  # shape [n,2]

    out = {
        **meta,
        "coords": coords,
    }

    if build_dist:
        out["dist"] = euc2d_distance_matrix(coords)

    return out


def euc2d_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    TSPLIB EUC_2D distance:
      d_ij = int(round( sqrt( (xi-xj)^2 + (yi-yj)^2 ) ))
    Vectorized with NumPy.

    Args:
        coords: np.ndarray of shape [n,2], float

    Returns:
        np.ndarray of shape [n,n], dtype=int
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"'coords' must be shape [n,2], got {coords.shape}")

    x = coords[:, 0][:, None]  # [n,1]
    y = coords[:, 1][:, None]  # [n,1]
    dx = x - x.T               # [n,n]
    dy = y - y.T               # [n,n]
    dist = np.rint(np.hypot(dx, dy)).astype(int)  # integer-rounded
    np.fill_diagonal(dist, 0)
    return dist
