import numpy as np
from scipy.signal import savgol_filter
try:
    from sklearn.linear_model import LinearRegression
    _HAVE_SKLEARN = True
except Exception:
    LinearRegression = None
    _HAVE_SKLEARN = False


def _largest_true_segment(mask):
    """Return (start_idx, end_idx) of the largest contiguous True segment in mask.
    If none found, return None.
    Indices are inclusive (start, end).
    """
    best_len = 0
    best_seg = None
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            length = j - i + 1
            if length > best_len:
                best_len = length
                best_seg = (i, j)
            i = j + 1
        else:
            i += 1
    return best_seg


def _expand_threshold_until_segment(d2y, mask_func, max_multiplier=8):
    """Try multiple threshold multipliers to obtain at least one contiguous segment.
    mask_func(threshold) should return a boolean mask array.
    Returns the first mask that yields a segment or the last mask.
    """
    base_std = np.nanstd(d2y)
    if np.isfinite(base_std) and base_std > 0:
        multipliers = [1, 2, 4, 8][: max_multiplier.bit_length()]
    else:
        multipliers = [1]
    last_mask = None
    for m in multipliers:
        thr = 0.5 * base_std * m
        mask = mask_func(thr)
        seg = _largest_true_segment(mask)
        last_mask = mask
        if seg is not None and (seg[1] - seg[0] + 1) >= 3:
            return mask
    return last_mask


def extract_linear_regions(x, y, smooth_window=None, polyorder=3, debug=False):
    """
    Detect left/right linear regions in y(x) (y should be ln(current) or similar).

    Returns a dict with keys:
      left_slope, right_slope, left_boundary, right_boundary,
      depletion_width, left_indices, right_indices

    Notes:
      - x, y are 1D arrays of same length.
      - The function smooths y with Savitzky-Golay filter (optional).
      - It uses second derivative (curvature) to detect low-curvature segments.
      - If no clear segment is found, it relaxes the threshold heuristically.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D arrays")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")

    n = x.size
    if n < 6:
        raise ValueError("Need at least 6 points for robust detection")

    # Ensure x is strictly increasing; if not, sort
    if not np.all(np.diff(x) > 0):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    # Smooth y with Savitzky-Golay to reduce noise in derivatives
    # Choose window_length automatically if not provided: odd and <= n-1
    if smooth_window is None:
        # window ~ max(5, odd fraction of n)
        w = max(5, int(np.clip(n // 15, 5, 51)))
        if w % 2 == 0:
            w += 1
        if w >= n:
            w = n - 1 if (n - 1) % 2 == 1 else n - 2
        smooth_window = w
    else:
        smooth_window = int(smooth_window)
        if smooth_window % 2 == 0:
            smooth_window += 1
        smooth_window = min(smooth_window, n - (1 if (n - 1) % 2 == 1 else 2))

    try:
        y_s = savgol_filter(y, smooth_window, polyorder)
    except Exception:
        # fallback: no smoothing
        y_s = y.copy()

    # First and second derivatives using numpy.gradient for non-uniform x
    dy = np.gradient(y_s, x)
    d2y = np.gradient(dy, x)

    # Mask where absolute second derivative is small
    base_std = np.nanstd(d2y)
    if not np.isfinite(base_std) or base_std == 0:
        base_std = np.nanstd(d2y + 0.0)

    def make_mask(thr):
        return np.abs(d2y) < thr

    mask = _expand_threshold_until_segment(d2y, make_mask)
    if mask is None:
        mask = make_mask(0.5 * base_std)

    # Locate peak index (highest y)
    peak_idx = int(np.argmax(y))

    # Split mask into left (indices <= peak_idx-1) and right (>= peak_idx+1)
    if peak_idx <= 1 or peak_idx >= n - 2:
        # Peak too close to edge; fallback to midpoint
        peak_idx = n // 2

    left_mask = mask[:peak_idx]
    right_mask = mask[peak_idx + 1 :]

    # Find largest contiguous True segments on each side
    left_seg = _largest_true_segment(left_mask)
    right_seg = _largest_true_segment(right_mask)

    # If no segment found on a side, relax threshold locally by increasing
    # the multiplier until we get something (handled by _expand_thresholdUntilSegment)
    if left_seg is None and left_mask.size >= 3:
        # try relaxers specifically on left d2y portion
        left_d2 = d2y[:peak_idx]
        mask_left = _expand_threshold_until_segment(left_d2, lambda t: np.abs(left_d2) < t)
        left_seg = _largest_true_segment(mask_left) if mask_left is not None else None
    if right_seg is None and right_mask.size >= 3:
        right_d2 = d2y[peak_idx + 1 :]
        mask_right = _expand_threshold_until_segment(right_d2, lambda t: np.abs(right_d2) < t)
        right_seg = _largest_true_segment(mask_right) if mask_right is not None else None

    # Translate segments back to global indices
    left_indices = None
    right_indices = None

    # Helper: fit linear regression and return slope, intercept
    def _fit_line(xseg, yseg):
        if len(xseg) < 3:
            return None, None
        if _HAVE_SKLEARN and LinearRegression is not None:
            lr = LinearRegression()
            lr.fit(xseg.reshape(-1, 1), yseg)
            slope = float(lr.coef_[0])
            intercept = float(lr.intercept_)
            return slope, intercept
        else:
            # Fallback to numpy.polyfit (linear)
            p = np.polyfit(xseg, yseg, 1)
            slope = float(p[0])
            intercept = float(p[1])
            return slope, intercept

    slope_left = None
    slope_right = None

    if left_seg is not None:
        i0, i1 = left_seg
        # indices in original arrays are the same (left segment ends at peak_idx-1)
        left_indices = (int(i0), int(i1))
        xseg = x[left_indices[0] : left_indices[1] + 1]
        yseg = y[left_indices[0] : left_indices[1] + 1]
        slope_left, _ = _fit_line(xseg, yseg)
    else:
        # fallback: choose contiguous region before peak with minimal mean |d2y|
        if peak_idx >= 4:
            L = peak_idx
            best = None
            for w in range(3, max(4, L // 2) + 1):
                for s in range(0, L - w + 1):
                    region = np.abs(d2y[s : s + w])
                    score = np.nanmean(region)
                    if best is None or score < best[0]:
                        best = (score, s, s + w - 1)
            if best is not None:
                left_indices = (int(best[1]), int(best[2]))
                xseg = x[left_indices[0] : left_indices[1] + 1]
                yseg = y[left_indices[0] : left_indices[1] + 1]
                slope_left, _ = _fit_line(xseg, yseg)

    if right_seg is not None:
        # right_seg indexes relative to right_mask, need offset
        i0, i1 = right_seg
        global_i0 = peak_idx + 1 + i0
        global_i1 = peak_idx + 1 + i1
        right_indices = (int(global_i0), int(global_i1))
        xseg = x[right_indices[0] : right_indices[1] + 1]
        yseg = y[right_indices[0] : right_indices[1] + 1]
        slope_right, _ = _fit_line(xseg, yseg)
    else:
        # fallback on right side using minimal mean |d2y| contiguous segment
        R = n - (peak_idx + 1)
        if R >= 4:
            best = None
            for w in range(3, max(4, R // 2) + 1):
                for s in range(peak_idx + 1, n - w + 1):
                    region = np.abs(d2y[s : s + w])
                    score = np.nanmean(region)
                    if best is None or score < best[0]:
                        best = (score, s, s + w - 1)
            if best is not None:
                right_indices = (int(best[1]), int(best[2]))
                xseg = x[right_indices[0] : right_indices[1] + 1]
                yseg = y[right_indices[0] : right_indices[1] + 1]
                slope_right, _ = _fit_line(xseg, yseg)

    # Boundaries in x-space: choose the inner edges (closest to peak)
    left_boundary = None
    right_boundary = None
    if left_indices is not None:
        # inner edge (closest to peak) for left is the maximum x in left segment
        left_boundary = float(x[left_indices[1]])
    if right_indices is not None:
        # inner edge for right is the minimum x in right segment
        right_boundary = float(x[right_indices[0]])

    depletion_width = None
    if left_boundary is not None and right_boundary is not None:
        depletion_width = float(right_boundary - left_boundary)

    if debug:
        return {
            "left_slope": slope_left,
            "right_slope": slope_right,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "depletion_width": depletion_width,
            "left_indices": left_indices,
            "right_indices": right_indices,
            "y_smoothed": y_s,
            "d2y": d2y,
            "mask": mask,
            "peak_idx": peak_idx,
        }

    return {
        "left_slope": slope_left,
        "right_slope": slope_right,
        "left_boundary": left_boundary,
        "right_boundary": right_boundary,
        "depletion_width": depletion_width,
        "left_indices": left_indices,
        "right_indices": right_indices,
    }


if __name__ == "__main__":
    # Quick synthetic test
    import math

    np.random.seed(0)
    n = 201
    x = np.linspace(0, 100, n)
    peak_pos = 50.0
    # build a synthetic log-current: two linear decays plus a Gaussian peak
    left_slope_true = -0.03
    right_slope_true = -0.02
    y = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi <= peak_pos:
            y[i] = left_slope_true * xi + 2.0
        else:
            y[i] = right_slope_true * xi + 2.5
    # add a sharp Gaussian peak around peak_pos
    y += 1.5 * np.exp(-0.5 * ((x - peak_pos) / 3.0) ** 2)
    # add noise
    y += np.random.normal(scale=0.05, size=n)

    res = extract_linear_regions(x, y, debug=True)
    print("Detected:")
    print("left_indices:", res["left_indices"])
    print("right_indices:", res["right_indices"])
    print("left_slope:", res["left_slope"])
    print("right_slope:", res["right_slope"])
    print("left_boundary:", res["left_boundary"])
    print("right_boundary:", res["right_boundary"])
    print("depletion_width:", res["depletion_width"])
    