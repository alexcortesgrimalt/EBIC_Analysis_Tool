
import numpy as np
import cv2
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import pearsonr

class JunctionAnalyzer:

    def __init__(self, pixel_size_m):
        self.pixel_size_m = pixel_size_m  # meters per pixel

    def detect(self, roi, manual_line, roi_current=None, weight_current=10.0, debug=False):
        h, w = roi.shape

        # --- resample manual_line to match ROI width ---
        if len(manual_line) != w:
            t_manual = np.linspace(0.0, 1.0, len(manual_line))
            t_target = np.linspace(0.0, 1.0, w)
            fx = interp1d(t_manual, manual_line[:, 0], kind='linear', fill_value='extrapolate')
            fy = interp1d(t_manual, manual_line[:, 1], kind='linear', fill_value='extrapolate')
            manual_line_rs = np.column_stack([fx(t_target), fy(t_target)])
        else:
            manual_line_rs = np.array(manual_line, dtype=float)

        results = []

        # --- Method : Canny with Bilateral pre-filtering, with post-processing ---
        try:
            filtered_roi = self._apply_preprocessing_filter(roi)
            # If an EBIC/current ROI is provided, preprocess it similarly
            filtered_current = None
            if roi_current is not None:
                try:
                    filtered_current = self._apply_preprocessing_filter(roi_current)
                except Exception:
                    # If preprocessing fails, fall back to raw current ROI
                    filtered_current = roi_current.astype(np.uint8)

            detected_roi_coords = self._detect_junction_canny(filtered_roi, roi_current=filtered_current, weight_current=weight_current, debug=debug)
            if detected_roi_coords.shape[0] != w:
                t_det = np.linspace(0.0, 1.0, detected_roi_coords.shape[0])
                t_new = np.linspace(0.0, 1.0, w)
                fcx = interp1d(t_det, detected_roi_coords[:, 0], kind='linear', fill_value='extrapolate')
                fcy = interp1d(t_det, detected_roi_coords[:, 1], kind='linear', fill_value='extrapolate')
                detected_roi_coords = np.column_stack([fcx(t_new), fcy(t_new)])

            # Apply spline post-processing to the detected points
            postprocessed_roi_coords = self._fit_line_postprocessing(detected_roi_coords)

            detected_image_coords = self._map_detected_to_image_coords(manual_line_rs, postprocessed_roi_coords,
                                                                       roi_height=h)
            metrics = self._compare_with_manual(manual_line_rs, detected_image_coords)
            results.append(("Canny (Filtered, Spline)", detected_image_coords, metrics))
        except Exception as e:
            print(f"[Canny (Filtered, Spline)] failed: {e}")

        return results

    def _apply_preprocessing_filter(self, roi):
        """
        Applies a Bilateral filter to the ROI to smooth noise while preserving edges.
        """
        # Convert the ROI to 8-bit for OpenCV's bilateral filter
        normalized_roi = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Apply the bilateral filter. Parameters (d, sigmaColor, sigmaSpace)
        # may need to be adjusted for optimal results.
        return cv2.bilateralFilter(normalized_roi, d=9, sigmaColor=75, sigmaSpace=9)

    def _apply_spline_postprocessing(self, detected_coords):
        """
        Applies a cubic spline interpolation to the detected coordinates to create a smooth line.
        """
        # Create a parameter `t` for the spline fit
        t = np.linspace(0, 1, len(detected_coords))

        # Fit a cubic spline (k=3) to the x and y coordinates
        tck_x = splrep(t, detected_coords[:, 0], s=0.5)  # s is a smoothing parameter
        tck_y = splrep(t, detected_coords[:, 1], s=0.5)

        # Generate new, denser points from the spline
        t_smooth = np.linspace(0, 1, 1000)
        x_smooth = splev(t_smooth, tck_x)
        y_smooth = splev(t_smooth, tck_y)

        # Return the new, smoothed coordinates
        return np.column_stack([x_smooth, y_smooth])

    def compute_gradient_stats(self, roi, roi_current):
        """
        Compute simple gradient statistics for SEM (roi) and EBIC/current (roi_current).

        Returns dicts for sem_stats and curr_stats containing max, mean, median of absolute gradients,
        and a ratios dict with max_ratio and mean_ratio (curr/sem).
        """
        sem = roi.astype(float)
        cur = roi_current.astype(float)

        # If shapes differ (SEM vs EBIC may have different ROI heights), crop to overlapping region
        h_sem, w_sem = sem.shape
        h_cur, w_cur = cur.shape
        h = min(h_sem, h_cur)
        w = min(w_sem, w_cur)
        if (h, w) != (h_sem, w_sem):
            sem = sem[:h, :w]
        if (h, w) != (h_cur, w_cur):
            cur = cur[:h, :w]

        # per-column absolute gradient magnitudes (along rows)
        sem_grads = np.abs(np.gradient(sem, axis=0))
        cur_grads = np.abs(np.gradient(cur, axis=0))

        # global stats
        sem_stats = {"max": float(np.max(sem_grads)),
                     "mean": float(np.mean(sem_grads)),
                     "median": float(np.median(sem_grads))}
        curr_stats = {"max": float(np.max(cur_grads)),
                      "mean": float(np.mean(cur_grads)),
                      "median": float(np.median(cur_grads))}

        # compute ratio arrays where sem nonzero
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_arr = np.where(sem_grads != 0, cur_grads / (sem_grads + 1e-12), 0.0)

        ratios = {"max_ratio": float(np.nanmax(ratio_arr)),
                  "mean_ratio": float(np.nanmean(ratio_arr))}

        return sem_stats, curr_stats, ratios
    
    # ---------------------------------------------------------------------
    # Canny detection
    # ---------------------------------------------------------------------
    def _detect_junction_canny(self, roi, roi_current=None, weight_current=10.0, debug=False):
        """Detect junction using Canny edge detection with Otsu's adaptive thresholds.

        Combines SEM and optional EBIC/current gradients per-column. When debug=True
        a per-column summary is printed.
        """
        H, W = roi.shape

        # Prepare output
        detected = np.zeros((W, 2), dtype=float)

        # Convert ROI to 8-bit for OpenCV processing
        roi_8bit = roi.astype(np.uint8)

        # Use Otsu's method to find the optimal threshold
        otsu_val, _ = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh = float(otsu_val)
        low_thresh = 0.5 * high_thresh

        # Perform Canny edge detection
        edges = cv2.Canny(roi_8bit, low_thresh, high_thresh)

        # Prepare current/EBIC processing
        if roi_current is not None:
            has_current = True
            roi_current_f = roi_current.astype(float)
            try:
                sem_stats, curr_stats, ratios = self.compute_gradient_stats(roi, roi_current_f)
                print("[JunctionAnalyzer] SEM grad max/mean/median:", sem_stats["max"], sem_stats["mean"], sem_stats["median"])
                print("[JunctionAnalyzer] CUR grad max/mean/median:", curr_stats["max"], curr_stats["mean"], curr_stats["median"])
                print(f"[JunctionAnalyzer] CUR/SEM max ratio: {ratios['max_ratio']:.6g}, mean ratio: {ratios['mean_ratio']:.6g}")
            except Exception:
                pass
        else:
            has_current = False
            roi_current_f = None

        per_column_debug = []

        for col in range(W):
            edge_rows = np.where(edges[:, col] > 0)[0]
            profile = roi[:, col].astype(float)
            grad = np.gradient(profile)

            curr_grad = None
            if has_current:
                curr_profile = roi_current_f[:, col]
                curr_grad = np.gradient(curr_profile)

            row_idx = 0
            eps = 1e-12

            if edge_rows.size > 0:
                if curr_grad is None:
                    sem_edge = np.abs(grad[edge_rows])
                    sem_norm = sem_edge / (np.max(sem_edge) + eps)
                    max_grad_idx = edge_rows[np.argmax(sem_norm)]
                    row_idx = int(max_grad_idx)
                    if debug:
                        per_column_debug.append((col, 'sem_only', int(row_idx)))
                else:
                    sem_scores = np.abs(grad[edge_rows])
                    curr_scores = np.abs(curr_grad[edge_rows])
                    # per-column normalization on the edge pixels
                    sem_norm = sem_scores / (np.max(sem_scores) + eps)
                    curr_norm = curr_scores / (np.max(curr_scores) + eps)

                    # neighborhood candidates
                    neigh = 3
                    candidates = []
                    for r in edge_rows:
                        r0 = max(0, r - neigh)
                        r1 = min(H - 1, r + neigh)
                        candidates.extend(range(r0, r1 + 1))
                    candidates = np.unique(candidates)

                    sem_col = np.abs(grad[candidates])
                    curr_col = np.abs(curr_grad[candidates])
                    sem_col_norm = sem_col / (np.max(sem_col) + eps)
                    curr_col_norm = curr_col / (np.max(curr_col) + eps)
                    comb_col = sem_col_norm + weight_current * curr_col_norm
                    best_idx = int(candidates[np.argmax(comb_col)])
                    row_idx = best_idx
                    if debug:
                        per_column_debug.append((col, 'combined_best', int(row_idx), float(np.max(sem_col_norm)), float(np.max(curr_col_norm))))
            else:
                if curr_grad is None:
                    col_abs = np.abs(grad)
                    col_norm = col_abs / (np.max(col_abs) + eps)
                    row_idx = int(np.argmax(col_norm))
                    if debug:
                        per_column_debug.append((col, 'sem_only_fallback', int(row_idx)))
                else:
                    sem_col = np.abs(grad)
                    curr_col = np.abs(curr_grad)
                    sem_col_norm = sem_col / (np.max(sem_col) + eps)
                    curr_col_norm = curr_col / (np.max(curr_col) + eps)
                    comb = sem_col_norm + weight_current * curr_col_norm
                    row_idx = int(np.argmax(comb))
                    if debug:
                        per_column_debug.append((col, 'combined_fallback', int(row_idx)))

            detected[col] = [col, row_idx]

        if debug:
            print(f"[JunctionAnalyzer DEBUG] per-column entries: {len(per_column_debug)} (showing up to 20)")
            for entry in per_column_debug[:20]:
                print(" ", entry)
            from collections import Counter
            kinds = [e[1] for e in per_column_debug]
            cnt = Counter(kinds)
            print(f"[JunctionAnalyzer DEBUG] counts: {dict(cnt)}")

        return detected


    # ---------------------------------------------------------------------
    def _map_detected_to_image_coords(self, manual_line_rs, detected_roi_coords, roi_height):
        """Map detected ROI coords (col,row) to image coordinates."""
        W = len(manual_line_rs)
        H = roi_height
        half = (H - 1) / 2.0
        image_coords = np.zeros((W, 2), dtype=float)
        tangents = np.zeros_like(manual_line_rs)
        tangents[1:-1] = (manual_line_rs[2:] - manual_line_rs[:-2]) / 2.0
        tangents[0] = manual_line_rs[1] - manual_line_rs[0]
        tangents[-1] = manual_line_rs[-1] - manual_line_rs[-2]
        perp_vectors = np.array([-tangents[:, 1], tangents[:, 0]]).T
        perp_norms = np.linalg.norm(perp_vectors, axis=1)
        perp_units = np.zeros_like(perp_vectors)
        non_zero = perp_norms != 0
        perp_units[non_zero] = perp_vectors[non_zero] / perp_norms[non_zero][:, np.newaxis]
        for i in range(W):
            cx, cy = manual_line_rs[i]
            row_idx = detected_roi_coords[i, 1]
            offset = float(row_idx) - half
            img_x = cx + perp_units[i, 0] * offset
            img_y = cy + perp_units[i, 1] * offset
            image_coords[i] = [img_x, img_y]
        return image_coords

    # ---------------------------------------------------------------------
    def _compare_with_manual(self, manual_line, detected_line):
        """Compute mean, std, max deviation (µm) and correlation."""
        if manual_line.shape != detected_line.shape:
            t_manual = np.linspace(0, 1, len(manual_line))
            t_detect = np.linspace(0, 1, len(detected_line))
            f_x = interp1d(t_detect, detected_line[:, 0], kind='linear', fill_value="extrapolate")
            f_y = interp1d(t_detect, detected_line[:, 1], kind='linear', fill_value="extrapolate")
            detected_line = np.column_stack([f_x(t_manual), f_y(t_manual)])
        diffs = np.linalg.norm(detected_line - manual_line, axis=1) * self.pixel_size_m * 1e6
        mean_dev = np.mean(diffs)
        std_dev = np.std(diffs)
        max_dev = np.max(diffs)
        corr_x, _ = pearsonr(manual_line[:, 0], detected_line[:, 0])
        return mean_dev, std_dev, max_dev

    def print_summary(self, results):
        print("\nSummary of junction detection:")
        print("Method | Mean Dev (µm) | Std Dev (µm) | Max Dev (µm)")
        print("-------|---------------|--------------|--------------")
        for name, _, (mean_dev, std_dev, max_dev, corr) in results:
            print(f"{name:<7}| {mean_dev:>13.3f} | {std_dev:>12.3f} | {max_dev:>12.3f} | {corr:>4.3f}")

    def visualize_results(self, image, manual_line, results):
        """Plot the detection result on the original image."""
        import matplotlib.pyplot as plt
        for name, line_imgcoords, metrics in results:
            # metrics may be a 3-tuple (mean,std,max) or 4-tuple; handle both
            if len(metrics) == 4:
                mean_dev, std_dev, max_dev, r2 = metrics
            else:
                mean_dev, std_dev, max_dev = metrics
                r2 = float('nan')

            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray', origin='upper')
            ax.plot(manual_line[:, 0], manual_line[:, 1], 'r--', label='Manual')
            ax.plot(line_imgcoords[:, 0], line_imgcoords[:, 1], '-', label=name)
            ax.set_title(f"{name}\nMean: {mean_dev:.2f} µm, Std: {std_dev:.2f} µm, Max: {max_dev:.2f} µm, R²: {r2 if not np.isnan(r2) else 'n/a'}")
            ax.legend()
            plt.show()


    def _fit_line_postprocessing(self, detected_coords):
        """
        Fits a straight line (y = ax + b) to the detected coordinates.
        Returns the fitted line coordinates across the full range of x.
        """
        x = detected_coords[:, 0]
        y = detected_coords[:, 1]

        # Fit line y = ax + b
        A = np.vstack([x, np.ones_like(x)]).T
        a, b = np.linalg.lstsq(A, y, rcond=None)[0]

        # Generate fitted line coordinates
        x_fit = np.linspace(x.min(), x.max(), len(x))
        y_fit = a * x_fit + b

        return np.column_stack([x_fit, y_fit])
