
import numpy as np
import cv2
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import pearsonr

class JunctionAnalyzer:

    def __init__(self, pixel_size_m):
        self.pixel_size_m = pixel_size_m  # meters per pixel

    def detect(self, roi, manual_line, roi_current=None, weight_current=10.0, debug=False, sweep_weights=None, _sweep_call=False):
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

        # If debug is requested, show the SEM and EBIC/current ROIs before processing
        if debug:
            try:
                import matplotlib.pyplot as plt
                if roi_current is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                    axes[0].imshow(roi, cmap='gray', origin='upper')
                    axes[0].set_title('SEM ROI')
                    axes[1].imshow(roi_current, cmap='viridis', origin='upper')
                    axes[1].set_title('EBIC / Current ROI')
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                    ax.imshow(roi, cmap='gray', origin='upper')
                    ax.set_title('SEM ROI')
                plt.tight_layout()
                plt.show()
            except Exception:
                # Do not fail detection if plotting is unavailable
                pass

        # If user requested a weight sweep and we're not already inside a sweep, invoke it.
        # Use _sweep_call to avoid recursive re-entry (visualize_weight_sweep calls detect()).
        if sweep_weights is not None and not _sweep_call and debug:
            try:
                # Show the sweep (per-weight debug popups) and continue with normal detection.
                self.visualize_weight_sweep(roi, manual_line, roi_current=roi_current, weights=tuple(sweep_weights), per_weight_debug=True, show=True)
            except Exception:
                pass

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
            # If debug is requested, show post-processed overlays: filtered ROI, detected ROI points,
            # post-processed spline, manual line and mapped detected image coordinates.
            if debug:
                try:
                    import matplotlib.pyplot as plt

                    # Show filtered ROI with overlays
                    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                    ax.imshow(filtered_roi, cmap='gray', origin='upper')
                    # detected_roi_coords are (col, row) in ROI coords
                    ax.plot(detected_roi_coords[:, 0], detected_roi_coords[:, 1], 'y.', markersize=2, label='detected (ROI)')
                    # postprocessed_roi_coords may be dense (spline)
                    ax.plot(postprocessed_roi_coords[:, 0], postprocessed_roi_coords[:, 1], 'c-', linewidth=1, label='postproc spline')
                    # mapped image coords are in image coordinates (x,y)
                    ax.plot(detected_image_coords[:, 0], detected_image_coords[:, 1], 'r-', linewidth=1, label='mapped detected')
                    # manual_line_rs is in image coords already (resampled manual line)
                    ax.plot(manual_line_rs[:, 0], manual_line_rs[:, 1], 'g--', linewidth=1, label='manual line')
                    # Overlay SEM Canny edges (compute with Otsu on the filtered ROI)
                    try:
                        roi8 = filtered_roi if filtered_roi.dtype == np.uint8 else cv2.normalize(filtered_roi, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                        otsu_r, _ = cv2.threshold(roi8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        edges_sem = cv2.Canny(roi8, 0.5 * float(otsu_r), float(otsu_r))
                        ys_sem, xs_sem = np.where(edges_sem > 0)
                        if ys_sem.size > 0:
                            ax.scatter(xs_sem, ys_sem, s=1, c='magenta', label='SEM Canny')
                    except Exception:
                        pass

                    ax.set_title('Filtered ROI with detected points and post-processed line')
                    ax.legend(loc='best', fontsize='small')
                    plt.tight_layout()
                    plt.show()

                    # If an EBIC/current ROI is available, show it side-by-side with its filtered version (if any)
                    if roi_current is not None:
                        try:
                            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
                            axes2[0].imshow(roi_current, cmap='viridis', origin='upper')
                            axes2[0].set_title('Raw EBIC / Current ROI')
                            # Overlay EBIC Canny on raw EBIC image
                            try:
                                roi_curr8_raw = roi_current if roi_current.dtype == np.uint8 else cv2.normalize(roi_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                otsu_cr, _ = cv2.threshold(roi_curr8_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                edges_curr_raw = cv2.Canny(roi_curr8_raw, 0.5 * float(otsu_cr), float(otsu_cr))
                                ys_c, xs_c = np.where(edges_curr_raw > 0)
                                if ys_c.size > 0:
                                    axes2[0].scatter(xs_c, ys_c, s=1, c='magenta', label='EBIC Canny')
                            except Exception:
                                pass

                            if filtered_current is not None:
                                axes2[1].imshow(filtered_current, cmap='viridis', origin='upper')
                                axes2[1].set_title('Filtered EBIC / Current ROI')
                                # Overlay EBIC Canny on filtered EBIC
                                try:
                                    roi_curr8 = filtered_current if filtered_current.dtype == np.uint8 else cv2.normalize(filtered_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                                    otsu_cc, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                    edges_curr = cv2.Canny(roi_curr8, 0.5 * float(otsu_cc), float(otsu_cc))
                                    ys2, xs2 = np.where(edges_curr > 0)
                                    if ys2.size > 0:
                                        axes2[1].scatter(xs2, ys2, s=1, c='magenta')
                                except Exception:
                                    pass

                            plt.tight_layout()
                            plt.show()
                        except Exception:
                            pass
                except Exception:
                    # plotting errors should not break the detection flow
                    pass

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
    def _detect_junction_canny(self, roi, roi_current=None, weight_current=10.0, debug=True):
        """Detect junction using Canny edge detection with Otsu's adaptive thresholds.

        Combines SEM and optional EBIC/current gradients per-column. When debug=True
        a per-column summary is printed.
        """
        H, W = roi.shape

        # Prepare output
        detected = np.zeros((W, 2), dtype=float)

        # Convert ROI to 8-bit for OpenCV processing
        roi_8bit = roi.astype(np.uint8)

        # Use Otsu's method to find the optimal threshold for SEM
        otsu_val, _ = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_thresh = float(otsu_val)
        low_thresh = 0.5 * high_thresh

        # Perform Canny edge detection on SEM
        edges = cv2.Canny(roi_8bit, low_thresh, high_thresh)

        # If EBIC/current ROI is provided, also compute Canny on it and combine edge pixels
        curr_edges = None
        if roi_current is not None:
            try:
                roi_curr_8 = roi_current.astype(np.uint8)
                otsu_cur, _ = cv2.threshold(roi_curr_8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                high_cur = float(otsu_cur)
                low_cur = 0.5 * high_cur
                curr_edges = cv2.Canny(roi_curr_8, low_cur, high_cur)
                if debug:
                    print(f"[JunctionAnalyzer] EBIC Canny thresholds: low={low_cur:.2f}, high={high_cur:.2f}")
            except Exception:
                curr_edges = None

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
            # start from SEM edge pixels
            edge_rows_sem = np.where(edges[:, col] > 0)[0]
            # include EBIC/current edge pixels if available
            if curr_edges is not None:
                edge_rows_curr = np.where(curr_edges[:, col] > 0)[0]
            else:
                edge_rows_curr = np.array([], dtype=int)

            # union of SEM and EBIC edge rows
            if edge_rows_sem.size == 0 and edge_rows_curr.size == 0:
                edge_rows = np.array([], dtype=int)
            else:
                edge_rows = np.unique(np.concatenate([edge_rows_sem, edge_rows_curr]))
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


    def visualize_weight_sweep(self, roi, manual_line, roi_current=None, weights=(0, 10, 100, 1000), figsize=(10, 6), save_path=None, show=True, per_weight_debug=True, save_per_weight_dir=None, separate_edges_plot=True):
        """Run detection for multiple EBIC weightings and plot all detected lines on one image.

        Parameters
        - roi: 2D SEM ROI (array-like)
        - manual_line: Nx2 array of manual coordinates (image coords)
        - roi_current: optional EBIC/current ROI (same shape as roi or compatible)
        - weights: iterable of weight_current values to test
        - figsize: figure size for plotting
        - save_path: optional path to save the figure instead of only showing
        - show: whether to call plt.show()

        Returns a list of (weight, detected_coords, metrics) tuples (detected_coords may be None on failure).
        """
        import matplotlib.pyplot as plt

        results = []
        for w in weights:
            try:
                # run detection without GUI popups (we'll save per-weight images explicitly if requested)
                print(f"[visualize_weight_sweep] running detect with weight={w}")
                # honor per_weight_debug: if True, detect will show its debug plots (plt.show())
                res = self.detect(roi, manual_line, roi_current=roi_current, weight_current=w, debug=per_weight_debug)
                if res and len(res) > 0:
                    # take the first method result
                    _, coords, metrics = res[0]
                    results.append((w, coords, metrics))
                else:
                    results.append((w, None, None))
            except Exception:
                print(f"[visualize_weight_sweep] detection failed for weight={w}")
                results.append((w, None, None))

            # If requested, save a per-weight diagnostic image (non-blocking)
            if save_per_weight_dir is not None:
                try:
                    import os
                    import matplotlib.pyplot as plt
                    os.makedirs(save_per_weight_dir, exist_ok=True)
                    fname = os.path.join(save_per_weight_dir, f"sweep_weight_{w}.png")
                    figw, axw = plt.subplots(1, 1, figsize=(8, 6))
                    axw.imshow(roi, cmap='gray', origin='upper')
                    # manual line
                    try:
                        ml = np.array(manual_line, dtype=float)
                        axw.plot(ml[:, 0], ml[:, 1], 'k--', linewidth=1.0, label='manual')
                    except Exception:
                        pass
                    # detected line
                    if res and len(res) > 0 and res[0][1] is not None:
                        try:
                            axw.plot(res[0][1][:, 0], res[0][1][:, 1], 'r-', linewidth=1.0, label=f'weight={w}')
                        except Exception:
                            pass
                    # EBIC edges
                    if roi_current is not None:
                        try:
                            roi_curr8 = roi_current if roi_current.dtype == np.uint8 else cv2.normalize(roi_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            otsu_c, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                            edges_curr = cv2.Canny(roi_curr8, 0.5 * float(otsu_c), float(otsu_c))
                            ys_c, xs_c = np.where(edges_curr > 0)
                            if ys_c.size > 0:
                                axw.scatter(xs_c, ys_c, s=2, c='yellow', alpha=0.9, label='EBIC Canny')
                        except Exception:
                            pass
                    axw.set_title(f"Weight {w}")
                    axw.legend(fontsize='small')
                    plt.tight_layout()
                    figw.savefig(fname, dpi=200)
                    plt.close(figw)
                    print(f"[visualize_weight_sweep] saved per-weight image: {fname}")
                except Exception as e:
                    print(f"[visualize_weight_sweep] failed to save per-weight image for weight={w}: {e}")

        # Plot all results on one image
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(roi, cmap='gray', origin='upper')
        # manual line (plot in black dashed)
        try:
            ml = np.array(manual_line, dtype=float)
            ax.plot(ml[:, 0], ml[:, 1], 'k--', linewidth=1.0, label='manual')
        except Exception:
            pass

        colors = ['r', 'g', 'b', 'm', 'c', 'y']
        for i, (w, coords, metrics) in enumerate(results):
            if coords is None:
                print(f"[visualize_weight_sweep] no detected line for weight={w}")
                continue
            col = colors[i % len(colors)]
            try:
                # plot line and small semi-transparent markers so overlapping lines are visible
                ax.plot(coords[:, 0], coords[:, 1], color=col, linewidth=1.2, alpha=0.9, label=f'weight={w}')
                ax.scatter(coords[:, 0], coords[:, 1], s=4, c=col, alpha=0.6)
                print(f"[visualize_weight_sweep] weight={w} detected {coords.shape[0]} points")
            except Exception:
                print(f"[visualize_weight_sweep] plotting failed for weight={w}")
                continue

        # If EBIC/current ROI is present, overlay its Canny edges on the combined plot for reference
        if roi_current is not None:
            try:
                roi_curr8 = roi_current if roi_current.dtype == np.uint8 else cv2.normalize(roi_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                otsu_c, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                edges_curr = cv2.Canny(roi_curr8, 0.5 * float(otsu_c), float(otsu_c))
                ys_c, xs_c = np.where(edges_curr > 0)
                if ys_c.size > 0:
                    ax.scatter(xs_c, ys_c, s=2, c='yellow', alpha=0.9, label='EBIC Canny')
                    print(f"[visualize_weight_sweep] EBIC Canny edges count: {ys_c.size}")
            except Exception:
                pass

        # Optionally create a separate plot that shows only the EBIC/Current edges (and detected lines)
        if roi_current is not None and separate_edges_plot:
            try:
                fig_e, ax_e = plt.subplots(1, 1, figsize=(6, 6))
                roi_curr8 = roi_current if roi_current.dtype == np.uint8 else cv2.normalize(roi_current, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                ax_e.imshow(roi_curr8, cmap='viridis', origin='upper')
                otsu_ce, _ = cv2.threshold(roi_curr8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                edges_ce = cv2.Canny(roi_curr8, 0.5 * float(otsu_ce), float(otsu_ce))
                ys_ce, xs_ce = np.where(edges_ce > 0)
                if ys_ce.size > 0:
                    ax_e.scatter(xs_ce, ys_ce, s=2, c='yellow', alpha=0.9, label='EBIC Canny')

                # overlay detected lines on the separate edges plot for quick comparison
                for i, (w, coords, metrics) in enumerate(results):
                    if coords is None:
                        continue
                    try:
                        ax_e.plot(coords[:, 0], coords[:, 1], '-', linewidth=0.9, alpha=0.7, label=f'weight={w}')
                    except Exception:
                        pass

                ax_e.set_title('EBIC Canny edges (separate) and detected lines')
                ax_e.legend(fontsize='small')
                plt.tight_layout()
                # if a save_path for the combined figure was given, save an edges-only sibling file
                if save_path:
                    try:
                        import os
                        base, ext = os.path.splitext(save_path)
                        fname_edges = base + '_edges' + (ext if ext else '.png')
                        fig_e.savefig(fname_edges, dpi=200)
                        print(f"[visualize_weight_sweep] saved edges-only image: {fname_edges}")
                    except Exception:
                        pass
                if show:
                    plt.show()
                else:
                    plt.close(fig_e)
            except Exception:
                pass

        ax.set_title(f"Detected lines (weights: {', '.join(str(x) for x in weights)})")
        ax.legend(fontsize='small', loc='best')
        plt.tight_layout()
        if save_path:
            try:
                fig.savefig(save_path, dpi=200)
            except Exception:
                pass
        if show:
            plt.show()
        else:
            plt.close(fig)

        return results


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
