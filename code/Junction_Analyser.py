
import numpy as np
import cv2
from scipy.interpolate import interp1d, splrep, splev
from scipy.stats import pearsonr

class JunctionAnalyzer:

    def __init__(self, pixel_size_m):
        self.pixel_size_m = pixel_size_m  # meters per pixel

    def detect(self, roi, manual_line):
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
            detected_roi_coords = self._detect_junction_canny(filtered_roi)
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
    
    # ---------------------------------------------------------------------
    # Canny detection
    # ---------------------------------------------------------------------
    def _detect_junction_canny(self, roi):
        """
        Detect junction using Canny edge detection with Otsu's method for adaptive thresholding.

        This function uses Otsu's method to automatically determine the optimal high and low
        thresholds for the Canny edge detector.
        """
        H, W = roi.shape
        detected = np.zeros((W, 2), dtype=float)

        # Convert ROI to 8-bit for OpenCV processing
        roi_8bit = roi.astype(np.uint8)

        # Use Otsu's method to find the optimal threshold
        otsu_val, _ = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Set Canny thresholds based on Otsu value
        high_thresh = float(otsu_val)
        low_thresh = 0.5 * high_thresh

        # Perform Canny edge detection
        edges = cv2.Canny(roi_8bit, low_thresh, high_thresh)

        # Iterate through each column to find the most significant edge pixel
        for col in range(W):
            edge_rows = np.where(edges[:, col] > 0)[0]
            if len(edge_rows) > 0:
                profile = roi[:, col].astype(float)
                grad = np.gradient(profile)
                max_grad_idx = edge_rows[np.argmax(np.abs(grad[edge_rows]))]
                row_idx = max_grad_idx
            else:
                profile = roi[:, col].astype(float)
                grad = np.gradient(profile)
                row_idx = int(np.argmax(np.abs(grad)))

            detected[col] = [col, row_idx]

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
        for name, line_imgcoords, metrics in results:
            mean_dev, std_dev, max_dev, r2 = metrics
            fig, ax = plt.subplots()
            ax.imshow(image, cmap='gray', origin='upper')
            ax.plot(manual_line[:, 0], manual_line[:, 1], 'r--', label='Manual')
            ax.plot(line_imgcoords[:, 0], line_imgcoords[:, 1], '-', label=name)
            ax.set_title(f"{name}\nMean: {mean_dev:.2f} µm, Std: {std_dev:.2f} µm, Max: {max_dev:.2f} µm, R²: {r2:.3f}")
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
