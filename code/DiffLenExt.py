import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np  
from scipy.optimize import curve_fit


def _maybe_close(fig):
    """
    Close a matplotlib figure only when running in a non-interactive / headless
    backend (e.g. 'Agg') or when matplotlib is not in interactive mode.
    This prevents saved figures from being immediately closed when the user
    runs the GUI interactively and expects windows to remain open.
    """
    try:
        backend = mpl.get_backend().lower()
        # Only auto-close when backend is clearly headless (Agg).
        # Do NOT close just because plt.isinteractive() is False — that
        # causes figures to be closed immediately in scripts where the
        # user expects interactive windows to remain open.
        if backend.startswith('agg'):
            plt.close(fig)
    except Exception:
        # If anything goes wrong, attempt a safe close and swallow errors
        try:
            plt.close(fig)
        except Exception:
            pass


class DiffusionLengthExtractor:
    def __init__(self, pixel_size=1.0, smoothing_sigma=None):
        self.pixel_size = pixel_size
        self.smoothing_sigma = smoothing_sigma
        self.profiles = []
        self.results = []
        self.inv_lambdas = []
        self.central_inv_lambdas = []

    def load_profiles(self, profiles):
        self.profiles = profiles

    @staticmethod
    def exp_rising(x, A, lam, y0, x0):
        return A * np.exp(lam * (x - x0)) + y0

    @staticmethod
    def exp_falling(x, A, lam, y0, x0):
        return A * np.exp(-lam * (x - x0)) + y0

    @staticmethod
    def exp_model(x, A, lam, y0, x0):
        return A * np.exp(-lam * (x - x0)) + y0

    # --- Helper function for SNR-based truncation ---
    def _find_snr_end_index(self, y_vals, snr_threshold=3.0, min_points=10):
        """
        Finds the index to truncate the data based on a Signal-to-Noise Ratio (SNR) threshold.
        """
        if len(y_vals) < min_points:
            return len(y_vals)

        noise_region = y_vals[-min_points:]
        y0_est = np.median(noise_region)
        sigma_est = np.std(noise_region)
        cutoff_value = y0_est + snr_threshold * sigma_est

        found_end_idx = 0
        for i in range(len(y_vals) - 1, min_points - 1, -1):
            if y_vals[i] > cutoff_value:
                found_end_idx = i + 1
                break

        if found_end_idx < min_points:
            found_end_idx = min_points

        return found_end_idx

    # --- Helper function for curve fitting ---
    def _fit_falling(self, x, y, shift, side="Right", profile_id=0):
        """
        Internal helper function to perform the actual curve fitting with more robust
        initial parameter guesses.
        """
        if len(x) < 3:
            return None

        # Initial parameter guesses
        # y0_init should be the minimum value of the truncated data, a more stable guess
        if y[-1] > 0:
            y0_init = y[-1]
        else:
            y0_init = 0

        # A_init should be the difference between the maximum and minimum, or peak height
        A_init = y[0] - y0_init

        # lam_init can be guessed from the initial decay rate
        lam_init = 1.0 / (np.max(x) - np.min(x))

        # x0_init should be the x-value corresponding to the peak (or start of the fit)
        x0_init = x[0]

        try:
            popt, _ = curve_fit(
                self.exp_model, x, y,
                p0=[A_init, lam_init, y0_init, x0_init],
                bounds=([0, 0, -np.inf, -np.inf],
                        [np.inf, np.inf, np.inf, np.inf]),
                maxfev=20000
            )
            fit_curve = self.exp_model(x, *popt)
            # x values passed to the fitter are in micrometers (dist_um). popt[1] (lam)
            # therefore has units 1/µm. Compute inv_lambda in µm and quantize to
            # the pixel grid given by pixel_size (stored in meters).
            pixel_size_um = self.pixel_size * 1e6
            inv_lambda_um = 1.0 / popt[1]
            # Quantize to pixel grid (in µm) and enforce at least one pixel
            inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
            r2 = self._calculate_r2(y, fit_curve)

            if shift == 0:
                self.central_inv_lambdas.append(inv_lambda)
            else:
                self.inv_lambdas.append(inv_lambda)

            return {
                'side': f'{side} (shift {shift})',
                'shift': shift,
                'x_vals': x,
                'y_vals': y,
                'fit_curve': fit_curve,
                'parameters': popt,
                'inv_lambda': inv_lambda,
                'r2': r2
            }
        except Exception as e:
            print(f"{side} fit failed (shift {shift}) for profile {profile_id}: {e}")
            return None

    @staticmethod
    def _calculate_r2(y, y_fit):
        """Calculates R-squared value."""
        residuals = y - y_fit
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def _detect_plateau_in_derivative(self, x_vals, deriv, min_plateau_length=10, 
                                      derivative_threshold=0.3, absolute_threshold=0.05,
                                      search_from_end=False, use_robust=True):
        """
        Detect plateau regions in the derivative where d(ln(I))/dx is approximately constant.
        A plateau indicates a linear region in log-space.
        
        Parameters
        ----------
        x_vals : array
            Distance values (µm)
        deriv : array
            Derivative d(ln(I))/dx
        min_plateau_length : int
            Minimum number of points to consider a plateau (default: 5)
        derivative_threshold : float
            Maximum allowed relative variation in derivative (as fraction of LOCAL median, default: 0.3)
        absolute_threshold : float
            Maximum allowed absolute variation in derivative (1/µm, default: 0.05)
        search_from_end : bool
            If True, search from the tail inward (away from junction)
            If False, search from junction outward
        use_robust : bool
            If True, use LOCAL statistics (median and MAD) instead of global
            This makes detection more robust to noise and outliers
        
        Returns
        -------
        plateau_start : int or None
            Start index of the detected plateau
        plateau_end : int or None
            End index of the detected plateau
        plateau_mean_deriv : float or None
            Mean derivative value in the plateau
        """
        if len(deriv) < min_plateau_length:
            return None, None, None
        
        # Remove NaN values
        valid_mask = ~np.isnan(deriv)
        if np.sum(valid_mask) < min_plateau_length:
            return None, None, None
        
        if not use_robust:
            # OLD METHOD: Use global median (sensitive to noise)
            deriv_abs = np.abs(deriv[valid_mask])
            median_deriv = np.median(deriv_abs)
            
            if median_deriv == 0:
                threshold_relative = 0.1
            else:
                threshold_relative = derivative_threshold * median_deriv
            
            threshold = max(threshold_relative, absolute_threshold)
        else:
            # NEW ROBUST METHOD: Use local statistics (not used for threshold, 
            # but for robust plateau detection - see below)
            threshold = absolute_threshold  # Will use local criteria in the loop
        
        # Use less smoothing to better match visible derivative
        from scipy.ndimage import uniform_filter1d
        window = min(3, len(deriv) // 5)
        if window < 3:
            window = 3
        deriv_smooth = uniform_filter1d(deriv.astype(float), size=window, mode='nearest')
        
        # Search for plateau
        n = len(deriv_smooth)
        best_plateau_start = None
        best_plateau_end = None
        best_plateau_length = 0
        best_plateau_std = np.inf
        
        if search_from_end:
            # Search from tail inward (typical for exponential decay tails)
            search_range = range(n - min_plateau_length, -1, -1)
        else:
            # Search from start outward (from junction)
            search_range = range(0, n - min_plateau_length + 1)
        
        for start in search_range:
            for length in range(min_plateau_length, min(n - start + 1, 50)):  # max 50 points
                end = start + length
                segment = deriv_smooth[start:end]
                
                if use_robust:
                    # ROBUST METHOD: Use local median and MAD (Median Absolute Deviation)
                    # MAD is much more robust to outliers than standard deviation
                    segment_median = np.median(segment)
                    segment_mad = np.median(np.abs(segment - segment_median))
                    
                    # Convert MAD to equivalent std (MAD * 1.4826 ≈ std for normal distribution)
                    segment_robust_std = segment_mad * 1.4826
                    
                    # Use interquartile range for spread (more robust than range)
                    q25, q75 = np.percentile(segment, [25, 75])
                    segment_iqr = q75 - q25
                    
                    # LOCAL threshold based on the segment itself (not global median)
                    # This adapts to the local derivative level
                    local_threshold_rel = derivative_threshold * np.abs(segment_median) if segment_median != 0 else absolute_threshold
                    local_threshold = max(local_threshold_rel, absolute_threshold)
                    
                    # Plateau criteria using robust statistics
                    is_plateau = (segment_robust_std < local_threshold * 0.7 and 
                                 segment_iqr < local_threshold * 1.2)
                else:
                    # OLD METHOD: Use mean, std, and range (sensitive to outliers)
                    segment_mean = np.mean(segment)
                    segment_std = np.std(segment)
                    segment_range = np.max(segment) - np.min(segment)
                    
                    # Plateau criteria: tighter constraints for better derivative matching
                    is_plateau = (segment_std < threshold * 0.8 and 
                                 segment_range < 1.5 * threshold)
                
                if is_plateau:
                    # Prefer plateau closest to zero (junction) with sufficient length
                    # Accept if: no plateau yet, or closer to zero with reasonable quality
                    if best_plateau_start is None:
                        # First valid plateau found
                        best_plateau_start = start
                        best_plateau_end = end
                        best_plateau_length = length
                        best_plateau_std = segment_robust_std if use_robust else segment_std
                    elif start < best_plateau_start and length >= min_plateau_length:
                        # Found a plateau closer to zero with acceptable length
                        best_plateau_start = start
                        best_plateau_end = end
                        best_plateau_length = length
                        best_plateau_std = segment_robust_std if use_robust else segment_std
        
        if best_plateau_start is not None:
            plateau_segment = deriv_smooth[best_plateau_start:best_plateau_end]
            plateau_mean_deriv = np.mean(plateau_segment)
            return best_plateau_start, best_plateau_end, plateau_mean_deriv
        
        return None, None, None

    # def fit_profile_sides(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
        #                   plot_left=True, plot_right=True):
        # """
        # Fit falling exponentials on both sides:
        # 1. Find the correct starting point by shifting until 1/λ stabilizes.
        # 2. With the starting point fixed, progressively truncate the tail
        #    and fit each case.
        # 3. Visualize only the tail truncation results.
        # """
        # x_vals = np.asarray(x_vals)
        # y_vals = np.asarray(y_vals)

        # base_idx = int(np.argmax(y_vals))
        # if intersection_idx is None:
        #     intersection_idx = base_idx

        # results = []
        # tolerance = self.pixel_size * 1e6

        # # --- LEFT side: find start index ---
        # prev_left_val, left_stable, best_left_idx = None, False, None
        # for shift in range(0, -16, -1):  # try shifts leftwards
        #     if left_stable:
        #         break
        #     start_idx_left = max(0, (intersection_idx if intersection_idx <= base_idx else base_idx) + shift)
        #     left_x_raw = x_vals[:start_idx_left + 1]
        #     y_left_raw = y_vals[:start_idx_left + 1]

        #     y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
        #     y_left_flipped = y_left_filtered[::-1]
        #     x_left_flipped = np.abs(left_x_raw)[::-1]

        #     cut_idx = self._find_snr_end_index(y_left_flipped)
        #     left_y_truncated = y_left_flipped[:cut_idx]
        #     left_x_truncated = x_left_flipped[:cut_idx]

        #     if len(left_x_truncated) > 2:
        #         left_fit = self._fit_falling(
        #             x=left_x_truncated, y=left_y_truncated,
        #             shift=shift, side="Left", profile_id=profile_id
        #         )
        #         if left_fit:
        #             curr_val = left_fit.get('inv_lambda', None)
        #             if curr_val is not None:
        #                 if prev_left_val is not None and abs(curr_val - prev_left_val) <= tolerance:
        #                     left_stable, best_left_idx = True, start_idx_left
        #                 prev_left_val = curr_val

        # # --- RIGHT side: find start index ---
        # prev_right_val, right_stable, best_right_idx = None, False, None
        # for shift in range(0, 16):  # try shifts rightwards
        #     if right_stable:
        #         break
        #     start_idx_right = min(len(x_vals) - 1,
        #                           (intersection_idx if intersection_idx >= base_idx else base_idx) + shift)
        #     right_x_raw = x_vals[start_idx_right:]
        #     y_right_raw = y_vals[start_idx_right:]

        #     y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)
        #     cut_idx = self._find_snr_end_index(y_right_filtered)
        #     right_y_truncated = y_right_filtered[:cut_idx]
        #     right_x_truncated = right_x_raw[:cut_idx]

        #     if len(right_x_truncated) > 2:
        #         right_fit = self._fit_falling(
        #             x=right_x_truncated, y=right_y_truncated,
        #             shift=shift, side="Right", profile_id=profile_id
        #         )
        #         if right_fit:
        #             curr_val = right_fit.get('inv_lambda', None)
        #             if curr_val is not None:
        #                 if prev_right_val is not None and abs(curr_val - prev_right_val) <= tolerance:
        #                     right_stable, best_right_idx = True, start_idx_right
        #                 prev_right_val = curr_val

        # # --- Tail truncations with fixed start indices ---
        # if best_left_idx is not None:
        #     left_x_raw = x_vals[:best_left_idx + 1]
        #     y_left_raw = y_vals[:best_left_idx + 1]
        #     y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
        #     y_left_flipped = y_left_filtered[::-1]
        #     x_left_flipped = np.abs(left_x_raw)[::-1]

        #     cut_idx = self._find_snr_end_index(y_left_flipped)
        #     left_y_truncated = y_left_flipped[:cut_idx]
        #     left_x_truncated = x_left_flipped[:cut_idx]

        #     results.extend(self._truncate_tail_and_fit(
        #         left_x_truncated, left_y_truncated,
        #         shift=0, side="Left", profile_id=profile_id
        #     ))

        # if best_right_idx is not None:
        #     right_x_raw = x_vals[best_right_idx:]
        #     y_right_raw = y_vals[best_right_idx:]
        #     y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)

        #     cut_idx = self._find_snr_end_index(y_right_filtered)
        #     right_y_truncated = y_right_filtered[:cut_idx]
        #     right_x_truncated = right_x_raw[:cut_idx]

        #     results.extend(self._truncate_tail_and_fit(
        #         right_x_truncated, right_y_truncated,
        #         shift=0, side="Right", profile_id=profile_id
        #     ))

        # # --- Visualization: only tail truncations ---
        # if plot_left and best_left_idx is not None:
        #     self._plot_tailcut_fits(results, x_vals, y_vals, 'Left', base_idx, profile_id)
        # if plot_right and best_right_idx is not None:
        #     self._plot_tailcut_fits(results, x_vals, y_vals, 'Right', base_idx, profile_id)

        # return results
    
    def fit_profile_sides(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                          plot_left=False, plot_right=False, prevent_cross_peak=False):
        """
        Fit falling exponentials on both sides:
        1. Find the correct starting point by shifting until 1/λ stabilizes.
        2. With the starting point fixed, progressively truncate the tail
           and fit each case.
        3. Visualize only the tail truncation results.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)
        # preserve original experimental data for plotting overlays
        y_raw = np.array(y_vals, dtype=float)

        # Do NOT apply low-pass smoothing before fitting: keep the
        # experimental data raw for all linear-on-log fitting per user request.
        # Preserve the original experimental data for plotting overlays.
        y_vals = np.asarray(y_vals)

        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx

        # Reference zero for fitting/selection/clipping: prefer the detected
        # `intersection_idx` (junction) when provided; otherwise use the peak.
        ref = float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx])
        # Reference zero for fitting/selection/clipping: prefer the detected
        # `intersection_idx` (junction) when provided; otherwise use the peak.
        ref = float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx])

        # Reference zero for fitting/selection/clipping: prefer the detected
        # `intersection_idx` (junction) when provided; otherwise use the peak.
        ref = float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx])

        # Reference zero for fitting/selection/clipping: prefer the detected
        # `intersection_idx` (junction) when provided; otherwise use the peak.
        ref = float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx])

        results = []
        tolerance = self.pixel_size * 1e6

        prev_left_val, left_stable, best_left_idx = None, False, None
        left_candidates = []
        for shift in np.arange(0, -11, -1):
            print(f"Trying left shift: {shift}")
            if left_stable:
                break

            start_idx_left = max(0, base_idx + shift)
            left_x_raw = x_vals[:start_idx_left + 1]
            y_left_raw = y_vals[:start_idx_left + 1]

            # Use raw data for fitting (no pre-filtering)
            y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]

            try:
                start_pos_left = float(left_x_raw[-1])
                x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
            except Exception:
                x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            if len(left_x_truncated) > 2:
                left_fit = self._fit_falling(
                    x=left_x_truncated, y=left_y_truncated,
                    shift=shift, side="Left", profile_id=profile_id
                )
                if left_fit:
                    curr_val = left_fit.get('inv_lambda', None)
                    try:
                        left_candidates.append((start_idx_left, left_fit))
                    except Exception:
                        pass
                    if curr_val is not None:
                        if (prev_left_val is not None
                        and abs(curr_val - prev_left_val) <= tolerance):
                            left_stable, best_left_idx = False, start_idx_left
                        prev_left_val = curr_val



        prev_right_val, right_stable, best_right_idx = None, False, None
        right_candidates = []

        for shift in range(0, 11, 1):
            if right_stable:
                break
            start_idx_right = min(len(x_vals) - 1, base_idx + shift)
            right_x_raw = x_vals[start_idx_right:]
            y_right_raw = y_vals[start_idx_right:]

            # Use raw data for fitting (no pre-filtering)
            right_y_raw_arr = np.array(y_right_raw, dtype=float)
            cut_idx = self._find_snr_end_index(right_y_raw_arr)
            right_y_truncated = right_y_raw_arr[:cut_idx]
            try:
                start_pos_right = float(right_x_raw[0])
                right_x_local = np.array(right_x_raw, dtype=float) - start_pos_right
                right_x_truncated = right_x_local[:cut_idx]
            except Exception:
                right_x_truncated = right_x_raw[:cut_idx]

            if len(right_x_truncated) > 2:
                right_fit = self._fit_falling(
                    x=right_x_truncated, y=right_y_truncated,
                    shift=shift, side="Right", profile_id=profile_id
                )
                if right_fit:
                    curr_val = right_fit.get('inv_lambda', None)
                    try:
                        right_candidates.append((start_idx_right, right_fit))
                    except Exception:
                        pass
                    if curr_val is not None:
                        if prev_right_val is not None and abs(curr_val - prev_right_val) <= tolerance:
                            right_stable, best_right_idx = False, start_idx_right
                        prev_right_val = curr_val

        # --- Choose best candidates by R² (if any) and then do tail truncations ---
        best_left_idx = None
        if left_candidates:
            # pick candidate with highest R²
            best_left_idx = max(left_candidates, key=lambda t: t[1].get('r2', -np.inf))[0]

        best_right_idx = None
        if right_candidates:
            best_right_idx = max(right_candidates, key=lambda t: t[1].get('r2', -np.inf))[0]

        # --- Choose best candidates by R² (and enforce side-start sign) ---
        best_left_idx = None
        if left_candidates:
            # prefer candidates whose start position is on the left side (<= ref)
            left_filtered = [(idx, f) for idx, f in left_candidates if float(x_vals[idx]) - ref <= 0]
            if left_filtered:
                best_left_idx = max(left_filtered, key=lambda t: t[1].get('r2', -np.inf))[0]
            else:
                # fallback: pick best overall but ensure it doesn't lie right of peak
                cand = max(left_candidates, key=lambda t: t[1].get('r2', -np.inf))
                if float(x_vals[cand[0]]) - ref <= 0:
                    best_left_idx = cand[0]

        best_right_idx = None
        if right_candidates:
            right_filtered = [(idx, f) for idx, f in right_candidates if float(x_vals[idx]) - ref >= 0]
            if right_filtered:
                best_right_idx = max(right_filtered, key=lambda t: t[1].get('r2', -np.inf))[0]
            else:
                cand = max(right_candidates, key=lambda t: t[1].get('r2', -np.inf))
                if float(x_vals[cand[0]]) - ref >= 0:
                    best_right_idx = cand[0]

        # --- Tail truncations with fixed start indices ---
        if best_left_idx is not None:
            left_x_raw = x_vals[:best_left_idx + 1]
            y_left_raw = y_vals[:best_left_idx + 1]
            # Use raw data for fitting (no pre-filtering)
            y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
            try:
                start_pos_left = float(left_x_raw[-1])
                x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
            except Exception:
                x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            results.extend(self._truncate_tail_and_fit(
                left_x_truncated, left_y_truncated,
                shift=0, side="Left", profile_id=profile_id,
                # pass original-profile x coordinates corresponding to the
                # truncated flipped array: left_x_raw[::-1] maps the flipped
                # distances back to profile positions starting at the start_idx
                global_x=left_x_raw[::-1][:cut_idx],
                prevent_cross_peak=prevent_cross_peak,
                ref=ref
            ))

        if best_right_idx is not None:
            right_x_raw = x_vals[best_right_idx:]
            y_right_raw = y_vals[best_right_idx:]
            # Use raw data for fitting (no pre-filtering)
            right_y_raw_arr = np.array(y_right_raw, dtype=float)
            cut_idx = self._find_snr_end_index(right_y_raw_arr)
            right_y_truncated = right_y_raw_arr[:cut_idx]
            try:
                start_pos_right = float(right_x_raw[0])
                right_x_local = np.array(right_x_raw, dtype=float) - start_pos_right
                right_x_truncated = right_x_local[:cut_idx]
            except Exception:
                right_x_truncated = right_x_raw[:cut_idx]

            # Pass the ORIGINAL profile coordinates as global_x so overlays
            # and depletion extraction map back to the profile axis correctly.
            results.extend(self._truncate_tail_and_fit(
                right_x_truncated, right_y_truncated,
                shift=0, side="Right", profile_id=profile_id,
                global_x=right_x_raw[:cut_idx],
                prevent_cross_peak=prevent_cross_peak,
                ref=ref
            ))

        # --- Visualization: only tail truncations ---
        if plot_left and best_left_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_raw, 'Left', base_idx, profile_id)
        if plot_right and best_right_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_raw, 'Right', base_idx, profile_id)

        return results

    def _plot_shifted_fits(self, results, x_vals, y_vals, side_to_plot, base_idx, profile_id):
        """Helper to create the overlay plots."""
        plt.figure(figsize=(8, 5))
        plt.title(f"Profile {profile_id} - {side_to_plot} shifted fits")
        # Center x-axis at the profile peak (x_vals[base_idx]) so 0 is the peak
        ref = float(x_vals[base_idx])
        # Determine the base data for the plot (and shift to peak-centered coords)
        if side_to_plot == 'Left':
            x_base, y_base = x_vals[:base_idx + 1] - ref, y_vals[:base_idx + 1]
        else:
            x_base, y_base = x_vals[base_idx:] - ref, y_vals[base_idx:]

        plt.plot(x_base, y_base, color='0.6', alpha=0.7, label=f'raw {side_to_plot.lower()}')
        y_filtered_base = self.apply_low_pass_filter(y_base, visualize=False)
        plt.plot(x_base, y_filtered_base, 'k-', lw=1.5, label=f'filtered {side_to_plot.lower()}')

        for res in results:
            if res['side'].startswith(side_to_plot):
                # prefer global_x_vals (already in profile coords centered where we set them);
                # otherwise map stored x_vals and shift by ref
                x_plot = res.get('global_x_vals', None)
                if x_plot is None:
                    if 'Left' in res['side']:
                        x_plot = -np.array(res['x_vals'], dtype=float) - ref
                    else:
                        x_plot = np.array(res['x_vals'], dtype=float) - ref
                else:
                    x_plot = np.array(x_plot, dtype=float) - ref
                y_plot = np.array(res.get('fit_curve', []), dtype=float)
                try:
                    order = np.argsort(x_plot)
                    x_plot = x_plot[order]
                    y_plot = y_plot[order]
                except Exception:
                    pass
                r2_display = res.get('_r2_linear', res.get('r2', float('nan')))
                plt.plot(x_plot, y_plot, lw=1.2,
                         label=f"shift {res['shift']} (R²={r2_display:.2f})")

        plt.xlabel("x (pixels)")
        plt.ylabel("Signal")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def _extract_depletion_region(self, fit_results, x_vals, base_idx):
        """
        Determine the depletion region from stabilized fit results.
        The left side is flipped during fitting, so we must invert back
        to get the correct starting coordinate.
        """
        # Choose the best Left and Right fits by R² (fit quality). This avoids
        # picking the last-fit in the list which may be a low-quality fit.
        left_candidates = [r for r in fit_results if "Left" in r['side'] and r.get('r2') is not None]
        right_candidates = [r for r in fit_results if "Right" in r['side'] and r.get('r2') is not None]

        best_left = max(left_candidates, key=lambda r: r['r2']) if left_candidates else None
        best_right = max(right_candidates, key=lambda r: r['r2']) if right_candidates else None

        # Map fit x-values to the profile axis and pick the inner extremities
        # (closest to the intersection): for the left side the fit x_vals are
        # stored as positive distances from the left side, so we negate them
        # to place them on the negative side of the profile axis. The inner
        # edge (most-right on the left side) is therefore the maximum of the
        # mapped x array. For the right side the inner edge is the minimum.
        if best_left is not None:
            try:
                # prefer global_x_vals if available (already in profile coords)
                if 'global_x_vals' in best_left:
                    left_start = float(np.max(np.array(best_left['global_x_vals'], dtype=float)))
                else:
                    # fall back to mapping stored x_vals (distances) to negative
                    left_x_mapped = -np.array(best_left['x_vals'], dtype=float)
                    left_start = float(np.max(left_x_mapped))
            except Exception:
                try:
                    left_start = float(-best_left['x_vals'][0])
                except Exception:
                    left_start = None
        else:
            left_start = None

        if best_right is not None:
            try:
                if 'global_x_vals' in best_right:
                    right_start = float(np.min(np.array(best_right['global_x_vals'], dtype=float)))
                else:
                    right_x_mapped = np.array(best_right['x_vals'], dtype=float)
                    right_start = float(np.min(right_x_mapped))
            except Exception:
                try:
                    right_start = float(best_right['x_vals'][0])
                except Exception:
                    right_start = None
        else:
            right_start = None

        if left_start is not None and right_start is not None:
            depletion_width = abs(right_start - left_start)
        else:
            depletion_width = None

        return {
            "left_start": left_start,
            "right_start": right_start,
            "depletion_width": depletion_width,
            # include chosen fits so callers can visualize the selected fit curves
            "best_left_fit": best_left,
            "best_right_fit": best_right,
        }

    def fit_all_profiles(self):
        """
        Run iterative head+tail fitting on all profiles.
        Computes depletion widths per profile.
        """
        self.results = []
        self.inv_lambdas = []
        self.central_inv_lambdas = []

        for i, prof in enumerate(self.profiles):
            intersection_idx = prof.get('intersection_idx', None)
            x_vals = np.array(prof['dist_um'])
            y_vals = np.array(prof['current'])

            # use iterative method instead of single-pass
            sides = self.fit_profile_sides_iterative(
                x_vals, y_vals,
                intersection_idx=intersection_idx,
                profile_id=i + 1

            )

            depletion_info = self._extract_depletion_region(
                sides, x_vals, np.argmax(y_vals)
            )

            self.results.append({
                'Profile': i + 1,
                'fit_sides': sides,
                'depletion': depletion_info
            })

    def visualize_depletion_regions(self):
        """
        Quick plot of profiles with vertical markers for depletion edges.
        """
        if not self.results:
            print("No results. Run fit_all_profiles() first.")
            return

        for res in self.results:
            profile_id = res['Profile']
            fit_sides = res['fit_sides']
            depletion = res['depletion']

            if depletion['depletion_width'] is None:
                continue

            profile_entry = self.profiles[profile_id - 1]
            x = np.array(profile_entry.get('dist_um', []))
            y = np.array(profile_entry.get('current', []))
            # attempt to fetch source name (file/stem) if available
            source_name = profile_entry.get('source_name', None)
            # center x axis at profile peak so x=0 corresponds to max(y)
            base_idx = int(np.argmax(y))
            ref = float(x[base_idx])
            x_plot = x - ref

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x_plot, y, 'k-', alpha=0.6, label="EBIC Profile")

            left_start = depletion.get('left_start', None)
            right_start = depletion.get('right_start', None)

            # Shade depletion region if both edges present
            if left_start is not None and right_start is not None:
                # shift edges to peak-centered plotting coords
                ax.axvspan(left_start - ref, right_start - ref, color='green', alpha=0.12, label='Depletion zone')

            # Draw vertical lines
            if left_start is not None:
                ax.axvline(left_start - ref, color='b', linestyle='--', label="Left Start")
            if right_start is not None:
                ax.axvline(right_start - ref, color='r', linestyle='--', label="Right Start")

            # Overlay best-fit curves if available
            best_left = depletion.get('best_left_fit', None)
            best_right = depletion.get('best_right_fit', None)

            if best_left is not None:
                # Prefer global_x_vals if present (already in profile coords).
                if 'global_x_vals' in best_left:
                    x_fit_left = np.array(best_left['global_x_vals'], dtype=float) - ref
                else:
                    x_fit_left = -np.array(best_left['x_vals']) - ref
                y_fit_left = np.array(best_left.get('fit_curve', []), dtype=float)
                try:
                    order = np.argsort(x_fit_left)
                    x_fit_left = x_fit_left[order]
                    y_fit_left = y_fit_left[order]
                except Exception:
                    pass
                ax.plot(x_fit_left, y_fit_left, 'b-', lw=1.6, alpha=0.9, label='Left fit (best)')
                # annotate R² near the first plotted point
                try:
                    ax.text(x_fit_left[0], y_fit_left[0], f"R²={best_left['r2']:.2f}", color='b')
                except Exception:
                    pass

            if best_right is not None:
                if 'global_x_vals' in best_right:
                    x_fit_right = np.array(best_right['global_x_vals'], dtype=float) - ref
                else:
                    x_fit_right = np.array(best_right['x_vals']) - ref
                y_fit_right = np.array(best_right.get('fit_curve', []), dtype=float)
                try:
                    order = np.argsort(x_fit_right)
                    x_fit_right = x_fit_right[order]
                    y_fit_right = y_fit_right[order]
                except Exception:
                    pass
                ax.plot(x_fit_right, y_fit_right, 'r-', lw=1.6, alpha=0.9, label='Right fit (best)')
                try:
                    ax.text(x_fit_right[0], y_fit_right[0], f"R²={best_right['r2']:.2f}", color='r')
                except Exception:
                    pass

            # Include source name in title when available
            if source_name:
                ax.set_title(f"{os.path.splitext(os.path.basename(str(source_name)))[0]} - Profile {profile_id} – Depletion Width = {depletion['depletion_width']:.2f} µm")
            else:
                ax.set_title(f"Profile {profile_id} – Depletion Width = {depletion['depletion_width']:.2f} µm")
            ax.set_xlabel("Distance (µm)")
            ax.set_ylabel("Current (nA)")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()
            # Ensure a folder to save the per-profile depletion plot so the
            # visualization is available even when no interactive window opens.
            try:
                out_dir = os.path.join(os.getcwd(), 'depletion_plots')
                os.makedirs(out_dir, exist_ok=True)
                # include source name in saved filename when available
                if source_name:
                    base = os.path.splitext(os.path.basename(str(source_name)))[0]
                    # sanitize base
                    import re
                    base = re.sub(r'[^A-Za-z0-9._-]', '_', base)
                    out_path = os.path.join(out_dir, f'{base}_profile_{profile_id:02d}_depletion.png')
                else:
                    out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_depletion.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
            except Exception:
                out_path = None

            # Show (if possible) and close to avoid many open windows
            try:
                plt.show()
            except Exception:
                pass
            _maybe_close(fig)
            if out_path:
                if source_name:
                    print(f"Saved depletion plot to {out_path} (source: {source_name})")
                else:
                    print(f"Saved depletion plot to {out_path}")

    def apply_low_pass_filter(self, y_vals, cutoff_fraction=0.1, visualize=True):
        """
        Apply a double-sided Gaussian low-pass filter in the frequency domain
        with edge padding to avoid tail artifacts.
        """
        n = len(y_vals)
        if n == 0:
            return y_vals

        # Remove DC offset
        y_centered = y_vals - np.mean(y_vals)

        # ---  Pad the signal symmetrically to reduce edge effects ---
        pad_width = n // 2
        y_padded = np.pad(y_centered, pad_width, mode='reflect')
        n_padded = len(y_padded)

        # FFT on padded signal
        dx = self.pixel_size * 1e6  # µm
        fft_data = np.fft.fft(y_padded)
        freqs = np.fft.fftfreq(n_padded, d=dx)

        # Nyquist frequency
        f_nyquist = 0.5 / dx

        # Gaussian filter
        sigma = cutoff_fraction * f_nyquist
        filter_mask = np.exp(-0.5 * (freqs / sigma) ** 2)

        # Apply mask
        filtered_fft = fft_data * filter_mask
        filtered_padded = np.fft.ifft(filtered_fft).real

        # --- Remove padding ---
        filtered_y = filtered_padded[pad_width:pad_width + n]
        # Restore mean exactly (compensate for numerical error)
        orig_mean = np.mean(y_vals)
        # filtered_y currently has zero-mean w.r.t. padded centered signal; adjust so mean matches original
        filtered_y += (orig_mean - np.mean(filtered_y))

        # Optional: visualize
        if visualize:
            mask = freqs >= 0

            fig, axs = plt.subplots(2, 1, figsize=(10, 10))

            # --- 1) Real-space signal before and after ---
            x_vals = np.arange(n) * dx
            axs[0].plot(x_vals, y_vals, 'b-', alpha=0.7, label='Original')
            axs[0].plot(x_vals, filtered_y, 'r-', linewidth=2, label='Filtered')
            axs[0].set_title("Signal Before and After Gaussian Low-pass Filter")
            axs[0].set_xlabel("Position (µm)")
            axs[0].set_ylabel("Signal (a.u.)")
            axs[0].legend()
            axs[0].grid(True, linestyle='--', alpha=0.5)

            # --- 2) Frequency-domain magnitude with mask ---
            axs[1].plot(freqs[mask], np.abs(fft_data[mask]), 'b-', label='Original FFT')
            axs[1].plot(freqs[mask], np.abs(filtered_fft[mask]), 'r-', label='Filtered FFT')
            axs[1].plot(freqs[mask], filter_mask[mask] * np.max(np.abs(fft_data)),
                        'k--', label='Gaussian Mask')
            axs[1].set_title("FFT Before and After Filtering (with Gaussian Mask)")
            axs[1].set_xlabel("Spatial frequency (1/µm)")
            axs[1].set_ylabel("Magnitude (a.u.)")
            axs[1].legend()
            axs[1].grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.show()
        return filtered_y

    def visualize_fitted_profiles(self):
        """Visualize central fits and bar plots (same as before)."""
        if not self.results:
            print("No fitted results to visualize.")
            return

        for res in self.results:
            profile_id = res['Profile']
            central_fits = [side for side in res['fit_sides'] if side['shift'] == 0]
            if not central_fits:
                continue
            # use original profile coords and center at its peak
            x = np.array(self.profiles[profile_id - 1]['dist_um'])
            y = np.array(self.profiles[profile_id - 1]['current'])
            base_idx = int(np.argmax(y))
            ref = float(x[base_idx])

            fig_fit, axes_fit = plt.subplots(1, 2, figsize=(12, 4))
            for ax, side in zip(axes_fit, central_fits):
                # prefer global_x if present, otherwise map stored x_vals
                if 'global_x_vals' in side:
                    x_side = np.array(side['global_x_vals'], dtype=float) - ref
                else:
                    if 'Left' in side['side']:
                        x_side = -np.array(side['x_vals'], dtype=float) - ref
                    else:
                        x_side = np.array(side['x_vals'], dtype=float) - ref

                ax.plot(x_side, side['y_vals'], 'r.', alpha=0.5, label='Raw EBIC')
                ax.plot(x_side, side['fit_curve'], 'b-', linewidth=2, label='Central Fit')
                r2 = self._calculate_r2(side['y_vals'], side['fit_curve'])
                ax.set_xlabel("Distance (µm)")
                ax.set_ylabel("Current (nA)")
                ax.set_title(f"{side['side']}\n1/λ={side['inv_lambda']:.3f}, R²={r2:.3f}")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()
            fig_fit.tight_layout()
            # Attempt to save the central-fit figure for later inspection
            try:
                out_dir = os.path.join(os.getcwd(), 'depletion_plots')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_central_fits.png')
                fig_fit.savefig(out_path, dpi=200, bbox_inches='tight')
            except Exception:
                out_path = None

            try:
                plt.show()
            except Exception:
                pass
            _maybe_close(fig_fit)
            if out_path:
                print(f"Saved central-fit plot to {out_path}")

    def visualize_log_profiles(self, base='10', smooth=True, cutoff_fraction=0.05,
                               floor_factor=0.1, subtract_baseline=True):
        """
        Plot log(EBIC) vs distance for each profile and save the plots.

        Parameters
        ----------
        base : {'10', 'e'}
            Logarithm base to use: '10' for log10, 'e' for natural log.
        smooth : bool
            Whether to apply the low-pass filter before taking the log.
        cutoff_fraction : float
            Cutoff fraction passed to apply_low_pass_filter when smoothing.
        floor_factor : float
            Fraction of the minimum positive EBIC value used as floor for
            non-positive or negative values (to allow taking the logarithm).
        subtract_baseline : bool
            If True, attempt to subtract a baseline (y0) estimated from the
            best fits (if available) or from the tail median before taking log.
        """
        # Allow plotting even if fits/results are not available: use raw loaded profiles
        data_source = None
        if self.results:
            data_source = 'results'
            iterable = self.results
        elif getattr(self, 'profiles', None):
            data_source = 'profiles'
            # build simple result-like dicts pointing to profiles
            iterable = []
            for i, prof in enumerate(self.profiles):
                iterable.append({'Profile': i + 1})
        else:
            print("No fitted results or profiles to visualize.")
            return

        for res in iterable:
            profile_id = res['Profile']
            x = np.array(self.profiles[profile_id - 1]['dist_um'])
            y = np.array(self.profiles[profile_id - 1]['current'])

            # Optionally smooth first (recommended to reduce high-frequency noise)
            if smooth:
                y_plot = self.apply_low_pass_filter(y, cutoff_fraction=cutoff_fraction, visualize=False)
            else:
                y_plot = y.copy()

            # Baseline subtraction: prefer fit-derived y0 if available and sensible
            baseline = 0.0
            if subtract_baseline:
                best_left = res.get('depletion', {}).get('best_left_fit', None)
                best_right = res.get('depletion', {}).get('best_right_fit', None)
                y0_cands = []
                for b in (best_left, best_right):
                    if b is not None and 'parameters' in b:
                        try:
                            y0_cands.append(float(b['parameters'][2]))
                        except Exception:
                            pass
                if y0_cands:
                    # Use the smaller baseline (more conservative)
                    baseline = max(min(y0_cands), 0.0)
                else:
                    # Fallback: median of last 10 points
                    tail = y_plot[-10:]
                    baseline = max(np.median(tail), 0.0)

            # Subtract baseline and floor
            y_corr = y_plot - baseline
            pos = y_corr[y_corr > 0]
            if pos.size > 0:
                floor = max(np.min(pos) * floor_factor, 1e-12)
            else:
                floor = 1e-12

            y_safe = np.maximum(y_corr, floor)

            # Choose log base
            if base == '10':
                logy = np.log10(y_safe)
                ylabel = 'log10(Current)'
            else:
                logy = np.log(y_safe)
                ylabel = 'ln(Current)'

            fig, ax = plt.subplots(figsize=(8, 4))
            # center x axis so peak is at 0
            base_idx = int(np.argmax(y))
            ref = float(x[base_idx])
            x_plot = x - ref
            ax.plot(x_plot, logy, 'k-', lw=1.2, label=f'{ylabel}')

            # Optionally overlay best-fit transformed to log domain
            best_left = res.get('depletion', {}).get('best_left_fit', None)
            best_right = res.get('depletion', {}).get('best_right_fit', None)
            for b, color, label in ((best_left, 'b', 'Left fit'), (best_right, 'r', 'Right fit')):
                if b is None:
                    continue
                try:
                    # prefer global mapping and center at peak
                    if 'global_x_vals' in b:
                        x_fit = np.array(b['global_x_vals'], dtype=float) - ref
                    else:
                        x_fit = np.array(b['x_vals'], dtype=float)
                    y_fit = np.array(b['fit_curve'])
                    # transform fit by the same baseline subtraction
                    y_fit_corr = y_fit - baseline
                    y_fit_corr = np.maximum(y_fit_corr, floor)
                    if base == '10':
                        y_fit_log = np.log10(y_fit_corr)
                    else:
                        y_fit_log = np.log(y_fit_corr)
                    # For left fit, x_fit was from flipped side
                    if 'Left' in b['side']:
                        x_fit_plot = -x_fit - ref if 'global_x_vals' not in b else x_fit
                    else:
                        x_fit_plot = x_fit - ref if 'global_x_vals' not in b else x_fit
                    # if global_x_vals was used we already subtracted ref above
                    if 'global_x_vals' in b:
                        x_fit_plot = np.array(b['global_x_vals'], dtype=float) - ref
                    try:
                        order = np.argsort(x_fit_plot)
                        x_fit_plot = np.array(x_fit_plot)[order]
                        y_fit_log = np.array(y_fit_log)[order]
                    except Exception:
                        pass
                    ax.plot(x_fit_plot, y_fit_log, color=color, lw=1.6, alpha=0.9, label=f'{label} (log)')
                except Exception:
                    pass

            ax.set_xlabel('Distance (µm)')
            ax.set_ylabel(ylabel)
            ax.set_title(f'Profile {profile_id} — {ylabel} vs distance (baseline={baseline:.3g})')
            ax.grid(True, linestyle='--', alpha=0.5)

            fig.tight_layout()
            try:
                out_dir = os.path.join(os.getcwd(), 'depletion_plots')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_log.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
            except Exception:
                out_path = None

            try:
                plt.show()
            except Exception:
                pass
            _maybe_close(fig)
            if out_path:
                print(f"Saved log profile plot to {out_path}")



# --- new visualization ---
    def visualize_inv_lambda_vs_shift(self):
        """
        Plot 1/λ vs shift for each profile and side.
        """
        if not self.results:
            print("No fitted results to visualize.")
            return

        for res in self.results:
            profile_id = res['Profile']
            fit_sides = res['fit_sides']

            if not fit_sides:
                continue

            # Group by side type (Left/Right)
            sides = {"Left": [], "Right": []}
            for side in fit_sides:
                if "Left" in side['side']:
                    sides["Left"].append(side)
                else:
                    sides["Right"].append(side)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            fig.suptitle(f"Profile {profile_id} – 1/λ vs Shift")

            for ax, (side_name, side_data) in zip(axes, sides.items()):
                if not side_data:
                    ax.set_title(f"{side_name} – No fits")
                    continue

                shifts = [s['shift'] for s in side_data]
                inv_lambdas = [s['inv_lambda'] for s in side_data]

                ax.plot(shifts, inv_lambdas, 'o-', label=f'{side_name} side')
                ax.axvline(0, color='k', linestyle='--', alpha=0.5)
                ax.set_xlabel("Shift (pixels)")
                ax.set_ylabel("1/λ (µm)")
                ax.set_title(f"{side_name} side")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()

            plt.tight_layout()
            # Save the figure showing 1/λ vs shift so users have a persisted copy
            try:
                out_dir = os.path.join(os.getcwd(), 'depletion_plots')
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_invlambda_vs_shift.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
            except Exception:
                out_path = None

            try:
                plt.show()
            except Exception:
                pass
            _maybe_close(fig)
            if out_path:
                print(f"Saved 1/λ vs shift plot to {out_path}")


    def compute_average_lengths(self, show_table=True):
        """
        Compute average depletion width and diffusion lengths
        across all fitted profiles, separated by side.
        Results are quantized to the pixel size (no smaller increments).

        Parameters
        ----------
        show_table : bool
            If True, show a matplotlib table with the averages.

        Returns
        -------
        dict with keys:
            'average_depletion_width' (µm)
            'average_diffusion_length_left' (µm)
            'average_diffusion_length_right' (µm)
            'average_diffusion_length_all' (µm)
        """
        if not hasattr(self, "results") or not self.results:
            print("No results found. Run fit_all_profiles() first.")
            return None

        pixel_size_um = self.pixel_size * 1e6

        # --- Depletion widths ---
        depletion_widths = [
            res['depletion']['depletion_width']
            for res in self.results
            if res['depletion']['depletion_width'] is not None
        ]

        # --- Diffusion lengths by side ---
        left_lengths, right_lengths = np.array([], dtype=float), np.array([], dtype=float)
        for res in self.results:
            for side in res['fit_sides']:
                if side.get('inv_lambda') is not None:
                    if "Left" in side['side']:
                        left_lengths = np.append(left_lengths, side['inv_lambda'])
                    elif "Right" in side['side']:
                        right_lengths = np.append(right_lengths, side['inv_lambda'])

        # Compute means
        avg_depletion = np.mean(depletion_widths) if depletion_widths else None
        avg_left = np.mean(left_lengths) if left_lengths.any() else None
        avg_right = np.mean(right_lengths) if right_lengths.any() else None
        # Combine left and right inv_lambda arrays into a single vector.
        # Use concatenation to avoid elementwise broadcasting for unequal lengths.
        if left_lengths.size and right_lengths.size:
            combined = np.concatenate([left_lengths, right_lengths])
        elif left_lengths.size:
            combined = left_lengths
        else:
            combined = right_lengths
        avg_all = np.mean(combined) if combined.any() else None

        # --- Quantize values to pixel size ---
        def quantize(val):
            if val is None:
                return None
            return round(val / pixel_size_um) * pixel_size_um

        avg_depletion = quantize(avg_depletion)
        avg_left = quantize(avg_left)
        avg_right = quantize(avg_right)
        avg_all = quantize(avg_all)

        results = {
            "average_depletion_width": avg_depletion,
            "average_diffusion_length_left": avg_left,
            "average_diffusion_length_right": avg_right,
            "average_diffusion_length_all": avg_all,
        }

        # --- Optional visualization ---
        if show_table:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.axis("off")

            table_data = [
                ["Depletion Width (µm)", f"{avg_depletion:.3f}" if avg_depletion else "N/A"],
                ["Diffusion Length Left (µm)", f"{avg_left:.3f}" if avg_left else "N/A"],
                ["Diffusion Length Right (µm)", f"{avg_right:.3f}" if avg_right else "N/A"],
                ["Diffusion Length (All average) (µm)", f"{avg_all:.3f}" if avg_all else "N/A"],
            ]

            table = ax.table(
                cellText=table_data,
                colLabels=["Metric", "Average Value"],
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Add pixel size note
            ax.text(
                1.0, -0.25,
                f"Pixel size = {pixel_size_um:.3f} µm",
                transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="gray"
            )

            # Attempt to collect source names from the loaded profiles and display
            try:
                sources = []
                for p in getattr(self, 'profiles', []) or []:
                    s = p.get('source_name') if isinstance(p, dict) else None
                    if s and s not in sources:
                        sources.append(s)
                if sources:
                    if len(sources) == 1:
                        src_text = f"Source: {os.path.splitext(os.path.basename(str(sources[0])))[0]}"
                    else:
                        # show up to first 3 names then ellipsize
                        short = [os.path.splitext(os.path.basename(str(s)))[0] for s in sources[:3]]
                        src_text = "Sources: " + ", ".join(short) + ("..." if len(sources) > 3 else "")
                    ax.text(0.01, -0.25, src_text, transform=ax.transAxes, ha='left', va='top', fontsize=9, color='gray')
            except Exception:
                pass

            plt.title("Average Lengths Summary", fontsize=12, pad=12)
            plt.tight_layout()
            plt.show()

        return results

    def _truncate_tail_and_fit(self, x, y, shift, side, profile_id=0, global_x=None,
                               prevent_cross_peak=False, ref=None):
        """
        Progressively truncate the tail and fit each case.
        Stops early if 1/λ stabilizes within tolerance.
        """
        results = []
        n = len(x)
        tolerance = self.pixel_size * 1e6  # same as in head shifting
        prev_val = None
        stable_reached = False

        for cut in range(n, max(3, n - 20), -1):  # try up to 20 tail cuts
            if stable_reached:
                break

            x_cut, y_cut = x[:cut], y[:cut]
            if len(x_cut) < 3:
                continue

            # Fit on the local-distance array (positive distances from the
            # fitting edge) to preserve the original fitting behavior. We
            # will attach the absolute profile coords (global_x) and also
            # recompute the fit curve evaluated on those global coords for
            # correct overlaying in plots.
            x_for_fit = np.array(x_cut, dtype=float)

            fit = self._fit_falling(
                x=x_for_fit, y=y_cut,
                shift=shift,
                side=f"{side}-tailcut{n - cut}",
                profile_id=profile_id
            )

            if fit:
                curr_val = fit.get('inv_lambda', None)
                if curr_val is not None:
                    if prev_val is not None and abs(curr_val - prev_val) <= tolerance:
                        stable_reached = True  # stop once stable
                    prev_val = curr_val
                    # attach global x coordinates if provided so callers
                    # can map fit curves to the original profile axis. Also
                    # preserve the local-distance `x_vals` for backward
                    # compatibility (tests and other code expect x_vals to
                    # be the positive distances measured from the fitting
                    # edge).
                    try:
                        if global_x is not None:
                            gl = np.array(global_x[:cut], dtype=float)
                        else:
                            gl = np.array(x_cut, dtype=float)
                        # Optionally prevent the fit segment from crossing
                        # the profile peak (ref). For left fits we clamp
                        # values to be <= ref; for right fits we clamp to
                        # be >= ref. This makes overlays stop at the peak.
                        if prevent_cross_peak and (ref is not None):
                            if side.startswith('Left'):
                                gl = np.minimum(gl, float(ref))
                            else:
                                gl = np.maximum(gl, float(ref))
                        fit['global_x_vals'] = gl
                    except Exception:
                        fit['global_x_vals'] = np.array(x_cut, dtype=float)

                    # Preserve the local-distance x_vals for backward compatibility
                    try:
                        fit['x_vals'] = np.array(x_cut, dtype=float)
                    except Exception:
                        fit['x_vals'] = np.array(x_cut, dtype=float)

                    # Recompute fit_curve evaluated at the mapped global
                    # coordinates so plotting overlays align directly with
                    # fit['global_x_vals'] (which plotting prefers).
                    try:
                        params = fit.get('parameters', None)
                        if params is not None:
                            # Map global profile coordinates back to the local
                            # distance domain used for fitting. The first
                            # element of global_x_vals corresponds to the
                            # fit start (closest to the intersection).
                            if 'global_x_vals' in fit:
                                gl = np.array(fit['global_x_vals'], dtype=float)
                                if gl.size > 0:
                                    start_pos = float(gl[0])
                                else:
                                    start_pos = 0.0
                                if 'Left' in side:
                                    x_for_global_eval = np.abs(start_pos - gl)
                                else:
                                    x_for_global_eval = gl - start_pos
                            else:
                                x_for_global_eval = np.array(x_cut, dtype=float)

                            # Evaluate on the same local-distance array used
                            # for fitting and keep that as the global-fit curve
                            # so each fit value corresponds elementwise to
                            # the entries of fit['global_x_vals'] (which is
                            # built to align with x_cut). Keep a local copy
                            # as well for debugging.
                            fit_local = self.exp_model(np.array(x_for_fit, dtype=float), *params)
                            fit['fit_curve_local'] = fit_local
                            fit['fit_curve'] = fit_local
                    except Exception:
                        pass
                    results.append(fit)

        return results

    def plot_inv_lambda_vs_tailcut(self, results=None, profile_id=None):
        """
        Plot 1/λ vs tailcut for each profile and side.
        - If `results` is given → use that list of fits directly.
        - If `results` is None → use self.results (batch of profiles).
        """
        if results is None:
            results = []
            for res in getattr(self, "results", []):
                pid = res.get("Profile", None)
                for fs in res.get("fit_sides", []):
                    results.append({**fs, "Profile": pid})

        if not results:
            print("No fitted results to visualize.")
            return

        # Group by profile
        profiles = {}
        for res in results:
            pid = res.get("Profile", profile_id if profile_id is not None else 0)
            profiles.setdefault(pid, []).append(res)

        # Plot per profile
        for pid, prof_results in profiles.items():
            left_tailcuts, left_inv = [], []
            right_tailcuts, right_inv = [], []

            for r in prof_results:
                side_str = r.get("side", "")
                inv_lambda = r.get("inv_lambda", None)
                if inv_lambda is None:
                    continue

                if "tailcut" in side_str:
                    try:
                        cut_val = int(side_str.split("tailcut")[-1].split()[0])
                    except Exception:
                        continue

                    if side_str.startswith("Left"):
                        left_tailcuts.append(cut_val)
                        left_inv.append(inv_lambda)
                    elif side_str.startswith("Right"):
                        right_tailcuts.append(cut_val)
                        right_inv.append(inv_lambda)

            # Plot if we have data
            plt.figure(figsize=(7, 5))
            if left_tailcuts:
                plt.plot(left_tailcuts, left_inv, "o-", label="Left side")
            if right_tailcuts:
                plt.plot(right_tailcuts, right_inv, "s-", label="Right side")

            plt.xlabel("Tailcut index (# points truncated)")
            plt.ylabel("1 / λ (1/pixels)")
            plt.title(f"Profile {pid} – 1/λ vs. tailcut")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def _plot_tailcut_fits(self, results, x_vals, y_vals, side_to_plot, base_idx, profile_id):
        """Overlay fits for tail truncation only."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.title(f"Profile {profile_id} – {side_to_plot} tail truncations")

        # Base data
        # center x axis at peak
        ref = float(x_vals[base_idx])
        if side_to_plot == 'Left':
            x_base, y_base = x_vals[:base_idx + 1] - ref, y_vals[:base_idx + 1]
        else:
            x_base, y_base = x_vals[base_idx:] - ref, y_vals[base_idx:]

        plt.plot(x_base, y_base, color='0.6', alpha=0.7, label=f'raw {side_to_plot.lower()}')
        y_filtered_base = self.apply_low_pass_filter(y_base, visualize=False)
        plt.plot(x_base, y_filtered_base, 'k-', lw=1.5, label=f'filtered {side_to_plot.lower()}')

        # Overlay tailcut fits
        for res in results:
            side_str = res.get('side', '')
            if side_str.startswith(side_to_plot) and "tailcut" in side_str:
                try:
                    cut_val = int(side_str.split("tailcut")[-1].split()[0])
                except Exception:
                    cut_val = -1
                # prefer global x mapped to profile coords when available
                # prefer global mapping; otherwise map and shift to peak-centered
                x_plot = res.get('global_x_vals', None)
                if x_plot is None:
                    if side_to_plot == 'Left':
                        x_plot = -np.array(res['x_vals'], dtype=float) - ref
                    else:
                        x_plot = np.array(res['x_vals'], dtype=float) - ref
                else:
                    x_plot = np.array(x_plot, dtype=float) - ref
                y_plot = np.array(res.get('fit_curve', []), dtype=float)
                try:
                    order = np.argsort(x_plot)
                    x_plot = x_plot[order]
                    y_plot = y_plot[order]
                except Exception:
                    pass
                plt.plot(x_plot, y_plot, lw=1.2,
                         label=f"tailcut {cut_val} (1/λ={res.get('inv_lambda', float('nan')):.2f})")

        plt.xlabel("x (pixels)")
        plt.ylabel("Signal")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        # Save tailcut overlay figure to disk
        try:
            out_dir = os.path.join(os.getcwd(), 'depletion_plots')
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_{side_to_plot}_tailcuts.png')
            plt.gcf().savefig(out_path, dpi=200, bbox_inches='tight')
        except Exception:
            out_path = None

        try:
            plt.show()
        except Exception:
            pass
        if out_path:
            print(f"Saved tailcut overlay plot to {out_path}")


    def fit_profile_sides_iterative(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                    plot_left=True, plot_right=True, max_iter=10, tol_factor=1.0):
        """
        Iterative version of fit_profile_sides:
        Alternates between shifting (head search) and tail truncation until 1/λ converges.
        """

        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        inv_lambda_old_left = None
        inv_lambda_old_right = None
        results = []

        for iteration in range(max_iter):
            # --- Run the normal head+tail procedure once ---
            results = self.fit_profile_sides(
                x_vals, y_vals,
                intersection_idx=intersection_idx,
                profile_id=profile_id,
                plot_left=plot_left,
                plot_right=plot_right
            )

            # Extract latest stable values (shift=0 fits)
            inv_left = [r['inv_lambda'] for r in results if "Left-tailcut" in r['side']]
            inv_right = [r['inv_lambda'] for r in results if "Right-tailcut" in r['side']]

            inv_left = inv_left[-1] if inv_left else None
            inv_right = inv_right[-1] if inv_right else None

            # --- Check convergence ---
            converged_left = (
                inv_left is not None
                and inv_lambda_old_left is not None
                and abs(inv_left - inv_lambda_old_left) <= self.pixel_size * tol_factor
            )
            converged_right = (
                inv_right is not None
                and inv_lambda_old_right is not None
                and abs(inv_right - inv_lambda_old_right) <= self.pixel_size * tol_factor
            )

            if converged_left and converged_right:
                print(f"Profile {profile_id}: converged after {iteration+1} iterations.")
                break

            inv_lambda_old_left = inv_left
            inv_lambda_old_right = inv_right

        return results


    

    def _fit_linear(self, x, y, shift, side="Right", profile_id=0):
        """
        Fit a linear model to ln(y) vs x (natural log). Returns a dict similar
        to _fit_falling but with slope/intercept and inv_lambda derived from the
        slope (inv_lambda = 1/|slope| in µm units when x is in µm).
        """
        if len(x) < 3:
            return None

        # Ensure positive values for log; floor tiny or non-positive values
        y_arr = np.array(y, dtype=float)
        pos = y_arr[y_arr > 0]
        if pos.size > 0:
            floor = max(np.min(pos) * 0.1, 1e-12)
        else:
            floor = 1e-12
        y_safe = np.maximum(y_arr, floor)

        try:
            # Default behavior: log-transform inside this routine for callers
            # that pass raw y. Keep existing semantics.
            logy = np.log(y_safe)
            return self._fit_linear_on_log(x, logy, shift=shift, side=side, profile_id=profile_id)
        except Exception as e:
            print(f"{side} linear fit failed (shift {shift}) for profile {profile_id}: {e}")
            return None

    def _fit_linear_on_log(self, x, logy, shift, side="Right", profile_id=0, baseline=0.0, x_already_positive=False):
        """
        Fit a linear model to precomputed natural-log values `logy`.
        Returns a dict with fit parameters, R², and fitted curve.
        
        Args:
            x_already_positive: If True, x is already positive distances from edge (for Left side).
                               If False, will negate x for Left side fitting.
        """
        try:
            # Ensure arrays
            x_arr = np.array(x, dtype=float)
            logy_arr = np.array(logy, dtype=float)

            # The executable contract used elsewhere: `x_vals` returned by
            # fitting routines are the local positive distances from the
            # fitting edge (0..). For the left side callers often pass a
            # flipped/positive-distance x already; ensure we keep that
            # convention here. We'll also flip for fitting to keep a
            # consistent increasing-x fit direction, and then flip back for
            # construction of fit_curve_local.
            # Determine whether the input x_arr is already local-positive
            # distances (starts at 0); we treat x_arr as the local domain.

            # For fitting consistency, flip the x-axis for left-side fits
            # so the polynomial sees increasing distances from the edge.
            # BUT: if x is already positive (pre-flipped by caller), don't flip again!
            if 'Left' in side and not x_already_positive:
                x_for_fit = -np.array(x_arr, dtype=float)
            else:
                x_for_fit = np.array(x_arr, dtype=float)

            # Linear fit in log-domain (slope, intercept)
            p = np.polyfit(x_for_fit, logy_arr, 1)
            slope = float(p[0])
            intercept = float(p[1])

            # For Left side with pre-flipped positive x, slope is already correct
            # For Left side with original x (negated for fitting), we need to negate slope
            if 'Left' in side and not x_already_positive:
                slope_profile = -slope
            else:
                slope_profile = slope

            if 'Left' in side and not x_already_positive:
                # x_for_fit == -x_arr, so -x_for_fit == x_arr
                log_fit = np.polyval(p, -x_for_fit)
            else:
                # For already-positive x (or Right side), evaluate directly
                log_fit = np.polyval(p, x_for_fit)

            # Reconstruct fitted curve in original units by adding the
            # baseline back (we assume logy = log(y - baseline)). Keep a
            # local copy (`fit_curve_local`) that is elementwise aligned
            # with `x_vals` (local distances). Callers may later compute a
            # `fit_curve` mapped to `global_x_vals` depending on mapping.
            fit_curve_local = np.exp(log_fit) + float(baseline)

            # Characteristic length: slope is d(ln y)/dx, so 1/|slope| in
            # the x-units (which here are µm when used as such). Convert
            # and quantize to pixel size for reporting, matching other
            # routines' behavior.
            inv_lambda_um = np.inf if slope_profile == 0 else 1.0 / abs(slope_profile)
            pixel_size_um = self.pixel_size * 1e6
            if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
            else:
                inv_lambda = inv_lambda_um

            # Compute R² in log-domain (consistent with linear fit)
            ss_res = np.sum((logy_arr - log_fit) ** 2)
            ss_tot = np.sum((logy_arr - np.mean(logy_arr)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            # Return a dict that mirrors the exponential fitter's keys so
            # downstream code can consume results uniformly. Note:
            # - 'x_vals' is the local distances array (as provided)
            # - 'y_vals' is reconstructed raw-domain values (exp(logy)+baseline)
            # - 'fit_curve' is set to fit_curve_local here; callers that
            #   attach 'global_x_vals' and need the fit mapped to global
            #   coordinates will recompute a mapped fit later (truncate
            #   helpers already do this).
            return {
                'side': f'{side} (shift {shift})',
                'shift': shift,
                'x_vals': np.array(x_arr, dtype=float),
                'y_vals': np.exp(logy_arr) + float(baseline),
                'fit_curve': fit_curve_local,
                'fit_curve_local': fit_curve_local,
                # parameters returned in profile-coordinates: (slope_profile, intercept_profile)
                'parameters': (slope_profile, intercept),
                'slope': slope_profile,
                'intercept': intercept,
                'inv_lambda': inv_lambda,
                'r2': r2
            }

        except Exception as e:
            print(f"{side} linear(on-log) fit failed (shift {shift}) for profile {profile_id}: {e}")
            return None


    def _truncate_tail_and_fit_linear(self, x, y, shift, side, profile_id=0, global_x=None,
                                      prevent_cross_peak=False, ref=None):
        """
        Progressively truncate the tail and fit each case for linear-on-log fits.
        """
        results = []
        n = len(x)
        tolerance = self.pixel_size * 1e6
        prev_val = None
        stable_reached = False

        for cut in range(n, max(3, n - 20), -1):
            if stable_reached:
                break

            x_cut, y_cut = x[:cut], y[:cut]
            if len(x_cut) < 3:
                continue

            # Pre-log the y data here so the linear fit operates on ln(y)
            # consistently with the requested behavior.
            y_arr = np.array(y[:cut], dtype=float)
            pos = y_arr[y_arr > 0]
            if pos.size > 0:
                floor = max(np.min(pos) * 0.1, 1e-12)
            else:
                floor = 1e-12
            y_safe = np.maximum(y_arr, floor)
            logy = np.log(y_safe)

            fit = self._fit_linear_on_log(
                x=x_cut, logy=logy,
                shift=shift,
                side=f"{side}-tailcut{n - cut}",
                profile_id=profile_id
            )

            if fit:
                curr_val = fit.get('inv_lambda', None)
                if curr_val is not None:
                    if prev_val is not None and abs(curr_val - prev_val) <= tolerance:
                        stable_reached = True
                    prev_val = curr_val
                    try:
                        if global_x is not None:
                            gl = np.array(global_x[:cut], dtype=float)
                        else:
                            gl = np.array(x_cut, dtype=float)
                        if prevent_cross_peak and (ref is not None):
                            if side.startswith('Left'):
                                gl = np.minimum(gl, float(ref))
                            else:
                                gl = np.maximum(gl, float(ref))
                        fit['global_x_vals'] = gl
                    except Exception:
                        fit['global_x_vals'] = np.array(x_cut, dtype=float)
                    # compute fit_curve consistent with global_x_vals for plotting
                    try:
                        params = fit.get('parameters', None)
                        if params is not None:
                            # linear-fit stores slope/intercept in parameters
                            slope, intercept = params[0], params[1]
                            # Map global coords back to the local fit domain
                            if 'global_x_vals' in fit:
                                gl = np.array(fit['global_x_vals'], dtype=float)
                                if gl.size > 0:
                                    start_pos = float(gl[0])
                                else:
                                    start_pos = 0.0
                                if 'Left' in side:
                                    x_eval = np.abs(start_pos - gl)
                                else:
                                    x_eval = gl - start_pos
                            else:
                                x_eval = np.array(x_cut, dtype=float)
                            # compute fit curve in linear domain (exp of poly)
                            log_fit = slope * x_eval + intercept
                            # Use the local-domain evaluation as the canonical
                            # fit_curve so it lines up elementwise with
                            # fit['global_x_vals'].
                            fit_local = np.exp(np.polyval([slope, intercept], np.array(x_cut, dtype=float)))
                            fit['fit_curve_local'] = fit_local
                            fit['fit_curve'] = fit_local
                    except Exception:
                        pass
                    results.append(fit)

        return results

    def fit_profile_sides_linear(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                 plot_left=False, plot_right=False, prevent_cross_peak=False,
                                 use_window_search=True, min_window=10, max_window=60, max_search=30,
                                 r2_threshold_window=0.75, min_inv_factor=0.2, max_inv_factor=500.0):
        """
        Same flow as fit_profile_sides but fits linear models on ln(current) for
        left and right sides. Uses the same shifting and tail-cut truncation logic.
        
        Key change: reduced min_window from 20 to 10, max_window from 80 to 60,
        and increased max_search from 20 to 30 to better capture short linear
        regions close to the junction.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx

        # Mirror the exponential fitter: allow the left-side start to move
        # rightwards up to the profile peak so we can try fitting closer to
        # the maximum when the intersection is offset.
        n = len(x_vals)
        relative = int(base_idx - intersection_idx)
        pos_allow = max(0, relative)
        pos_allow = min(pos_allow, n - 1)
        neg_allow = min(0, relative)
        neg_allow = max(neg_allow, -(n - 1))

        results = []
        tolerance = self.pixel_size * 1e6

        # --- LEFT side: find start index ---
        # We'll try a broader shift range and collect candidate fits. We
        # prefer starting indices that are <= base_idx so the left start is
        # never to the right of the peak. We transform y to ln(y) before
        # fitting so the curve fit is linear in the log-domain.
        left_candidates = []
        max_shift = min(max(30, n // 4), n - 1)
        for shift in range(pos_allow, -max_shift - 1, -1):
            # clamp start index so it never lies to the right of the peak
            start_idx_left = max(0, min(base_idx, base_idx + shift))
            left_x_raw = x_vals[:start_idx_left + 1]
            y_left_raw = y_vals[:start_idx_left + 1]

            # Use raw data for fitting (no pre-filtering)
            y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
            try:
                start_pos_left = float(left_x_raw[-1])
                x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
            except Exception:
                x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            if len(left_x_truncated) > 2:
                # pre-log the truncated data for linear fitting
                y_arr = np.array(left_y_truncated, dtype=float)
                # estimate baseline from tail median of the truncated segment
                tail_n = min(10, max(3, len(y_arr) // 5))
                if len(y_arr) <= max_window:
                    baseline = 0.0
                else:
                    baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                # subtract baseline then floor small/negative residuals
                y_corr = y_arr - baseline
                pos = y_corr[y_corr > 0]
                if pos.size > 0:
                    # use a small fraction for floor to avoid changing shape
                    floor = max(np.min(pos) * 1e-6, 1e-12)
                else:
                    floor = 1e-12
                logy = np.log(np.maximum(y_corr, floor))

                left_fit = self._fit_linear_on_log(
                    x=left_x_truncated, logy=logy,
                    shift=shift, side="Left", profile_id=profile_id,
                    baseline=baseline
                )
                if left_fit and left_fit.get('r2') is not None:
                    left_candidates.append((start_idx_left, left_fit))

        # If requested, perform a local sliding-window search anchored at
        # the detected junction to find small linear regimes close to the
        # intersection. This handles short linear regions that long
        # left-shift fits miss.
        def _search_local_region(side, length_penalty=0.002):
            """Search for linear-on-log regions by trying different start positions and window sizes.
            
            Key improvement: detect the END of linearity by checking if extending the window
            significantly degrades the fit quality (residuals increase).
            """
            best_candidate = None
            ref_idx = int(intersection_idx) if intersection_idx is not None else base_idx
            if side == 'Left':
                # Search starting from various positions moving left from junction
                left_start = min(ref_idx, base_idx - 1)
                start_range = range(left_start, max(left_start - max_search, 0) - 1, -1)
            else:
                # For right side, start just after the junction/peak
                right_start = max(ref_idx, base_idx + 1)
                start_range = range(right_start, min(right_start + max_search, n - 1) + 1)

            for s in start_range:
                try:
                    if side == 'Left':
                        left_x_raw = x_vals[:s + 1]
                        y_left_raw = y_vals[:s + 1]
                        # Use raw data for fitting (no pre-filtering)
                        y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
                        try:
                            start_pos_left = float(left_x_raw[-1])
                            x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
                        except Exception:
                            x_left_flipped = np.abs(left_x_raw)[::-1]
                        max_w = min(max_window, len(y_left_flipped))
                        if max_w < min_window:
                            continue
                        # Try a range of window sizes - look for the longest window
                        # that maintains good linearity (high R²)
                        best_local = (None, -np.inf, None, -np.inf)  # (w, r2_lin, fit, score)
                        prev_r2 = -np.inf
                        degradation_count = 0
                        max_degradation = 4  # Stop if R² degrades for this many consecutive steps (more permissive)
                        degradation_threshold = 0.03  # More permissive threshold for degradation detection
                        
                        for w in range(min_window, max_w + 1):
                            x_win = x_left_flipped[:w]
                            y_win = y_left_flipped[:w]
                            y_arr = np.array(y_win, dtype=float)
                            tail_n = min(10, max(3, len(y_arr) // 5))
                            if len(y_arr) <= max_window:
                                baseline = 0.0
                            else:
                                baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                            y_corr = y_arr - baseline
                            pos = y_corr[y_corr > 0]
                            # Require reasonable fraction positive (more permissive)
                            if pos.size < max(3, int(0.3 * len(y_arr))):
                                continue
                            floor = max(np.min(pos) * 1e-6, 1e-12)
                            logy = np.log(np.maximum(y_corr, floor))
                            fit = self._fit_linear_on_log(x=x_win, logy=logy, shift=0, side='Left', profile_id=profile_id, baseline=baseline, x_already_positive=True)
                            if fit is None:
                                continue
                            # Compute linear-domain R² on y vs fit_curve_local
                            try:
                                y_true = np.exp(logy) + baseline
                                y_fit = np.array(fit.get('fit_curve_local', []), dtype=float)
                                r2_lin = self._calculate_r2(y_true, y_fit)
                            except Exception:
                                r2_lin = -np.inf
                            
                            # Check if R² is degrading - this indicates we've left the linear region
                            if r2_lin < prev_r2 - degradation_threshold:
                                degradation_count += 1
                                if degradation_count >= max_degradation:
                                    # Stop here - we've likely left the linear region
                                    break
                            else:
                                degradation_count = 0
                                prev_r2 = r2_lin
                            
                            # Score: favor high R² with very small penalty for window length
                            score = (r2_lin - length_penalty * float(w))
                            # Reject fits that imply an unphysically small
                            # characteristic length (inv_lambda) which often
                            # results from near-vertical slopes on tiny
                            # windows. Enforce min/max acceptable inv_lambda
                            try:
                                pixel_size_um = self.pixel_size * 1e6
                                inv_lambda = fit.get('inv_lambda', None)
                            except Exception:
                                pixel_size_um = self.pixel_size * 1e6
                                inv_lambda = None
                            # Allow more permissive lower bound by scaling the
                            # pixel size. The default factor (0.2) permits
                            # shorter characteristic lengths while still
                            # protecting against extremely small / noisy
                            # near-vertical fits. The upper bound remains
                            # large by default.
                            min_inv = float(min_inv_factor) * pixel_size_um
                            max_inv = float(max_inv_factor) * pixel_size_um
                            # If inv_lambda is None (unexpected), accept the
                            # candidate so downstream logic can evaluate it.
                            if inv_lambda is not None and np.isfinite(inv_lambda):
                                if inv_lambda < min_inv or inv_lambda > max_inv:
                                    continue
                            if r2_lin > best_local[1] or score > best_local[3]:
                                best_local = (w, r2_lin, fit, score)
                            if r2_lin >= r2_threshold_window:
                                # prefer the smallest window meeting threshold
                                # but don't return immediately — record and
                                # allow the search to continue so we can
                                # compare other starts/windows and avoid
                                # selecting a trivial tiny-window at the
                                # junction.
                                best_local = (w, r2_lin, fit, score)
                                break
                        # update global best if this start produced a good fit
                        if best_local[3] > (best_candidate[3] if best_candidate else -np.inf):
                            # store (s, w, fit, r2_lin, score)
                            best_candidate = (s, best_local[0], best_local[2], best_local[1], best_local[3])
                    else:
                        right_x_raw = x_vals[s:]
                        y_right_raw = y_vals[s:]
                        # Use raw data for fitting (no pre-filtering)
                        y_right_raw_arr = np.array(y_right_raw, dtype=float)
                        cut = len(y_right_raw_arr)
                        max_w = min(max_window, cut)
                        if max_w < min_window:
                            continue
                        best_local = (None, -np.inf, None, -np.inf)
                        prev_r2 = -np.inf
                        degradation_count = 0
                        max_degradation = 4  # More permissive
                        degradation_threshold = 0.03  # More permissive threshold
                        try:
                            start_pos_right = float(right_x_raw[0])
                            x_right_local = (np.array(right_x_raw, dtype=float) - start_pos_right)
                        except Exception:
                            x_right_local = np.array(right_x_raw, dtype=float)
                        for w in range(min_window, max_w + 1):
                            x_win = x_right_local[:w]
                            y_win = np.array(y_right_raw_arr[:w], dtype=float)
                            y_arr = np.array(y_win, dtype=float)
                            tail_n = min(10, max(3, len(y_arr) // 5))
                            if len(y_arr) <= max_window:
                                baseline = 0.0
                            else:
                                baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                            y_corr = y_arr - baseline
                            pos = y_corr[y_corr > 0]
                            # More permissive: require only 40% positive
                            if pos.size < max(3, int(0.4 * len(y_arr))) or y_corr[-1] <= 0:
                                continue
                            floor = max(np.min(pos) * 1e-6, 1e-12)
                            logy = np.log(np.maximum(y_corr, floor))
                            fit = self._fit_linear_on_log(x=x_win, logy=logy, shift=0, side='Right', profile_id=profile_id, baseline=baseline)
                            if fit is None:
                                continue
                            try:
                                y_true = np.exp(logy) + baseline
                                y_fit = np.array(fit.get('fit_curve_local', []), dtype=float)
                                r2_lin = self._calculate_r2(y_true, y_fit)
                            except Exception:
                                r2_lin = -np.inf
                            
                            # Check if R² is degrading - indicates end of linear region
                            if r2_lin < prev_r2 - degradation_threshold:
                                degradation_count += 1
                                if degradation_count >= max_degradation:
                                    break
                            else:
                                degradation_count = 0
                                prev_r2 = r2_lin
                            
                            # Score: favor high R² with small penalty for window length
                            score = (r2_lin - length_penalty * float(w))
                            try:
                                pixel_size_um = self.pixel_size * 1e6
                                inv_lambda = fit.get('inv_lambda', None)
                            except Exception:
                                pixel_size_um = self.pixel_size * 1e6
                                inv_lambda = None
                            min_inv = float(min_inv_factor) * pixel_size_um
                            max_inv = float(max_inv_factor) * pixel_size_um
                            if inv_lambda is not None and np.isfinite(inv_lambda):
                                if inv_lambda < min_inv or inv_lambda > max_inv:
                                    continue
                            if r2_lin > best_local[1] or score > best_local[3]:
                                best_local = (w, r2_lin, fit, score)
                            if r2_lin >= r2_threshold_window:
                                best_local = (w, r2_lin, fit, score)
                                break
                        if best_local[3] > (best_candidate[3] if best_candidate else -np.inf):
                            best_candidate = (s, best_local[0], best_local[2], best_local[1], best_local[3])
                except Exception:
                    continue
            if best_candidate:
                # Try to prefer a smaller window if a near-junction smaller
                # window reaches a large fraction of the best R² found.
                try:
                    best_r2_overall = best_candidate[3]
                    small_radius = min(6, max_search)
                    # search small neighborhood near the junction for smaller windows
                    ref_idx_local = int(intersection_idx) if intersection_idx is not None else base_idx
                    neighborhood = range(ref_idx_local, max(ref_idx_local - small_radius, 0) - 1, -1) if side == 'Left' else range(ref_idx_local, min(ref_idx_local + small_radius, n - 1) + 1)
                    for s2 in neighborhood:
                        try:
                            if side == 'Left':
                                left_x_raw = x_vals[:s2 + 1]
                                y_left_raw = y_vals[:s2 + 1]
                                # Use raw data for fitting (no pre-filtering)
                                y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
                                try:
                                    start_pos_left = float(left_x_raw[-1])
                                    x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
                                except Exception:
                                    x_left_flipped = np.abs(left_x_raw)[::-1]
                                max_w2 = min(max_window, len(y_left_flipped))
                                for w2 in range(min_window, max_w2 + 1):
                                    x_win = x_left_flipped[:w2]
                                    y_win = y_left_flipped[:w2]
                                    y_arr = np.array(y_win, dtype=float)
                                    tail_n = min(10, max(3, len(y_arr) // 5))
                                    if len(y_arr) <= max_window:
                                        baseline = 0.0
                                    else:
                                        baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                                    y_corr = y_arr - baseline
                                    pos = y_corr[y_corr > 0]
                                    if pos.size < max(3, int(0.4 * len(y_arr))):
                                        continue
                                    floor = max(np.min(pos) * 1e-6, 1e-12)
                                    logy = np.log(np.maximum(y_corr, floor))
                                    fit2 = self._fit_linear_on_log(x=x_win, logy=logy, shift=0, side='Left', profile_id=profile_id, baseline=baseline)
                                    if fit2 is None:
                                        continue
                                    try:
                                        y_true = np.exp(logy) + baseline
                                        y_fit = np.array(fit2.get('fit_curve_local', []), dtype=float)
                                        r2_lin2 = self._calculate_r2(y_true, y_fit)
                                    except Exception:
                                        r2_lin2 = -np.inf
                                    if r2_lin2 >= 0.9 * best_r2_overall and w2 < best_candidate[1]:
                                        return (s2, w2, fit2, r2_lin2)
                            else:
                                right_x_raw = x_vals[s2:]
                                y_right_raw = y_vals[s2:]
                                # Use raw data for fitting (no pre-filtering)
                                y_right_raw_arr = np.array(y_right_raw, dtype=float)
                                max_w2 = min(max_window, len(y_right_raw_arr))
                                try:
                                    start_pos_right = float(right_x_raw[0])
                                    x_right_local = (np.array(right_x_raw, dtype=float) - start_pos_right)
                                except Exception:
                                    x_right_local = np.array(right_x_raw, dtype=float)
                                for w2 in range(min_window, max_w2 + 1):
                                    x_win = x_right_local[:w2]
                                    y_win = np.array(y_right_raw_arr[:w2], dtype=float)
                                    y_arr = np.array(y_win, dtype=float)
                                    tail_n = min(10, max(3, len(y_arr) // 5))
                                    if len(y_arr) <= max_window:
                                        baseline = 0.0
                                    else:
                                        baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                                    y_corr = y_arr - baseline
                                    pos = y_corr[y_corr > 0]
                                    if pos.size < max(3, int(0.6 * len(y_arr))) or y_corr[-1] <= 0:
                                        continue
                                    floor = max(np.min(pos) * 1e-6, 1e-12)
                                    logy = np.log(np.maximum(y_corr, floor))
                                    fit2 = self._fit_linear_on_log(x=x_win, logy=logy, shift=0, side='Right', profile_id=profile_id, baseline=baseline)
                                    if fit2 is None:
                                        continue
                                    try:
                                        y_true = np.exp(logy) + baseline
                                        y_fit = np.array(fit2.get('fit_curve_local', []), dtype=float)
                                        r2_lin2 = self._calculate_r2(y_true, y_fit)
                                    except Exception:
                                        r2_lin2 = -np.inf
                                    if r2_lin2 >= 0.9 * best_r2_overall and w2 < best_candidate[1]:
                                        return (s2, w2, fit2, r2_lin2)
                        except Exception:
                            continue
                except Exception:
                    pass
                return best_candidate
            return None

        # Run window search first if requested
        if use_window_search:
            left_win = _search_local_region('Left')
            right_win = _search_local_region('Right')
            if left_win is not None:
                if len(left_win) == 5:
                    s, w, fit, r2_lin, _score = left_win
                else:
                    s, w, fit, r2_lin = left_win
                # attach some metadata and append
                fit['side'] = f'Left-window(w={w})'
                fit['_r2_linear'] = r2_lin
                # global mapping: map to original coordinates
                left_x_raw = x_vals[:s + 1]
                # produce an ascending, properly ordered global x
                # vector so plotting maps slopes in the right
                # orientation.
                gx = np.array(left_x_raw[::-1][:w], dtype=float)
                fit['global_x_vals'] = np.sort(gx)
                results.append(fit)
            if right_win is not None:
                if len(right_win) == 5:
                    s, w, fit, r2_lin, _score = right_win
                else:
                    s, w, fit, r2_lin = right_win
                fit['side'] = f'Right-window(w={w})'
                fit['_r2_linear'] = r2_lin
                right_x_raw = x_vals[s:]
                fit['global_x_vals'] = np.array(right_x_raw[:w], dtype=float)
                results.append(fit)
            # If window search produced at least one windowed fit, prefer
            # those and return early to avoid appending older long-tail fits.
            if left_win is not None or right_win is not None:
                # Debug output to show what was fitted
                ref_pos = float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx])
                if left_win is not None:
                    s_l, w_l = (left_win[0], left_win[1]) if len(left_win) >= 2 else (None, None)
                    r2_l = left_win[3] if len(left_win) >= 4 else None
                    if s_l is not None:
                        start_pos = float(x_vals[s_l])
                        end_pos = float(x_vals[max(0, s_l - w_l)])
                        print(f"Profile {profile_id} Left: window={w_l}pts, R²={r2_l:.3f}, range=[{end_pos:.2f}, {start_pos:.2f}]µm (junction at {ref_pos:.2f}µm)")
                if right_win is not None:
                    s_r, w_r = (right_win[0], right_win[1]) if len(right_win) >= 2 else (None, None)
                    r2_r = right_win[3] if len(right_win) >= 4 else None
                    if s_r is not None:
                        start_pos = float(x_vals[s_r])
                        end_pos = float(x_vals[min(len(x_vals)-1, s_r + w_r)])
                        print(f"Profile {profile_id} Right: window={w_r}pts, R²={r2_r:.3f}, range=[{start_pos:.2f}, {end_pos:.2f}]µm (junction at {ref_pos:.2f}µm)")
                return results

        # --- RIGHT side: find start index ---
        right_candidates = []
        for shift in range(neg_allow, max_shift + 1):  # try a broader rightward shift
            start_idx_right = min(len(x_vals) - 1, max(base_idx, base_idx + shift))
            right_x_raw = x_vals[start_idx_right:]
            y_right_raw = y_vals[start_idx_right:]

            # Use raw data for fitting (no pre-filtering)
            right_y_raw_arr = np.array(y_right_raw, dtype=float)
            cut_idx = self._find_snr_end_index(right_y_raw_arr)
            right_y_truncated = right_y_raw_arr[:cut_idx]
            try:
                start_pos_right = float(right_x_raw[0])
                right_x_truncated = (np.array(right_x_raw, dtype=float) - start_pos_right)[:cut_idx]
            except Exception:
                right_x_truncated = right_x_raw[:cut_idx]

            if len(right_x_truncated) > 2:
                y_arr = np.array(right_y_truncated, dtype=float)
                tail_n = min(10, max(3, len(y_arr) // 5))
                baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                y_corr = y_arr - baseline
                pos = y_corr[y_corr > 0]
                if pos.size > 0:
                    floor = max(np.min(pos) * 1e-6, 1e-12)
                else:
                    floor = 1e-12
                logy = np.log(np.maximum(y_corr, floor))

                right_fit = self._fit_linear_on_log(
                    x=right_x_truncated, logy=logy,
                    shift=shift, side="Right", profile_id=profile_id,
                    baseline=baseline
                )
                if right_fit and right_fit.get('r2') is not None:
                    right_candidates.append((start_idx_right, right_fit))

        # If an explicit intersection_idx was provided and lies to the right
        # of the peak, also try a left-side start exactly at the intersection
        # so we have candidates anchored at the detected junction.
        try:
            if intersection_idx is not None and int(intersection_idx) > base_idx and int(intersection_idx) < len(x_vals):
                start_idx_left = int(intersection_idx)
                left_x_raw = x_vals[:start_idx_left + 1]
                y_left_raw = y_vals[:start_idx_left + 1]
                # Use raw data for fitting (no pre-filtering)
                y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
                try:
                    start_pos_left = float(left_x_raw[-1])
                    x_left_flipped = (start_pos_left - np.array(left_x_raw, dtype=float))[::-1]
                except Exception:
                    x_left_flipped = np.abs(left_x_raw)[::-1]
                cut_idx = self._find_snr_end_index(y_left_flipped)
                left_y_truncated = y_left_flipped[:cut_idx]
                left_x_truncated = x_left_flipped[:cut_idx]
                if len(left_x_truncated) > 2:
                    y_arr = np.array(left_y_truncated, dtype=float)
                    tail_n = min(10, max(3, len(y_arr) // 5))
                    baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                    y_corr = y_arr - baseline
                    pos = y_corr[y_corr > 0]
                    if pos.size > 0:
                        floor = max(np.min(pos) * 1e-6, 1e-12)
                    else:
                        floor = 1e-12
                    logy = np.log(np.maximum(y_corr, floor))
                    left_fit = self._fit_linear_on_log(
                        x=left_x_truncated, logy=logy,
                        shift=0, side="Left", profile_id=profile_id,
                        baseline=baseline
                    )
                    if left_fit and left_fit.get('r2') is not None:
                        left_candidates.append((start_idx_left, left_fit))
        except Exception:
            pass
        # If intersection_idx lies to the left of the peak, try a right-side
        # start anchored at the intersection so right-side fits can be
        # evaluated relative to the detected junction.
        try:
            if intersection_idx is not None and int(intersection_idx) < base_idx and int(intersection_idx) >= 0:
                start_idx_right = int(intersection_idx)
                right_x_raw = x_vals[start_idx_right:]
                y_right_raw = y_vals[start_idx_right:]
                # Use raw data for fitting (no pre-filtering)
                right_y_raw_arr = np.array(y_right_raw, dtype=float)
                cut_idx = self._find_snr_end_index(right_y_raw_arr)
                right_y_truncated = right_y_raw_arr[:cut_idx]
                try:
                    start_pos_right = float(right_x_raw[0])
                    right_x_truncated = (np.array(right_x_raw, dtype=float) - start_pos_right)[:cut_idx]
                except Exception:
                    right_x_truncated = right_x_raw[:cut_idx]
                if len(right_x_truncated) > 2:
                    y_arr = np.array(right_y_truncated, dtype=float)
                    tail_n = min(10, max(3, len(y_arr) // 5))
                    baseline = float(max(np.median(y_arr[-tail_n:]), 0.0))
                    y_corr = y_arr - baseline
                    pos = y_corr[y_corr > 0]
                    if pos.size > 0:
                        floor = max(np.min(pos) * 1e-6, 1e-12)
                    else:
                        floor = 1e-12
                    logy = np.log(np.maximum(y_corr, floor))
                    right_fit = self._fit_linear_on_log(
                        x=right_x_truncated, logy=logy,
                        shift=0, side="Right", profile_id=profile_id,
                        baseline=baseline
                    )
                    if right_fit and right_fit.get('r2') is not None:
                        right_candidates.append((start_idx_right, right_fit))
        except Exception:
            pass

        # --- Evaluate tail-fit results for each candidate and select by
        # distance-to-junction (prefer close) but require a minimum R².
        best_left_idx = None
        best_right_idx = None

        # Helper to compute tail-fit summary (best R²) for a candidate
        def _candidate_tail_summary(side, start_idx):
            try:
                if side == 'Left':
                    left_x_raw = x_vals[:start_idx + 1]
                    y_left_raw = y_vals[:start_idx + 1]
                    # Use raw data for fitting (no pre-filtering)
                    y_left_flipped = np.array(y_left_raw, dtype=float)[::-1]
                    x_left_flipped = np.abs(left_x_raw)[::-1]
                    cut_idx = self._find_snr_end_index(y_left_flipped)
                    left_y_truncated = y_left_flipped[:cut_idx]
                    left_x_truncated = x_left_flipped[:cut_idx]
                    tail_results = self._truncate_tail_and_fit_linear(
                        left_x_truncated, left_y_truncated,
                        shift=0, side='Left', profile_id=profile_id,
                        global_x=left_x_raw[::-1][:cut_idx],
                        prevent_cross_peak=prevent_cross_peak,
                        ref=(float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx]))
                    )
                else:
                    right_x_raw = x_vals[start_idx:]
                    y_right_raw = y_vals[start_idx:]
                    # Use raw data for fitting (no pre-filtering)
                    right_y_raw_arr = np.array(y_right_raw, dtype=float)
                    cut_idx = self._find_snr_end_index(right_y_raw_arr)
                    right_y_truncated = right_y_raw_arr[:cut_idx]
                    try:
                        start_pos_right = float(right_x_raw[0])
                        right_x_truncated = (np.array(right_x_raw, dtype=float) - start_pos_right)[:cut_idx]
                    except Exception:
                        right_x_truncated = right_x_raw[:cut_idx]
                    tail_results = self._truncate_tail_and_fit_linear(
                        right_x_truncated, right_y_truncated,
                        shift=0, side='Right', profile_id=profile_id,
                        global_x=right_x_raw[:cut_idx],
                        prevent_cross_peak=prevent_cross_peak,
                        ref=(float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx]))
                    )
                # determine best R² among tail results. Use linear-domain
                # R² (original y vs fitted curve) because log-domain R²
                # can be misleading when baseline/subtraction or floors
                # affect the log transform. Keep log-domain R² as well.
                best_r2 = -np.inf
                if tail_results:
                    for f in tail_results:
                        try:
                            y_true = np.array(f.get('y_vals', []), dtype=float)
                            y_fit = np.array(f.get('fit_curve_local', []), dtype=float)
                            r2_lin = self._calculate_r2(y_true, y_fit)
                        except Exception:
                            r2_lin = -np.inf
                        # store linear r2 for later inspection
                        f['_r2_linear'] = r2_lin
                        f['r2_linear'] = r2_lin
                        # prefer linear-domain R² for selection
                        if r2_lin is not None and r2_lin > best_r2:
                            best_r2 = r2_lin
                return tail_results, best_r2
            except Exception:
                return [], -np.inf

        ref_val = (float(x_vals[intersection_idx]) if intersection_idx is not None else float(x_vals[base_idx]))

        # Evaluate left candidates' tail fits
        left_candidate_summaries = {}
        for start_idx, _ in left_candidates:
            tail_results, best_r2 = _candidate_tail_summary('Left', start_idx)
            left_candidate_summaries[start_idx] = {'tail_results': tail_results, 'best_r2': best_r2}

        # Evaluate right candidates' tail fits
        right_candidate_summaries = {}
        for start_idx, _ in right_candidates:
            tail_results, best_r2 = _candidate_tail_summary('Right', start_idx)
            right_candidate_summaries[start_idx] = {'tail_results': tail_results, 'best_r2': best_r2}

        # Selection policy: prefer nearest-to-junction candidate among those
        # with r2 >= threshold; fallback to highest-r2 candidate if none.
        r2_threshold = 0.85

        if left_candidate_summaries:
            # candidates on left side (start <= ref_val)
            left_filtered = {idx: v for idx, v in left_candidate_summaries.items() if float(x_vals[idx]) - ref_val <= 0}
            if left_filtered:
                good = [(idx, v) for idx, v in left_filtered.items() if v['best_r2'] >= r2_threshold]
                if good:
                    best_left_idx = min(good, key=lambda t: abs(float(x_vals[t[0]]) - ref_val))[0]
                else:
                    # choose candidate with max r2 among left_filtered
                    best_left_idx = max(left_filtered.items(), key=lambda t: t[1]['best_r2'])[0]
            else:
                # fallback: choose best overall left candidate by r2
                best_left_idx = max(left_candidate_summaries.items(), key=lambda t: t[1]['best_r2'])[0]

        if right_candidate_summaries:
            right_filtered = {idx: v for idx, v in right_candidate_summaries.items() if float(x_vals[idx]) - ref_val >= 0}
            if right_filtered:
                good = [(idx, v) for idx, v in right_filtered.items() if v['best_r2'] >= r2_threshold]
                if good:
                    best_right_idx = min(good, key=lambda t: abs(float(x_vals[t[0]]) - ref_val))[0]
                else:
                    best_right_idx = max(right_filtered.items(), key=lambda t: t[1]['best_r2'])[0]
            else:
                best_right_idx = max(right_candidate_summaries.items(), key=lambda t: t[1]['best_r2'])[0]

        # Append the selected candidates' tail-fit results to final results
        if best_left_idx is not None:
            results.extend(left_candidate_summaries.get(best_left_idx, {}).get('tail_results', []))
        if best_right_idx is not None:
            results.extend(right_candidate_summaries.get(best_right_idx, {}).get('tail_results', []))

        # tail-fitting results for selected candidates were already
        # computed above and appended to `results`.

        # --- Visualization: only tail truncations ---
        if plot_left and best_left_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Left', base_idx, profile_id)
        if plot_right and best_right_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Right', base_idx, profile_id)

        return results

    def fit_profile_sides_plateau_based(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                        gradient_window=9, min_plateau_length=5, 
                                        derivative_threshold=0.2, absolute_threshold=0.05):
        """
        NEW METHOD: Fit linear regions using derivative plateau detection.
        
        This method:
        1. Computes d(ln(I))/dx using windowed gradient
        2. Detects plateaus in the derivative (constant deriv = linear region in log)
        3. Fits straight lines to ln(I) within the detected plateau regions
        4. Extracts depletion width from the gap between left and right plateaus
        
        Parameters
        ----------
        gradient_window : int
            Window size for gradient computation (must be odd)
        min_plateau_length : int
            Minimum number of points to consider a plateau
        derivative_threshold : float
            Maximum relative variation in derivative (as fraction of median) for plateau detection
        absolute_threshold : float
            Maximum absolute variation in derivative (1/µm) for plateau detection
            Ensures robust detection even when derivative is small
        
        Returns
        -------
        results : list
            List with best_left and best_right fit dicts
        """
        from .perpendicular import gradient_with_window
        
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        
        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx
        
        results = []
        
        # Prepare log-transformed data. Use the RAW ln(I) for derivative
        # computation and plateau detection — do not use any filtered
        # current for depletion-region detection per user request.
        pos = y_vals[y_vals > 0]
        floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
        y_safe = np.maximum(y_vals, floor)
        ln_y = np.log(y_safe)

        # Compute derivative d(ln(I))/dx using windowed gradient on the
        # RAW ln(I) (no filtering).
        try:
            dlnI_dx = gradient_with_window(x_vals, ln_y, window=gradient_window)
        except Exception as e:
            print(f"Profile {profile_id}: Failed to compute gradient on raw ln: {e}")
            return results
        
        # --- LEFT SIDE: Detect plateau from junction moving leftward ---
        left_x = x_vals[:intersection_idx + 1]
        left_ln_y = ln_y[:intersection_idx + 1]
        left_deriv = dlnI_dx[:intersection_idx + 1]
        
        # Flip to search from tail inward
        left_x_flip = left_x[::-1]
        left_ln_y_flip = left_ln_y[::-1]
        left_deriv_flip = left_deriv[::-1]
        
        plat_start_l, plat_end_l, mean_deriv_l = self._detect_plateau_in_derivative(
            left_x_flip, left_deriv_flip, 
            min_plateau_length=min_plateau_length,
            derivative_threshold=derivative_threshold,
            absolute_threshold=absolute_threshold,
            search_from_end=False  # search from tail inward
        )
        
        if plat_start_l is not None and plat_end_l is not None:
            # Extract plateau region
            x_plateau_l = left_x_flip[plat_start_l:plat_end_l]
            ln_y_plateau_l = left_ln_y_flip[plat_start_l:plat_end_l]
            y_plateau_l = np.exp(ln_y_plateau_l)
            
            # Validate: Check that plateau is not in the noise floor
            peak_signal = np.max(y_vals)
            plateau_mean_signal = np.mean(y_plateau_l)
            signal_ratio = plateau_mean_signal / peak_signal
            
            # Reject if plateau is too weak (< 10% of peak signal = likely noise floor)
            if signal_ratio < 0.10:
                # print(f"Profile {profile_id} Left: Rejected plateau at noise floor (signal={signal_ratio:.1%} of peak)")
                plat_start_l, plat_end_l = None, None
            
            # Fit straight line to ln(I) in the plateau
            if plat_start_l is not None and len(x_plateau_l) >= 2:
                p = np.polyfit(x_plateau_l, ln_y_plateau_l, 1)
                slope_log = float(p[0])
                intercept_log = float(p[1])
                
                # Compute fit curve in log-space and linear space
                ln_fit = np.polyval(p, x_plateau_l)
                fit_curve_local = np.exp(ln_fit)
                
                # Compute R²
                r2 = self._calculate_r2(ln_y_plateau_l, ln_fit)
                
                # Characteristic length
                inv_lambda_um = np.inf if slope_log == 0 else 1.0 / abs(slope_log)
                pixel_size_um = self.pixel_size * 1e6
                if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                    inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
                else:
                    inv_lambda = inv_lambda_um
                
                # Map back to original profile coordinates (un-flip)
                # Original indices: [0, 1, ..., intersection_idx]
                # Flipped indices: [intersection_idx, ..., 1, 0]
                # Plateau in flipped: [plat_start_l:plat_end_l]
                # Map back: intersection_idx - plat_end_l + 1, intersection_idx - plat_start_l + 1
                orig_start = intersection_idx - plat_end_l + 1
                orig_end = intersection_idx - plat_start_l + 1
                global_x_vals = x_vals[orig_start:orig_end]
                
                # Evaluate the fit at the GLOBAL x coordinates for proper plotting
                # The fit was done on flipped x (x_plateau_l), but we need to evaluate
                # it at the original x coordinates (global_x_vals) for plotting
                ln_fit_global = np.polyval(p, global_x_vals[::-1])[::-1]  # Flip to match, then flip back
                fit_curve_global = np.exp(ln_fit_global)
                
                results.append({
                    'side': 'Left (plateau)',
                    'shift': 0,
                    'x_vals': x_plateau_l,
                    'y_vals': y_plateau_l,
                    'fit_curve': fit_curve_global,  # Use global-coordinate fit for plotting
                    'fit_curve_local': fit_curve_local,
                    'ln_fit_curve': ln_fit_global,  # Store log-space fit at global coords for plotting
                    'global_x_vals': global_x_vals,
                    'parameters': (slope_log, intercept_log),
                    'slope': slope_log,
                    'intercept': intercept_log,
                    'inv_lambda': inv_lambda,
                    'r2': r2,
                    'plateau_indices': (orig_start, orig_end)
                })
                # print(f"Profile {profile_id} Left: Detected plateau from index {orig_start} to {orig_end}, "
                #       f"slope={slope_log:.3g}, R²={r2:.3f}, inv_lambda={inv_lambda:.3g} µm")
        else:
            pass  # print(f"Profile {profile_id}: No left plateau detected")
        
        # --- RIGHT SIDE: Detect plateau from junction moving rightward ---
        right_x = x_vals[intersection_idx:]
        right_ln_y = ln_y[intersection_idx:]
        right_deriv = dlnI_dx[intersection_idx:]
        
        # Limit search to avoid far tail noise floor (use first 70% of data)
        search_limit = max(min_plateau_length + 5, int(len(right_x) * 0.7))
        right_x_limited = right_x[:search_limit]
        right_ln_y_limited = right_ln_y[:search_limit]
        right_deriv_limited = right_deriv[:search_limit]
        
        plat_start_r, plat_end_r, mean_deriv_r = self._detect_plateau_in_derivative(
            right_x_limited, right_deriv_limited,
            min_plateau_length=min_plateau_length,
            derivative_threshold=derivative_threshold,
            absolute_threshold=absolute_threshold,
            search_from_end=False  # search from junction outward to avoid tail noise
        )
        
        if plat_start_r is not None and plat_end_r is not None:
            # Extract plateau region
            x_plateau_r = right_x[plat_start_r:plat_end_r]
            ln_y_plateau_r = right_ln_y[plat_start_r:plat_end_r]
            y_plateau_r = np.exp(ln_y_plateau_r)
            
            # Validate: Check that plateau is not in the noise floor
            # Compare plateau signal level to peak signal
            peak_signal = np.max(y_vals)
            plateau_mean_signal = np.mean(y_plateau_r)
            signal_ratio = plateau_mean_signal / peak_signal
            
            # Reject if plateau is too weak (< 10% of peak signal = likely noise floor)
            if signal_ratio < 0.10:
                # print(f"Profile {profile_id} Right: Rejected plateau at noise floor (signal={signal_ratio:.1%} of peak)")
                plat_start_r, plat_end_r = None, None
            
            # Fit straight line to ln(I) in the plateau
            if plat_start_r is not None and len(x_plateau_r) >= 2:
                # Transform to local coordinates (starting from 0)
                x_local_r = x_plateau_r - x_plateau_r[0]
                p = np.polyfit(x_local_r, ln_y_plateau_r, 1)
                slope_log = float(p[0])
                intercept_log = float(p[1])
                
                # Compute fit curve in log-space and linear space
                ln_fit = np.polyval(p, x_local_r)
                fit_curve_local = np.exp(ln_fit)
                
                # Compute R²
                r2 = self._calculate_r2(ln_y_plateau_r, ln_fit)
                
                # Characteristic length
                inv_lambda_um = np.inf if slope_log == 0 else 1.0 / abs(slope_log)
                pixel_size_um = self.pixel_size * 1e6
                if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                    inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
                else:
                    inv_lambda = inv_lambda_um
                
                # Map to original profile coordinates
                orig_start = intersection_idx + plat_start_r
                orig_end = intersection_idx + plat_end_r
                global_x_vals = x_vals[orig_start:orig_end]
                
                # Evaluate the fit at the GLOBAL x coordinates for proper plotting
                # Convert global x to local coordinates relative to plateau start
                x_local_global = global_x_vals - global_x_vals[0]
                ln_fit_global = np.polyval(p, x_local_global)
                fit_curve_global = np.exp(ln_fit_global)
                
                results.append({
                    'side': 'Right (plateau)',
                    'shift': 0,
                    'x_vals': x_local_r,
                    'y_vals': y_plateau_r,
                    'fit_curve': fit_curve_global,  # Use global-coordinate fit for plotting
                    'fit_curve_local': fit_curve_local,
                    'ln_fit_curve': ln_fit_global,  # Store log-space fit at global coords for plotting
                    'global_x_vals': global_x_vals,
                    'parameters': (slope_log, intercept_log),
                    'slope': slope_log,
                    'intercept': intercept_log,
                    'inv_lambda': inv_lambda,
                    'r2': r2,
                    'plateau_indices': (orig_start, orig_end)
                })
                # print(f"Profile {profile_id} Right: Detected plateau from index {orig_start} to {orig_end}, "
                #       f"slope={slope_log:.3g}, R²={r2:.3f}, inv_lambda={inv_lambda:.3g} µm")
        else:
            pass  # print(f"Profile {profile_id}: No right plateau detected")
        
        return results

    def fit_profile_sides_plateau_with_shift(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                             gradient_window=3, min_plateau_length=5, 
                                             derivative_threshold=0.4, absolute_threshold=0.1,
                                             max_expansion=500, consecutive_drops=10):
        """
        HYBRID METHOD: Combines plateau detection with iterative expansion.
        
        This method:
        1. Uses plateau detection to find initial linear regions
        2. Iteratively expands boundaries outward as long as R² improves
        3. Tolerates noise by requiring consecutive_drops decreases before stopping
        4. Returns optimized linear fits with maximum valid extent
        
        Parameters
        ----------
        shift_range : int
            Initial number of pixels to shift for quick refinement (default: 5)
        max_expansion : int
            Maximum number of pixels to expand beyond initial plateau (default: 30)
        consecutive_drops : int
            Number of consecutive R² decreases before stopping expansion (default: 3)
        
        Other parameters same as fit_profile_sides_plateau_based
        """
        from .perpendicular import gradient_with_window
        
        x_vals = np.asarray(x_vals, dtype=float)
        y_vals = np.asarray(y_vals, dtype=float)
        
        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx
        
        results = []
        
        # Prepare log-transformed data
        pos = y_vals[y_vals > 0]
        floor = max(np.min(pos) * 0.1, 1e-12) if pos.size > 0 else 1e-12
        y_safe = np.maximum(y_vals, floor)
        ln_y = np.log(y_safe)

        # Compute derivative
        try:
            dlnI_dx = gradient_with_window(x_vals, ln_y, window=gradient_window)
        except Exception as e:
            print(f"Profile {profile_id}: Failed to compute gradient: {e}")
            return results
        
        # --- LEFT SIDE: Detect initial plateau ---
        left_x = x_vals[:intersection_idx + 1]
        left_ln_y = ln_y[:intersection_idx + 1]
        left_deriv = dlnI_dx[:intersection_idx + 1]
        
        left_x_flip = left_x[::-1]
        left_ln_y_flip = left_ln_y[::-1]
        left_deriv_flip = left_deriv[::-1]
        
        plat_start_l, plat_end_l, mean_deriv_l = self._detect_plateau_in_derivative(
            left_x_flip, left_deriv_flip, 
            min_plateau_length=min_plateau_length,
            derivative_threshold=derivative_threshold,
            absolute_threshold=absolute_threshold,
            search_from_end=False
        )
        
        if plat_start_l is not None and plat_end_l is not None:
            # Map back to original indices
            orig_start = intersection_idx - plat_end_l + 1
            orig_end = intersection_idx - plat_start_l + 1
            
            # Helper function to fit a region and return R²
            def fit_region_left(start_idx, end_idx):
                if end_idx - start_idx < min_plateau_length:
                    return None
                x_test = x_vals[start_idx:end_idx]
                ln_y_test = ln_y[start_idx:end_idx]
                y_test = np.exp(ln_y_test)
                
                # Check signal level
                peak_signal = np.max(y_vals)
                test_mean_signal = np.mean(y_test)
                if test_mean_signal / peak_signal < 0.10:
                    return None
                
                try:
                    p = np.polyfit(x_test, ln_y_test, 1)
                    ln_fit = np.polyval(p, x_test)
                    r2 = self._calculate_r2(ln_y_test, ln_fit)
                    return {'start': start_idx, 'end': end_idx, 'r2': r2, 
                            'x': x_test, 'ln_y': ln_y_test, 'y': y_test,
                            'slope': float(p[0]), 'intercept': float(p[1]), 'ln_fit': ln_fit}
                except Exception:
                    return None
            
            # Start with plateau region
            best_fit_l = fit_region_left(orig_start, orig_end)
            if best_fit_l is None:
                best_fit_l = None
            else:
                best_r2_l = best_fit_l['r2']
                
                # Iteratively expand both boundaries
                drops_start = 0
                drops_end = 0
                
                # Expand toward tail (decrease start_idx) while R² improves
                for expansion in range(1, max_expansion + 1):
                    new_start = max(0, orig_start - expansion)
                    if new_start == best_fit_l['start']:  # Can't expand further
                        break
                    
                    candidate = fit_region_left(new_start, best_fit_l['end'])
                    if candidate is None:
                        drops_start += 1
                    elif candidate['r2'] > best_r2_l:
                        # Improvement! Update best fit and reset counter
                        best_fit_l = candidate
                        best_r2_l = candidate['r2']
                        drops_start = 0
                    else:
                        # R² decreased
                        drops_start += 1
                    
                    if drops_start >= consecutive_drops:
                        break
                
                # Expand toward junction (increase end_idx) while R² improves
                for expansion in range(1, max_expansion + 1):
                    new_end = min(intersection_idx + 1, best_fit_l['end'] + expansion)
                    if new_end == best_fit_l['end']:  # Can't expand further
                        break
                    
                    candidate = fit_region_left(best_fit_l['start'], new_end)
                    if candidate is None:
                        drops_end += 1
                    elif candidate['r2'] > best_r2_l:
                        # Improvement! Update best fit and reset counter
                        best_fit_l = candidate
                        best_r2_l = candidate['r2']
                        drops_end = 0
                    else:
                        # R² decreased
                        drops_end += 1
                    
                    if drops_end >= consecutive_drops:
                        break
                
                # Build final fit dict
                if best_fit_l is not None:
                    slope_log = best_fit_l['slope']
                    intercept_log = best_fit_l['intercept']
                    inv_lambda_um = np.inf if slope_log == 0 else 1.0 / abs(slope_log)
                    pixel_size_um = self.pixel_size * 1e6
                    if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                        inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
                    else:
                        inv_lambda = inv_lambda_um
                    
                    fit_curve_global = np.exp(best_fit_l['ln_fit'])
                    
                    best_fit_l = {
                        'side': 'Left (plateau+expand)',
                        'shift': 0,
                        'x_vals': best_fit_l['x'] - best_fit_l['x'][-1],
                        'y_vals': best_fit_l['y'],
                        'fit_curve': fit_curve_global,
                        'fit_curve_local': fit_curve_global,
                        'ln_fit_curve': best_fit_l['ln_fit'],
                        'global_x_vals': best_fit_l['x'],
                        'parameters': (slope_log, intercept_log),
                        'slope': slope_log,
                        'intercept': intercept_log,
                        'inv_lambda': inv_lambda,
                        'r2': best_r2_l,
                        'plateau_indices': (best_fit_l['start'], best_fit_l['end'])
                    }
                    results.append(best_fit_l)
        
        # --- RIGHT SIDE: Detect initial plateau and shift ---
        right_x = x_vals[intersection_idx:]
        right_ln_y = ln_y[intersection_idx:]
        right_deriv = dlnI_dx[intersection_idx:]
        
        search_limit = max(min_plateau_length + 5, int(len(right_x) * 0.7))
        right_x_limited = right_x[:search_limit]
        right_ln_y_limited = right_ln_y[:search_limit]
        right_deriv_limited = right_deriv[:search_limit]
        
        plat_start_r, plat_end_r, mean_deriv_r = self._detect_plateau_in_derivative(
            right_x_limited, right_deriv_limited,
            min_plateau_length=min_plateau_length,
            derivative_threshold=derivative_threshold,
            absolute_threshold=absolute_threshold,
            search_from_end=False
        )
        
        if plat_start_r is not None and plat_end_r is not None:
            # Map to original indices
            orig_start = intersection_idx + plat_start_r
            orig_end = intersection_idx + plat_end_r
            
            # Helper function to fit a region and return R²
            def fit_region_right(start_idx, end_idx):
                if end_idx - start_idx < min_plateau_length:
                    return None
                x_test = x_vals[start_idx:end_idx]
                ln_y_test = ln_y[start_idx:end_idx]
                y_test = np.exp(ln_y_test)
                
                # Check signal level
                peak_signal = np.max(y_vals)
                test_mean_signal = np.mean(y_test)
                if test_mean_signal / peak_signal < 0.10:
                    return None
                
                try:
                    x_local = x_test - x_test[0]
                    p = np.polyfit(x_local, ln_y_test, 1)
                    ln_fit = np.polyval(p, x_local)
                    r2 = self._calculate_r2(ln_y_test, ln_fit)
                    return {'start': start_idx, 'end': end_idx, 'r2': r2,
                            'x': x_test, 'x_local': x_local, 'ln_y': ln_y_test, 'y': y_test,
                            'slope': float(p[0]), 'intercept': float(p[1]), 'ln_fit': ln_fit}
                except Exception:
                    return None
            
            # Start with plateau region
            best_fit_r = fit_region_right(orig_start, orig_end)
            if best_fit_r is None:
                best_fit_r = None
            else:
                best_r2_r = best_fit_r['r2']
                
                # Iteratively expand both boundaries
                drops_start = 0
                drops_end = 0
                
                # Expand toward junction (decrease start_idx) while R² improves
                for expansion in range(1, max_expansion + 1):
                    new_start = max(intersection_idx, best_fit_r['start'] - expansion)
                    if new_start == best_fit_r['start']:  # Can't expand further
                        break
                    
                    candidate = fit_region_right(new_start, best_fit_r['end'])
                    if candidate is None:
                        drops_start += 1
                    elif candidate['r2'] > best_r2_r:
                        # Improvement! Update best fit and reset counter
                        best_fit_r = candidate
                        best_r2_r = candidate['r2']
                        drops_start = 0
                    else:
                        # R² decreased
                        drops_start += 1
                    
                    if drops_start >= consecutive_drops:
                        break
                
                # Expand toward tail (increase end_idx) while R² improves
                for expansion in range(1, max_expansion + 1):
                    new_end = min(len(x_vals), best_fit_r['end'] + expansion)
                    if new_end == best_fit_r['end']:  # Can't expand further
                        break
                    
                    candidate = fit_region_right(best_fit_r['start'], new_end)
                    if candidate is None:
                        drops_end += 1
                    elif candidate['r2'] > best_r2_r:
                        # Improvement! Update best fit and reset counter
                        best_fit_r = candidate
                        best_r2_r = candidate['r2']
                        drops_end = 0
                    else:
                        # R² decreased
                        drops_end += 1
                    
                    if drops_end >= consecutive_drops:
                        break
                
                # Build final fit dict
                if best_fit_r is not None:
                    slope_log = best_fit_r['slope']
                    intercept_log = best_fit_r['intercept']
                    inv_lambda_um = np.inf if slope_log == 0 else 1.0 / abs(slope_log)
                    pixel_size_um = self.pixel_size * 1e6
                    if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                        inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)
                    else:
                        inv_lambda = inv_lambda_um
                    
                    fit_curve_global = np.exp(best_fit_r['ln_fit'])
                    
                    best_fit_r = {
                        'side': 'Right (plateau+expand)',
                        'shift': 0,
                        'x_vals': best_fit_r['x_local'],
                        'y_vals': best_fit_r['y'],
                        'fit_curve': fit_curve_global,
                        'fit_curve_local': fit_curve_global,
                        'ln_fit_curve': best_fit_r['ln_fit'],
                        'global_x_vals': best_fit_r['x'],
                        'parameters': (slope_log, intercept_log),
                        'slope': slope_log,
                        'intercept': intercept_log,
                        'inv_lambda': inv_lambda,
                        'r2': best_r2_r,
                        'plateau_indices': (best_fit_r['start'], best_fit_r['end'])
                    }
                    results.append(best_fit_r)
        
        return results

    def fit_profile_sides_iterative_linear(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                           plot_left=False, plot_right=False, max_iter=10, tol_factor=1.0):
        """
        Iterative linear-on-log version of fit_profile_sides_iterative.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        inv_lambda_old_left = None
        inv_lambda_old_right = None
        results = []

        for iteration in range(max_iter):
            results = self.fit_profile_sides_linear(
                x_vals, y_vals,
                intersection_idx=intersection_idx,
                profile_id=profile_id,
                plot_left=plot_left,
                plot_right=plot_right
            )

            inv_left = [r['inv_lambda'] for r in results if "Left-tailcut" in r['side']]
            inv_right = [r['inv_lambda'] for r in results if "Right-tailcut" in r['side']]

            inv_left = inv_left[-1] if inv_left else None
            inv_right = inv_right[-1] if inv_right else None

            converged_left = (
                inv_left is not None
                and inv_lambda_old_left is not None
                and abs(inv_left - inv_lambda_old_left) <= self.pixel_size * tol_factor
            )
            converged_right = (
                inv_right is not None
                and inv_lambda_old_right is not None
                and abs(inv_right - inv_lambda_old_right) <= self.pixel_size * tol_factor
            )

            if converged_left and converged_right:
                print(f"Profile {profile_id}: linear fits converged after {iteration+1} iterations.")
                break

            inv_lambda_old_left = inv_left
            inv_lambda_old_right = inv_right

        return results

    def fit_all_profiles_linear(self, use_plateau_detection=True, use_shifting=False,
                                 gradient_window=9, min_plateau_length=5,
                                 derivative_threshold=0.3, absolute_threshold=0.03,
                                 max_expansion=30, consecutive_drops=3):
        """
        Run linear-on-log fitting for all loaded profiles and populate self.results
        similarly to fit_all_profiles.
        
        Parameters
        ----------
        use_plateau_detection : bool
            If True, use derivative plateau detection (NEW METHOD - recommended)
            If False, use the old iterative window search method
        use_shifting : bool
            If True, use hybrid plateau+shifting method to refine fits
            Only applies when use_plateau_detection=True
        gradient_window : int
            Window size for gradient computation in plateau detection
        min_plateau_length : int
            Minimum number of points for a valid plateau
        derivative_threshold : float
            Maximum relative variation in derivative for plateau detection
        absolute_threshold : float
            Maximum absolute variation in derivative (1/µm)
        max_expansion : int
            Maximum number of pixels to expand beyond initial plateau (default: 30)
        consecutive_drops : int
            Number of consecutive R² decreases before stopping expansion (default: 3)
        """
        self.results = []
        self.inv_lambdas = []
        self.central_inv_lambdas = []
        
        # Store parameters for use in plotting
        self.plateau_params = {
            'gradient_window': gradient_window,
            'min_plateau_length': min_plateau_length,
            'derivative_threshold': derivative_threshold,
            'absolute_threshold': absolute_threshold
        }

        for i, prof in enumerate(self.profiles):
            intersection_idx = prof.get('intersection_idx', None)
            x_vals = np.array(prof['dist_um'])
            y_vals = np.array(prof['current'])

            if use_plateau_detection and use_shifting:
                # HYBRID: Use plateau detection + iterative expansion for refinement
                sides = self.fit_profile_sides_plateau_with_shift(
                    x_vals, y_vals,
                    intersection_idx=intersection_idx,
                    profile_id=i + 1,
                    gradient_window=gradient_window,
                    min_plateau_length=min_plateau_length,
                    derivative_threshold=derivative_threshold,
                    absolute_threshold=absolute_threshold,
                    max_expansion=max_expansion,
                    consecutive_drops=consecutive_drops
                )
            elif use_plateau_detection:
                # NEW: Use derivative plateau detection only
                sides = self.fit_profile_sides_plateau_based(
                    x_vals, y_vals,
                    intersection_idx=intersection_idx,
                    profile_id=i + 1,
                    gradient_window=gradient_window,
                    min_plateau_length=min_plateau_length,
                    derivative_threshold=derivative_threshold,
                    absolute_threshold=absolute_threshold
                )
            else:
                # OLD: Use iterative window search
                sides = self.fit_profile_sides_iterative_linear(
                    x_vals, y_vals,
                    intersection_idx=intersection_idx,
                    profile_id=i + 1
                )

            depletion_info = self._extract_depletion_region(
                sides, x_vals, np.argmax(y_vals)
            )

            self.results.append({
                'Profile': i + 1,
                'fit_sides': sides,
                'depletion': depletion_info
            })
