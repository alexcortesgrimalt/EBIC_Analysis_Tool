import matplotlib.pyplot as plt
import os
import numpy as np  
from scipy.optimize import curve_fit


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

    def fit_profile_sides(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                          plot_left=True, plot_right=True):
        """
        Fit falling exponentials on both sides:
        1. Find the correct starting point by shifting until 1/λ stabilizes.
        2. With the starting point fixed, progressively truncate the tail
           and fit each case.
        3. Visualize only the tail truncation results.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx

        # allow overlap up to the profile maximum: permit the left-side start to
        # move rightwards up to the profile peak (base_idx) and the right-side
        # start to move leftwards down to the peak. This ensures we can begin
        # fits closer to the true peak when the intersection is offset.
        n = len(x_vals)
        relative = int(base_idx - intersection_idx)
        # positive allowed shift for left side (how far right we can go)
        pos_allow = max(0, relative)
        pos_allow = min(pos_allow, n - 1)
        # negative allowed shift for right side (how far left we can go)
        neg_allow = min(0, relative)
        neg_allow = max(neg_allow, -(n - 1))

        results = []
        tolerance = self.pixel_size * 1e6

        # --- LEFT side: find start index ---
        prev_left_val, left_stable, best_left_idx = None, False, None
        # try shifts: allow starting to the right up to the peak (pos_allow),
        # then move leftwards (negative shifts) to search for stable head.
        for shift in range(pos_allow, -16, -1):
            if left_stable:
                break
            start_idx_left = max(0, (intersection_idx if intersection_idx <= base_idx else base_idx) + shift)
            left_x_raw = x_vals[:start_idx_left + 1]
            y_left_raw = y_vals[:start_idx_left + 1]

            y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
            y_left_flipped = y_left_filtered[::-1]
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
                    if curr_val is not None:
                        if prev_left_val is not None and abs(curr_val - prev_left_val) <= tolerance:
                            left_stable, best_left_idx = True, start_idx_left
                        prev_left_val = curr_val

        # --- RIGHT side: find start index ---
        prev_right_val, right_stable, best_right_idx = None, False, None
        # try shifts: allow starting to the left down to the peak (neg_allow),
        # then move rightwards (positive shifts) to search for stable head.
        for shift in range(neg_allow, 16):
            if right_stable:
                break
            start_idx_right = min(len(x_vals) - 1,
                                  (intersection_idx if intersection_idx >= base_idx else base_idx) + shift)
            right_x_raw = x_vals[start_idx_right:]
            y_right_raw = y_vals[start_idx_right:]

            y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)
            cut_idx = self._find_snr_end_index(y_right_filtered)
            right_y_truncated = y_right_filtered[:cut_idx]
            right_x_truncated = right_x_raw[:cut_idx]

            if len(right_x_truncated) > 2:
                right_fit = self._fit_falling(
                    x=right_x_truncated, y=right_y_truncated,
                    shift=shift, side="Right", profile_id=profile_id
                )
                if right_fit:
                    curr_val = right_fit.get('inv_lambda', None)
                    if curr_val is not None:
                        if prev_right_val is not None and abs(curr_val - prev_right_val) <= tolerance:
                            right_stable, best_right_idx = True, start_idx_right
                        prev_right_val = curr_val

        # --- Tail truncations with fixed start indices ---
        if best_left_idx is not None:
            left_x_raw = x_vals[:best_left_idx + 1]
            y_left_raw = y_vals[:best_left_idx + 1]
            y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
            y_left_flipped = y_left_filtered[::-1]
            x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            results.extend(self._truncate_tail_and_fit(
                left_x_truncated, left_y_truncated,
                shift=0, side="Left", profile_id=profile_id
            ))

        if best_right_idx is not None:
            right_x_raw = x_vals[best_right_idx:]
            y_right_raw = y_vals[best_right_idx:]
            y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)

            cut_idx = self._find_snr_end_index(y_right_filtered)
            right_y_truncated = y_right_filtered[:cut_idx]
            right_x_truncated = right_x_raw[:cut_idx]

            results.extend(self._truncate_tail_and_fit(
                right_x_truncated, right_y_truncated,
                shift=0, side="Right", profile_id=profile_id
            ))

        # --- Visualization: only tail truncations ---
        if plot_left and best_left_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Left', base_idx, profile_id)
        if plot_right and best_right_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Right', base_idx, profile_id)

        return results

    def _plot_shifted_fits(self, results, x_vals, y_vals, side_to_plot, base_idx, profile_id):
        """Helper to create the overlay plots."""
        plt.figure(figsize=(8, 5))
        plt.title(f"Profile {profile_id} - {side_to_plot} shifted fits")

        # Determine the base data for the plot
        if side_to_plot == 'Left':
            x_base, y_base = x_vals[:base_idx + 1], y_vals[:base_idx + 1]
        else:
            x_base, y_base = x_vals[base_idx:], y_vals[base_idx:]

        plt.plot(x_base, y_base, color='0.6', alpha=0.7, label=f'raw {side_to_plot.lower()}')
        y_filtered_base = self.apply_low_pass_filter(y_base, visualize=False)
        plt.plot(x_base, y_filtered_base, 'k-', lw=1.5, label=f'filtered {side_to_plot.lower()}')

        for res in results:
            if res['side'].startswith(side_to_plot):
                plt.plot(res['x_vals'], res['fit_curve'], lw=1.2,
                         label=f"shift {res['shift']} (R²={res['r2']:.2f})")

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

        left_start = -best_left['x_vals'][0] if best_left is not None else None
        right_start = best_right['x_vals'][0] if best_right is not None else None

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

            x = np.array(self.profiles[profile_id - 1]['dist_um'])
            y = np.array(self.profiles[profile_id - 1]['current'])

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(x, y, 'k-', alpha=0.6, label="EBIC Profile")

            left_start = depletion.get('left_start', None)
            right_start = depletion.get('right_start', None)

            # Shade depletion region if both edges present
            if left_start is not None and right_start is not None:
                ax.axvspan(left_start, right_start, color='green', alpha=0.12, label='Depletion zone')

            # Draw vertical lines
            if left_start is not None:
                ax.axvline(left_start, color='b', linestyle='--', label="Left Start")
            if right_start is not None:
                ax.axvline(right_start, color='r', linestyle='--', label="Right Start")

            # Overlay best-fit curves if available
            best_left = depletion.get('best_left_fit', None)
            best_right = depletion.get('best_right_fit', None)

            if best_left is not None:
                # best_left['x_vals'] are in positive distances measured from the fit side
                # map them to the profile x-axis: for left side we negate to place on the left
                x_fit_left = -np.array(best_left['x_vals'])
                y_fit_left = np.array(best_left['fit_curve'])
                ax.plot(x_fit_left, y_fit_left, 'b-', lw=1.6, alpha=0.9, label='Left fit (best)')
                # annotate R²
                ax.text(x_fit_left[0], y_fit_left[0], f"R²={best_left['r2']:.2f}", color='b')

            if best_right is not None:
                x_fit_right = np.array(best_right['x_vals'])
                y_fit_right = np.array(best_right['fit_curve'])
                ax.plot(x_fit_right, y_fit_right, 'r-', lw=1.6, alpha=0.9, label='Right fit (best)')
                ax.text(x_fit_right[0], y_fit_right[0], f"R²={best_right['r2']:.2f}", color='r')

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
                out_path = os.path.join(out_dir, f'profile_{profile_id:02d}_depletion.png')
                fig.savefig(out_path, dpi=200, bbox_inches='tight')
            except Exception:
                out_path = None

            # Show (if possible) and close to avoid many open windows
            try:
                plt.show()
            except Exception:
                pass
            plt.close(fig)
            if out_path:
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

            fig_fit, axes_fit = plt.subplots(1, 2, figsize=(12, 4))
            for ax, side in zip(axes_fit, central_fits):
                ax.plot(side['x_vals'], side['y_vals'], 'r.', alpha=0.5, label='Raw EBIC')
                ax.plot(side['x_vals'], side['fit_curve'], 'b-', linewidth=2, label='Central Fit')
                r2 = self._calculate_r2(side['y_vals'], side['fit_curve'])
                ax.set_xlabel("Distance (µm)")
                ax.set_ylabel("Current (nA)")
                ax.set_title(f"{side['side']}\n1/λ={side['inv_lambda']:.3f}, R²={r2:.3f}")
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend()
            fig_fit.tight_layout()
            plt.show()
            plt.close(fig_fit)

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
            ax.plot(x, logy, 'k-', lw=1.2, label=f'{ylabel}')

            # Optionally overlay best-fit transformed to log domain
            best_left = res.get('depletion', {}).get('best_left_fit', None)
            best_right = res.get('depletion', {}).get('best_right_fit', None)
            for b, color, label in ((best_left, 'b', 'Left fit'), (best_right, 'r', 'Right fit')):
                if b is None:
                    continue
                try:
                    x_fit = np.array(b['x_vals'])
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
                        x_fit_plot = -x_fit
                    else:
                        x_fit_plot = x_fit
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
            plt.close(fig)
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
            plt.show()


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

            plt.title("Average Lengths Summary", fontsize=12, pad=12)
            plt.tight_layout()
            plt.show()

        return results

    def _truncate_tail_and_fit(self, x, y, shift, side, profile_id=0):
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

            fit = self._fit_falling(
                x=x_cut, y=y_cut,
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
        if side_to_plot == 'Left':
            x_base, y_base = x_vals[:base_idx + 1], y_vals[:base_idx + 1]
        else:
            x_base, y_base = x_vals[base_idx:], y_vals[base_idx:]

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
                plt.plot(res['x_vals'], res['fit_curve'], lw=1.2,
                         label=f"tailcut {cut_val} (1/λ={res['inv_lambda']:.2f})")

        plt.xlabel("x (pixels)")
        plt.ylabel("Signal")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


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
            # fit linear model to natural log
            logy = np.log(y_safe)
            p = np.polyfit(x, logy, 1)
            slope = float(p[0])
            intercept = float(p[1])
            log_fit = np.polyval(p, x)
            fit_curve = np.exp(log_fit)

            # slope has units 1/µm (x in µm). inv_lambda in µm is 1/|slope|.
            if slope == 0:
                inv_lambda_um = np.inf
            else:
                inv_lambda_um = 1.0 / abs(slope)

            pixel_size_um = self.pixel_size * 1e6
            inv_lambda = inv_lambda_um
            if np.isfinite(inv_lambda_um) and pixel_size_um > 0:
                inv_lambda = max(round(inv_lambda_um / pixel_size_um) * pixel_size_um, pixel_size_um)

            # compute R^2 on log-domain
            ss_res = np.sum((logy - log_fit) ** 2)
            ss_tot = np.sum((logy - np.mean(logy)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

            return {
                'side': f'{side} (shift {shift})',
                'shift': shift,
                'x_vals': x,
                'y_vals': y,
                'fit_curve': fit_curve,
                'parameters': (slope, intercept),
                'slope': slope,
                'intercept': intercept,
                'inv_lambda': inv_lambda,
                'r2': r2
            }
        except Exception as e:
            print(f"{side} linear fit failed (shift {shift}) for profile {profile_id}: {e}")
            return None

    def _truncate_tail_and_fit_linear(self, x, y, shift, side, profile_id=0):
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

            fit = self._fit_linear(
                x=x_cut, y=y_cut,
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
                    results.append(fit)

        return results

    def fit_profile_sides_linear(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                 plot_left=True, plot_right=True):
        """
        Same flow as fit_profile_sides but fits linear models on ln(current) for
        left and right sides. Uses the same shifting and tail-cut truncation logic.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        base_idx = int(np.argmax(y_vals))
        if intersection_idx is None:
            intersection_idx = base_idx

        results = []
        tolerance = self.pixel_size * 1e6

        # --- LEFT side: find start index ---
        prev_left_val, left_stable, best_left_idx = None, False, None
        for shift in range(0, -16, -1):  # try shifts leftwards
            if left_stable:
                break
            start_idx_left = max(0, (intersection_idx if intersection_idx <= base_idx else base_idx) + shift)
            left_x_raw = x_vals[:start_idx_left + 1]
            y_left_raw = y_vals[:start_idx_left + 1]

            y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
            y_left_flipped = y_left_filtered[::-1]
            x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            if len(left_x_truncated) > 2:
                left_fit = self._fit_linear(
                    x=left_x_truncated, y=left_y_truncated,
                    shift=shift, side="Left", profile_id=profile_id
                )
                if left_fit:
                    curr_val = left_fit.get('inv_lambda', None)
                    if curr_val is not None:
                        if prev_left_val is not None and abs(curr_val - prev_left_val) <= tolerance:
                            left_stable, best_left_idx = True, start_idx_left
                        prev_left_val = curr_val

        # --- RIGHT side: find start index ---
        prev_right_val, right_stable, best_right_idx = None, False, None
        for shift in range(0, 16):  # try shifts rightwards
            if right_stable:
                break
            start_idx_right = min(len(x_vals) - 1,
                                  (intersection_idx if intersection_idx >= base_idx else base_idx) + shift)
            right_x_raw = x_vals[start_idx_right:]
            y_right_raw = y_vals[start_idx_right:]

            y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)
            cut_idx = self._find_snr_end_index(y_right_filtered)
            right_y_truncated = y_right_filtered[:cut_idx]
            right_x_truncated = right_x_raw[:cut_idx]

            if len(right_x_truncated) > 2:
                right_fit = self._fit_linear(
                    x=right_x_truncated, y=right_y_truncated,
                    shift=shift, side="Right", profile_id=profile_id
                )
                if right_fit:
                    curr_val = right_fit.get('inv_lambda', None)
                    if curr_val is not None:
                        if prev_right_val is not None and abs(curr_val - prev_right_val) <= tolerance:
                            right_stable, best_right_idx = True, start_idx_right
                        prev_right_val = curr_val

        # --- Tail truncations with fixed start indices ---
        if best_left_idx is not None:
            left_x_raw = x_vals[:best_left_idx + 1]
            y_left_raw = y_vals[:best_left_idx + 1]
            y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)
            y_left_flipped = y_left_filtered[::-1]
            x_left_flipped = np.abs(left_x_raw)[::-1]

            cut_idx = self._find_snr_end_index(y_left_flipped)
            left_y_truncated = y_left_flipped[:cut_idx]
            left_x_truncated = x_left_flipped[:cut_idx]

            results.extend(self._truncate_tail_and_fit_linear(
                left_x_truncated, left_y_truncated,
                shift=0, side="Left", profile_id=profile_id
            ))

        if best_right_idx is not None:
            right_x_raw = x_vals[best_right_idx:]
            y_right_raw = y_vals[best_right_idx:]
            y_right_filtered = self.apply_low_pass_filter(y_right_raw, visualize=False)

            cut_idx = self._find_snr_end_index(y_right_filtered)
            right_y_truncated = y_right_filtered[:cut_idx]
            right_x_truncated = right_x_raw[:cut_idx]

            results.extend(self._truncate_tail_and_fit_linear(
                right_x_truncated, right_y_truncated,
                shift=0, side="Right", profile_id=profile_id
            ))

        # --- Visualization: only tail truncations ---
        if plot_left and best_left_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Left', base_idx, profile_id)
        if plot_right and best_right_idx is not None:
            self._plot_tailcut_fits(results, x_vals, y_vals, 'Right', base_idx, profile_id)

        return results

    def fit_profile_sides_iterative_linear(self, x_vals, y_vals, intersection_idx=None, profile_id=0,
                                           plot_left=True, plot_right=True, max_iter=10, tol_factor=1.0):
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

    def fit_all_profiles_linear(self):
        """
        Run linear-on-log fitting for all loaded profiles and populate self.results
        similarly to fit_all_profiles.
        """
        self.results = []
        self.inv_lambdas = []
        self.central_inv_lambdas = []

        for i, prof in enumerate(self.profiles):
            intersection_idx = prof.get('intersection_idx', None)
            x_vals = np.array(prof['dist_um'])
            y_vals = np.array(prof['current'])

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
