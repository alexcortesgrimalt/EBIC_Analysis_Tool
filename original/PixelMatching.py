"""
This version is the best fitting yet
"""
from scipy.signal import savgol_filter
from tifffile import imread
import glob
import re
import xml.etree.ElementTree as ET
import pandas as pd
from matplotlib import patches
from matplotlib.widgets import Button
from scipy.ndimage import map_coordinates, gaussian_filter
import matplotlib
from matplotlib.lines import Line2D
import os

"""
this version does the fitting but still have problems with overshooting and the tail
"""

matplotlib.use('TkAgg')


# ==================== SEM TIFF PROCESSOR ====================
class Metadata:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.data = self._parse_metadata()

    def _parse_metadata(self):
        ns = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'image': 'http://ns.pointelectronic.com/Image/1.0/',
            'efa': 'http://ns.pointelectronic.com/EFA/1.0/',
        }

        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        desc = root.find('.//rdf:Description', ns)

        def get_nested_value(tag):
            elem = desc.find(tag, ns)
            if elem is not None:
                val = elem.find('rdf:value', ns)
                if val is not None:
                    return float(val.text)
            return 0.0

        return {
            'PixelSizeX': float(desc.findtext('image:PixelSizeX', '1e-6', namespaces=ns)),
            'Contrast': get_nested_value('efa:Contrast'),
            'EffectiveAmpGain': float(desc.findtext('efa:EffectiveAmpGain', '1e6', namespaces=ns)),
            'OutputOffset': get_nested_value('efa:OutputOffset'),
            'InputOffset': get_nested_value('efa:InputOffset'),
            'InverseEnabled': bool(int(desc.findtext('efa:InverseEnabled', '0', namespaces=ns))),
            'BiasEnabled': bool(int(desc.findtext('efa:BiasEnabled', '0', namespaces=ns))),
            'BiasVoltage': get_nested_value('efa:Bias')
        }


class PixelMap:
    def __init__(self, image, metadata: Metadata, output_dir):
        self.data = np.array(image).astype(np.float64)
        self.metadata = metadata
        self.output_dir = output_dir
        self.pixel_size = metadata.data['PixelSizeX']

    def save(self):
        path = os.path.join(self.output_dir, "pixel_map.csv")
        pd.DataFrame(self.data).to_csv(path, header=False, index=False)


class CurrentMap:
    def __init__(self, pixel_map: PixelMap):
        self.metadata = pixel_map.metadata.data
        self.pixel_size = pixel_map.pixel_size
        self.data = self._compute_current(pixel_map.data)

    def _compute_current(self, pixels):
        C = self.metadata['Contrast']
        G = self.metadata['EffectiveAmpGain']
        O = self.metadata['OutputOffset']
        I = self.metadata['InputOffset']
        inv = self.metadata['InverseEnabled']
        bias_enabled = self.metadata['BiasEnabled']
        bias_voltage = self.metadata['BiasVoltage']

        scale = 1  # mV
        offset = -0.5  # mV
        voltage = (pixels / 65535) * scale + offset

        # if bias_enabled:
        #    voltage -= bias_voltage

        if inv:
            return (((voltage - O) / C) + I) / G * -1e9
        else:
            return (((voltage - O) / C) - I) / G * +1e9

    def save_csv(self, folder_path):
        path = os.path.join(folder_path, "current_map.csv")
        pd.DataFrame(self.data).to_csv(path, header=False, index=False)


class SEMTiffProcessor:
    def __init__(self):
        self.tiff_path = ""
        self.output_dir = ""
        self.metadata = None

    def _extract_xmp(self):
        with open(self.tiff_path, "rb") as f:
            content = f.read()
        pattern = re.compile(rb'<\?xpacket begin=.*?\?>.*?<\?xpacket end=.*?\?>', re.DOTALL)
        match = pattern.search(content)
        if not match:
            raise ValueError("No XML metadata found.")
        xmp_bytes = match.group(0)
        xml_str = xmp_bytes.decode("utf-8", errors="ignore").strip()
        xml_path = self.tiff_path.replace('.tif', '_metadata.xml')
        with open(xml_path, "w", encoding="utf-8") as out_file:
            out_file.write('<?xml version="1.0"?>\n' + xml_str)
        return xml_path

    def load_maps(self, output_root, sample_name):
        frame_dirs = sorted(glob.glob(os.path.join(output_root, sample_name, "frame_*")))
        pixel_maps = []
        current_maps = []
        frame_sizes = []  # store (width, height) for each frame
        pixel_size = None

        for frame_dir in frame_dirs:
            pixel_path = os.path.join(frame_dir, "pixel_map.csv")
            current_path = os.path.join(frame_dir, "current_map.csv")

            if not os.path.exists(pixel_path):
                print(f"Warning: pixel_map.csv not found in {frame_dir}")
                pixel_data = None
                width, height = 0, 0
            else:
                pixel_data = np.loadtxt(pixel_path, delimiter=",")
                height, width = pixel_data.shape

            if not os.path.exists(current_path):
                current_data = None
            else:
                current_data = np.loadtxt(current_path, delimiter=",")

            pixel_maps.append(pixel_data)
            current_maps.append(current_data)
            frame_sizes.append((width, height))  # store frame dimensions

            if pixel_size is None and pixel_data is not None:
                pixel_size = self.metadata.data['PixelSizeX']

        return pixel_maps, current_maps, pixel_size, sample_name, frame_sizes

    def process_tiff(self, tiff_path, output_root, sample_name):
        """
        Process a 2-frame TIFF:
        - Frame 1 = pixel map
        - Frame 2 = current map
        Saves them into frame_1/ and frame_2/ folders.
        Returns arrays ready for SEMViewer.
        """
        import os
        from tifffile import imread

        self.tiff_path = tiff_path
        self.output_dir = os.path.join(output_root, sample_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # --- Extract metadata ---
        xml_path = self._extract_xmp()
        self.metadata = Metadata(xml_path)

        # --- Read TIFF ---
        image = imread(tiff_path)

        if image.ndim != 3 or image.shape[0] < 2:
            raise ValueError("Expected a 2-frame TIFF: [pixel_map, current_map]")

        # --- Frame 1: Pixel map ---
        frame1 = image[0]
        frame1_dir = os.path.join(self.output_dir, "frame_1")
        os.makedirs(frame1_dir, exist_ok=True)

        pixel_map = PixelMap(frame1, self.metadata, frame1_dir)
        pixel_map.save()

        # --- Frame 2: Current map ---
        frame2 = image[1]
        frame2_dir = os.path.join(self.output_dir, "frame_2")
        os.makedirs(frame2_dir, exist_ok=True)

        pixel_map2 = PixelMap(frame2, self.metadata, frame2_dir)
        current_map = CurrentMap(pixel_map2)
        current_map.save_csv(frame2_dir)

        # --- Reload using existing loader ---
        return self.load_maps(output_root, sample_name)

    def map_all_pixels(self, dataset_dict, window_size, max_disp, ref):
        """
        Compute pixel correspondences between a reference dataset and all others.

        Args:
            dataset_dict: dict of {name: (pixel_maps, current_maps)} for each dataset
            window_size: window size for SSIM
            max_disp: search radius
            ref: reference dataset name
        """
        from skimage.exposure import match_histograms
        import pickle

        pixel_maps_ref, _ = dataset_dict[ref]
        image_ref = pixel_maps_ref[0]  # frame_1 = pixel map

        mapping_save = {}

        # Histogram match all datasets to reference (only frame_1!)
        matched_dict = {}
        for key, (pix, cur) in dataset_dict.items():
            matched_pix = pix.copy()
            matched_pix[0] = match_histograms(pix[0], image_ref)
            matched_dict[key] = (matched_pix, cur)

        # Compute mapping for each dataset
        for key, (pix, cur) in matched_dict.items():
            if key == ref:
                continue
            image_other = pix[0]  # frame_1 = pixel map
            assert image_other.ndim == 2, f"{key} frame_1 is not 2D, got shape {image_other.shape}"
            mapping = self.map_pixels(image_ref, image_other,
                                      window_size=window_size,
                                      search_radius=max_disp)
            mapping_save[key] = mapping

        # Save result
        with open("data_coordinates.pkl", "wb") as file:
            pickle.dump(mapping_save, file)

        print("Pixel mapping saved to data_coordinates.pkl")
        return mapping_save

    @staticmethod
    def map_pixels(img1, img2, window_size=11, search_radius=15):
        """
        Find pixel correspondences between two images using SSIM.
        Returns dict {(row, col): (row2, col2)}.
        """
        import numpy as np
        from skimage.metrics import structural_similarity as ssim

        assert img1.ndim == 2 and img2.ndim == 2, "Images must be 2D arrays"
        assert img1.shape == img2.shape, "Images must have the same dimensions"
        assert window_size % 2 == 1, "Window size must be odd"

        global_min = min(img1.min(), img2.min())
        global_max = max(img1.max(), img2.max())
        data_range = global_max - global_min
        result_dict = {}

        pad = window_size // 2
        height, width = img1.shape

        img1_padded = np.pad(img1, pad, mode="reflect")
        img2_padded = np.pad(img2, pad, mode="reflect")

        for i in range(height):
            for j in range(width):
                pi, pj = i + pad, j + pad
                window1 = img1_padded[pi - pad:pi + pad + 1, pj - pad:pj + pad + 1]

                best_score = -np.inf
                best_pos = (i, j)

                min_i = max(pad, pi - search_radius)
                max_i = min(img2_padded.shape[0] - pad, pi + search_radius + 1)
                min_j = max(pad, pj - search_radius)
                max_j = min(img2_padded.shape[1] - pad, pj + search_radius + 1)

                for x in range(min_i, max_i):
                    for y in range(min_j, max_j):
                        window2 = img2_padded[x - pad:x + pad + 1, y - pad:y + pad + 1]
                        score = ssim(window1, window2, data_range=data_range,
                                     gaussian_weights=True, win_size=window_size,
                                     use_sample_covariance=False)

                        if score > best_score:
                            best_score = score
                            best_pos = (x - pad, y - pad)

                result_dict[(i, j)] = best_pos

        return result_dict

    def match_line_only(self, image1, image2, line_coords,
                        window_size=11, search_radius=15, step=5):
        """
        Match only the pixels along a single polyline between two images.
        To speed up, only every `step`-th pixel is matched directly, and the
        intermediate positions are linearly interpolated.

        Args:
            image1, image2: 2D numpy arrays
            line_coords: (N,2) array of [x,y] coords
            window_size: SSIM window size (must be odd)
            search_radius: max displacement in pixels
            step: stride for matching (default=5 -> match 1 in 5 pixels)

        Returns:
            dict { (row,col) in image1 : (row2,col2) in image2 }
        """
        from skimage.metrics import structural_similarity as ssim
        import numpy as np
        from scipy.interpolate import interp1d

        pad = window_size // 2
        img1_padded = np.pad(image1, pad, mode="reflect")
        img2_padded = np.pad(image2, pad, mode="reflect")

        # indices to match
        sampled_idx = np.arange(0, len(line_coords), step)
        if sampled_idx[-1] != len(line_coords) - 1:
            sampled_idx = np.append(sampled_idx, len(line_coords) - 1)

        mapping_sampled = {}

        # --- Match only sampled pixels ---
        for idx in sampled_idx:
            x, y = line_coords[idx].astype(int)
            pi, pj = y + pad, x + pad  # row,col
            window1 = img1_padded[pi - pad:pi + pad + 1,
                      pj - pad:pj + pad + 1]

            best_score = -np.inf
            best_pos = (y, x)

            min_i = max(pad, pi - search_radius)
            max_i = min(img2_padded.shape[0] - pad, pi + search_radius + 1)
            min_j = max(pad, pj - search_radius)
            max_j = min(img2_padded.shape[1] - pad, pj + search_radius + 1)

            for i in range(min_i, max_i):
                for j in range(min_j, max_j):
                    window2 = img2_padded[i - pad:i + pad + 1,
                              j - pad:j + pad + 1]
                    score = ssim(window1, window2,
                                 data_range=image2.max() - image2.min(),
                                 gaussian_weights=True,
                                 win_size=window_size,
                                 use_sample_covariance=False)
                    if score > best_score:
                        best_score = score
                        best_pos = (i - pad, j - pad)

            mapping_sampled[idx] = best_pos  # store with index

        # --- Interpolate skipped pixels ---
        all_rows = []
        all_cols = []
        idxs = sorted(mapping_sampled.keys())
        rows = [mapping_sampled[i][0] for i in idxs]
        cols = [mapping_sampled[i][1] for i in idxs]

        f_row = interp1d(idxs, rows, kind="linear")
        f_col = interp1d(idxs, cols, kind="linear")

        mapping = {}
        for i, (x, y) in enumerate(line_coords.astype(int)):
            row2 = float(f_row(i)) if i >= idxs[0] and i <= idxs[-1] else rows[-1]
            col2 = float(f_col(i)) if i >= idxs[0] and i <= idxs[-1] else cols[-1]
            mapping[(y, x)] = (int(round(row2)), int(round(col2)))

        return mapping

    def match_junction_and_perps(self, image1, image2, junction_line, perpendiculars,
                                 window_size=11, search_radius=10):
        """
        Match junction line and perpendicular lines separately.
        Returns:
            mapping_junction: dict for junction
            mapping_perps: list of dicts (one per perpendicular)
        """
        mapping_junction = self.match_line_only(image1, image2, junction_line,
                                                window_size, search_radius)

        mapping_perps = []
        for perp in perpendiculars:
            m = self.match_line_only(image1, image2, perp,
                                     window_size, search_radius)
            mapping_perps.append(m)

        return mapping_junction, mapping_perps


import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt


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

    # --- exponential models ---
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
        y0_init = np.min(y)

        # A_init should be the difference between the maximum and minimum, or peak height
        A_init = np.max(y) - y0_init

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
            inv_lambda = max(round((1.0 / popt[1]) / self.pixel_size) * self.pixel_size, self.pixel_size)
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
                          plot_left=True, plot_right=True, per_shift_plots=False):
        """
        Fit falling exponentials on both sides with shifted windows, using SNR-based truncation.
        The starting point for each fit is determined by the intersection index or the maximum.
        Stabilization: stop shifting for a side when consecutive inv_lambda values for that side
        change less than or equal to self.pixel_size.
        """
        x_vals = np.asarray(x_vals)
        y_vals = np.asarray(y_vals)

        base_idx = int(np.argmax(y_vals))

        if intersection_idx is None:
            intersection_idx = base_idx

        results = []

        tolerance = self.pixel_size * 1e6

        # --- LEFT side ---
        prev_left_val = None
        left_stable = False
        for shift in range(0, -16, -1):  # 0, -1, -2, ..., -15
            if left_stable:
                break

            if intersection_idx <= base_idx:
                start_idx_left = intersection_idx + shift
            else:
                start_idx_left = base_idx + shift

            start_idx_left = max(0, start_idx_left)

            left_x_raw = x_vals[:start_idx_left + 1]
            y_left_raw = y_vals[:start_idx_left + 1]

            y_left_filtered = self.apply_low_pass_filter(y_left_raw, visualize=False)

            # Flip to make a falling curve
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
                    results.append(left_fit)
                    curr_val = left_fit.get('inv_lambda', None)
                    if curr_val is not None:
                        if prev_left_val is not None and abs(curr_val - prev_left_val) <= tolerance:
                            left_stable = True
                        prev_left_val = curr_val

        # --- RIGHT side ---
        prev_right_val = None
        right_stable = False
        for shift in range(0, 16):  # 0, 1, 2, ..., 15
            if right_stable:
                break

            if intersection_idx >= base_idx:
                start_idx_right = intersection_idx + shift
            else:
                start_idx_right = base_idx + shift

            start_idx_right = min(len(x_vals) - 1, start_idx_right)

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
                    results.append(right_fit)
                    curr_val = right_fit.get('inv_lambda', None)
                    if curr_val is not None:
                        if prev_right_val is not None and abs(curr_val - prev_right_val) <= tolerance:
                            right_stable = True
                        prev_right_val = curr_val

        # --- Optional overlay plots ---
        if plot_left:
            self._plot_shifted_fits(results, x_vals, y_vals, 'Left', base_idx, profile_id)
        if plot_right:
            self._plot_shifted_fits(results, x_vals, y_vals, 'Right', base_idx, profile_id)

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
        left_start = None
        right_start = None

        for res in fit_results:
            if "Left" in res['side']:
                # take the last x-value (unflipped end = real starting coordinate)
                left_start = -res['x_vals'][0]
            elif "Right" in res['side']:
                # take the first x-value (true start on the right side)
                right_start = res['x_vals'][0]

        if left_start is not None and right_start is not None:
            depletion_width = abs(right_start - left_start)
        else:
            depletion_width = None

        return {
            "left_start": left_start,
            "right_start": right_start,
            "depletion_width": depletion_width
        }

    def fit_all_profiles(self):
        """
        Extend to also compute depletion widths per profile.
        """
        self.results = []
        self.inv_lambdas = []
        self.central_inv_lambdas = []

        for i, prof in enumerate(self.profiles):
            intersection_idx = prof.get('intersection_idx', None)
            x_vals = np.array(prof['dist_um'])
            y_vals = np.array(prof['current'])

            sides = self.fit_profile_sides(x_vals, y_vals,
                                           intersection_idx=intersection_idx,
                                           profile_id=i + 1)

            depletion_info = self._extract_depletion_region(sides, x_vals, np.argmax(y_vals))

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

            plt.figure(figsize=(8, 4))
            plt.plot(x, y, 'k-', alpha=0.6, label="EBIC Profile")

            if depletion['left_start'] is not None:
                plt.axvline(depletion['left_start'], color='b', linestyle='--', label="Left Start")
            if depletion['right_start'] is not None:
                plt.axvline(depletion['right_start'], color='r', linestyle='--', label="Right Start")

            plt.title(f"Profile {profile_id} – Depletion Width = {depletion['depletion_width']:.2f} µm")
            plt.xlabel("Distance (µm)")
            plt.ylabel("Current (nA)")
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

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

        # Restore mean
        filtered_y += np.mean(y_vals)

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

    import numpy as np
    import matplotlib.pyplot as plt

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

    def visualize_fft_of_profile(self, profile_id=1, cutoff_fraction=0.1):
        """
        Compute and visualize FFT of the *entire* EBIC profile (before slicing).
        Also show the effect of the low-pass filter.
        """
        if not self.profiles or profile_id > len(self.profiles):
            print("Invalid profile ID.")
            return

        profile = self.profiles[profile_id - 1]
        x = np.array(profile['dist_um'])
        y = np.array(profile['current'])

        # FFT
        n = len(y)
        dx = np.mean(np.diff(x)) if len(x) > 1 else 1.0
        fft_vals = np.fft.fft(y - np.mean(y))
        freqs = np.fft.fftfreq(n, d=dx)

        # Gaussian filter
        f_nyquist = 0.5 / dx
        sigma = cutoff_fraction * f_nyquist
        filter_mask = np.exp(-0.5 * (freqs / sigma) ** 2)

        filtered_fft = fft_vals * filter_mask
        y_filtered = np.fft.ifft(filtered_fft).real + np.mean(y)

        # Plot FFT before/after
        mask = freqs >= 0
        plt.figure(figsize=(10, 6))
        plt.plot(freqs[mask], np.abs(fft_vals[mask]), 'b-', label='Original FFT')
        plt.plot(freqs[mask], np.abs(filtered_fft[mask]), 'r-', label='Filtered FFT')
        plt.plot(freqs[mask], filter_mask[mask] * np.max(np.abs(fft_vals[mask])),
                 'k--', label='Gaussian Mask')
        plt.xlabel("Spatial frequency (1/µm)")
        plt.ylabel("Magnitude (a.u.)")
        plt.title(f"Profile {profile_id} – FFT (whole profile)")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

        # Plot original vs filtered profile
        plt.figure(figsize=(10, 4))
        plt.plot(x, y, 'gray', alpha=0.5, label="Original")
        plt.plot(x, y_filtered, 'r-', linewidth=2, label="Filtered")
        plt.xlabel("Distance (µm)")
        plt.ylabel("Current (nA)")
        plt.title(f"Profile {profile_id} – Original vs Filtered")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


from scipy.optimize import curve_fit
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
            postprocessed_roi_coords = self._apply_spline_postprocessing(detected_roi_coords)

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


# ==================== SEM VIEWER ====================
class SEMViewer:
    def __init__(self, pixel_maps, current_maps, pixel_size, sample_name,
                 frame_sizes=None, dpi=100, pixel_mapping=None):
        self.detected_line_obj_mapped = None
        self.detected_junction_line_mapped = None
        self.manual_line_dense = None
        self.detected_junction_coords_mapped = None
        self.perpendicular_profiles = None
        self.pixel_maps = pixel_maps
        self.current_maps = current_maps
        self.pixel_size = pixel_size
        self.sample_name = sample_name
        self.index = 0
        self.map_type = 'current'
        self.overlay_mode = False
        self.line = None
        self.line_coords = None
        self._clicks = []
        self.frame_sizes = frame_sizes or [(pm.shape[1], pm.shape[0]) for pm in pixel_maps]
        self.junction_line = None
        self.junction_coords = None
        self.junction_position = None
        self.pixel_mapping = pixel_mapping  # NEW: store mapping

        # DPI and figure size to match the first frame
        width_px, height_px = self.frame_sizes[0]
        self.dpi = dpi
        fig_width_in = width_px / dpi
        fig_height_in = height_px / dpi
        self.fig = plt.figure(figsize=(fig_width_in, fig_height_in), dpi=dpi)
        # Axes positioned exactly for image, bottom-left fixed
        self.ax = self.fig.add_axes([0, 0, 1, 1])  # [left, bottom, width, height] in figure fraction
        self.cbar = None
        self.cbar_ax = None

        self.zoom_stack = []
        self.cid_scroll = self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.zoom_rectangle = None
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self._dragging = False
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        self.current_colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'rainbow']
        self.current_cmap_index = 0
        self.zoom_mode_enabled = False
        self.zoom_rect_start = None
        self.zoom_rect = None

        # Initialize UI
        self._init_ui()
        self._update_display()
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.junction_analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)

    def _init_ui(self):
        # Fixed image axes: occupy full height, leave space for buttons on the right
        img_width_frac = 0.85  # fraction of figure width for image
        img_height_frac = 1.0  # full height
        self.ax.set_position([0, 0, img_width_frac, img_height_frac])

        # Button layout parameters
        button_w = 0.10
        button_h = 0.05
        spacing = 0.02
        x_pos = img_width_frac + spacing  # buttons start to the right of image
        y_start = 0.80  # top of first button (fraction of figure height)
        y = y_start

        # Overlay button
        ax_overlay = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_overlay = Button(ax_overlay, 'Overlay')
        self.b_overlay.on_clicked(self._toggle_overlay)
        y -= (button_h + spacing)

        # Toggle Map button
        ax_toggle = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_toggle = Button(ax_toggle, 'Toggle Map')
        self.b_toggle.on_clicked(self._toggle_map_type)
        y -= (button_h + spacing)

        # Previous Frame button
        ax_prev = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_prev = Button(ax_prev, 'Previous')
        self.b_prev.on_clicked(self._prev_frame)
        y -= (button_h + spacing)

        # Next Frame button
        ax_next = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_next = Button(ax_next, 'Next')
        self.b_next.on_clicked(self._next_frame)
        y -= (button_h + spacing)

        # Palette button
        ax_palette = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_palette = Button(ax_palette, 'Palette')
        self.b_palette.on_clicked(self._cycle_colormap)
        y -= (button_h + spacing)

        # Perpendiculars button
        ax_perp = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_perp = Button(ax_perp, 'Perpendiculars')
        self.b_perp.on_clicked(self._show_perpendicular_input)
        y -= (button_h + spacing)

        # Reset Zoom button
        ax_reset = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_reset = Button(ax_reset, 'Reset Zoom')
        self.b_reset.on_clicked(self._reset_zoom)
        y -= (button_h + spacing)

        # Reset Lines button
        ax_reset_overlays = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_reset_overlays = Button(ax_reset_overlays, 'Reset Lines')
        self.b_reset_overlays.on_clicked(self._reset_overlays)
        y -= (button_h + spacing)

        # Zoom button at the bottom
        ax_zoom = self.fig.add_axes([x_pos, 0.01, button_w, button_h])
        self.zoom_btn = Button(ax_zoom, 'Zoom')
        self.zoom_btn.on_clicked(self.enable_zoom)

        # Fit Profiles button
        ax_fit = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_fit = Button(ax_fit, 'Fit Profiles')
        self.b_fit.on_clicked(self._fit_profiles)
        y -= (button_h + spacing)

        # --- Junction Detection button ---
        ax_junc = self.fig.add_axes([x_pos, y, button_w, button_h])
        self.b_junction = Button(ax_junc, 'Junction Detect')
        self.b_junction.on_clicked(self._detect_junction_button)
        y -= (button_h + spacing)

        # Connect close event
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def set_line_from_mapping(self, junction_line, mapping_junction,
                              perps=None, mapping_perps=None, show=True,
                              straight=True):
        """
        Map junction + perpendiculars from image1 -> image2 using
        pixel-to-pixel mapping dictionaries.

        - Junction and perpendiculars are mapped via mapping_junction / mapping_perps.
        - Perpendiculars are densified (sample every pixel along start→end).
        - intersection_idx is remapped via pixel matching.
        - Each perpendicular profile contains: id, points, line_coords,
          intersection_idx, dist_um, current, sem.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import map_coordinates

        def fit_line(coords, n_pts=None):
            """Least-squares straight fit returning same number of points."""
            if coords is None or len(coords) < 2:
                return coords
            if n_pts is None:
                n_pts = len(coords)
            x = coords[:, 0]
            y = coords[:, 1]
            A = np.vstack([x, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            x_fit = np.linspace(x.min(), x.max(), n_pts)
            y_fit = a * x_fit + b
            return np.column_stack([x_fit, y_fit])

        # ---------- Map junction ----------
        mapped_junc_list = []
        for (x, y) in junction_line.astype(int):
            key = (int(y), int(x))
            if key in mapping_junction:
                r2, c2 = mapping_junction[key]
                mapped_junc_list.append([c2, r2])
            else:
                # fallback: try neighbors
                found = False
                for dy in (-1, 0, 1):
                    if found: break
                    for dx in (-1, 0, 1):
                        k = (int(y) + dy, int(x) + dx)
                        if k in mapping_junction:
                            r2, c2 = mapping_junction[k]
                            mapped_junc_list.append([c2, r2])
                            found = True
                            break
                if not found:
                    mapped_junc_list.append([np.nan, np.nan])

        mapped_junction = np.array(mapped_junc_list, dtype=float)

        # interpolate NaNs if any
        idxs = np.arange(len(mapped_junction))
        valid = ~np.isnan(mapped_junction[:, 0])
        if valid.sum() >= 2:
            mapped_junction[:, 0] = np.interp(idxs, idxs[valid], mapped_junction[valid, 0])
            mapped_junction[:, 1] = np.interp(idxs, idxs[valid], mapped_junction[valid, 1])
        else:
            mapped_junction = None

        if mapped_junction is not None and len(mapped_junction) > 1:
            if straight:
                mapped_junction = fit_line(mapped_junction, n_pts=len(mapped_junction))
            self.detected_junction_line = mapped_junction
            self.line_coords = (mapped_junction[0], mapped_junction[-1])
        else:
            self.detected_junction_line = None
            self.line_coords = None

        # ---------- Map perpendiculars ----------
        self.perpendicular_profiles = []
        if perps is None or mapping_perps is None:
            return

        sem_map = self.pixel_maps[0] if len(self.pixel_maps) > 0 else None
        current_map = self.current_maps[1] if len(self.current_maps) > 1 else None

        for i, (orig_perp, m) in enumerate(zip(perps, mapping_perps)):
            orig_perp = np.asarray(orig_perp, dtype=float)
            n_pts = len(orig_perp)
            if n_pts < 2:
                continue

            # Map each original perp sample
            mapped_pts = np.full((n_pts, 2), np.nan, dtype=float)
            for k_idx, (xx, yy) in enumerate(orig_perp.astype(int)):
                key = (int(yy), int(xx))
                if key in m:
                    r2, c2 = m[key]
                    mapped_pts[k_idx] = [c2, r2]
                else:
                    # fallback: small neighborhood
                    found = False
                    for rsearch in (1, 2):
                        if found: break
                        for dy in range(-rsearch, rsearch + 1):
                            if found: break
                            for dx in range(-rsearch, rsearch + 1):
                                kk = (int(yy) + dy, int(xx) + dx)
                                if kk in m:
                                    rr, cc = m[kk]
                                    mapped_pts[k_idx] = [cc, rr]
                                    found = True
                                    break

            # interpolate missing points
            idxs = np.arange(n_pts)
            valid = ~np.isnan(mapped_pts[:, 0])
            if valid.sum() < 2:
                continue
            mapped_pts[:, 0] = np.interp(idxs, idxs[valid], mapped_pts[valid, 0])
            mapped_pts[:, 1] = np.interp(idxs, idxs[valid], mapped_pts[valid, 1])

            # optionally fit straight line
            if straight:
                mapped_pts = fit_line(mapped_pts, n_pts=n_pts)

            # densify: use only endpoints
            (x0, y0), (x1, y1) = mapped_pts[0], mapped_pts[-1]
            length_px = int(np.hypot(x1 - x0, y1 - y0)) + 1
            xs = np.linspace(x0, x1, length_px)
            ys = np.linspace(y0, y1, length_px)

            # intersection_idx: nearest to junction
            intersection_idx = length_px // 2
            if self.detected_junction_line is not None:
                dists = np.linalg.norm(
                    np.column_stack([xs, ys])[:, None, :] - self.detected_junction_line[None, :, :],
                    axis=2
                )
                perp_min_dists = dists.min(axis=1)
                intersection_idx = int(np.argmin(perp_min_dists))

            # sample sem + current
            sem_vals = None
            cur_vals = None
            if sem_map is not None:
                sem_vals = map_coordinates(sem_map, [ys, xs], order=1, mode="nearest")
            if current_map is not None:
                cur_vals = map_coordinates(current_map, [ys, xs], order=1, mode="nearest")

            dist_um = np.linspace(-((length_px - 1) * self.pixel_size * 1e6) / 2,
                                  ((length_px - 1) * self.pixel_size * 1e6) / 2,
                                  length_px)

            prof = {
                "id": i,
                "line_coords": ((x0, y0), (x1, y1)),
                "points": np.column_stack([xs, ys]),
                "intersection_idx": intersection_idx,
                "dist_um": dist_um,
                "current": cur_vals if cur_vals is not None else np.zeros_like(dist_um),
                "sem": sem_vals if sem_vals is not None else np.zeros_like(dist_um)
            }
            self.perpendicular_profiles.append(prof)

        # ---------- Visualization ----------
        if show:
            fig, ax = plt.subplots()
            ax.imshow(self.pixel_maps[0], cmap="gray", origin="upper")
            if self.detected_junction_line is not None:
                ax.plot(self.detected_junction_line[:, 0],
                        self.detected_junction_line[:, 1],
                        color="green", linewidth=2, label="Mapped junction")
            for prof in self.perpendicular_profiles:
                pts = prof["points"]
                ax.plot(pts[:, 0], pts[:, 1], color="cyan", linewidth=1)
                idx = prof["intersection_idx"]
                if idx is not None and 0 <= idx < len(pts):
                    ax.plot(pts[idx, 0], pts[idx, 1], 'ro', markersize=5)
            ax.set_title("Mapped + Densified Perpendiculars")
            ax.legend()
            plt.show()

    def _fit_profiles(self, event):
        if not hasattr(self, "perpendicular_profiles") or not self.perpendicular_profiles:
            print("No perpendicular profiles available. Draw them first.")
            return

        intersections = [prof['intersection_idx'] for prof in self.perpendicular_profiles]
        extractor = DiffusionLengthExtractor(self.pixel_size, smoothing_sigma=1)
        extractor.load_profiles(self.perpendicular_profiles)

        # 3. Fit all profiles (now using globally filtered data)
        extractor.fit_all_profiles()

        # 4. Visualize fitted results
        extractor.visualize_fitted_profiles()
        extractor.visualize_inv_lambda_vs_shift()
        extractor.visualize_depletion_regions()

    def enable_zoom(self, event):
        """Toggle zoom mode on/off"""
        self.zoom_mode_enabled = not self.zoom_mode_enabled

        if self.zoom_mode_enabled:
            self.zoom_btn.label.set_text("Zooming...")
            print("Zoom mode enabled: Click and drag to draw a zoom rectangle")
        else:
            self.zoom_btn.label.set_text("Zoom")
            # Clean up any existing rectangle
            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_rect_start = None

        self.fig.canvas.draw_idle()

    def _get_current_data(self):
        return self.pixel_maps if self.map_type == 'pixel' else self.current_maps

    def _toggle_overlay(self, event):
        self.overlay_mode = not self.overlay_mode
        self._update_display()

    def _toggle_map_type(self, event):
        self.map_type = 'pixel' if self.map_type == 'current' else 'current'
        self._update_display()

    def _next_frame(self, event):
        self.index = (self.index + 1) % len(self.pixel_maps)
        self._update_display()

    def _prev_frame(self, event):
        self.index = (self.index - 1) % len(self.pixel_maps)
        self._update_display()

    def _update_display(self):
        # Store current view limits before clearing
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        had_zoom = len(current_xlim) == 2 and current_xlim != (0.0, 1.0)

        # Clear previous state while preserving figure properties
        self.ax.clear()
        self._safe_remove_colorbar()

        # Reset ticks and labels
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Handle empty data case
        if not self.pixel_maps or not self.current_maps:
            self.ax.set_title("No data available")
            self.fig.canvas.draw_idle()
            return

        try:
            if self.overlay_mode:
                # Overlay mode implementation
                if len(self.pixel_maps) < 1 or len(self.current_maps) < 2:
                    self.ax.set_title("Overlay: Not enough frames.")
                else:
                    pixel = self.pixel_maps[0]
                    current = self.current_maps[1]

                    if pixel is not None:
                        self.im = self.ax.imshow(pixel, cmap='gray', extent=[0, pixel.shape[1], pixel.shape[0], 0])
                        if not had_zoom:  # Only set full view if not zoomed
                            self.ax.set_xlim(0, pixel.shape[1])
                            self.ax.set_ylim(pixel.shape[0], 0)

                    if current is not None and pixel is not None:
                        # Create colorbar axis if it doesn't exist
                        if self.cbar_ax is None or not self.cbar_ax in self.fig.axes:
                            self.cbar_ax = self.fig.add_axes([0.82, 0.1, 0.02, 0.3])  # Adjust position as needed

                        self.im = self.ax.imshow(current,
                                                 cmap=self.current_colormaps[self.current_cmap_index],
                                                 alpha=0.6,
                                                 extent=[0, pixel.shape[1], pixel.shape[0], 0])
                        self.cbar = self.fig.colorbar(self.im, cax=self.cbar_ax)
                        self.cbar.set_label("Current (nA)")

                    self.ax.set_title(f"{self.sample_name} - Overlay: Pixel Map + Current Map")
            else:
                # Single map mode implementation
                data_list = self._get_current_data()
                data = data_list[self.index]

                if data is None:
                    if self.map_type == 'current':
                        self.map_type = 'pixel'
                        data = self.pixel_maps[self.index]
                        title = f"{self.sample_name} - (No current map) Showing pixel map - Frame {self.index + 1}"
                    else:
                        title = f"{self.sample_name} - No data available - Frame {self.index + 1}"
                    self.ax.set_title(title)

                if data is not None:
                    cmap = 'gray' if self.map_type == 'pixel' else self.current_colormaps[self.current_cmap_index]

                    if self.map_type == 'current':
                        # Create colorbar axis if it doesn't exist
                        if self.cbar_ax is None or not self.cbar_ax in self.fig.axes:
                            self.cbar_ax = self.fig.add_axes([0.82, 0.1, 0.02, 0.3])  # Adjust position as needed

                        self.im = self.ax.imshow(data, cmap=cmap, extent=[0, data.shape[1], data.shape[0], 0])
                        self.cbar = self.fig.colorbar(self.im, cax=self.cbar_ax)
                        self.cbar.set_label("Current (nA)")
                    else:
                        self.im = self.ax.imshow(data, cmap=cmap, extent=[0, data.shape[1], data.shape[0], 0])

                    # Set proper view limits based on image dimensions only if not zoomed
                    if not had_zoom:
                        self.ax.set_xlim(0, data.shape[1])
                        self.ax.set_ylim(data.shape[0], 0)  # Inverted y-axis

                    self.ax.set_title(f"{self.sample_name} - {self.map_type.capitalize()} Map - Frame {self.index + 1}")

            # Restore zoom state if we had one
            if had_zoom:
                self.ax.set_xlim(current_xlim)
                self.ax.set_ylim(current_ylim)

            # Restore line if it exists
            if self.line_coords:
                (x0, y0), (x1, y1) = self.line_coords
                self.line = Line2D([x0, x1], [y0, y1], color='red', linewidth=2)
                self.ax.add_line(self.line)

            # Draw scale bar
            self._draw_scalebar()

            self.fig.canvas.draw_idle()


        except Exception as e:
            print(f"Error in display update: {str(e)}")
            import traceback
            traceback.print_exc()

    def _draw_scalebar(self, ax=None, length_um=1.0, height_px=5, color='white', font_size=12):
        """
        Draw a scalebar on the current axes.

        length_um: desired scalebar length in microns
        height_px: height of the bar in pixels
        """
        if ax is None:
            ax = self.ax

        # Convert pixel size to microns
        pixel_size_m = self.pixel_size  # from XML
        pixel_size_um = pixel_size_m * 1e6  # meters -> microns

        # Calculate scalebar length in pixels
        length_px = length_um / pixel_size_um

        # Position the scalebar 10% above bottom-left corner
        x0 = 0.05 * ax.get_xlim()[1]
        y0 = 0.95 * ax.get_ylim()[0]  # inverted y-axis

        # Draw rectangle
        rect = plt.Rectangle((x0, y0), length_px, height_px, color=color, clip_on=False)
        ax.add_patch(rect)

        # Add label
        ax.text(x0 + length_px / 2, y0 - height_px * 1.5, f"{length_um} µm",
                color=color, ha='center', va='bottom', fontsize=font_size, clip_on=False)

    def _safe_remove_colorbar(self):
        if self.cbar:
            try:
                self.cbar.remove()
            except Exception:
                pass
            self.cbar = None
        if hasattr(self, 'cbar_ax') and self.cbar_ax is not None:
            try:
                self.cbar_ax.remove()
            except Exception:
                pass
            self.cbar_ax = None

    def _plot_line_profile(self):
        if self.line_coords is None:
            return

        (x0, y0), (x1, y1) = self.line_coords
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length < 2:
            return

        # Coordinates along the line
        x = np.linspace(x0, x1, length)
        y = np.linspace(y0, y1, length)

        # SEM contrast from frame 1
        sem_data = self.pixel_maps[0]  # frame 1
        if sem_data is None:
            print("Warning: Missing SEM data for frame 1.")
            return

        # Current from frame 2
        current_data = self.current_maps[1]  # frame 2
        if current_data is None:
            print("Warning: Missing current data for frame 2.")

        # Clip coordinates to valid array range (based on SEM frame size)
        x = np.clip(x, 0, sem_data.shape[1] - 1)
        y = np.clip(y, 0, sem_data.shape[0] - 1)

        # Interpolate values
        sem_values = map_coordinates(sem_data, [y, x], order=1, mode='nearest')
        if current_data is not None:
            current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')

        # Distances in microns
        distances_um = np.linspace(0, (length - 1) * self.pixel_size * 1e6, length)

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Distance (µm)")
        ax1.set_ylabel("Pixel Value (SEM)", color='tab:blue')
        ax1.plot(distances_um, sem_values, color='tab:blue', label='SEM Contrast')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, linestyle='--', alpha=0.5)

        if current_data is not None:
            ax2 = ax1.twinx()
            ax2.set_ylabel("Current (nA)", color='tab:red')
            ax2.plot(distances_um, current_values, color='tab:red', label='Current')
            ax2.tick_params(axis='y', labelcolor='tab:red')

            # Combine legends from both axes
            lines = ax1.get_lines() + ax2.get_lines()
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='best')
        else:
            ax1.legend(loc='best')

        plt.title("Line Profile - SEM (frame 1) + Current (frame 2)")
        plt.tight_layout()
        plt.show()

    def show(self):
        plt.show()

    def _cycle_colormap(self, event=None):
        if not self.current_colormaps:
            return
        self.current_cmap_index = (self.current_cmap_index + 1) % len(self.current_colormaps)
        self._update_display()

    def _on_click(self, event):
        if event.inaxes != self.ax or event.button != 1:
            return

        if self.zoom_mode_enabled:
            return

        if not hasattr(self, '_clicks'):
            self._clicks = []

        if len(self._clicks) == 0:
            # First click
            self._clicks.append((event.xdata, event.ydata))
            self._dragging = True

            if hasattr(self, 'click_marker') and self.click_marker in self.ax.collections:
                self.click_marker.remove()
            self.click_marker = self.ax.scatter(
                [event.xdata], [event.ydata],
                color='red', s=50, marker='o', zorder=10)

        elif len(self._clicks) == 1:
            # Second click - finish line
            self._dragging = False

            if hasattr(self, '_temp_line') and self._temp_line in self.ax.lines:
                self._temp_line.remove()

            x0, y0 = self._clicks[0]
            x1, y1 = event.xdata, event.ydata
            self.line_coords = ((x0, y0), (x1, y1))

            if self.line and self.line in self.ax.lines:
                self.line.remove()

            self.line = Line2D([x0, x1], [y0, y1],
                               color='red',
                               linewidth=2)
            self.ax.add_line(self.line)

            # === Generate dense manual line ===
            line_length = int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
            xs = np.linspace(x0, x1, line_length)
            ys = np.linspace(y0, y1, line_length)
            self.manual_line_dense = np.column_stack([xs, ys])

            # Cleanup click markers
            self._clicks = []
            if hasattr(self, 'click_marker'):
                self.click_marker.remove()
                del self.click_marker

            self.fig.canvas.draw_idle()

            # Automatically run junction detection
            try:
                width_um = self._ask_junction_width()
                if width_um is not None:
                    half_width_px = int(np.round(width_um * 1e-6 / self.pixel_size))
                    half_width_px = max(1, half_width_px)

                    pixel_map = self.pixel_maps[self.index]
                    roi, roi_coords = self.extract_line_rectangle(pixel_map,
                                                                  self.manual_line_dense,
                                                                  half_width_px)
                    analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)
                    results = analyzer.detect(roi, self.manual_line_dense)

                    if results:
                        # **New Code:**
                        # Get the best result (which is the only one returned by Canny)
                        best_result = results[0]
                        detected_coords = best_result[1]

                        # Store the detected line and its visual object as class attributes
                        self.detected_junction_line = detected_coords
                        self.detected_line_obj = Line2D(
                            self.detected_junction_line[:, 0],
                            self.detected_junction_line[:, 1],
                            color='green',
                            linewidth=2
                        )
                        self.ax.add_line(self.detected_line_obj)

                    else:
                        print("Junction detection failed.")
            except Exception as e:
                print(f"Auto junction detection error: {e}")

    def _on_close(self, event):
        print("Figure closing, saving images and data...")

        # Prepare save folder
        save_root = os.path.join("tiff_test_output", self.sample_name, "saved_on_close")
        os.makedirs(save_root, exist_ok=True)

        # Save current displayed frame image with overlay line if any
        img_save_path = os.path.join(save_root, f"{self.sample_name}_frame{self.index + 1}_overlay.png")

        # Draw the overlay line on a copy of the current image and save
        fig, ax = plt.subplots()
        data = (self.pixel_maps if self.map_type == 'pixel' else self.current_maps)[self.index]

        if data is None:
            print("No data to save for the current frame.")
            plt.close(fig)
            return

        cmap = 'gray' if self.map_type == 'pixel' else self.current_colormaps[self.current_cmap_index]
        ax.imshow(data, cmap=cmap)
        if self.line is not None:
            ax.add_line(Line2D(self.line.get_xdata(), self.line.get_ydata(), color='red'))

        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(img_save_path)
        plt.close(fig)
        print(f"Saved overlay image to {img_save_path}")

    def _save_line_profile_plot(self, save_root):
        (x0, y0), (x1, y1) = self.line_coords
        length = int(np.hypot(x1 - x0, y1 - y0))
        if length < 2:
            print("Line too short to save profile plot.")
            return

        x = np.linspace(x0, x1, length)
        y = np.linspace(y0, y1, length)

        current_data = self.current_maps[self.index]
        pixel_data_first = self.pixel_maps[0]

        if current_data is None or pixel_data_first is None:
            print("Current or pixel map not available for saving profile.")
            return

        x = np.clip(x, 0, current_data.shape[1] - 1)
        y = np.clip(y, 0, current_data.shape[0] - 1)

        current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')
        pixel_values = map_coordinates(pixel_data_first, [y, x], order=1, mode='nearest')

        distances_um = np.linspace(0, (length - 1) * self.pixel_size * 1e6, length)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.set_xlabel("Distance (µm)")
        ax1.set_ylabel("Pixel Value (1st Frame)", color='tab:blue')
        ax1.plot(distances_um, pixel_values, color='tab:blue', label='Pixel (Frame 1)')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel("Current (nA)", color='tab:red')
        ax2.plot(distances_um, current_values, color='tab:red', label='Current')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        fig.suptitle(f"Line Profile: {os.path.basename(self.sample_name[self.index])}", fontsize=12)
        fig.tight_layout()

        profile_save_path = os.path.join(save_root, f"{self.sample_name}_frame{self.index + 1}_line_profile.png")
        fig.savefig(profile_save_path)
        plt.close(fig)
        print(f"Saved line profile plot to {profile_save_path}")

    def _on_scroll(self, event):
        if event.inaxes != self.ax:
            return

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()

        zoom_factor = 1.2 if event.button == 'up' else 0.8
        cx, cy = event.xdata, event.ydata

        new_width = (x_max - x_min) * zoom_factor
        new_height = (y_max - y_min) * zoom_factor

        new_xmin = cx - new_width / 2
        new_xmax = cx + new_width / 2
        new_ymin = cy - new_height / 2
        new_ymax = cy + new_height / 2

        # Save current view for reset
        self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))

        # Apply new limits
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)

        # 🔑 Force equal aspect ratio (no stretching)
        self.ax.set_aspect('equal', adjustable='box')

        self.fig.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # Left-click zoom rectangle
        if self.zoom_mode_enabled and event.button == 1:
            x = max(0, min(event.xdata, width))
            y = max(0, min(event.ydata, height))
            self.zoom_rect_start = (x, y)
            return

        # Right-click drag
        if event.button == 3:
            x = max(0, min(event.xdata, width))
            y = max(0, min(event.ydata, height))
            self._drag_start = (x, y)

    def _on_release(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # Left-click zoom rectangle
        if self.zoom_mode_enabled and self.zoom_rect_start and event.button == 1:
            x0, y0 = self.zoom_rect_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))
                self.ax.set_xlim(min(x0, x1), max(x0, x1))
                self.ax.set_ylim(max(y0, y1), min(y0, y1))

            if self.zoom_rect:
                self.zoom_rect.remove()
                self.zoom_rect = None
            self.zoom_rect_start = None
            self.fig.canvas.draw_idle()
            return

        # Right-click drag zoom
        if hasattr(self, "_drag_start") and event.button == 3:
            x0, y0 = self._drag_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            if abs(x1 - x0) > 2 and abs(y1 - y0) > 2:
                self.zoom_stack.append((self.ax.get_xlim(), self.ax.get_ylim()))
                self.ax.set_xlim(min(x0, x1), max(x0, x1))
                self.ax.set_ylim(max(y0, y1), min(y0, y1))
                self.fig.canvas.draw_idle()
            del self._drag_start

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return

        width, height = self.frame_sizes[self.index]

        # === Case 1: zoom rectangle ===
        if self.zoom_mode_enabled and self.zoom_rect_start:
            if self.zoom_rect:
                self.zoom_rect.remove()

            x0, y0 = self.zoom_rect_start
            x1 = max(0, min(event.xdata, width))
            y1 = max(0, min(event.ydata, height))

            self.zoom_rect = patches.Rectangle(
                (min(x0, x1), min(y0, y1)),
                abs(x1 - x0), abs(y1 - y0),
                linewidth=1, edgecolor='r', facecolor='none', linestyle='--'
            )
            self.ax.add_patch(self.zoom_rect)
            self.fig.canvas.draw_idle()
            return

        # === Case 2: live preview of line ===
        if self._dragging and len(self._clicks) == 1:
            x0, y0 = self._clicks[0]
            x1, y1 = event.xdata, event.ydata
            if x1 is None or y1 is None:
                return

            # Remove previous temp line
            if hasattr(self, '_temp_line') and self._temp_line in self.ax.lines:
                self._temp_line.remove()

            # Draw new temp line
            self._temp_line = Line2D([x0, x1], [y0, y1],
                                     color='red', linewidth=1, linestyle='--')
            self.ax.add_line(self._temp_line)
            self.fig.canvas.draw_idle()

    def _reset_zoom(self, event=None):
        self.zoom_stack.clear()

        data_list = self._get_current_data()
        if not data_list or data_list[self.index] is None:
            return

        data = data_list[self.index]

        # Reset to full image extents
        self.ax.set_xlim(0, data.shape[1])
        self.ax.set_ylim(data.shape[0], 0)  # inverted y-axis

        self._update_display()
        self.fig.canvas.draw_idle()

    def _show_perpendicular_input(self, event):
        """Ask user for number of perpendiculars and length, then calculate and plot them."""
        from tkinter import simpledialog, Tk

        # Use a temporary root for dialogs, then destroy it
        root = Tk()
        root.withdraw()  # hide the root window
        root.update()  # process events

        try:
            num_lines = simpledialog.askinteger(
                "Input", "Number of perpendicular lines:", parent=root, minvalue=1, maxvalue=100
            )
            length_um = simpledialog.askfloat(
                "Input", "Length of each perpendicular line (µm):", parent=root, minvalue=0.1, maxvalue=1e5
            )
        except Exception:
            print("Invalid input. Operation canceled.")
            root.destroy()
            return

        root.destroy()

        if num_lines is None or length_um is None:
            print("Operation canceled by user.")
            return

        if self.line_coords is None:
            print("Main line not drawn.")
            return

        # === Calculate perpendicular profiles ===
        profiles = self._calculate_perpendicular_profiles(num_lines, length_um, line_coords=self.line_coords)

        # === Plot perpendicular profiles ===
        self._plot_perpendicular_profiles(profiles)

    def _calculate_perpendicular_profiles(self, num_lines, length_um, line_coords=None):
        """
        Calculate perpendicular profiles along the manual line.
        Adds the intersection with detected junction if available.
        """
        if line_coords is None:
            line_coords = self.line_coords  # fallback

        # Create dense line
        if isinstance(line_coords, np.ndarray) and line_coords.shape[1] == 2:
            x0, y0 = line_coords[0]
            x1, y1 = line_coords[-1]
            dense_line = line_coords
        else:
            (x0, y0), (x1, y1) = line_coords
            line_length = int(np.ceil(np.hypot(x1 - x0, y1 - y0)))
            xs = np.linspace(x0, x1, line_length)
            ys = np.linspace(y0, y1, line_length)
            dense_line = np.column_stack([xs, ys])

        main_vec = np.array([x1 - x0, y1 - y0], dtype=float)
        main_len = np.linalg.norm(main_vec)
        if main_len == 0:
            print("Line is too short.")
            return []

        main_unit = main_vec / main_len
        perp_unit = np.array([-main_unit[1], main_unit[0]])
        length_px = length_um / (self.pixel_size * 1e6)  # µm -> px

        # Pick points along dense line
        distances = np.linspace(0, len(dense_line) - 1, num_lines).astype(int)

        sem_data = self.pixel_maps[0]
        current_data = self.current_maps[1]
        if sem_data is None or current_data is None:
            print("Missing SEM or current data.")
            return []

        profiles = []
        for i, idx in enumerate(distances):
            center = dense_line[idx]
            p_start = center - perp_unit * (length_px / 2)
            p_end = center + perp_unit * (length_px / 2)

            x = np.linspace(p_start[0], p_end[0], int(length_px))
            y = np.linspace(p_start[1], p_end[1], int(length_px))
            x = np.clip(x, 0, sem_data.shape[1] - 1)
            y = np.clip(y, 0, sem_data.shape[0] - 1)

            sem_values = map_coordinates(sem_data, [y, x], order=1, mode='nearest')
            current_values = map_coordinates(current_data, [y, x], order=1, mode='nearest')
            dist_um_line = np.linspace(-length_um / 2, length_um / 2, len(sem_values))

            # --- Find intersection with detected junction ---
            intersection = None
            intersection_idx = None
            if hasattr(self, 'detected_junction_line') and self.detected_junction_line is not None:
                intersection, intersection_idx = self._find_perpendicular_junction_intersection(
                    p_start, p_end, self.detected_junction_line, len(sem_values)
                )

            profiles.append({
                "id": i,
                "dist_um": dist_um_line,
                "current": current_values,
                "sem": sem_values,
                "line_coords": (p_start, p_end),
                "intersection": intersection,
                "intersection_idx": intersection_idx
            })

        self.perpendicular_profiles = profiles
        return profiles

    def _find_perpendicular_junction_intersection(self, p_start, p_end, junction_line, profile_length):
        line_vec = p_end - p_start
        line_len2 = np.sum(line_vec ** 2)
        if line_len2 == 0:
            return None, None

        # Project each junction point onto the perpendicular line
        t = np.dot(junction_line - p_start, line_vec) / line_len2
        t = np.clip(t, 0, 1)
        projection = p_start + np.outer(t, line_vec)

        # Find closest junction point
        distances = np.linalg.norm(junction_line - projection, axis=1)
        idx_min = np.argmin(distances)
        intersection = projection[idx_min]

        # Map intersection to closest index along profile array
        profile_idx = int(round(t[idx_min] * (profile_length - 1)))
        profile_idx = np.clip(profile_idx, 0, profile_length - 1)

        return intersection, profile_idx

    def _plot_perpendicular_profiles(self, profiles):
        """
        Plot perpendicular lines for manual line with intersection markers.
        """
        if not profiles:
            print("No profiles to plot.")
            return

        # --- Draw perpendicular lines on main viewer ---
        if hasattr(self, 'ax') and self.ax:
            self.perp_lines = getattr(self, 'perp_lines', [])
            for prof in profiles:
                p_start, p_end = prof["line_coords"]
                line, = self.ax.plot([p_start[0], p_end[0]], [p_start[1], p_end[1]], 'r--', linewidth=1)
                self.perp_lines.append(line)
                # Draw intersection if available
                if prof.get("intersection") is not None:
                    inter = prof["intersection"]
                    self.ax.scatter(inter[0], inter[1], color='lime', s=50, marker='x', zorder=20)

            if hasattr(self, 'fig') and self.fig:
                self.fig.canvas.draw_idle()

        # --- Scrollable profile plot window ---
        win = tk.Toplevel()
        win.title("Perpendicular Profiles (Manual Line)")
        win.geometry("1200x800")
        win.resizable(True, True)
        win.lift()
        win.focus_force()

        canvas = tk.Canvas(win)
        scrollbar = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas)
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for prof in profiles:
            dist_um = prof["dist_um"]
            sem_vals = prof["sem"]
            cur_vals = prof["current"]

            sem_vals_norm = (sem_vals - np.min(sem_vals)) / (np.ptp(sem_vals) + 1e-12)

            fig, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(dist_um, sem_vals_norm, color='tab:blue', linewidth=2, label='SEM (norm)')
            ax2 = ax1.twinx()
            ax2.plot(dist_um, cur_vals, color='tab:red', linewidth=1.5, label='Current (nA)')

            # Mark intersection point
            if prof.get("intersection_idx") is not None:
                idx = prof["intersection_idx"]
                ax1.scatter(dist_um[idx], sem_vals_norm[idx], color='lime', s=50, marker='x', zorder=10)
                ax2.scatter(dist_um[idx], cur_vals[idx], color='lime', s=50, marker='x', zorder=10)

            ax1.set_xlabel("Distance (µm)")
            ax1.set_ylabel("SEM Contrast (norm)", color='tab:blue')
            ax2.set_ylabel("Current (nA)", color='tab:red')
            ax1.set_title(f"Perpendicular {prof['id'] + 1}")
            ax1.legend(loc='upper left')
            fig.tight_layout()

            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            plot_canvas = FigureCanvasTkAgg(fig, master=scroll_frame)
            widget = plot_canvas.get_tk_widget()
            widget.pack(pady=10)

    def _reset_overlays(self, event=None):
        """Remove main line and perpendicular lines, reset line coordinates."""
        # Remove main line
        if self.line is not None and hasattr(self.line, "remove"):
            self.line.remove()
            self.line = None

        # Reset coordinates
        self.line_coords = None
        self._clicks = []

        # Remove perpendicular lines if tracked
        if hasattr(self, 'perp_lines'):
            new_list = []
            for pl in self.perp_lines:
                try:
                    if hasattr(pl, "remove"):
                        pl.remove()
                except Exception as e:
                    print(f"Warning: could not remove line: {e}")
                    new_list.append(pl)  # keep if cannot remove
            self.perp_lines = new_list

        # Redraw image without overlays
        self._update_display()
        self.fig.canvas.draw_idle()

    def _ask_junction_width(self):
        """Ask user for the half-width of the junction analysis region (in microns)."""
        from tkinter import simpledialog, Tk

        root = Tk()
        root.withdraw()  # hide root window
        root.update()

        try:
            width_um = simpledialog.askfloat(
                "Junction Width",
                "Half-width of the junction region (µm):",
                parent=root,
                minvalue=1e-6, maxvalue=1e5
            )
        except Exception:
            print("Invalid input. Operation canceled.")
            root.destroy()
            return None

        root.destroy()
        return width_um

    def _detect_junction_button(self, event):
        """Callback for Junction Detection button using JunctionAnalyzer (Canny only)."""
        if self.line_coords is None:
            print("Please draw the rough line along the junction first.")
            return

        if not hasattr(self, 'manual_line_dense'):
            print("Manual line points not generated. Draw the line first.")
            return

        # Ask user for half-width
        width_um = self._ask_junction_width()
        if width_um is None:
            print("Junction detection canceled by user.")
            return

        # Convert to pixels
        half_width_px = int(np.round(width_um * 1e-6 / self.pixel_size))
        half_width_px = max(1, half_width_px)

        # Extract ROI from pixel map
        pixel_map = self.pixel_maps[self.index]
        roi, roi_coords = self.extract_line_rectangle(pixel_map, self.manual_line_dense, half_width_px)

        # Create analyzer (Canny only)
        analyzer = JunctionAnalyzer(pixel_size_m=self.pixel_size)

        # Detect junction (returns a single result in a list)
        results = analyzer.detect(roi, self.manual_line_dense)

        if not results:
            print("Junction detection failed.")
            return

        # Visualize on the original pixel_map (image)
        analyzer.visualize_results(pixel_map, self.manual_line_dense, results)

    def extract_line_rectangle(self, pixel_map, line_dense, half_width_px):
        """
        Extract a rectangular ROI around a dense line.
        """
        import numpy as np

        if line_dense is None or len(line_dense) < 2:
            raise ValueError("Line must have at least 2 points.")

        height, width = pixel_map.shape
        roi_coords = []


        for i in range(len(line_dense) - 1):
            x, y = line_dense[i]
            x_next, y_next = line_dense[i + 1]

            dx = x_next - x
            dy = y_next - y
            norm = np.hypot(dx, dy)

            if norm < 1e-8:
                print(f"[DEBUG] Skipping degenerate segment {i}: "
                      f"({x:.1f},{y:.1f}) -> ({x_next:.1f},{y_next:.1f})")
                continue

            # Perpendicular vector
            perp_vec = np.array([-dy, dx]) / norm

            for offset in range(-half_width_px, half_width_px + 1):
                px = int(np.round(x + offset * perp_vec[0]))
                py = int(np.round(y + offset * perp_vec[1]))

                if 0 <= px < width and 0 <= py < height:
                    roi_coords.append((py, px))

        if not roi_coords:
            raise ValueError("ROI extraction failed: no valid coords found.")

        ys, xs = zip(*roi_coords)
        ys = np.array(ys)
        xs = np.array(xs)

        roi = pixel_map[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

        return roi, roi_coords






# ==================== INTEGRATED WORKFLOW ====================
if __name__ == "__main__":
    import pickle

    processor1 = SEMTiffProcessor()
    pixel_maps1, current_maps1, pixel_size1, sample1, frame_sizes1 = processor1.process_tiff(
        "MidMag/MidMag_DISS_Vb0V_bis.tif", "output_root", "sample1"
    )

    viewer1 = SEMViewer(pixel_maps1, current_maps1, pixel_size1, sample1, frame_sizes1)
    viewer1.show()
    # Save detected junction + perpendiculars from viewer1
    saved_line = viewer1.detected_junction_line.copy()
    perps = [np.column_stack([np.linspace(p0[0], p1[0], 50),
                              np.linspace(p0[1], p1[1], 50)])
             for prof in viewer1.perpendicular_profiles
             for p0, p1 in [prof["line_coords"]]]

    # === Dataset 2 ===
    processor2 = SEMTiffProcessor()
    pixel_maps2, current_maps2, pixel_size2, sample2, frame_sizes2 = processor2.process_tiff(
        "MidMag/MidMag_DISS_Vb0.4V_bis.tif", "output_root", "sample2"
    )

    # === Match only along the drawn line ===
    image1 = pixel_maps1[0]  # frame 1 (pixel map) from dataset 1
    image2 = pixel_maps2[0]  # frame 1 (pixel map) from dataset 2

    mapping_junction, mapping_perps = processor1.match_junction_and_perps(
        image1, image2, saved_line, perps,
        window_size=11, search_radius=10
    )

    # === Apply mapped line to dataset 2 ===
    # Apply mapped junction line to dataset 2
    viewer2 = SEMViewer(pixel_maps2, current_maps2, pixel_size2, sample2, frame_sizes2)
    viewer2.set_line_from_mapping(saved_line, mapping_junction,
                                  perps=perps, mapping_perps=mapping_perps)
    # viewer2.show()



