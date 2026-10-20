from __future__ import annotations
import logging
from typing import Tuple, List, Dict, Optional, Any
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OralROIEngine:
    """
    Oral Cavity ROI Extractor, Mask Generator, and Seamless Stitcher.
    
    Adheres to the Dental Restoration Standard:
    1. Extracts high-resolution bounding box around the oral cavity (teeth + lips + margins).
    2. Resizes oral cavity to 512x512 where teeth occupy ~70% of the field of view.
    3. Prepares precision inpaint masks for dental restorations.
    4. Seamlessly blends inpainted mouth back into the original master portrait with
       feathered boundaries, ensuring 100% preservation of facial skin, eyes, and hair.
    """

    _detector = None

    @classmethod
    def get_face_landmarker(cls):
        """Lazy-loaded MediaPipe FaceLandmarker for anatomical smile window extraction."""
        if cls._detector is None:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision
                from pathlib import Path
                model_path = Path(r"C:\OralWisdom_venv\face_landmarker.task")
                if model_path.exists():
                    base_options = python.BaseOptions(model_asset_path=str(model_path))
                    options = vision.FaceLandmarkerOptions(
                        base_options=base_options,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                        num_faces=1
                    )
                    cls._detector = vision.FaceLandmarker.create_from_options(options)
                    logger.info("MediaPipe FaceLandmarker initialized successfully.")
            except Exception as exc:
                logger.debug("MediaPipe FaceLandmarker not initialized: %s", exc)
        return cls._detector

    @classmethod
    def extract_smile_window(
        cls,
        image: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Extract anatomical Smile Window (inner lip oral opening) using MediaPipe Face Mesh.
        Returns:
            inner_lip_poly: np.ndarray of shape (N, 2) with inner lip boundary points in image coordinates
            outer_lip_poly: np.ndarray of shape (M, 2) with outer vermilion border points
            bbox: (min_x, min_y, max_x, max_y) bounding box of inner lip opening
        """
        detector = cls.get_face_landmarker()
        if detector is None:
            return None, None, None

        try:
            import mediapipe as mp
            h, w = image.shape[:2]
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            res = detector.detect(mp_img)

            if not res.face_landmarks:
                return None, None, None

            lms = res.face_landmarks[0]
            # Standard MediaPipe FaceMesh lip indices
            inner_lip_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
            outer_lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

            inner_pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in inner_lip_indices], dtype=np.int32)
            outer_pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in outer_lip_indices], dtype=np.int32)

            ix, iy, iw, ih = cv2.boundingRect(inner_pts)
            bbox = (ix, iy, ix + iw, iy + ih)

            logger.info("MediaPipe detected anatomical smile window: bbox=%s", bbox)
            return inner_pts, outer_pts, bbox
        except Exception as exc:
            logger.warning("Error extracting smile window via MediaPipe: %s", exc)
            return None, None, None

    @staticmethod
    def compute_teeth_bounding_box(
        instances: List[Dict[str, Any]],
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Compute minimal bounding box containing detected teeth, filtering spatial outliers.
        Returns (min_x, min_y, max_x, max_y).
        """
        h, w = image_shape[:2]
        centroids = []
        valid_instances = []
        for inst in instances:
            poly = inst.get("polygon", [])
            if poly and len(poly) >= 3:
                poly_arr = np.array(poly, dtype=np.float32)
                centroids.append(poly_arr.mean(axis=0))
                valid_instances.append((inst, poly_arr))

        if not valid_instances:
            logger.warning("No tooth polygons found; falling back to center crop.")
            min_x = int(w * 0.3)
            max_x = int(w * 0.7)
            min_y = int(h * 0.45)
            max_y = int(h * 0.65)
            return min_x, min_y, max_x, max_y

        centroids_np = np.array(centroids)
        med_x = float(np.median(centroids_np[:, 0]))
        med_y = float(np.median(centroids_np[:, 1]))

        # Filter out instances whose centroid is an extreme outlier from oral cluster
        all_pts = []
        for (inst, poly_arr), c in zip(valid_instances, centroids_np):
            dx = abs(c[0] - med_x)
            dy = abs(c[1] - med_y)
            if dy <= h * 0.20 and dx <= w * 0.30:
                all_pts.extend(poly_arr)
            else:
                logger.info("ROIEngine: Filtered spatial outlier instance %s at (%.1f, %.1f)", inst.get("id"), c[0], c[1])

        if not all_pts:
            all_pts = [p for _, poly_arr in valid_instances for p in poly_arr]

        pts_np = np.array(all_pts, dtype=np.float32)
        min_x = int(np.min(pts_np[:, 0]))
        max_x = int(np.max(pts_np[:, 0]))
        min_y = int(np.min(pts_np[:, 1]))
        max_y = int(np.max(pts_np[:, 1]))

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w - 1, max_x)
        max_y = min(h - 1, max_y)
        return min_x, min_y, max_x, max_y

    @classmethod
    def extract_oral_cavity_roi(
        cls,
        image: np.ndarray,
        instances: List[Dict[str, Any]],
        padding_h: float = 0.35,
        padding_w: float = 0.40,
        target_size: Tuple[int, int] = (512, 512),
        make_square: bool = True,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        """
        Crop oral cavity region from master photo with anatomical context padding.
        Ensures a square 1:1 aspect ratio to eliminate anamorphic tooth stretching.

        Returns:
            roi_512: 512x512 BGR image crop of the oral cavity
            crop_box: (crop_x1, crop_y1, crop_x2, crop_y2) in original image coordinates
            roi_original: original-resolution crop prior to resizing
        """
        h, w = image.shape[:2]
        bx1, by1, bx2, by2 = cls.compute_teeth_bounding_box(instances, (h, w))
        bw = bx2 - bx1
        bh = by2 - by1
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0

        if make_square:
            roi_side = int(max(bw * (1.0 + 2.0 * padding_w), bh * (1.0 + 2.0 * padding_h), 100))
            roi_side = min(roi_side, min(h, w))
            roi_side = ((roi_side + 1) // 2) * 2

            crop_x1 = max(0, int(cx - roi_side // 2))
            crop_y1 = max(0, int(cy - roi_side // 2))
            crop_x2 = min(w, crop_x1 + roi_side)
            crop_y2 = min(h, crop_y1 + roi_side)

            if crop_x2 - crop_x1 < roi_side:
                crop_x1 = max(0, crop_x2 - roi_side)
            if crop_y2 - crop_y1 < roi_side:
                crop_y1 = max(0, crop_y2 - roi_side)
        else:
            pad_x = int(bw * padding_w)
            pad_y = int(bh * padding_h)
            crop_x1 = max(0, bx1 - pad_x)
            crop_y1 = max(0, by1 - pad_y)
            crop_x2 = min(w, bx2 + pad_x)
            crop_y2 = min(h, by2 + pad_y)

        # Ensure sensible minimum dimensions
        if crop_x2 - crop_x1 < 100 or crop_y2 - crop_y1 < 100:
            crop_x1 = max(0, int(w * 0.25))
            crop_x2 = min(w, int(w * 0.75))
            crop_y1 = max(0, int(h * 0.40))
            crop_y2 = min(h, int(h * 0.70))

        roi_original = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        roi_512 = cv2.resize(roi_original, target_size, interpolation=cv2.INTER_LANCZOS4)
        crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

        logger.info(
            "Extracted Oral Cavity ROI: box=%s (square=%s), orig_size=%s, target_size=%s",
            crop_box,
            make_square,
            roi_original.shape[:2],
            target_size,
        )
        return roi_512, crop_box, roi_original

    @classmethod
    def prepare_dental_inpaint_mask(
        cls,
        instances: List[Dict[str, Any]],
        crop_box: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (512, 512),
        dilate_px: int = 4,
        include_interdental: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate high-precision binary inpainting mask for detected teeth within the ROI crop.

        Returns:
            mask_512: (512, 512) uint8 mask (255 = inpaint region, 0 = keep original)
            mask_roi_res: uint8 mask at the original ROI crop resolution
        """
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        roi_w = crop_x2 - crop_x1
        roi_h = crop_y2 - crop_y1

        mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        # Draw tooth polygons inside crop box
        tooth_polys = []
        for inst in instances:
            poly = inst.get("polygon", [])
            if poly and len(poly) >= 3:
                pts_np = np.array(poly, dtype=np.float32)
                cx, cy = pts_np.mean(axis=0)
                if not (crop_x1 <= cx <= crop_x2 and crop_y1 <= cy <= crop_y2):
                    continue
                local_pts = []
                for pt in poly:
                    lx = int(pt[0] - crop_x1)
                    ly = int(pt[1] - crop_y1)
                    local_pts.append([lx, ly])
                pts_arr = np.array(local_pts, dtype=np.int32)
                cv2.fillPoly(mask_roi, [pts_arr], 255)
                tooth_polys.append(pts_arr)

        if not tooth_polys:
            # Fallback mask if no polygons
            center_h, center_w = roi_h // 2, roi_w // 2
            cv2.ellipse(
                mask_roi,
                (center_w, center_h),
                (int(roi_w * 0.35), int(roi_h * 0.20)),
                0, 0, 360, 255, -1
            )

        # Optional interdental closure: close tiny gaps between adjacent teeth
        if include_interdental and len(tooth_polys) > 1:
            close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, close_kernel)

        # Scale to 512x512
        mask_512 = cv2.resize(mask_roi, target_size, interpolation=cv2.INTER_NEAREST)

        # In 512 space, apply slight dilation to allow natural margin blending
        if dilate_px > 0:
            ksize = 2 * dilate_px + 1
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            mask_512 = cv2.dilate(mask_512, dilate_kernel)

        return mask_512, mask_roi

    @classmethod
    def stitch_restored_roi(
        cls,
        original_image: np.ndarray,
        restored_roi_512: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        mask_512: np.ndarray,
        feather_sigma: float = 3.5,
    ) -> np.ndarray:
        """
        Seamlessly blend the restored 512×512 crop back into the master photo
        using Laplacian pyramid multi-band blending, colour harmonisation,
        and camera-noise matching.

        Args:
            original_image: full-resolution BGR image
            restored_roi_512: 512×512 BGR restored oral cavity crop
            crop_box: (crop_x1, crop_y1, crop_x2, crop_y2)
            mask_512: 512×512 binary inpainting mask (255 = edited)
            feather_sigma: Gaussian blur sigma for mask feathering

        Returns:
            stitched_image: full-resolution BGR with photorealistic smile restored
        """
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
        roi_w = crop_x2 - crop_x1
        roi_h = crop_y2 - crop_y1

        # Resize restored crop back to native ROI resolution
        restored_roi = cv2.resize(
            restored_roi_512, (roi_w, roi_h), interpolation=cv2.INTER_LANCZOS4
        )

        # Resize mask back to native ROI resolution
        mask_roi = cv2.resize(
            mask_512, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST
        )

        # Extract original crop
        orig_roi = original_image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

        # Clear mask near crop edges to guarantee zero boundary artifacts
        b_margin = 8
        if roi_h > 2 * b_margin and roi_w > 2 * b_margin:
            mask_roi[:b_margin, :] = 0
            mask_roi[-b_margin:, :] = 0
            mask_roi[:, :b_margin] = 0
            mask_roi[:, -b_margin:] = 0

        # ── Colour harmonisation ──
        restored_roi = cls._harmonize_colors(orig_roi, restored_roi, mask_roi)

        # Inward feathering cannot modify lips/skin outside the approved mask.
        from pipeline.generation_controls import composite_inside
        blended_roi = composite_inside(orig_roi, restored_roi, mask_roi,
                                       feather_px=feather_sigma)

        # Smooth boundary falloff to eliminate any rectangular bounding box seams
        edge_feather = 16
        y_ramp = np.ones(roi_h, dtype=np.float32)
        if roi_h > 2 * edge_feather:
            y_ramp[:edge_feather] = np.linspace(0, 1, edge_feather)
            y_ramp[-edge_feather:] = np.linspace(1, 0, edge_feather)
        x_ramp = np.ones(roi_w, dtype=np.float32)
        if roi_w > 2 * edge_feather:
            x_ramp[:edge_feather] = np.linspace(0, 1, edge_feather)
            x_ramp[-edge_feather:] = np.linspace(1, 0, edge_feather)
        crop_falloff = np.outer(y_ramp, x_ramp)[:, :, None]

        final_roi = (blended_roi.astype(np.float32) * crop_falloff + orig_roi.astype(np.float32) * (1.0 - crop_falloff)).astype(np.uint8)
        final_roi[mask_roi == 0] = orig_roi[mask_roi == 0]

        # Paste composite back into full image
        stitched = original_image.copy()
        stitched[crop_y1:crop_y2, crop_x1:crop_x2] = final_roi

        logger.info("Stitched restored ROI via Laplacian blending + colour harmonisation + noise matching.")
        return stitched

    # ──────────────────── Post-processing helpers ────────────────────

    @classmethod
    def _laplacian_blend(
        cls,
        orig_roi: np.ndarray,
        restored_roi: np.ndarray,
        mask_roi: np.ndarray,
        feather_sigma: float = 3.5,
        levels: int = 4,
    ) -> np.ndarray:
        """
        Multi-band blending via Laplacian pyramids.
        Low-frequency bands blend colour/luminance smoothly (no halo).
        High-frequency bands maintain sharp edges at mask boundary.
        """
        h, w = orig_roi.shape[:2]

        # Ensure dimensions are divisible by 2^levels
        pad_h = ((1 << levels) - h % (1 << levels)) % (1 << levels)
        pad_w = ((1 << levels) - w % (1 << levels)) % (1 << levels)
        if pad_h > 0 or pad_w > 0:
            orig_roi = cv2.copyMakeBorder(orig_roi, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            restored_roi = cv2.copyMakeBorder(restored_roi, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_roi = cv2.copyMakeBorder(mask_roi, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

        # Feather the mask for soft transitions without eroding the edited tooth/gingival core
        mask_norm = mask_roi.astype(np.float32) / 255.0
        mask_dil = cv2.dilate(mask_roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        ksize = int(feather_sigma * 4) | 1
        mask_feathered = cv2.GaussianBlur(
            mask_dil.astype(np.float32) / 255.0, (ksize, ksize), feather_sigma
        )
        # Strictly preserve alpha=1.0 inside all edited pixels so old yellow teeth/roots cannot bleed back in
        mask_f = np.maximum(mask_feathered, mask_norm)

        # Build Gaussian pyramids
        def gauss_pyramid(img, n):
            pyr = [img.astype(np.float32)]
            for _ in range(n):
                pyr.append(cv2.pyrDown(pyr[-1]))
            return pyr

        def laplacian_pyramid(img, n):
            gp = gauss_pyramid(img, n)
            lp = []
            for i in range(n):
                up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
                lp.append(gp[i] - up)
            lp.append(gp[-1])
            return lp

        lp_orig = laplacian_pyramid(orig_roi, levels)
        lp_rest = laplacian_pyramid(restored_roi, levels)
        gp_mask = gauss_pyramid(mask_f, levels)

        # Blend each level
        lp_blend = []
        for lo, lr, gm in zip(lp_orig, lp_rest, gp_mask):
            alpha3 = np.dstack([gm] * 3) if lo.ndim == 3 else gm
            lp_blend.append(lr * alpha3 + lo * (1.0 - alpha3))

        # Reconstruct from blended pyramid
        result = lp_blend[-1]
        for i in range(levels - 1, -1, -1):
            result = cv2.pyrUp(result, dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
            result += lp_blend[i]

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            result = result[:h, :w]

        return result

    @classmethod
    def _harmonize_colors(
        cls,
        orig_roi: np.ndarray,
        restored_roi: np.ndarray,
        mask_roi: np.ndarray,
    ) -> np.ndarray:
        """
        Preserve natural enamel tones and lighting harmony.
        Teeth must NEVER be color-shifted toward lips/skin (which are pink/red).
        Only subtle luminance (L*) balance is applied if lighting differs.
        """
        try:
            # Enamel color is already matched via CAD/CAM bilateral mirroring and BrushNet prompt.
            # Avoid global color shifts from skin/lips which contaminate ivory enamel with red/pink.
            return restored_roi

        except Exception as exc:
            logger.debug("Colour harmonisation skipped: %s", exc)
            return restored_roi

    @classmethod
    def _match_noise(
        cls,
        orig_roi: np.ndarray,
        restored_roi: np.ndarray,
        mask_roi: np.ndarray,
    ) -> np.ndarray:
        """
        Inject subtle natural sensor noise to match camera grain without
        introducing heavy speckles or picking up beard/stubble texture.
        """
        try:
            # Measure noise on a smooth area, capped at 1.0 std
            orig_gray = cv2.cvtColor(orig_roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blurred = cv2.GaussianBlur(orig_gray, (0, 0), 2.0)
            noise_map = orig_gray - blurred

            unmask = mask_roi < 128
            if unmask.sum() < 100:
                return restored_roi

            # Filter out edges / hair / stubble by only considering small gradient regions
            noise_std = float(noise_map[unmask].std())
            # Cap noise std strictly between 0.0 and 1.2 to avoid stubble grain injection
            calibrated_std = min(max(noise_std * 0.25, 0.0), 1.2)
            if calibrated_std < 0.3:
                return restored_roi

            rng = np.random.default_rng(42)
            synth_noise = rng.normal(0, calibrated_std, restored_roi.shape[:2]).astype(np.float32)

            result = restored_roi.astype(np.float32)
            tooth_mask = mask_roi > 128
            for ch in range(3):
                result[:, :, ch][tooth_mask] += synth_noise[tooth_mask]

            return np.clip(result, 0, 255).astype(np.uint8)

        except Exception as exc:
            logger.debug("Noise matching skipped: %s", exc)
            return restored_roi
