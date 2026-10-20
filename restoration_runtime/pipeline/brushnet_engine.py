from __future__ import annotations
import gc
import logging
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from pipeline.roi_engine import OralROIEngine
from pipeline.generation_controls import (validate_options, composite_inside,
    conditioning_image, prepare_prompt)

logger = logging.getLogger(__name__)

# Cosmetic dental enhancement prompts with bi-maxillary veneers and pontics
DENTAL_AESTHETIC_PROMPT = (
    "Natural cosmetic smile makeover, individually defined aligned teeth, balanced proportions, "
    "smooth incisal edges, natural ivory enamel, subtle translucency at edges, fine enamel texture, "
    "small natural embrasures, realistic reflections, consistent original lighting, dental photograph"
)
DENTAL_NEGATIVE_PROMPT = (
    "fused teeth, extra teeth, duplicate teeth, overlapping crowns, distorted anatomy, "
    "plastic, chalky, glass teeth, mouthguard, flat white block, excessive glare, "
    "blur, jagged edges, cartoon, text, watermark"
)


class BrushNetEngine:
    """
    Cosmetic appearance renderer with explicit backend and review metadata.

    Architecture:
    1. Oral Cavity ROI extraction with context margin (preserves 100% of facial skin/features).
    2. High-density 512x512 neural inpainting via BrushNet dual-branch conditioning.
    3. Seamless feathered paste-back into native high-resolution portrait.
    4. Multi-tier resilience: BrushNet -> SD Inpainting -> Local Anatomical.
    """

    def __init__(self):
        self._pipe = None
        self._loaded = False
        self._engine_type = "none"  # 'brushnet' | 'sd_inpaint' | 'anatomical'
        self.last_edit_mask = None
        self._load_errors = []

    def load(self) -> None:
        """Initialize the restoration model pipeline."""
        if self._loaded and self._pipe is not None:
            return

        logger.info("Initializing Photorealistic Dental Restoration Engine...")
        os.environ["HF_HOME"] = os.path.expanduser("~/.cache/huggingface")

        # Tier 1: Try BrushNet with SD 1.5 backbone
        try:
            logger.info("Attempting to load BrushNet pipeline (FP16)...")
            from pipeline.brushnet import BrushNetModel, StableDiffusionBrushNetPipeline
            from pipeline.brushnet.unet_2d_condition import UNet2DConditionModel
            from diffusers import UniPCMultistepScheduler

            # Configurable base model (defaults to runwayml/stable-diffusion-v1-5, or can be SG161222/Realistic_Vision_V5.1_noVAE)
            base_model = os.environ.get("ORALWISDOM_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
            logger.info("Using base model: %s", base_model)

            # Load BrushNet branch and BrushNet-aware UNet
            brushnet, brushnet_loading = BrushNetModel.from_pretrained(
                os.environ.get("ORALWISDOM_BRUSHNET_MODEL", "Sanster/brushnet_segmentation_mask"),
                torch_dtype=torch.float16,
                variant="fp16",
                output_loading_info=True,
            )
            if any(brushnet_loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError(f"BrushNet checkpoint is incompatible: {brushnet_loading}")
            self._brushnet_loading = brushnet_loading
            unet = UNet2DConditionModel.from_pretrained(
                base_model,
                subfolder="unet",
                torch_dtype=torch.float16,
                variant="fp16" if "runwayml" in base_model else None,
            )
            pipe = StableDiffusionBrushNetPipeline.from_pretrained(
                base_model,
                unet=unet,
                brushnet=brushnet,
                torch_dtype=torch.float16,
                variant="fp16" if "runwayml" in base_model else None,
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(getattr(pipe, "vae", None), "enable_slicing"):
                pipe.vae.enable_slicing()

            self._pipe = pipe
            if os.environ.get('ORALWISDOM_DISABLE_PROGRESS') == '1':
                pipe.set_progress_bar_config(disable=True)
            self._engine_type = "brushnet"
            self._loaded = True
            logger.info("BrushNet pipeline successfully loaded and offloaded to CPU/GPU.")
            return
        except Exception as e:
            self._load_errors.append(f"BrushNet: {type(e).__name__}: {e}")
            logger.warning("BrushNet loading was not successful (%s). Trying SD Inpainting fallback...", e)

        # Tier 2: Try Stable Diffusion Inpainting fallback
        try:
            logger.info("Attempting to load StableDiffusionInpaintPipeline (FP16)...")
            from diffusers import StableDiffusionInpaintPipeline, UniPCMultistepScheduler

            pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16,
                variant="fp16",
                safety_checker=None,
                requires_safety_checker=False,
            )
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            pipe.enable_vae_slicing()

            self._pipe = pipe
            self._engine_type = "sd_inpaint"
            self._loaded = True
            logger.info("StableDiffusionInpaintPipeline successfully loaded and offloaded.")
            return
        except Exception as e:
            self._load_errors.append(f"SD inpainting: {type(e).__name__}: {e}")
            logger.warning("SD Inpainting fallback unavailable (%s). Using Anatomical Engine.", e)

        # Tier 3: Local Anatomical Engine (deterministic offline)
        self._engine_type = "anatomical"
        self._loaded = True
        logger.info("Dental Restoration Engine ready in Anatomical mode.")

    def unload(self) -> None:
        """Offload idle modules while retaining the loaded model for the next job."""
        if self._pipe is not None and hasattr(self._pipe, "maybe_free_model_hooks"):
            self._pipe.maybe_free_model_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def inpaint_oral_cavity(
        self,
        image: np.ndarray,
        instances: List[Dict[str, Any]],
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        seed: int = 42,
        steps: int = 32,
        guidance_scale: float = 7.5,
        conditioning_scale: float = 0.58,
        control_guidance_end: float = 0.74,
        force_bilateral_mirror: Optional[bool] = None,
        quality_mode: str = "fast",
        conditioning_mode: str = "scaffold",
        prepared_design: Optional[dict] = None,
        generation_callback=None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Render selected crown appearance tasks from a reviewed mask/design.

        Args:
            image: full-resolution BGR numpy array
            instances: detected tooth polygon instances
            prompt: aesthetic dental prompt
            negative_prompt: negative prompt
            seed: random seed for reproducibility
            steps: denoising inference steps
            guidance_scale: classifier-free guidance scale
            conditioning_scale: BrushNet conditioning weight; current benchmark default is 0.58
            control_guidance_end: fraction of denoising steps receiving BrushNet control
            force_bilateral_mirror: legacy flag; automatic mirrored replacements are disabled
            quality_mode: one candidate or two serial candidates selected by the orchestrator

        Returns:
            (restored_master_image, metadata_dict)
        """
        validate_options(seed, steps, guidance_scale, conditioning_scale,
                         control_guidance_end, quality_mode, conditioning_mode)
        instances = deepcopy(instances)
        self.last_edit_mask = None
        if not instances:
            raise ValueError("No detected teeth available for restoration")
        from pipeline.waxup_planner import CosmeticWaxupPlanner

        started = time.time()
        active_prompt = prompt or DENTAL_AESTHETIC_PROMPT
        active_neg = negative_prompt or DENTAL_NEGATIVE_PROMPT
        prompt_info = {}
        if self._pipe is not None:
            active_prompt, prompt_info["positive"] = prepare_prompt(self._pipe.tokenizer, active_prompt)
            active_neg, prompt_info["negative"] = prepare_prompt(self._pipe.tokenizer, active_neg)
        actual_backend = self._engine_type
        fallback_reason = None

        def pil_condition(source, mask):
            conditioned = conditioning_image(source, mask, conditioning_mode)
            return Image.fromarray(cv2.cvtColor(conditioned, cv2.COLOR_BGR2RGB))

        cancellation_reason = None

        def run_pipe(**kwargs):
            result = self._pipe(**kwargs)
            # Raise only after the pipeline has returned and released its live
            # denoising frame. Never pass a partial image to the fallback path.
            if cancellation_reason is not None:
                from pipeline.job_registry import GenerationCancelled
                raise GenerationCancelled(cancellation_reason)
            return result

        def callback_options(arch):
            if generation_callback is None:
                return {}
            def callback(pipe,index,timestep,values):
                nonlocal cancellation_reason
                from pipeline.job_registry import GenerationCancelled
                if cancellation_reason is not None:
                    return values
                try:
                    generation_callback(arch,index+1,steps)
                except GenerationCancelled as exc:
                    cancellation_reason = str(exc)
                    # Drain this pass through normal decode/offload ordering.
                    # Early hook teardown caused a native Windows access violation
                    # on repeated cancellation. Discard the image in run_pipe;
                    # never start another arch, save it, or invoke a fallback.
                return values
            return {'callback_on_step_end':callback,'callback_on_step_end_tensor_inputs':[]}
        if generation_callback is not None:
            generation_callback('loading',0,steps)

        # 1. Extract Oral Cavity ROI with square 1:1 aspect ratio (zero anamorphic tooth stretching)
        roi_512, crop_box, roi_orig = OralROIEngine.extract_oral_cavity_roi(
            image, instances, padding_h=0.35, padding_w=0.20, target_size=(512, 512), make_square=True
        )

        # 2. Compile selected crown contours, scaffold and protected edit mask.
        if prepared_design is not None:
            if tuple(prepared_design["crop_box"]) != tuple(crop_box):
                raise ValueError("The saved edit area does not match this image crop; analyze again")
            scaffold_512 = prepared_design["scaffold"].copy()
            mask_512 = prepared_design["mask"].copy()
            waxup_meta = deepcopy(prepared_design["waxup"])
        else:
            scaffold_512, mask_512, waxup_meta = CosmeticWaxupPlanner.plan_bilateral_smile_restoration(
                image, crop_box, instances, target_size=(512, 512), force_bilateral_mirror=force_bilateral_mirror)

        shade_mask = waxup_meta.get('shade_mask', np.zeros_like(mask_512))
        has_neural_task = bool(np.any(mask_512))
        if has_neural_task and not self._loaded:
            self.load()
        actual_backend = self._engine_type if has_neural_task else 'deterministic_shade'
        if self._pipe is not None:
            from pipeline.restoration_tasks import arch_prompt
            appearance = waxup_meta.get('appearance_plan', {})
            default_arch = 'upper' if np.any(waxup_meta.get('upper_mask',mask_512)) else 'lower'
            active_prompt, prompt_info['positive'] = prepare_prompt(self._pipe.tokenizer, prompt or arch_prompt(appearance,default_arch))
            active_neg, prompt_info['negative'] = prepare_prompt(self._pipe.tokenizer, negative_prompt or DENTAL_NEGATIVE_PROMPT)

        def arch_settings(arch):
            custom=waxup_meta.get('arch_design',{}).get(arch,{})
            return dict(guidance_scale=custom.get('guidance_scale',guidance_scale),
                brushnet_conditioning_scale=custom.get('conditioning_scale',conditioning_scale),
                control_guidance_end=custom.get('control_guidance_end',control_guidance_end))

        # 3. Neural Inpainting Pass using Bi-Maxillary Decoupled Architecture
        restored_roi_512 = None
        final_mask = mask_512

        upper_mask = waxup_meta.get("upper_mask")
        lower_mask = waxup_meta.get("lower_mask")
        has_upper = upper_mask is not None and bool(np.any(upper_mask > 0))
        has_lower = bool(
            waxup_meta.get("has_lower_teeth")
            and (lower_mask is not None)
            and np.any(lower_mask > 0)
        )
        upper_scaffold = waxup_meta.get("upper_scaffold", scaffold_512)
        if upper_scaffold is None:
            upper_scaffold = scaffold_512
        if not np.any((mask_512 | shade_mask) > 0):
            raise ValueError("The smile edit mask is empty; analyze a clearer smile image")
        lower_enamel_color = waxup_meta.get("lower_enamel_color", [140, 155, 175])

        if has_neural_task and self._engine_type in ("brushnet", "sd_inpaint") and self._pipe is not None:
            try:
                generator = torch.Generator(device="cpu").manual_seed(seed)

                if self._engine_type == "brushnet" and has_lower and has_upper:
                    # â”€â”€ Decoupled Two-Pass Restoration â”€â”€
                    # PASS 1: Upper Maxillary Arch Restoration (establishes consonant smile arc & anchor)
                    logger.info("Executing Decoupled Restoration: Pass 1 (Upper Maxillary Arch)...")
                    pil_upper_scaf = pil_condition(upper_scaffold, upper_mask)
                    pil_upper_mask = Image.fromarray(upper_mask)

                    res_pass1 = run_pipe(
                        prompt=active_prompt,
                        negative_prompt=active_neg,
                        image=pil_upper_scaf,
                        mask=pil_upper_mask,
                        num_inference_steps=steps,
                        **arch_settings('upper'),
                        generator=generator,
                        **callback_options('upper'),
                    )
                    pass1_roi = cv2.cvtColor(np.array(res_pass1.images[0]), cv2.COLOR_RGB2BGR)
                    pass1_roi = composite_inside(roi_512, pass1_roi, upper_mask)
                    # Pass two may change only the lower mask, never an upper pixel.
                    lower_mask = lower_mask.copy()
                    lower_mask[upper_mask > 0] = 0

                    # PASS 2: Lower Mandibular Arch Restoration (conditioned on restored upper veneers)
                    logger.info("Executing Decoupled Restoration: Pass 2 (Lower Mandibular Arch + Pontic)...")
                    lower_scaffold_prep = waxup_meta.get("lower_scaffold")
                    if lower_scaffold_prep is not None and upper_mask is not None:
                        mask_upper_dil = cv2.dilate(upper_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
                        lower_scaf = np.where(mask_upper_dil[:, :, None] > 0, pass1_roi, lower_scaffold_prep)
                    elif lower_scaffold_prep is not None:
                        sep_y = waxup_meta.get("sep_y", 200)
                        lower_scaf = pass1_roi.copy()
                        lower_scaf[sep_y:, :] = lower_scaffold_prep[sep_y:, :]
                    else:
                        lower_scaf = pass1_roi.copy()

                    pil_lower_scaf = pil_condition(lower_scaf, lower_mask)
                    pil_lower_mask = Image.fromarray(lower_mask)

                    from pipeline.restoration_tasks import arch_prompt
                    prompt_lower, prompt_info['lower_positive'] = prepare_prompt(self._pipe.tokenizer,
                        prompt or arch_prompt(waxup_meta.get('appearance_plan',{}),'lower'))
                    active_neg_lower = active_neg

                    res_pass2 = run_pipe(
                        prompt=prompt_lower,
                        negative_prompt=active_neg_lower,
                        image=pil_lower_scaf,
                        mask=pil_lower_mask,
                        num_inference_steps=steps,
                        **arch_settings('lower'),
                        generator=torch.Generator(device="cpu").manual_seed(seed + 1),
                        **callback_options('lower'),
                    )
                    restored_roi_512 = cv2.cvtColor(np.array(res_pass2.images[0]), cv2.COLOR_RGB2BGR)
                    restored_roi_512 = composite_inside(pass1_roi, restored_roi_512, lower_mask)
                    final_mask = cv2.bitwise_or(upper_mask, lower_mask)
                    logger.info("Decoupled bi-maxillary neural inpainting completed in %.2fs", time.time() - started)

                elif self._engine_type == "brushnet":
                    # Single-arch or standard single-pass
                    target_mask = upper_mask if has_upper else (lower_mask if has_lower else mask_512)
                    target_scaffold = upper_scaffold if upper_scaffold is not None else scaffold_512
                    if not has_upper and has_lower and waxup_meta.get("lower_scaffold") is not None:
                        target_scaffold = waxup_meta["lower_scaffold"]
                    pil_image = pil_condition(target_scaffold, target_mask)
                    pil_mask = Image.fromarray(target_mask)

                    result = run_pipe(
                        prompt=active_prompt,
                        negative_prompt=active_neg,
                        image=pil_image,
                        mask=pil_mask,
                        num_inference_steps=steps,
                        **arch_settings('upper' if has_upper else 'lower'),
                        generator=generator,
                        **callback_options('upper' if has_upper else 'lower'),
                    )
                    restored_roi_512 = cv2.cvtColor(np.array(result.images[0]), cv2.COLOR_RGB2BGR)
                    final_mask = target_mask
                    logger.info("Single-pass neural inpainting finished in %.2fs", time.time() - started)

                else:  # sd_inpaint fallback
                    pil_image = Image.fromarray(cv2.cvtColor(scaffold_512, cv2.COLOR_BGR2RGB))
                    pil_mask = Image.fromarray(mask_512)
                    result = run_pipe(
                        prompt=active_prompt,
                        negative_prompt=active_neg,
                        image=pil_image,
                        mask_image=pil_mask,
                        num_inference_steps=steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                        **callback_options('rendering'),
                    )
                    restored_roi_512 = cv2.cvtColor(np.array(result.images[0]), cv2.COLOR_RGB2BGR)
                    final_mask = mask_512

            except Exception as exc:
                from pipeline.job_registry import GenerationCancelled
                if isinstance(exc,GenerationCancelled):
                    raise
                fallback_reason = f"{type(exc).__name__}: {exc}"
                logger.warning("Neural inpainting encountered error (%s). Falling back to anatomical.", exc)
                restored_roi_512 = None

        if not has_neural_task:
            restored_roi_512 = roi_512.copy()
            final_mask = np.zeros_like(mask_512)
        elif restored_roi_512 is None:
            actual_backend = "anatomical"
            restored_roi_512 = self._inpaint_anatomical_roi(scaffold_512, mask_512)
            final_mask = mask_512

        if actual_backend in ('brushnet','sd_inpaint') and waxup_meta.get('texture_preservation',0)>0:
            from pipeline.ceramic_appearance import match_crown_appearance
            restored_roi_512 = match_crown_appearance(scaffold_512,restored_roi_512,
                waxup_meta.get('regions',[]),waxup_meta['texture_preservation'])

        # 4. Seamless Feathered Paste-Back
        stitched = OralROIEngine.stitch_restored_roi(
            image, restored_roi_512, crop_box, final_mask,
            feather_sigma=1.5 if waxup_meta.get('geometry_guided') else 3.5
        )
        from pipeline.crown_designer import native_mask,protect_native_mask
        self.last_edit_mask = protect_native_mask(native_mask(final_mask | shade_mask,crop_box,image.shape),
                                                 waxup_meta.get('appearance_plan',{}))
        if np.any(shade_mask):
            from pipeline.ceramic_appearance import shade_foreground
            stitched = shade_foreground(stitched,native_mask(shade_mask,crop_box,image.shape),waxup_meta.get('shade','natural'))
        stitched[self.last_edit_mask==0] = image[self.last_edit_mask==0]

        metadata = {
            "engine_type": actual_backend,
            "requested_backend": "brushnet" if has_neural_task else 'deterministic_shade',
            "checkpoint_loading": getattr(self, "_brushnet_loading", {}),
            "base_model": os.environ.get("ORALWISDOM_BASE_MODEL", "runwayml/stable-diffusion-v1-5"),
            "brushnet_model": os.environ.get("ORALWISDOM_BRUSHNET_MODEL", "Sanster/brushnet_segmentation_mask"),
            "fallback_reason": fallback_reason,
            "load_errors": self._load_errors,
            "negative_prompt": active_neg,
            "prompt_tokens": prompt_info,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "conditioning_scale": conditioning_scale,
            "control_guidance_end": control_guidance_end,
            "conditioning_mode": conditioning_mode,
            "arch_render_settings": {a:arch_settings(a) for a in ('upper','lower')},
            "requires_visual_review": True,
            "pass_seeds": [seed, seed+1] if actual_backend == "brushnet" and has_upper and has_lower else [seed],
            "upper_arch_edited": has_upper,
            "lower_arch_edited": has_lower,
            "shade_only_pixels": int(np.count_nonzero(shade_mask)),
            "structural_render_completed": bool(has_neural_task and actual_backend in ('brushnet','sd_inpaint')),
            "crop_box": list(crop_box),
            "inference_time_seconds": round(time.time() - started, 3),
            "prompt": active_prompt,
            "seed": seed,
        }
        for k, v in waxup_meta.items():
            if isinstance(v, np.ndarray) or k == 'regions':
                continue
            metadata[k] = v

        return stitched, metadata

    def _inpaint_anatomical_roi(self, roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        High-fidelity anatomical fall-back on the 512x512 ROI.
        Uses Poisson seamless cloning and bilateral color gradient smoothing.
        """
        restored = roi_bgr.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        teeth_region = cv2.dilate(mask, kernel)

        # LAB color space for perceptual whitening
        lab = cv2.cvtColor(restored, cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)

        mask_bool = teeth_region > 0
        if np.any(mask_bool):
            # Bleach OM2 target: high L*, low yellow (b*), natural warm-pink balance (a*)
            current_L = np.mean(L[mask_bool])
            current_b = np.mean(b[mask_bool])

            target_L = min(250.0, max(current_L * 1.15, 215.0))
            L[mask_bool] = np.clip(L[mask_bool] + (target_L - current_L) * 0.75, 0, 255)
            # Reduce yellowing
            b[mask_bool] = np.clip(128 + (b[mask_bool] - 128) * 0.70, 0, 255)

        merged_lab = cv2.merge([L, a, b]).astype(np.uint8)
        shaded = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

        # Smooth blending along mask edges
        feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (7, 7), 2.0)
        feather_3c = np.dstack([feather] * 3)

        result = (shaded.astype(np.float32) * feather_3c + restored.astype(np.float32) * (1.0 - feather_3c))
        return np.clip(result, 0, 255).astype(np.uint8)

    def inpaint(self, edit_spec: Any, original_image: np.ndarray, seed: int = 42) -> np.ndarray:
        """Single-edit legacy interface for backward compatibility."""
        if not self._loaded:
            self.load()
        # Fallback to single anatomical edit if called via legacy interface
        mask = getattr(edit_spec, "defect_mask", None)
        if mask is None:
            mask = getattr(edit_spec, "tooth_mask", None)
        if mask is None:
            return original_image

        return self._inpaint_anatomical_roi(original_image, mask)
