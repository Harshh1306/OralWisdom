"""
orchestrator.py
───────────────
End-to-end pipeline coordination for smile restoration.

Orchestrates the 5-layer pipeline:
  Layer A: Geometry Engine (tooth numbering + geometric analysis + color)
  Layer B: VLM Agent (manual input for now, API later)
  Layer C: Edit Planner (confidence filter + mask compiler + prompt gen)
  Layer D: BrushNet Engine (inpainting + post-processing)
  Layer E: Verification (re-segment + compare + pass/fail)

The pipeline has two user-facing phases:
  1. analyze() — Runs Layers A+B+C, returns proposed edits for review
  2. restore() — Runs Layers D+E, executes approved edits
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from uuid import uuid4
import hashlib
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from config.model_config import (
    OUTPUT_DIR,
    RESTORE_DIR,
    ANALYSIS_DIR,
    CROPS_DIR,
    MAX_RESTORATION_ATTEMPTS,
)

logger = logging.getLogger(__name__)


class SmileRestorationPipeline:
    """
    Orchestrates the full 5-layer smile restoration pipeline.
    """

    def __init__(self):
        self._vlm_agent = None
        self._edit_planner = None
        self._brushnet_engine = None
        self._verifier = None
        self._initialized = False

        # In-memory cached specs from analyze() to restore()
        self._cached_edit_specs = []
        self._cached_image_path = None
        self._cached_instances = []
        self._cached_geometry = {}
        self._plans = {}

    def _ensure_initialized(self) -> None:
        """Lazy initialization of pipeline components."""
        if self._initialized:
            return

        from pipeline.vlm_agent import create_vlm_agent
        from pipeline.edit_planner import EditPlanner
        from pipeline.verification import RestorationVerifier

        self._vlm_agent = create_vlm_agent()
        self._edit_planner = EditPlanner()
        self._verifier = RestorationVerifier()
        self._initialized = True

        for d in [RESTORE_DIR, ANALYSIS_DIR, CROPS_DIR]:
            d.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        image_path: Path,
        instances: list[dict],
        manual_analysis: Optional[dict] = None,
        design_options: Optional[dict] = None,
    ) -> dict:
        """
        Run geometric analysis + VLM reasoning + edit planning.
        """
        self._ensure_initialized()
        started = time.time()
        instances = deepcopy(instances)

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        logger.info("=== ANALYZE: Starting smile analysis ===")

        # ── Layer A: Geometry ────────────────────────────────────────────
        from pipeline.tooth_numbering import assign_tooth_numbers
        instances = assign_tooth_numbers(instances, image.shape[:2])

        from pipeline.geometry_engine import GeometryEngine
        engine = GeometryEngine(instances, image.shape[:2], image_bgr=image)
        geometry_result = engine.analyze()
        geometry_dict = geometry_result.to_dict()
        geometry_text = engine.get_geometry_summary_text()

        from pipeline.color_analysis import extract_tooth_colors, compute_color_deltas
        color_data = extract_tooth_colors(image, instances)
        color_data = compute_color_deltas(color_data, instances)
        geometry_dict["color_analysis"] = color_data

        crops = self._generate_tooth_crops(image, instances, job_id=image_path.stem)

        # ── Layer B: VLM Analysis ────────────────────────────────────────
        if manual_analysis is not None:
            from config.schemas import normalize_vlm_output
            manual_analysis = normalize_vlm_output(manual_analysis, instances)
            from pipeline.vlm_agent import ManualInputAgent
            if isinstance(self._vlm_agent, ManualInputAgent):
                is_valid, errors = self._vlm_agent.set_analysis(manual_analysis, instances=instances)
                if not is_valid:
                    return {
                        "error": "Invalid analysis JSON",
                        "validation_errors": errors,
                        "geometry": geometry_dict,
                        "geometry_text": geometry_text,
                    }
            vlm_output = manual_analysis
        else:
            # Manual analysis belongs only to the image submitted with it.
            from pipeline.vlm_agent import ManualInputAgent
            if isinstance(self._vlm_agent, ManualInputAgent):
                self._vlm_agent = ManualInputAgent()
            try:
                vlm_output = self._vlm_agent.analyze(
                    image_path, geometry_dict, instances, crops
                )
            except (NotImplementedError, ValueError):
                # Auto-generate clinical aesthetic analysis from detected tooth geometry
                vlm_output = self._generate_auto_aesthetic_analysis(
                    geometry_dict, instances
                )


        # Cosmetic operations, masks and explanations share one executable plan.
        from config.restoration_plan import validate_design_options
        if design_options is None:
            design_options = vlm_output.get('appearance_request')
            if design_options is None and manual_analysis is not None:
                operations = {}
                for tooth in vlm_output.get('teeth', []):
                    intent = tooth.get('edit_intent', {})
                    operation = ('veneer_preview' if intent.get('apply_porcelain_veneer') else
                        ('repair_crown' if intent.get('restore_incisal_contour') else
                         ('shade_only' if intent.get('match_neighboring_shade') else 'preserve')))
                    operations[str(tooth['tooth_id'])] = operation
                design_options = dict(upper_mode='preserve', lower_mode='preserve', tooth_operations=operations)
        design_options = validate_design_options(design_options)

        self._cached_image_path = image_path
        self._cached_instances = instances
        self._cached_geometry = geometry_dict
        plan_id = uuid4().hex
        from pipeline.roi_engine import OralROIEngine
        from pipeline.crown_designer import compile_design, native_mask
        _, crop_box, _ = OralROIEngine.extract_oral_cavity_roi(
            image, instances, padding_h=0.35, padding_w=0.20)
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        design = compile_design(image, crop_box, instances, design_options, plan_id, image_hash)
        scaffold, waxup = design['scaffold'], design['waxup']
        mask = design['mask'] | waxup['shade_mask']
        from pipeline.edit_planner import EditSpecification
        edit_specs = []
        for crown in design['plan']['crowns']:
            if crown['operation'] == 'preserve':
                continue
            edit_specs.append(EditSpecification(tooth_id=crown['tooth_id'], fdi_number=crown['fdi_number'],
                tooth_type=crown['tooth_type'], edit_type=crown['operation'], confidence=0.0,
                severity='not_assessed', description=f"{crown['arch'].title()} {crown['operation'].replace('_',' ')}; preserve cervical tissue"))
        self._cached_edit_specs = edit_specs
        mask_filename = f"{image_path.stem}_{plan_id}_mask.png"
        cv2.imwrite(str(ANALYSIS_DIR / mask_filename), mask)
        overlay = scaffold.copy()
        overlay[mask > 0] = (0.65 * overlay[mask > 0] + 0.35 * np.array([30,180,255])).astype(np.uint8)
        preview_filename = f"{image_path.stem}_{plan_id}_area.png"
        cv2.imwrite(str(ANALYSIS_DIR / preview_filename), overlay)
        scaffold_filename = f"{image_path.stem}_{plan_id}_scaffold.png"
        cv2.imwrite(str(ANALYSIS_DIR / scaffold_filename), scaffold)
        source_roi = cv2.resize(image[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2]], (512,512))
        for crown in design['plan']['crowns']:
            visible_points=crown['source_polygon'] or crown['target_polygon']
            if not visible_points:
                continue
            pts = (np.asarray(visible_points) - np.asarray(crop_box[:2])) * np.array([512/(crop_box[2]-crop_box[0]),512/(crop_box[3]-crop_box[1])])
            if crown['source_polygon']: cv2.polylines(source_roi,[pts.astype(np.int32)],True,(60,210,255),1)
            if crown['operation']!='preserve' and crown['target_polygon']:
                target_pts=(np.asarray(crown['target_polygon'])-np.asarray(crop_box[:2]))*np.array([512/(crop_box[2]-crop_box[0]),512/(crop_box[3]-crop_box[1])])
                cv2.polylines(source_roi,[target_pts.astype(np.int32)],True,(60,255,60),1)
            center = tuple(np.mean(pts,axis=0).astype(int))
            cv2.putText(source_roi,crown['tooth_id'],center,cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1,cv2.LINE_AA)
        selection_filename = f"{image_path.stem}_{plan_id}_selection.png"
        cv2.imwrite(str(ANALYSIS_DIR / selection_filename), source_roi)
        self._plans[plan_id] = {
            "image_hash": image_hash,
            "instances": deepcopy(instances), "geometry": geometry_dict,
            "edit_specs": edit_specs,
            "design": design,
        }
        while len(self._plans) > 8:
            self._plans.pop(next(iter(self._plans)))

        analysis_result = {
            "plan_id": plan_id,
            "scope": "full_smile_cosmetic_preview",
            "requires_visual_review": True,
            "edit_area_filename": preview_filename,
            "mask_filename": mask_filename,
            "scaffold_filename": scaffold_filename,
            "selection_filename": selection_filename,
            "appearance_plan": design['plan'],
            "design_warnings": waxup['design_warnings'],
            "input_quality": __import__('pipeline.input_quality',fromlist=['assess_input']).assess_input(image,instances),
            "status": "plan_ready",
            "geometry": geometry_dict,
            "vlm_analysis": vlm_output,
            "proposed_edits": [spec.to_dict() for spec in edit_specs],
            "num_edits": len(edit_specs),
            "geometry_text": geometry_text,
            "runtime_seconds": round(time.time() - started, 3),
        }

        analysis_path = ANALYSIS_DIR / f"{image_path.stem}_analysis.json"
        analysis_path.write_text(
            json.dumps(analysis_result, indent=2, default=str),
            encoding="utf-8",
        )

        return analysis_result

    def restore(
        self,
        image_path: Path,
        instances: list[dict],
        analysis: Optional[dict] = None,
        force_bilateral_mirror: Optional[bool] = None,
        quality_mode: str = "fast",
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        brushnet_conditioning_scale: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        control_guidance_end: Optional[float] = None,
        steps: Optional[int] = None,
        seed: int = 42,
        conditioning_mode: str = "scaffold",
        generation_callback=None,
    ) -> dict:
        """
        Execute BrushNet inpainting + verification.
        """
        self._ensure_initialized()
        started = time.time()
        from pipeline.generation_controls import validate_options
        options = validate_options(
            seed=seed, steps=32 if steps is None else steps,
            guidance_scale=7.5 if guidance_scale is None else guidance_scale,
            conditioning_scale=0.58 if brushnet_conditioning_scale is None else brushnet_conditioning_scale,
            control_guidance_end=0.74 if control_guidance_end is None else control_guidance_end,
            quality_mode=quality_mode, conditioning_mode=conditioning_mode,
        )
        if force_bilateral_mirror is not None and not isinstance(force_bilateral_mirror, bool):
            raise ValueError("force_bilateral_mirror must be true, false or null")
        if not instances:
            return {"error": "No teeth detected. Use a sharper, closer smile photograph."}
        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        plan = self._plans.get((analysis or {}).get("plan_id"))
        if analysis is not None and (plan is None or plan['image_hash'] != image_hash):
            raise ValueError('This plan is expired or belongs to another image. Analyze again.')
        if plan is None:
            # Rebuild from this image, never reuse another patient's cached plan.
            analysis = self.analyze(image_path, deepcopy(instances))
            if "error" in analysis:
                return analysis
            plan = self._plans[analysis["plan_id"]]
        instances = deepcopy(plan["instances"])

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        if not any(inst.get("fdi_number") is not None for inst in instances):
            from pipeline.tooth_numbering import assign_tooth_numbers
            instances = assign_tooth_numbers(instances, image.shape[:2])

        edit_specs = plan["edit_specs"]
        if not edit_specs:
            return {"error": "Could not plan restorative edits for detected teeth."}

        logger.info("=== RESTORE: Inpainting with %d edits (mode=%s) ===", len(edit_specs), quality_mode)

        # ── Layer D: Photorealistic Oral Cavity ROI Inpainting ───────────
        from pipeline.brushnet_engine import BrushNetEngine

        if self._brushnet_engine is None:
            self._brushnet_engine = BrushNetEngine()

        if force_bilateral_mirror is True:
            raise ValueError('Select a crown repair or reviewed replacement slot; automatic mirroring is retired.')
        if np.any(plan['design']['mask']):
            if generation_callback is not None:
                generation_callback('loading',0,options['steps'])
            self._brushnet_engine.load()
        inpaint_kwargs = {
            "image": image,
            "instances": instances,
            "seed": options["seed"],
            "force_bilateral_mirror": force_bilateral_mirror,
            "quality_mode": quality_mode,
            "conditioning_mode": conditioning_mode,
            "prepared_design": plan["design"],
            "generation_callback": generation_callback,
        }
        if prompt:
            inpaint_kwargs["prompt"] = prompt
        if negative_prompt:
            inpaint_kwargs["negative_prompt"] = negative_prompt
        if brushnet_conditioning_scale is not None:
            cond_val = float(brushnet_conditioning_scale)
            inpaint_kwargs["conditioning_scale"] = cond_val
        else:
            inpaint_kwargs["conditioning_scale"] = 0.58
        if guidance_scale is not None:
            inpaint_kwargs["guidance_scale"] = float(guidance_scale)
        if control_guidance_end is not None:
            inpaint_kwargs["control_guidance_end"] = float(control_guidance_end)
        else:
            inpaint_kwargs["control_guidance_end"] = 0.74
        if steps is not None:
            inpaint_kwargs["steps"] = int(steps)

        try:
            restored, restore_meta = self._brushnet_engine.inpaint_oral_cavity(**inpaint_kwargs)
            candidate_mask = self._brushnet_engine.last_edit_mask.copy()
            second_candidate = None
            if quality_mode == "quality" and restore_meta["engine_type"] == "brushnet":
                inpaint_kwargs["seed"] = (options["seed"] + 1009) % (2**32 - 1)
                second_image, second_meta = self._brushnet_engine.inpaint_oral_cavity(**inpaint_kwargs)
                second_candidate = (second_image, second_meta, self._brushnet_engine.last_edit_mask.copy())
        finally:
            self._brushnet_engine.unload()

        # ── Layer E: Verification ────────────────────────────────────────
        geometry = plan["geometry"]
        verification = self._verifier.verify(
            image, restored, edit_specs, instances, geometry,
            actual_mask=candidate_mask,
        )
        self._verifier.check_design_regions(image, restored, plan['design'], verification, restore_meta)

        # ── Save outputs ─────────────────────────────────────────────────
        RESTORE_DIR.mkdir(parents=True, exist_ok=True)
        job_id = uuid4().hex[:12]
        restored_filename = f"{image_path.stem}_{job_id}_restored.png"
        restored_path = RESTORE_DIR / restored_filename
        if not cv2.imwrite(str(restored_path), restored):
            raise OSError("Could not save the restored image")
        mask_filename = f"{image_path.stem}_{job_id}_allowed_mask.png"
        cv2.imwrite(str(RESTORE_DIR / mask_filename),candidate_mask)

        comparison_filename = f"{image_path.stem}_{job_id}_comparison.jpg"
        comparison_path = RESTORE_DIR / comparison_filename
        comparison = self._create_comparison(image, restored)
        cv2.imwrite(str(comparison_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 95])

        import os
        try:
            with open(restored_path, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            with open(comparison_path, "rb+") as f:
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

        result = {
            "restored_filename": restored_filename,
            "comparison_filename": comparison_filename,
            "restored_path": str(restored_path),
            "allowed_mask_path": str(RESTORE_DIR / mask_filename),
            "allowed_mask_filename": mask_filename,
            "metadata_filename": f"{image_path.stem}_{job_id}_metadata.json",
            "comparison_path": str(comparison_path),
            "passed": bool(verification.passed),
            "verification": verification.to_dict(),
            "requires_visual_review": True,
            "scope": "full_smile_cosmetic_preview",
            "appearance_plan": plan['design']['plan'],
            "result_status": 'review_required' if verification.passed else 'needs_revision',
            "planned_edits": [s.to_dict() for s in edit_specs],
            "num_edits_applied": None,
            "restore_meta": restore_meta,
            "runtime_seconds": round(time.time() - started, 3),
        }
        result["restore_meta"].update(image_sha256=image_hash, plan_id=analysis["plan_id"])
        result["candidates"] = [{"restored_filename": restored_filename,
            "comparison_filename": comparison_filename, "seed": options["seed"],
            "verification": verification.to_dict(), "engine_type": restore_meta["engine_type"],
            "passed": bool(verification.passed), "restore_meta": restore_meta}]
        if second_candidate is not None:
            second_image, second_meta, second_mask = second_candidate
            second_check = self._verifier.verify(image, second_image, edit_specs, instances,
                                                 geometry, actual_mask=second_mask)
            self._verifier.check_design_regions(image, second_image, plan['design'], second_check, second_meta)
            second_name = f"{image_path.stem}_{job_id}_candidate2.png"
            second_comparison = f"{image_path.stem}_{job_id}_candidate2_comparison.jpg"
            if not cv2.imwrite(str(RESTORE_DIR / second_name), second_image):
                raise OSError("Could not save second candidate")
            cv2.imwrite(str(RESTORE_DIR / second_comparison), self._create_comparison(image, second_image))
            result["candidates"].append(dict(restored_filename=second_name,
                comparison_filename=second_comparison, seed=second_meta["seed"],
                verification=second_check.to_dict(), engine_type=second_meta["engine_type"],
                restore_meta=second_meta, passed=bool(second_check.passed)))
            if not verification.passed and second_check.passed:
                result.update(restored_filename=second_name,comparison_filename=second_comparison,
                    restored_path=str(RESTORE_DIR/second_name),comparison_path=str(RESTORE_DIR/second_comparison),
                    verification=second_check.to_dict(),restore_meta=second_meta,passed=True,result_status='review_required')
                result['restore_meta'].update(image_sha256=image_hash,plan_id=analysis['plan_id'])
        (RESTORE_DIR / f"{image_path.stem}_{job_id}_metadata.json").write_text(
            json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    def _generate_auto_aesthetic_analysis(
        self,
        geometry_dict: dict,
        instances: list[dict],
    ) -> dict:
        """
        Describe estimated observations and default cosmetic appearance preferences.
        This adapter makes no clinical damage, numbering or treatment claims.
        """
        teeth_list = []
        for idx, inst in enumerate(instances):
            fdi_num = inst.get("fdi_number")
            t_id = inst.get("id", f"T{idx+1:02d}")
            ttype = inst.get("tooth_type", "tooth")

            is_max_anterior = (
                (fdi_num in (13, 12, 11, 21, 22, 23))
                or (ttype in ("central", "lateral", "canine"))
            )

            teeth_list.append({
                "tooth_id": t_id,
                "fdi_number": int(fdi_num) if fdi_num else None,
                "visibility": "unknown",
                "observations": {},
                "confidence": {"overall": 0.0},
                "source": "user_requested_cosmetic_preset",
                "aesthetic_priority": "high",
                "edit_intent": {
                    "apply_porcelain_veneer": inst.get('arch','upper') == 'upper',
                    "restore_incisal_contour": is_max_anterior,
                    "restore_vertical_dimension": False,
                    "reduce_apparent_rotation": False,
                    "match_neighboring_shade": True,
                    "target_shade": "natural ivory",
                    "close_gap": False,
                    "smooth_contour": True,
                    "restore_missing_structure": False,
                    "preserve_gingival_margin": True,
                    "preserve_tooth_position": True,
                    "preserve_root_axis": True
                }
            })

        smile_geo = geometry_dict.get("smile", {})
        return {
            "smile_analysis": {
                "smile_arc": smile_geo.get("arc_type", "consonant"),
                "midline_shift_px": round(float(smile_geo.get("midline_shift_px", 0.0)), 2),
                "midline_cant_degrees": round(float(smile_geo.get("midline_cant_deg", 0.0)), 2),
                "overall_symmetry": "not_assessed",
                "confidence": 0.0,
                "source": "cosmetic_preference_not_diagnosis",
                "procedure_type": "Full-smile cosmetic veneer preview",
                "target_shade": "natural ivory",
                "restore_vertical_dimension": False
            },
            "teeth": teeth_list
        }

    def _generate_tooth_crops(
        self,
        image: np.ndarray,
        instances: list[dict],
        padding_pct: float = 0.15,
        job_id: str = "preview",
    ) -> List[Path]:
        CROPS_DIR.mkdir(parents=True, exist_ok=True)
        crop_paths = []
        h, w = image.shape[:2]

        for inst in instances:
            bbox = inst.get("bbox", [0, 0, 0, 0])
            bx, by, bw, bh = [int(v) for v in bbox]

            pad_x = int(bw * padding_pct)
            pad_y = int(bh * padding_pct)
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = f"{job_id}_{inst.get('id', 'unknown')}_crop.jpg"
            crop_path = CROPS_DIR / crop_name
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(crop_path)

        return crop_paths

    @staticmethod
    def _create_comparison(original: np.ndarray, restored: np.ndarray) -> np.ndarray:
        h, w = original.shape[:2]
        before = original.copy()
        after = restored.copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(1.2, w / 800))
        thickness = max(1, int(font_scale * 2))

        cv2.putText(before, "BEFORE", (15, 35), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(before, "BEFORE", (15, 35), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        cv2.putText(after, "AFTER", (15, 35), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(after, "AFTER", (15, 35), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

        divider = np.ones((h, 3, 3), dtype=np.uint8) * 200
        comp = np.hstack([before, divider, after])

        # Bound comparison image width to 2560px for smooth, fast browser rendering
        max_w = 2560
        if comp.shape[1] > max_w:
            scale = max_w / float(comp.shape[1])
            new_h = int(comp.shape[0] * scale)
            comp = cv2.resize(comp, (max_w, new_h), interpolation=cv2.INTER_AREA)

        return comp
