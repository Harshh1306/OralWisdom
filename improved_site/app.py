from __future__ import annotations

import json
import math
import os
import time
import secrets
from threading import RLock
from functools import wraps
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from shade_engine import apply_shade_from_preset, apply_shade_to_image, get_preset_params, list_presets


ROOT = Path(__file__).resolve().parent
os.environ["ORALWISDOM_SITE_ROOT"] = str(ROOT)
import sys
# Strip current dir from front of sys.path to prevent FAT32 USB corruption conflicts
sys.path = [p for p in sys.path if p not in ("", ".", str(ROOT))]
C_SITE_PACKAGES = Path(r"C:\OralWisdom_venv\Lib\site-packages")
if C_SITE_PACKAGES.exists() and str(C_SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(C_SITE_PACKAGES))
sys.path.append(str(ROOT))
# Use the recovered, project-owned implementation before environment packages.
RUNTIME_ROOT = Path(os.environ.get("ORALWISDOM_RUNTIME_ROOT",
    str(ROOT.parent / "restoration_runtime")))
sys.path.insert(0, str(RUNTIME_ROOT))
MODEL_PATH = Path(os.environ.get("ORALWISDOM_MODEL_PATH",
    str(ROOT / "models" / "best.pt") if (ROOT / "models" / "best.pt").exists()
    else r"D:\OralWisdom\projfiles\v3_inference_site\models\best.pt"))
OUTPUT_DIR = Path(os.environ.get("ORALWISDOM_OUTPUT_DIR", r"C:\OralWisdom_outputs"))
UPLOAD_DIR = OUTPUT_DIR / "uploads"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
JSON_DIR = OUTPUT_DIR / "json"
SHADE_DIR = OUTPUT_DIR / "shaded"
RESTORE_DIR = OUTPUT_DIR / "restored"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
CROPS_DIR = OUTPUT_DIR / "crops"
ULTRALYTICS_CONFIG_DIR = ROOT / ".ultralytics"

# ── In-memory cache: stem → {"image_path": Path, "instances": list} ──────────
_prediction_cache: dict[str, dict] = {}

os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {0: "tooth", 1: "canine", 2: "incisor"}
CLASS_COLORS_BGR = {
    0: (255, 0, 0),    # tooth: blue
    1: (0, 0, 255),    # canine: red
    2: (0, 200, 0),    # incisor: green
}
CLASS_COLORS_HEX = {
    0: "#2563eb",
    1: "#ef4444",
    2: "#22c55e",
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024


@app.before_request
def optional_access_control():
    user,password=os.environ.get('ORALWISDOM_AUTH_USER'),os.environ.get('ORALWISDOM_AUTH_PASSWORD')
    if not user and not password:
        return None
    credentials=request.authorization
    if not (user and password and credentials and secrets.compare_digest(credentials.username or '',user)
            and secrets.compare_digest(credentials.password or '',password)):
        return 'Authentication required',401,{'WWW-Authenticate':'Basic realm="OralWisdom"'}


_job_registry=None


def _get_jobs():
    global _job_registry
    from pipeline.job_registry import JobRegistry
    folder=OUTPUT_DIR/'jobs'
    if _job_registry is None or _job_registry.folder!=folder:
        _job_registry=JobRegistry(folder)
    return _job_registry
model = None
_gpu_lock = RLock()


def serialized_gpu(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not _gpu_lock.acquire(blocking=False):
            return jsonify({"error": "Another image is being processed. Please retry shortly."}), 409
        try:
            return fn(*args, **kwargs)
        finally:
            _gpu_lock.release()
    return wrapped


def ensure_dirs() -> None:
    for folder in (UPLOAD_DIR, OVERLAY_DIR, JSON_DIR, SHADE_DIR, RESTORE_DIR, ANALYSIS_DIR, CROPS_DIR, ROOT / "models", ULTRALYTICS_CONFIG_DIR, ROOT / ".matplotlib"):
        folder.mkdir(parents=True, exist_ok=True)


def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found. Put your trained V3 checkpoint at: {MODEL_PATH}"
        )
    if model is None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Could not load Ultralytics. Run 01_setup_env.bat, then restart the site with 02_run_site.bat. "
                f"Details: {exc}"
            ) from exc
        model = YOLO(str(MODEL_PATH))
    return model


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def polygon_metrics(poly: list[list[float]]) -> dict:
    pts = np.array(poly, dtype=np.float32)
    area = abs(cv2.contourArea(pts))
    moments = cv2.moments(pts)
    if abs(moments["m00"]) > 1e-6:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = float(pts[:, 0].mean())
        cy = float(pts[:, 1].mean())

    x, y, w, h = cv2.boundingRect(pts.astype(np.int32))
    rect = cv2.minAreaRect(pts)
    rw, rh = rect[1]

    centered = pts - pts.mean(axis=0)
    if len(pts) >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        vx, vy = vh[0]
        angle = math.degrees(math.atan2(vy, vx))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
    else:
        angle = 0.0

    return {
        "polygon": poly,
        "area_px": float(area),
        "centroid": [float(cx), float(cy)],
        "bbox": [float(x), float(y), float(w), float(h)],
        "rotated_width_px": float(max(rw, rh)),
        "rotated_height_px": float(min(rw, rh)),
        "principal_axis_angle_deg": float(angle),
    }


def prediction_to_instances(result) -> list[dict]:
    if result.masks is None or result.boxes is None:
        return []

    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()
    instances = []

    for idx, polygon in enumerate(result.masks.xy, start=1):
        if len(polygon) < 3:
            continue
        class_id = int(classes[idx - 1]) if idx - 1 < len(classes) else 0
        confidence = float(confidences[idx - 1]) if idx - 1 < len(confidences) else None
        poly = polygon.astype(float).tolist()
        item = polygon_metrics(poly)
        item.update(
            {
                "id": f"T{idx:02d}",
                "class_id": class_id,
                "class_name": CLASS_NAMES.get(class_id, f"class_{class_id}"),
                "confidence": confidence,
                "color": CLASS_COLORS_HEX.get(class_id, "#60a5fa"),
            }
        )
        instances.append(item)

    return instances


def draw_custom_overlay(image_path: Path, instances: list[dict], output_path: Path, fill_alpha: float = 0.28) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]

    # Calculate average tooth dimension dynamically to scale borders, text, and dots
    if instances:
        areas = [inst.get("area_px", 0) for inst in instances if inst.get("area_px", 0) > 0]
        if areas:
            avg_dim = math.sqrt(float(np.median(areas)))
        else:
            avg_dim = min(h, w) * 0.05
    else:
        avg_dim = min(h, w) * 0.05

    # Adaptive scaling rules for clean visualization on any resolution or distance
    line_thickness = max(1, min(3, int(round(avg_dim * 0.04))))
    dot_radius     = max(1, min(3, int(round(avg_dim * 0.035))))
    font_scale     = max(0.28, min(0.50, avg_dim * 0.007))
    font_thick     = 1 if font_scale < 0.42 else 2

    # Draw semi-transparent filled mask
    fill_layer = image.copy()
    for instance in instances:
        pts = np.asarray(instance["polygon"], dtype=np.int32)
        class_id = int(instance.get("class_id", 0))
        color = CLASS_COLORS_BGR.get(class_id, (255, 0, 0))
        cv2.fillPoly(fill_layer, [pts], color=color)

    image = cv2.addWeighted(fill_layer, fill_alpha, image, 1.0 - fill_alpha, 0)

    # Draw scaled outlines & text labels
    for instance in instances:
        pts = np.asarray(instance["polygon"], dtype=np.int32)
        class_id = int(instance.get("class_id", 0))
        color = CLASS_COLORS_BGR.get(class_id, (255, 0, 0))

        # Dynamic polyline border
        cv2.polylines(image, [pts], True, color, line_thickness, cv2.LINE_AA)

        cx, cy = (int(v) for v in instance["centroid"])

        # Centroid dot
        if dot_radius >= 1:
            cv2.circle(image, (cx, cy), dot_radius, (255, 255, 255), -1, cv2.LINE_AA)

        # Compact label on full-face / distant shots to prevent text overlap
        if avg_dim < 35:
            label = instance["id"]
        else:
            label = f'{instance["id"]} {instance["class_name"]}'

        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        tx = max(2, min(w - text_size[0] - 2, cx + 3))
        ty = max(text_size[1] + 2, min(h - 2, cy - 3))

        # High-contrast text with dark shadow outline
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 1, cv2.LINE_AA)
        cv2.putText(image, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    cv2.imwrite(str(output_path), image)


def run_prediction(image_path: Path) -> dict:
    yolo = load_model()
    started = time.time()
    result = yolo.predict(
        source=str(image_path),
        imgsz=1024,
        conf=0.20,
        iou=0.65,
        retina_masks=True,
        verbose=False,
    )[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    instances = prediction_to_instances(result)
    overlay_path = OVERLAY_DIR / f"{image_path.stem}_v3_overlay.jpg"
    json_path = JSON_DIR / f"{image_path.stem}_v3_predictions.json"
    draw_custom_overlay(image_path, instances, overlay_path)

    class_counts = {name: 0 for name in CLASS_NAMES.values()}
    for instance in instances:
        class_counts[instance["class_name"]] = class_counts.get(instance["class_name"], 0) + 1

    output = {
        "image": image_path.name,
        "image_size": {"width": width, "height": height},
        "model": str(MODEL_PATH),
        "num_instances": len(instances),
        "class_counts": class_counts,
        "instances": instances,
        "visualization": {
            "tooth": "blue mask",
            "canine": "red mask",
            "incisor": "green mask",
            "boxes": "not drawn",
            "confidence_labels": "not drawn",
        },
        "runtime_seconds": round(time.time() - started, 3),
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    # ── Cache instances so /shade can reuse them without re-running YOLO ──
    _prediction_cache[image_path.stem] = {
        "image_path": image_path,
        "instances": instances,
    }
    while len(_prediction_cache) > 64:
        _prediction_cache.pop(next(iter(_prediction_cache)))

    return {
        "image_name": image_path.name,
        "image_stem": image_path.stem,
        "overlay_url": url_for("output_file", kind="overlays", filename=overlay_path.name),
        "json_url": url_for("output_file", kind="json", filename=json_path.name),
        "upload_url": url_for("output_file", kind="uploads", filename=image_path.name),
        "overlay_path": str(overlay_path),
        "json_path": str(json_path),
        "counts": class_counts,
        "num_instances": len(instances),
        "runtime_seconds": output["runtime_seconds"],
        "shade_presets": list_presets(),
    }


@app.route("/")
def index():
    model_ready = MODEL_PATH.exists()
    return render_template("index.html", model_ready=model_ready, model_path=str(MODEL_PATH))


@app.route("/predict", methods=["POST"])
@serialized_gpu
def predict():
    ensure_dirs()
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Use a JPG, PNG, WEBP, or BMP image."}), 400

    original_name = secure_filename(file.filename)
    image_id = f"{Path(original_name).stem}_{uuid4().hex[:8]}{Path(original_name).suffix.lower()}"
    image_path = UPLOAD_DIR / image_id
    file.save(image_path)
    try:
        from PIL import Image, ImageOps
        with Image.open(image_path) as uploaded:
            if uploaded.width * uploaded.height > 24_000_000:
                return jsonify({"error": "Image is too large. Please use an image below 24 megapixels."}), 400
            normalized = ImageOps.exif_transpose(uploaded).convert("RGB")
            canonical_path = image_path.with_suffix(".png")
            normalized.save(canonical_path, format="PNG")
            image_path = canonical_path
    except Exception:
        return jsonify({"error": "Could not decode the uploaded image."}), 400
    try:
        with open(image_path, "rb+") as f:
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass

    test_img = cv2.imread(str(image_path))
    if test_img is None:
        return jsonify({"error": "The uploaded image could not be decoded. Please upload a valid JPG or PNG image."}), 400

    try:
        return jsonify(run_prediction(image_path))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/outputs/<kind>/<path:filename>")
def output_file(kind: str, filename: str):
    if kind not in {"uploads", "overlays", "json", "shaded", "restored", "analysis", "crops"}:
        return "Not found", 404
    primary_path = OUTPUT_DIR / kind / filename
    target_dir = OUTPUT_DIR / kind if primary_path.exists() else ROOT / "outputs" / kind
    if not (target_dir / filename).exists():
        return "Not found", 404
    as_attachment = request.args.get("download", "0") == "1"
    mimetype = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else None
    return send_from_directory(
        target_dir,
        filename,
        as_attachment=as_attachment,
        download_name=filename,
        mimetype=mimetype,
    )


@app.route("/presets")
def presets():
    """Return all available VITA shade presets."""
    return jsonify({"presets": list_presets()})


@app.route("/shade", methods=["POST"])
def shade():
    """
    Apply a shade change to a previously predicted image.

    JSON body fields (all optional except image_stem):
        image_stem  : str   — stem of the uploaded image (from /predict response)
        preset      : str   — VITA preset name  (default: 'A1')
        intensity   : float — 0.0–1.0 preset strength  (default: 1.0)
        brightness  : float — L* boost override  (used when preset='custom')
        yellow_red  : float — b* reduction override
        warmth      : float — a* shift override
    """
    ensure_dirs()
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    image_stem = body.get("image_stem", "")
    if not isinstance(image_stem, str) or not image_stem:
        return jsonify({"error": "image_stem is required"}), 400

    cached = _prediction_cache.get(image_stem)
    if cached is None:
        return jsonify({"error": f"No prediction cached for '{image_stem}'. Run /predict first."}), 404

    image_path: Path = cached["image_path"]
    instances: list[dict] = cached["instances"]

    image = cv2.imread(str(image_path))
    if image is None:
        return jsonify({"error": f"Source image not found: {image_path}"}), 500

    preset = body.get("preset", "A1")
    try:
        if not isinstance(preset, str) or preset not in list_presets() + ["custom"]:
            raise ValueError("Unknown shade preset")
        intensity = float(body.get("intensity", 1.0))
        if not math.isfinite(intensity) or not 0 <= intensity <= 1:
            raise ValueError("Shade intensity must be between 0 and 1")
        custom_values = [float(body.get(key, 0.0)) for key in ("brightness", "yellow_red", "warmth")]
        if not all(math.isfinite(v) and -255 <= v <= 255 for v in custom_values):
            raise ValueError("Shade adjustments must be finite numbers between -255 and 255")
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400

    if preset == "custom":
        params = get_preset_params("original")          # start from neutral
        brightness = float(body.get("brightness", 0.0))
        yellow_red = float(body.get("yellow_red", 0.0))
        warmth     = float(body.get("warmth",     0.0))
        shaded = apply_shade_to_image(image, instances, brightness, yellow_red, warmth)
    else:
        shaded = apply_shade_from_preset(image, instances, preset, intensity)

    # Save clean (no-overlay) shaded image
    shade_filename = f"{image_stem}_shade_{preset}_{uuid4().hex[:8]}.png"
    shade_path = SHADE_DIR / shade_filename
    if not cv2.imwrite(str(shade_path), shaded):
        return jsonify({"error": "Could not save the shade preview"}), 500

    return jsonify({
        "shaded_url": url_for("output_file", kind="shaded", filename=shade_filename),
        "shade_path": str(shade_path),
        "preset": preset,
        "intensity": intensity,
    })


# ── Smile Restoration Pipeline ─────────────────────────────────────────────
# Lazy-loaded to avoid importing heavy dependencies at startup
_restoration_pipeline = None


def _get_pipeline():
    global _restoration_pipeline
    if _restoration_pipeline is None:
        from pipeline.orchestrator import SmileRestorationPipeline
        _restoration_pipeline = SmileRestorationPipeline()
    return _restoration_pipeline


# In-memory cache for analysis results (keyed by image_stem)
_analysis_cache: dict[str, dict] = {}


@app.route("/analyze", methods=["POST"])
@serialized_gpu
def analyze():
    """
    Run Layer A (geometry) + Layer B (VLM) + Layer C (edit planning).
    Returns proposed edits for user review.

    JSON body:
        image_stem      : str   — stem of the uploaded image (from /predict)
        manual_analysis : dict  — (optional) VLM analysis JSON for manual testing

    If manual_analysis is not provided and VLM is in manual mode,
    returns geometry data only. The user can then POST manual_analysis
    in a subsequent call.
    """
    ensure_dirs()
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    image_stem = body.get("image_stem", "")
    if not isinstance(image_stem, str) or not image_stem:
        return jsonify({"error": "image_stem is required"}), 400

    cached = _prediction_cache.get(image_stem)
    if cached is None:
        return jsonify({
            "error": f"No prediction cached for '{image_stem}'. Run /predict first."
        }), 404

    image_path: Path = cached["image_path"]
    instances: list[dict] = cached["instances"]
    manual_analysis = body.get("manual_analysis")

    try:
        from config.restoration_plan import validate_design_options
        design_options = validate_design_options(body.get('design_options')) if 'design_options' in body else None
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400

    try:
        pipeline = _get_pipeline()
        result = pipeline.analyze(image_path, instances, manual_analysis, design_options=design_options)
        if 'error' in result:
            return jsonify(result),400

        # Cache for /restore
        _analysis_cache[image_stem] = result
        if "edit_area_filename" in result:
            result["edit_area_url"] = url_for("output_file", kind="analysis", filename=result["edit_area_filename"])
        for key in ('selection','scaffold','mask'):
            if key+'_filename' in result:
                result[key+'_url'] = url_for('output_file',kind='analysis',filename=result[key+'_filename'])

        return jsonify(result)
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/restore", methods=["POST"])
@serialized_gpu
def restore():
    """
    Execute Layer D (BrushNet) + Layer E (Verification).
    Call after /analyze has been run for this image.

    JSON body:
        image_stem : str — stem of the uploaded image (from /predict)
    """
    ensure_dirs()
    body = request.get_json(force=True, silent=True) or {}
    if not isinstance(body, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    image_stem = body.get("image_stem", "")
    force_bilateral_mirror = body.get("force_bilateral_mirror")  # None = auto clinical check, True = force, False = disable
    quality_mode = body.get("quality_mode", "fast")  # "fast" or "quality" (two-pass)
    prompt = body.get("prompt")
    negative_prompt = body.get("negative_prompt")
    brushnet_conditioning_scale = body.get("brushnet_conditioning_scale", body.get("brushnet_strength"))
    guidance_scale = body.get("guidance_scale")
    control_guidance_end = body.get("control_guidance_end")
    steps = body.get("steps")

    if not isinstance(image_stem, str) or not image_stem:
        return jsonify({"error": "image_stem is required"}), 400

    cached = _prediction_cache.get(image_stem)
    if cached is None:
        return jsonify({
            "error": f"No prediction cached for '{image_stem}'. Run /predict first."
        }), 404

    analysis = _analysis_cache.get(image_stem)
    if body.get('plan_id') is not None and (analysis is None or body['plan_id'] != analysis.get('plan_id')):
        return jsonify({'error':'The design changed or expired. Update the preview before generating.'}),409

    from pipeline.generation_controls import validate_options
    try:
        if force_bilateral_mirror is not None and not isinstance(force_bilateral_mirror, bool):
            raise ValueError("force_bilateral_mirror must be true, false or null")
        validate_options(
            seed=body.get("seed", 42), steps=32 if steps is None else steps,
            guidance_scale=7.5 if guidance_scale is None else guidance_scale,
            conditioning_scale=0.58 if brushnet_conditioning_scale is None else brushnet_conditioning_scale,
            control_guidance_end=0.74 if control_guidance_end is None else control_guidance_end,
            quality_mode=quality_mode, conditioning_mode=body.get("conditioning_mode", "scaffold"),
        )
        for key in ("prompt", "negative_prompt"):
            if body.get(key) is not None and (not isinstance(body[key], str) or len(body[key]) > 8000):
                raise ValueError(f"{key} must be text of at most 8000 characters")
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    image_path: Path = cached["image_path"]
    instances: list[dict] = cached["instances"]

    job_id=None
    try:
        pipeline = _get_pipeline()
        jobs=_get_jobs()
        job_id=jobs.start(image_stem,(analysis or {}).get('plan_id'),body.get('job_id'))
        callback=jobs.callback(job_id,timeout_seconds=int(os.environ.get('ORALWISDOM_JOB_TIMEOUT_SECONDS','600')))
        result = pipeline.restore(
            image_path,
            instances,
            analysis,
            force_bilateral_mirror=force_bilateral_mirror,
            quality_mode=quality_mode,
            prompt=prompt,
            negative_prompt=negative_prompt,
            brushnet_conditioning_scale=brushnet_conditioning_scale,
            guidance_scale=guidance_scale,
            control_guidance_end=control_guidance_end,
            steps=steps,
            seed=body.get("seed", 42),
            conditioning_mode=body.get("conditioning_mode", "scaffold"),
            generation_callback=callback,
        )

        if "error" in result:
            jobs.update(job_id,status='failed',error=result['error'])
            return jsonify(result), 400

        # Add URLs for the frontend
        if "restored_filename" in result:
            result["restored_url"] = url_for(
                "output_file", kind="restored", filename=result["restored_filename"]
            )
        if "comparison_filename" in result:
            result["comparison_url"] = url_for(
                "output_file", kind="restored", filename=result["comparison_filename"]
            )
        for candidate in result.get("candidates", []):
            candidate["restored_url"] = url_for("output_file", kind="restored", filename=candidate["restored_filename"])
            candidate["comparison_url"] = url_for("output_file", kind="restored", filename=candidate["comparison_filename"])

        result['job_id']=job_id
        artifacts=set()
        for candidate in [result]+result.get('candidates',[]):
            for key in ('restored_filename','comparison_filename','allowed_mask_filename','metadata_filename'):
                filename=candidate.get(key)
                if filename: artifacts.add(filename)
        jobs.update(job_id,status='completed' if result.get('passed',True) else 'needs_revision',
                    restored_filename=result.get('restored_filename'),artifacts=sorted(artifacts),
                    result_status=result.get('result_status','review_required'))

        return jsonify(result)
    except ValueError as exc:
        if job_id: _get_jobs().update(job_id,status='failed',error=str(exc))
        return jsonify({'error':str(exc)}),400
    except Exception as exc:
        from pipeline.job_registry import GenerationCancelled
        if job_id: _get_jobs().update(job_id,status='cancelled' if isinstance(exc,GenerationCancelled) else 'failed',error=str(exc))
        if isinstance(exc,GenerationCancelled): return jsonify({'error':str(exc),'job_id':job_id}),409
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route('/jobs/<job_id>',methods=['GET'])
def job_status(job_id):
    try:
        record=_get_jobs().read(job_id)
        return jsonify(record) if record else (jsonify({'error':'Unknown job'}),404)
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400


@app.route('/jobs/<job_id>/cancel',methods=['POST'])
def cancel_job(job_id):
    try:
        cancelled=_get_jobs().cancel(job_id)
        return jsonify({'cancel_requested':cancelled}),200 if cancelled else 409
    except ValueError as exc:
        return jsonify({'error':str(exc)}),400


if __name__ == "__main__":
    ensure_dirs()
    app.run(host="127.0.0.1", port=7860, debug=False)
