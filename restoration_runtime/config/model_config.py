"""
Configuration for all model paths, hyperparameters, and environment settings.
"""

from pathlib import Path
import os

ROOT = Path(os.environ.get("ORALWISDOM_SITE_ROOT", r"D:\OralWisdom\projfiles\v3_inference_site"))

# YOLO
YOLO_MODEL_PATH = ROOT / "models" / "best.pt"
YOLO_IMG_SIZE = 1024
YOLO_CONF = 0.20
YOLO_IOU = 0.65

# VLM (for future use)
VLM_PROVIDER = os.environ.get("VLM_PROVIDER", "manual")  # "manual" | "api"
VLM_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
VLM_API_MODEL = "qwen/qwen3-vl-8b"
VLM_API_BASE_URL = "https://openrouter.ai/api/v1"
VLM_MAX_RETRIES = 3
VLM_TEMPERATURE = 0.1
VLM_MAX_TOKENS = 4096

# BrushNet
BRUSHNET_CHECKPOINT = os.environ.get("ORALWISDOM_BRUSHNET_MODEL", "Sanster/brushnet_segmentation_mask")
BASE_DIFFUSION_MODEL = os.environ.get("ORALWISDOM_BASE_MODEL", "runwayml/stable-diffusion-v1-5")
BRUSHNET_STEPS = 32
BRUSHNET_GUIDANCE_SCALE = 7.5
BRUSHNET_PRESERVATION_SCALE = 0.58
BRUSHNET_RESOLUTION = 512

# ControlNet
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
USE_CONTROLNET = False  # No ControlNet branch is loaded by the current pipeline.

# Verification
VERIFICATION_SSIM_THRESHOLD = 0.85
MAX_RESTORATION_ATTEMPTS = 3

# Output directories (stored on fixed NTFS NVMe drive to prevent FAT32 USB corruption)
OUTPUT_DIR = Path(os.environ.get("ORALWISDOM_OUTPUT_DIR", r"C:\OralWisdom_outputs"))
RESTORE_DIR = OUTPUT_DIR / "restored"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"
CROPS_DIR = OUTPUT_DIR / "crops"

