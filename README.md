# OralWisdom cosmetic smile preview

Start with Start_OralWisdom.bat on the verified Windows environment. The bundle contains
the application, custom BrushNet runtime, Waitress and YOLO model. It excludes patient
photos, output history, Python/Torch and diffusion model weights. Set ORALWISDOM_PYTHON
to a compatible Python interpreter if the existing C:\OralWisdom_venv is elsewhere.
Startup checks file hashes, syntax, CUDA and locally cached diffusion models.

Default server: http://127.0.0.1:7860. Default output: C:\OralWisdom_outputs.
Set ORALWISDOM_OUTPUT_DIR to separate test and personal data. Local jobs have durable
status, cancellation after the current render pass, and interrupted-worker recovery.
Use prune_restoration_jobs.py OUTPUT --days 30 to preview retention; --apply deletes
expired job records and their explicitly tracked render artifacts only. Originals and
historical untracked outputs require separate review. Retention is not automatic.

Network access requires ORALWISDOM_AUTH_USER and ORALWISDOM_AUTH_PASSWORD and a HTTPS
reverse proxy. This is a single-owner local worker, without multi-tenant isolation,
distributed scheduling, or a public-service security audit.

Keep baseline_before_smile and evaluation_runs outside this bundle for rollback and review.
The interface controls cosmetic veneer appearance, not diagnosis, implant placement,
physical ceramic shade or a clinical treatment plan. Every result needs visual review.
