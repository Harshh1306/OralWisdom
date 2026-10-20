"""Local WSGI entry point. One GPU job at a time, four HTTP threads."""
from pathlib import Path
import sys
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'serving_dependencies'))
from waitress import serve
from app import app, ensure_dirs

if __name__ == '__main__':
    ensure_dirs()
    host=os.environ.get('ORALWISDOM_HOST','127.0.0.1')
    if host not in ('127.0.0.1','localhost','::1') and not (os.environ.get('ORALWISDOM_AUTH_USER') and os.environ.get('ORALWISDOM_AUTH_PASSWORD')):
        raise RuntimeError('Network serving requires ORALWISDOM_AUTH_USER and ORALWISDOM_AUTH_PASSWORD, behind HTTPS')
    serve(app, host=host, port=int(os.environ.get('ORALWISDOM_PORT','7860')),
          threads=4, max_request_body_size=30 * 1024 * 1024)
