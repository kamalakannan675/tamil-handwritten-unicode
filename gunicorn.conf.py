import os
import sys

# Get port from environment variable, default to 10000 for Render
port = os.environ.get("PORT", "10000")
bind = f"0.0.0.0:{port}"

# Free-tier single worker with bounded threads and timeout
workers = 1
threads = 2
timeout = 120
keepalive = 5

# Ensure stdout/stderr logs are immediately flushed to Render's live console
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

print(f"[gunicorn.conf.py] Starting Gunicorn bound to {bind} (workers={workers}, timeout={timeout})", file=sys.stderr, flush=True)
