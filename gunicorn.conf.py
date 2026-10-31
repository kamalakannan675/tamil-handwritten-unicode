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

# Memory management is handled deterministically via malloc_trim(0) in app/main.py.
# Worker recycling is disabled (0) so single-worker environments do not experience downtime during worker respawn.
max_requests = 0

# Ensure stdout/stderr logs are immediately flushed to Render's live console
accesslog = "-"
errorlog = "-"
loglevel = "info"
capture_output = True
enable_stdio_inheritance = True

print(f"[gunicorn.conf.py] Starting Gunicorn bound to {bind} (workers={workers}, timeout={timeout})", file=sys.stderr, flush=True)
