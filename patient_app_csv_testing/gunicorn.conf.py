import os
bind = f"0.0.0.0:{os.getenv('PORT','8080')}"
worker_class = "gthread"
workers = 2
threads = 2
timeout = 60
graceful_timeout = 30
keepalive = 75
max_requests = 1000
max_requests_jitter = 100
accesslog = "-"
errorlog = "-"