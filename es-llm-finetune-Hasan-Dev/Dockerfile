FROM python:3.11-slim

# 1) Build-Tools (für pytrec_eval) + git
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) pip updaten
RUN pip install -U pip

# 3) Torch CPU pinnen (wichtig!)
RUN printf "torch==2.8.0+cpu\n" > /tmp/constraints.txt \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -c /tmp/constraints.txt \
      "git+https://github.com/allenai/olmes.git"
