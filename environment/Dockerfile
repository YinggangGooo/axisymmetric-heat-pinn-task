FROM mirror.gcr.io/library/python:3.11-slim@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893
ENV DEBIAN_FRONTEND=noninteractive OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
COPY constraints.txt /tmp/runtime-constraints.txt
RUN python -m venv /opt/venv && \
    /opt/venv/bin/python -m pip install --no-cache-dir -c /tmp/runtime-constraints.txt numpy==2.2.6 && \
    /opt/venv/bin/python -m pip install --no-cache-dir -c /tmp/runtime-constraints.txt torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN useradd --create-home --uid 1000 agent && mkdir -p /app /logs/verifier && \
    chown -R agent:agent /opt/venv /app && chmod 700 /logs/verifier
ENV PATH="/opt/venv/bin:${PATH}"
COPY --chown=agent:agent app/ /app/
WORKDIR /app
CMD ["/bin/bash"]
