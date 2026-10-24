FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements-runtime.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt -r requirements-api.txt \
    && useradd --create-home --uid 10001 receiptops \
    && mkdir /state && chown receiptops:receiptops /state
COPY src ./src
COPY configs/evaluate.yaml ./configs/evaluate.yaml
COPY models/receipt-kie-lora-v2-highres ./models/receipt-kie-lora-v2-highres
COPY artifacts/experiments/quality_gate/quality_gate.json ./artifacts/experiments/quality_gate/quality_gate.json
ENV PYTHONPATH=/app/src HF_HOME=/state/huggingface RECEIPT_STATE_DIR=/state
ENV RECEIPT_HOST=0.0.0.0 RECEIPT_PORT=8000 RECEIPT_IMAGE_EDGE=1024
USER receiptops
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=180s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
CMD ["python", "-m", "receipt_kie.service"]
