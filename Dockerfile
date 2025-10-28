FROM public.ecr.aws/amazonlinux/amazonlinux:2023

WORKDIR /app

# Set PATH to use our custom symlink first
ENV PATH=/usr/local/bin:$PATH

# Install all system dependencies in a single layer
RUN dnf update -y && \
    dnf install -y gcc-c++ python3.12-devel && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    python3 -m ensurepip && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    dnf clean all && \
    rm -rf /var/cache/dnf

# Copy and install requirements
COPY ./requirements.txt requirements.txt
RUN pip install -U --no-cache-dir -r requirements.txt && \
    # Ensure bytecode is compiled for all modules (for performance)
    python -m compileall -q /usr/local/lib/python3.12/site-packages && \
    # Clean up unnecessary files
    rm -rf ~/.cache/pip /tmp/pip* && \
    find /usr/local \
        -type d -name "*.dist-info" -o -name "tests" -o -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /usr/local -type f -name "*.md" -o -name "*.txt" -o -name "*.html" -delete 2>/dev/null || true

# Copy application files
COPY ./parakeet_transcribe.py parakeet_transcribe.py 

# Cache the model file during build
RUN python3 -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"

CMD ["python3", "parakeet_transcribe.py"]