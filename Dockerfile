FROM public.ecr.aws/amazonlinux/amazonlinux:2023

WORKDIR /app

# Set PATH to use our custom symlink first
ENV PATH=/usr/local/bin:$PATH

# Install all system dependencies in a single layer
RUN dnf update -y && \
    dnf install -y gcc-c++ python3.12-devel tar xz && \
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && \
    python3 -m ensurepip && \
    python3 -m pip install --no-cache-dir --upgrade pip && \
    dnf clean all && \
    rm -rf /var/cache/dnf
    
# Install ffmpeg
RUN curl -L https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz | \
    tar xJ && \
    mv ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf ffmpeg-master-latest-linux64-gpl

# Copy and install requirements
COPY ./requirements.txt requirements.txt
RUN pip install -U --no-cache-dir -r requirements.txt && \
    # Clean up pip cache only
    rm -rf ~/.cache/pip /tmp/pip* && \
    python3 -m compileall -q /usr/local/lib/python3.12/site-packages

# Copy application files
COPY ./parakeet_transcribe.py parakeet_transcribe.py 

# Cache the model file during build
RUN python3 -c "from nemo.collections.asr.models import ASRModel; ASRModel.from_pretrained('nvidia/parakeet-tdt-0.6b-v3')"

CMD ["python3", "parakeet_transcribe.py"]