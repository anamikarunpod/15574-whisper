FROM runpod/base:0.6.2-cuda12.4.1

SHELL ["/bin/bash", "-c"]
WORKDIR /

# Update and upgrade the system packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y ffmpeg wget git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create cache directory for model
RUN mkdir -p /cache

# Copy requirements file first to leverage Docker cache
COPY builder/requirements.txt /builder/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --upgrade pip hf_transfer && \
    python3 -m pip install -r /builder/requirements.txt

# Copy the download script
COPY builder/download_models.sh /builder/download_models.sh
RUN chmod +x /builder/download_models.sh

# Set environment variables
ENV HF_HOME=/cache 

# RUN echo "Listing root directory:" && ls -la /
# RUN echo "Listing src directory:" && ls -la /src

# Copy source code
COPY src .

# Create a startup script that downloads the model then runs the handler
RUN echo '#!/bin/bash\n\
    echo "Starting CrisperWhisper worker..."\n\
    # Download model using the HF token from RunPod secrets\n\
    /builder/download_models.sh\n\
    \n\
    # If model download was successful, start the handler\n\
    if [ $? -eq 0 ]; then\n\
    echo "Model downloaded successfully, starting handler..."\n\
    exec python3 -u /rp_handler.py\n\
    else\n\
    echo "Model download failed, exiting..."\n\
    exit 1\n\
    fi' > /start.sh \
    && chmod +x /start.sh

CMD [ "/start.sh" ]