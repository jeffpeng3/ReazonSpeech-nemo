FROM nvcr.io/nvidia/nemo:25.11
RUN apt-get update && apt-get install -y ffmpeg
WORKDIR /app