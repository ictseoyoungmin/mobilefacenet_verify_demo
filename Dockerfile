FROM ubuntu:20.04

WORKDIR /app


# Copy files
# COPY . .
# COPY requirements.txt /app/requirements.txt
# COPY build/demo.py /app/demo.py

# area
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
# Install dependencies
RUN apt-get update
RUN apt-get install -y python3.9 python3-pip python3.9-dev
RUN apt-get install -y python3 
RUN apt install sudo && \
sudo apt-get update && \
sudo apt-get install -y libgl1-mesa-glx && \ 
sudo apt-get install -y libglib2.0-0


COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy files
COPY . .

# CMD ["python3", "build/demo.py"]