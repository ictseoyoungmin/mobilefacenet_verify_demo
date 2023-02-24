FROM ubuntu:20.04

WORKDIR /app


# Copy files
# COPY . .
# COPY requirements.txt /app/requirements.txt
# COPY build/demo.py /app/demo.py

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3 python3-pip python3-dev && \
    apt-get install -y python3

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy files
COPY . .

# CMD ["python3", "build/demo.py"]