FROM python:3.12

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src

CMD ["python", "-m", "src.scalable_sys.app", "--prompt", "Who won the Nobel Prize in Physics in 1921?"]
