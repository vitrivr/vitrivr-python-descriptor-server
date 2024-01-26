FROM python:3.9

WORKDIR ./vitrivr-python-descriptors

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888

CMD ["python3", "./main.py", "--host", "0.0.0.0"]
