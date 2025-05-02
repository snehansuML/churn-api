FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
RUN python train_model.py
EXPOSE 5000
CMD ["python", "run_this.py"]
