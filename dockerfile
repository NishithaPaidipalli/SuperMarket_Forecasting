# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first (this speeds up building)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code and the trained model
COPY src/ ./src/
COPY models/ ./models/

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API
CMD ["python", "src/app.py"]
