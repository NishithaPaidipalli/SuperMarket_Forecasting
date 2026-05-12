Supermarket Sales Forecasting System 🛒📈 : A Deep Learning-based time series forecasting application that predicts future supermarket sales using an LSTM (Long Short-Term Memory) neural network.

🚀 Features Deep Learning Model: LSTM architecture designed for sequential pattern recognition.

Automated Pipeline: Full data preprocessing, scaling, and sequence generation.

REST API: Built with FastAPI for high-performance model serving.

Containerized: Docker-ready for consistent deployment across environments.

Monitoring: Integrated logging and error handling for production stability.

📊 API Usage Endpoint: POST /predict

Payload Format:

JSON { "last_7_days_sales": [120.5, 450.2, 310.0, 290.4, 500.1, 410.2, 380.0] }
