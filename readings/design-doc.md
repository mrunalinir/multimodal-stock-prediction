# Multi-Modal Stock Prediction System Design Document
This document outlines the design of a multi-modal stock prediction system that integrates various data sources and machine learning techniques to forecast stock prices.

## Overview
The system aims to leverage both structured data (like historical stock prices) and unstructured data (like news articles, social media sentiment, and financial reports) to improve the accuracy of stock price predictions.

## Literature Review
-- To be updated --

## Components
1. **Data Ingestion Module**
   - **Structured Data Sources**: Historical stock prices, trading volumes, financial statements.
   - **Unstructured Data Sources**: News articles, social media posts, analyst reports.
   - **APIs and Scrapers**: For real-time data collection from financial news websites and social media platforms.
    - **Data Storage**: Use a combination of SQL databases for structured data and NoSQL databases (like MongoDB) for unstructured data.
2. **Data Preprocessing Module**
   - **Structured Data**: Clean and normalize historical stock data, handle missing values, and perform feature engineering (e.g., moving averages, volatility).
   - **Unstructured Data**: Text preprocessing (tokenization, stop-word removal, stemming), sentiment analysis, and feature extraction using NLP techniques (e.g., TF-IDF, word embeddings).
3. **Feature Engineering Module**
   - Combine features from both structured and unstructured data.
   - Create composite features that capture relationships between different data types (e.g., sentiment scores combined with historical price trends).
   - Use dimensionality reduction techniques (like PCA) if necessary to reduce feature space.
4. **Model Training Module**
   - **Machine Learning Models**: Implement various models such as:
     - Linear Regression
     - Decision Trees
     - Random Forests
     - Gradient Boosting Machines (GBM)
     - Recurrent Neural Networks (RNN) for time series prediction
     - Transformer-based models for handling sequential data.
     - LSTM (Long Short-Term Memory) networks for capturing long-term dependencies in time series data.
     - Multi-head Attention Mechanisms to integrate multiple data modalities effectively.
   - **Model Evaluation**: Use metrics like RMSE, MAE, and R-squared to evaluate model performance. Implement cross-validation techniques to ensure robustness.
   - **Hyperparameter Tuning**: Use grid search or random search to optimize model parameters for better performance.
   - **Ensemble Methods**: Combine predictions from multiple models to improve accuracy and robustness.
   - **Model Deployment**: Use containerization (e.g., Docker) for deploying models in a production environment. Implement APIs for real-time predictions.
   - **Monitoring and Maintenance**: Set up monitoring for model performance and retrain models periodically as new data becomes available. Implement logging and alerting mechanisms to track anomalies in predictions.
   - **User Interface Module**
   - Develop a web-based dashboard for users to visualize stock predictions, historical data, and sentiment analysis results.
   - Provide interactive features for users to select stocks, view trends, and access detailed reports.
   - Develop a feature to select which of the models to use for predictions, allowing users to choose based on their preferences or specific stock characteristics. 
   - Create a drop-down to select a stock, get the predictions - predicted prive over the next 10 days, and a summary of latest financial statements, news and social media sentiment.
5. **Prediction Module**
    - Implement a prediction engine that takes user input (selected stock) and provides:
      - Predicted stock prices for the next 10 days.
      - Summary of latest financial statements.
      - Sentiment analysis results from news and social media.
    - Use the selected model from the user interface to generate predictions.
6. **User Interface Module**
   - Develop a web-based dashboard for users to visualize stock predictions, historical data, and sentiment analysis results.
   - Provide interactive features for users to select stocks, view trends, and access detailed reports.
   - Implement a feature to select which model to use for predictions, allowing users to choose based on their preferences or specific stock characteristics.
   - Create a drop-down to select a stock, get the predictions (predicted price over the next 10 days), and a summary of the latest financial statements, news, and social media sentiment.

## Model Selection 
-- Thoughts on each model using documentation and validation scores. To be updated --

## Technologies
- **Data Ingestion**: Python, BeautifulSoup, Scrapy, Tweepy (for social media data)
- **Data Storage**: PostgreSQL (for structured data), MongoDB (for unstructured data)
- **Data Preprocessing**: Pandas, NumPy, NLTK, SpaCy
- **Machine Learning**: Scikit-learn, TensorFlow, Keras, PyTorch
- **Web Framework**: Flask or Django for the backend, React or Angular for the frontend
- **Deployment**: Docker, Kubernetes for container orchestration
- **Monitoring**: Prometheus, Grafana for performance monitoring and visualization

## Future Enhancements
- *Advanced NLP Techniques*: Implement more sophisticated NLP models like BERT or GPT for better sentiment analysis.
- *Data Augmentation*: Explore additional data sources such as economic indicators, insider trading data, and macroeconomic news.
- *Data backfeeding*: Re-train the model at the end of the day based on predicted vs current day OHLC (Open, High, Low, Close) values to improve future predictions.

## Limitations 
- **Data Quality**: The accuracy of predictions heavily relies on the quality and timeliness of the data collected.
- **Model Complexity**: More complex models may lead to overfitting, especially with limited data.
- **Market Volatility**: Sudden market changes or events (like economic crises) can significantly impact stock prices, making predictions less reliable.

## Conclusion
This multi-modal stock prediction system aims to provide a comprehensive solution for forecasting stock prices by integrating various data sources and advanced machine learning techniques. By continuously improving the model with new data and user feedback, the system can adapt to changing market conditions and provide valuable insights for investors. By using different models and allowing users to select their preferred model, the system can help understand the strengths and weaknesses of each approach, leading to better-informed decisions.

## References
- **Pati, S.P., Mishra, D. (2025)**
  - Title: A Multi-Modal Approach Using a Hybrid Vision Transformer and Temporal Fusion Transformer Model for Stock Price Movement Classification
  - Journal: IEEE Access
  - Benchmarked against: BERT-LSTM, CNN-GRU
  - Achieved 91.2% F1 on S&P500 dataset.

- **Papasotiriou, K., Sood, S., Reynolds, S. (2024)**
   -Title: AI in Investment Analysis: LLMs for Equity Stock Ratings
   - Conference: ACM KDD 2024
   - Uses GPT-style LLMs with structured financials + sentiment data.
   - Sets a new benchmark for earnings-based valuation predictions.

- **Cao, Y., Chen, Z., Pei, Q. (2024)**
   -Title: ECC Analyzer: Extracting Trading Signal from Earnings Conference Calls using LLMs
   - Venue: ACM Proceedings
   - Benchmarks multimodal model using financial call transcripts + market data.

- **Joshi, A., Koda, J.K., Hadimlioglu, A. (2025)**
   -  Title: Real-Time Adaptive Multi-Modal Stock Prediction with Temporal Graph Attention and Dynamic Interaction Networks
   -  Venue: IEEE Conference on Data Engineering
   - Introduces a dynamic graph-based fusion network
   - Outperforms T-GAT and XGBoost baselines by 23.5%

- **Joshi, A., Koda, J.K. (2024)**
   - Title: A Multi-Modal Transformer Architecture Combining Sentiment Dynamics, Temporal Market Data, and Macroeconomic Indicators
   - Conference: IEEE Big Data 2024
   - Validated on 3 public datasets; achieves 34% improvement over BERT-GRU

- **Yue, D. (2025)**
   - Title: Dynamic Weighted Multimodal Financial Forecasting Models: Fusion Strategies and Market Validation
   - Venue: IEEE ICAI 2025
   - Proposes a DWMF framework (Dynamic Weight Fusion)
   - Shows 19.7% RMSE reduction over static-fusion baselines

