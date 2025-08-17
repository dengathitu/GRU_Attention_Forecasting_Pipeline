## Forecasting Bitcoin Price Movements Using GRU-Attention Networks and Sentiment-Enhanced Features

This repository presents an end-to-end machine learning pipeline for forecasting Bitcoin price movements. It combines GRU-Attention neural networks with sentiment analysis to enhance predictive accuracy and interpretability.

### Research Question

Can the integration of sentiment analysis with GRU-Attention neural networks improve the predictive accuracy of forecasting future Bitcoin price movements, and how do these sentiment features contribute to the model's interpretability?

### Pipeline Overview

The pipeline consists of the following stages:

1. **Environment Setup**  
   Installation and import of all required libraries for data processing, modeling, and diagnostics.

2. **Data Loading & Preprocessing**  
   - Hourly OHLCV Bitcoin data and sentiment scores are loaded.
   - Data is aligned chronologically and merged on timestamp.

3. **Feature Engineering**  
   - Log returns and rolling volatility are computed.
   - Sentiment features are scaled using MinMaxScaler.

4. **Sequence Construction**  
   - Time-series sequences are created for model input.
   - Target variable: next-step log return.

5. **Train-Test Split**  
   - Chronological 80/20 split to prevent data leakage.

6. **Model Architecture**  
   - GRU layer processes sequences.
   - Attention layer highlights relevant time steps.
   - Dense layer outputs log return prediction.

7. **Model Training**  
   - Early stopping prevents overfitting.
   - Training monitored via validation loss.

8. **Evaluation**  
   - Metrics: RMSE, MAE, R²
   - GRU-Attention model compared against:
     - GARCH baseline
     - Ablation model (without sentiment)

### Final Results Summary

| Model                              | RMSE      | MAE       | R²         |
|-----------------------------------|-----------|-----------|------------|
| GRU-Attention (With Sentiment)    | 0.011546  | 0.004522  | -3.672144  |
| GARCH Baseline                    | 0.268393  | 0.192494  | 0.004864   |
| Ablation (Without Sentiment)      | 0.006142  | 0.003596  | -0.322207  |


### Data Sources

- **Bitcoin OHLCV**: [GitHub Dataset](https://github.com/mouadja02/bitcoin-hourly-ohclv-dataset)
- **Sentiment Scores**: [Augmento Dataset](https://www.augmento.ai/download/2317/)

### Dependencies

```bash
numpy
pandas
scikit-learn
tensorflow
arch
matplotlib
seaborn
vaderSentiment
textblob
```

### How to Run

```bash
# Clone the repo
git clone https://github.com/dengathitu/bitcoin-forecasting-gru-attention-sentiment.git
cd bitcoin-forecasting-gru-attention-sentiment

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python train_model.py
```

### Future Work

- Integrate transformer-based sentiment models.
- Expand to multi-asset forecasting.
- Deploy real-time prediction dashboard.
