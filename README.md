## Forecasting Bitcoin Price Movements Using GRU-Attention Networks and Sentiment-Enhanced Features

This repository contains the complete, end-to-end machine learning pipeline for the Master's dissertation by Den Peters Ngotho Gathitu. The project investigates whether the integration of systematically engineered sentiment data can improve the predictive accuracy of a GRU-Attention deep learning model for forecasting hourly Bitcoin price movements.

## Abstract

The forecasting of volatile cryptocurrency markets, such as Bitcoin, presents a significant challenge for traditional financial models. This research introduces and validates a novel forecasting pipeline that enhances a Gated Recurrent Unit with Attention mechanism (GRU-Attention) model with a robust, data-driven sentiment feature. Over 200 distinct sentiment indicators from online social media were distilled into a single feature using Principal Component Analysis (PCA). A controlled experiment was conducted to compare the performance of this sentiment-enhanced model against an identical baseline model. The findings confirm that the integration of methodically engineered sentiment data provides a significant and quantifiable improvement in forecasting performance.

## Pipeline Overview

The research is structured as a controlled experiment. The pipeline ingests raw price and sentiment data, processes it, engineers features, and then trains two models in parallel for a direct comparison.

![Conceptual Research Framework](figure_3_1_conceptualframework.png)

The key stages are:
1.  *Data Acquisition*: Sourcing hourly Bitcoin OHLCV data and high-frequency sentiment data.
2.  *Preprocessing*: Standardizing timestamps, merging the two datasets with an inner join, and imputing missing values using a forward-fill method.
3.  *Feature Engineering*:
    - Defining the target variable as the future hourly percentage return.
    - Applying Principal Component Analysis (PCA) to over 200 raw sentiment indicators to create a single, robust `sentiment_pca` feature.
4.  *Controlled Experiment*:
    - *Baseline Model*: A GRU-Attention network trained only on historical price and volume data.
    - *Enhanced Model*: An identical GRU-Attention network trained on price/volume data *plus* the `sentiment_pca` feature.
5.  *Evaluation*: Comparing the performance of both models on an unseen test set using standard regression metrics.

## Datasets

The two datasets used in this project were sourced from:
- *BTC Prices*: [Bitcoin Hourly OHCLV Dataset by Mouad Jaouad](https://github.com/mouadja02/bitcoin-hourly-ohclv-dataset)
- *Sentiment Data*: [Aggregated Sentiment Metrics for Bitcoin from Augmento.ai](https://www.augmento.ai/download/2317/)

## Installation and Setup

This project was developed in a Google Colab environment using Python 3. To replicate the environment, you can install the necessary libraries using pip.

1.  Clone the repository:
    ```bash
    git clone [https://github.com/dengathitu/GRU-Attention-Forecasting-Pipeline.git](https://github.com/dengathitu/GRU-Attention-Forecasting-Pipeline.git)
    cd GRU-Attention-Forecasting-Pipeline
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

requirements.txt`
pandas==2.0.3
numpy==1.25.2
scikit-learn==1.3.0
tensorflow==2.15.0

## Usage

The entire pipeline is contained within the Jupyter Notebook (`GRU_Attention_Forecasting_Pipeline.ipynb`).

1.  Open the notebook in a compatible environment (like Google Colab, Jupyter Lab, or VS Code).
2.  Ensure the datasets (`btc_prices.csv` and `sentiment.csv`) are accessible to the notebook (e.g., by placing them in the same directory or mounting Google Drive).
3.  Run the notebook cells sequentially from top to bottom. The notebook is fully documented and will execute all steps from data loading to the final model comparison.

## Results

The controlled experiment demonstrates that the sentiment-enhanced model consistently outperforms the baseline model across all evaluation metrics on the unseen test data.

| Metric | Baseline Model (Price Only) | Enhanced Model (Price + Sentiment) | Percentage Improvement |
| :--- | :--- | :--- | :--- |
| *MSE* | 0.00003043 | *0.00002933* | 3.61% |
| *MAE* | 0.003715 | *0.003570* | 3.90% |
| *RMSE* | 0.005516 | *0.005415* | 1.83% |

These results provide strong empirical evidence that the systematic integration of an engineered sentiment feature demonstrably improves the performance of the GRU-Attention model for forecasting hourly Bitcoin price movements.

## Citation

If you use this work, please cite the author:

Den Peters Ngotho Gathitu, "Forecasting Bitcoin Price Movements Using GRU-Attention Networks and Sentiment-Enhanced Features," Master's Dissertation, Gisma University of Applied Sciences, 2025.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
