
# ETL Pipeline and Sentiment Analysis

This repository contains an ETL (Extract, Transform, Load) pipeline for processing customer feedback data and performing sentiment analysis. The project integrates data extraction from a SQL Server database, transformation and loading into a data warehouse, and machine learning-based sentiment analysis. The results are exported as CSV files for further analysis.

## Table of Contents

- [Setup and Requirements](#setup-and-requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Data Pipeline Overview](#data-pipeline-overview)
- [Sentiment Analysis Model](#sentiment-analysis-model)
- [Results Export](#results-export)
- [Troubleshooting](#troubleshooting)

## Setup and Requirements

### Prerequisites

- **Python Version:** Python 3.7 or higher
- **SQL Server:** A running instance of SQL Server with two databases:
  - `FeedbackDB`: for raw data
  - `FeedbackDW`: for the data warehouse

### Required Python Libraries

The required libraries can be installed using the following command:

```bash
pip install pandas numpy sklearn nltk pyodbc prefect
```

### NLTK Data Setup

After installing NLTK, download the necessary datasets:

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
```

### SQL Server Setup

1. Ensure you have SQL Server installed and running.
2. Create two databases: `FeedbackDB` (raw data) and `FeedbackDW` (data warehouse).
3. Adjust the connection string in the code to match your SQL Server configuration:

   ```plaintext
   Driver={ODBC Driver 17 for SQL Server}; Server=localhost; Database=FeedbackDB; Trusted_Connection=yes;
   ```

   Modify the `Server` and `Driver` if needed based on your system settings.

### Data

Place the `feedback_data.csv` file in the root directory of the project. This file is used for loading raw data into the database.

## Project Structure

```
├── DataGenerator.py           # Script for generating synthetic data (if needed)
├── ETL_Pipeline.py            # Main ETL pipeline script
├── feedback_data.csv          # Sample CSV file with feedback data
├── README.md                  # This README file
```

## Usage

1. **Run the ETL Pipeline:**
   
   Execute the ETL pipeline script to load data into the raw database, transform it, and populate the data warehouse:

   ```bash
   python ETL_Pipeline.py
   ```

   The script will:
   - Load the `feedback_data.csv` file into the `FeedbackDB`.
   - Transform the data and populate the `FeedbackDW` data warehouse.
   - Perform sentiment analysis on the reviews.
   - Export the results as CSV files.

2. **Configure and Run Sentiment Analysis:**

   The script includes a built-in sentiment analysis model using logistic regression. The model will be trained on the review data and export predictions.

## Data Pipeline Overview

The ETL pipeline consists of the following stages:

1. **Extraction:** Load raw data from the `feedback_data.csv` file into the `FeedbackDB` database.
2. **Transformation:** Clean and transform data, merging information from the feedback, customer, and category tables.
3. **Loading:** Populate the `FeedbackDW` data warehouse with dimensional tables (`CustomerDim`, `CategoryDim`, etc.) and the fact table (`FeedbackFact`).
4. **Sentiment Analysis:** Use a machine learning model to predict the sentiment of reviews based on text features.
5. **Exporting Results:** Save the dimension and fact tables, along with sentiment predictions, as CSV files.

## Sentiment Analysis Model

- The sentiment analysis model is built using the `LogisticRegression` classifier from `scikit-learn`.
- Preprocessing includes text tokenization, lemmatization, and stopword removal.
- Features are generated using TF-IDF vectorization.
- The model is trained to classify sentiment as "positive", "neutral", or "negative".

## Results Export

The following CSV files are generated in the output directory:
- `CustomerDim.csv`
- `CategoryDim.csv`
- `DateDim.csv`
- `OrderDim.csv`
- `ProductDim.csv`
- `FeedbackFact.csv`
- `predicted_sentiments.csv` (containing sentiment predictions)

## Troubleshooting

- **Database Connection Issues:** Ensure that the connection string matches your SQL Server configuration.
- **Missing NLTK Data:** Make sure you have downloaded the required NLTK data.
- **Missing Feedback Data:** Ensure the `feedback_data.csv` file is in the correct directory.

## Contributing

Feel free to open issues or submit pull requests to improve the code.
