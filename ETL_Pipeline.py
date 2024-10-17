import random
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import pyodbc
from prefect import task, flow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Helper function to insert into SQL Server using pyodbc with column existence check
def insert_into_table(cursor, table_name, columns, values, primary_key=None):
    """
    Insert values into the specified table, with optional primary key checking.
    Handles None values by replacing them with default values.
    """
    existing_columns = [col for col in columns if col in values.keys() and values[col] is not None]
    filtered_values = [values[col] if values[col] is not None else get_default_value(col) for col in existing_columns]

    if primary_key:
        primary_key_value = values.get(primary_key)
        if primary_key_value is None:
            print(f"Skipping insert for {table_name} as primary key {primary_key} is None.")
            return
        query_check = f"SELECT COUNT(*) FROM {table_name} WHERE {primary_key} = ?"
        cursor.execute(query_check, primary_key_value)
        count = cursor.fetchone()[0]
        if count == 0:
            placeholders = ', '.join(['?' for _ in existing_columns])
            query_insert = f"INSERT INTO {table_name} ({', '.join(existing_columns)}) VALUES ({placeholders})"
            cursor.execute(query_insert, filtered_values)
        else:
            print(f"Duplicate entry found for {primary_key} = {primary_key_value} in {table_name}, skipping insert.")
    else:
        placeholders = ', '.join(['?' for _ in existing_columns])
        query_insert = f"INSERT INTO {table_name} ({', '.join(existing_columns)}) VALUES ({placeholders})"
        cursor.execute(query_insert, filtered_values)

def get_default_value(column_name):
    """
    Provides a default value based on the column name.
    For numeric columns, returns 0, and for strings, returns an empty string.
    """
    if 'ID' in column_name or 'Number' in column_name or column_name in ['Age', 'Rating']:
        return 0
    return ''

# SQL table creation commands for Raw Database
raw_table_creation_queries = '''
IF OBJECT_ID('dbo.Customer', 'U') IS NULL
CREATE TABLE Customer (
    CustomerID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100),
    PhoneNumber VARCHAR(50),
    Gender VARCHAR(10),
    Age INT,
    Location VARCHAR(100)
);

IF OBJECT_ID('dbo.Feedback', 'U') IS NULL
CREATE TABLE Feedback (
    FeedbackID INT PRIMARY KEY,
    CustomerID INT,
    FeedbackDate DATE,
    ReviewText TEXT,
    Sentiment VARCHAR(20),
    Rating INT CHECK (Rating BETWEEN 1 AND 5),
    OrderID INT,
    OrderDate DATE,
    OrderStatus VARCHAR(20),
    TotalAmount DECIMAL(10, 2),
    CategoryID INT,
    ProductID INT,
    ProductName VARCHAR(100),
    ProductPrice DECIMAL(10, 2),
    FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
);

IF OBJECT_ID('dbo.Category', 'U') IS NULL
CREATE TABLE Category (
    CategoryID INT PRIMARY KEY,
    CategoryNumber INT,
    CategoryName VARCHAR(50)
);
'''

# SQL table creation commands for Data Warehouse
warehouse_table_creation_queries = '''
IF OBJECT_ID('dbo.CustomerDim', 'U') IS NULL
CREATE TABLE CustomerDim (
    CustomerID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    Email VARCHAR(100),
    PhoneNumber VARCHAR(50),
    Gender VARCHAR(10),
    Age INT,
    Location VARCHAR(100),
    ReviewText TEXT,
    Sentiment VARCHAR(20)
);

IF OBJECT_ID('dbo.CategoryDim', 'U') IS NULL
CREATE TABLE CategoryDim (
    CategoryID INT PRIMARY KEY,
    CategoryNumber INT,
    CategoryName VARCHAR(50)
);

IF OBJECT_ID('dbo.DateDim', 'U') IS NULL
CREATE TABLE DateDim (
    DateID INT PRIMARY KEY IDENTITY(1,1),
    FeedbackDate DATE,
    Year INT,
    Month INT,
    Day INT,
    Week INT,
    Quarter INT
);

IF OBJECT_ID('dbo.OrderDim', 'U') IS NULL
CREATE TABLE OrderDim (
    OrderID INT PRIMARY KEY,
    OrderDate DATE,
    OrderStatus VARCHAR(20),
    TotalAmount DECIMAL(10, 2)
);

IF OBJECT_ID('dbo.ProductDim', 'U') IS NULL
CREATE TABLE ProductDim (
    ProductID INT PRIMARY KEY,
    ProductName VARCHAR(100),
    ProductPrice DECIMAL(10, 2),
    CategoryID INT,
    FOREIGN KEY (CategoryID) REFERENCES CategoryDim(CategoryID)
);

IF OBJECT_ID('dbo.FeedbackFact', 'U') IS NULL
CREATE TABLE FeedbackFact (
    FeedbackID INT PRIMARY KEY,
    CustomerID INT,
    CategoryID INT,
    DateID INT,
    OrderID INT,
    ProductID INT,
    Rating INT CHECK (Rating BETWEEN 1 AND 5),
    FOREIGN KEY (CustomerID) REFERENCES CustomerDim(CustomerID),
    FOREIGN KEY (CategoryID) REFERENCES CategoryDim(CategoryID),
    FOREIGN KEY (DateID) REFERENCES DateDim(DateID),
    FOREIGN KEY (OrderID) REFERENCES OrderDim(OrderID),
    FOREIGN KEY (ProductID) REFERENCES ProductDim(ProductID)
);
'''

# Task 1: Extract raw data and load into Raw Database
@task
def extract_load_raw_data():
    df = pd.read_csv('feedback_data.csv')

    with pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                        'Server=localhost;'
                        'Database=FeedbackDB;'
                        'Trusted_Connection=yes;') as conn:
        cursor = conn.cursor()

        for query in raw_table_creation_queries.strip().split(';'):
            if query.strip():
                cursor.execute(query)
        conn.commit()

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            customer_columns = ['CustomerID', 'FirstName', 'LastName', 'Email', 'PhoneNumber', 'Gender', 'Age', 'Location']
            insert_into_table(cursor, 'Customer', customer_columns, row_dict, primary_key='CustomerID')

            feedback_columns = ['FeedbackID', 'CustomerID', 'FeedbackDate', 'ReviewText', 'Sentiment', 'Rating',
                                'OrderID', 'OrderDate', 'OrderStatus', 'TotalAmount', 'CategoryID',
                                'ProductID', 'ProductName', 'ProductPrice']
            insert_into_table(cursor, 'Feedback', feedback_columns, row_dict, primary_key='FeedbackID')

            category_columns = ['CategoryID', 'CategoryNumber', 'CategoryName']
            insert_into_table(cursor, 'Category', category_columns, row_dict, primary_key='CategoryID')

        conn.commit()

    print("Raw data loaded into FeedbackDB successfully.")
    return df

# Task 2: Transform data and load into Data Warehouse
@task
def transform_and_load_to_warehouse():
    with pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                        'Server=localhost;'
                        'Database=FeedbackDB;'
                        'Trusted_Connection=yes;') as conn:
        feedback_df = pd.read_sql("SELECT * FROM Feedback", conn)
        customer_df = pd.read_sql("SELECT * FROM Customer", conn)
        category_df = pd.read_sql("SELECT * FROM Category", conn)

    feedback_df['CustomerID'] = feedback_df['CustomerID'].astype(str)
    customer_df['CustomerID'] = customer_df['CustomerID'].astype(str)
    feedback_df['CategoryID'] = feedback_df['CategoryID'].astype(str)
    category_df['CategoryID'] = category_df['CategoryID'].astype(str)

    try:
        df_cleaned = feedback_df.merge(customer_df, on='CustomerID', how='left')
        print(f"Rows after merging with Customer: {len(df_cleaned)}")

        df_cleaned = df_cleaned.merge(category_df, on='CategoryID', how='left')
        print(f"Rows after merging with Category: {len(df_cleaned)}")

    except Exception as e:
        print(f"Error during merging: {e}")
        return

    print(f"Columns in df_cleaned after merging: {df_cleaned.columns.tolist()}")

    df_cleaned['FeedbackDate'] = pd.to_datetime(df_cleaned['FeedbackDate'], errors='coerce')
    df_cleaned['OrderDate'] = pd.to_datetime(df_cleaned['OrderDate'], errors='coerce')
    df_cleaned['Year'] = df_cleaned['FeedbackDate'].dt.year
    df_cleaned['Month'] = df_cleaned['FeedbackDate'].dt.month
    df_cleaned['Day'] = df_cleaned['FeedbackDate'].dt.day
    df_cleaned['Week'] = df_cleaned['FeedbackDate'].dt.isocalendar().week
    df_cleaned['Quarter'] = df_cleaned['FeedbackDate'].dt.quarter

    with pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                        'Server=localhost;'
                        'Database=FeedbackDW;'
                        'Trusted_Connection=yes;') as conn_warehouse:
        cursor_warehouse = conn_warehouse.cursor()

        for query in warehouse_table_creation_queries.strip().split(';'):
            if query.strip():
                cursor_warehouse.execute(query)
        conn_warehouse.commit()

        unique_dates = df_cleaned[['FeedbackDate', 'Year', 'Month', 'Day', 'Week', 'Quarter']].drop_duplicates()
        for _, row in unique_dates.iterrows():
            date_dim_columns = ['FeedbackDate', 'Year', 'Month', 'Day', 'Week', 'Quarter']
            insert_into_table(cursor_warehouse, 'DateDim', date_dim_columns, row.to_dict())

        for _, row in df_cleaned.iterrows():
            row_dict = row.to_dict()
            cursor_warehouse.execute("SELECT DateID FROM DateDim WHERE FeedbackDate = ?", row_dict['FeedbackDate'])
            date_id = cursor_warehouse.fetchone()[0]

            # Include ReviewText and Sentiment in CustomerDim
            customer_dim_columns = [
                'CustomerID', 'FirstName', 'LastName', 'Email', 'PhoneNumber', 'Gender', 
                'Age', 'Location', 'ReviewText', 'Sentiment'
            ]
            insert_into_table(cursor_warehouse, 'CustomerDim', customer_dim_columns, row_dict, primary_key='CustomerID')

            category_dim_columns = ['CategoryID', 'CategoryNumber', 'CategoryName']
            insert_into_table(cursor_warehouse, 'CategoryDim', category_dim_columns, row_dict, primary_key='CategoryID')

            product_dim_columns = ['ProductID', 'ProductName', 'ProductPrice', 'CategoryID']
            insert_into_table(cursor_warehouse, 'ProductDim', product_dim_columns, row_dict, primary_key='ProductID')

            order_dim_columns = ['OrderID', 'OrderDate', 'OrderStatus', 'TotalAmount']
            insert_into_table(cursor_warehouse, 'OrderDim', order_dim_columns, row_dict, primary_key='OrderID')

            feedback_fact_columns = ['FeedbackID', 'CustomerID', 'CategoryID', 'ProductID', 'OrderID', 'Rating', 'DateID']
            row_dict['DateID'] = date_id
            insert_into_table(cursor_warehouse, 'FeedbackFact', feedback_fact_columns, row_dict, primary_key='FeedbackID')

        conn_warehouse.commit()

    print("Data transformation and loading completed successfully.")
    return df_cleaned


# Task 3: Sentiment Analysis Model
@task
def sentiment_analysis():
    # Connect to Data Warehouse to load data
    with pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                        'Server=localhost;'
                        'Database=FeedbackDW;'
                        'Trusted_Connection=yes;') as conn_warehouse:
        df_feedback_fact = pd.read_sql("SELECT * FROM FeedbackFact", conn_warehouse)
        df_reviews = pd.read_sql("SELECT * FROM CustomerDim", conn_warehouse)

    # Preprocess the text data
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words]
        return " ".join(tokens)

    df_reviews['processed_text'] = df_reviews['ReviewText'].apply(preprocess_text)

    # Prepare the data for modeling
    X = df_reviews['processed_text']
    y = df_reviews['Sentiment']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create TF-IDF features
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Train a Logistic Regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    # Evaluate the model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save predictions to a DataFrame
    df_reviews['Predicted_Sentiment'] = model.predict(tfidf_vectorizer.transform(df_reviews['processed_text']))

    # Export predictions to CSV
    df_reviews[['CustomerID', 'ReviewText', 'Sentiment', 'Predicted_Sentiment']].to_csv('predicted_sentiments.csv', index=False)
    print("Sentiment analysis predictions exported to predicted_sentiments.csv.")
    return df_reviews

# Task 4: Export Data from Data Warehouse
@task
def export_warehouse_data():
    with pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};'
                        'Server=localhost;'
                        'Database=FeedbackDW;'
                        'Trusted_Connection=yes;') as conn_warehouse:
        cursor_warehouse = conn_warehouse.cursor()

        # Export all tables to CSV for further analysis
        for table in ['CustomerDim', 'CategoryDim', 'DateDim', 'OrderDim', 'ProductDim', 'FeedbackFact']:
            df = pd.read_sql(f"SELECT * FROM {table}", conn_warehouse)
            df.to_csv(f"{table}.csv", index=False)
            print(f"{table} data exported to {table}.csv successfully.")

# Define the flow
@flow
def etl_feedback_pipeline():
    # Step 1: Extract and Load raw data into Raw Database
    raw_data = extract_load_raw_data()

    # Step 2: Transform and Load data into Data Warehouse
    transformed_data = transform_and_load_to_warehouse()

    # Step 3: Perform Sentiment Analysis
    sentiment_analysis_results = sentiment_analysis()

    # Step 4: Export Data from Warehouse
    export_warehouse_data()

if __name__ == "__main__":
    etl_feedback_pipeline()
