import random
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

# Initialize Faker and seed random for reproducibility
fake = Faker()

"""
Faker.seed(random.randrange(0,100))
random.seed(random.randrange(0,100))
np.random.seed(random.randrange(0,100))
"""

Faker.seed(42)
random.seed(42)
np.random.seed(42)
# Number of records to generate
num_records = 162980

# Helper functions
def add_noise_to_date(date, noise_days=30):
    noise = timedelta(days=random.randint(-noise_days, noise_days))
    return max(date + noise, date)

def generate_realistic_age():
    age = int(np.random.normal(35, 12))
    return max(0, min(age, 200))

def generate_realistic_price(base_price, volatility=0.2):
    price = base_price * (1 + np.random.normal(0, volatility))
    return round(max(0.01, price), 2)

def generate_realistic_rating(sentiment):
    if sentiment == "positive":
        return max(1, min(5, int(np.random.normal(4.5, 0.5))))
    elif sentiment == "neutral":
        return max(1, min(5, int(np.random.normal(3, 0.7))))
    else:  # negative
        return max(1, min(5, int(np.random.normal(1.5, 0.5))))

def generate_sentiment_based_review(product_name, product_category):
    sentiment = random.choices(["positive", "neutral", "negative"], weights=[0.6, 0.3, 0.1])[0]
    
    sentiment_vocab = {
        "positive": ["excellent", "outstanding", "fantastic", "superb", "delighted", "impressed"],
        "neutral": ["okay", "decent", "average", "satisfactory", "fair", "reasonable"],
        "negative": ["disappointing", "frustrated", "terrible", "poor", "unsatisfactory", "awful"]
    }
    
    category_specific_comments = {
        "Electronics": {
            "positive": [f"The {product_name} has amazing battery life.", f"The display quality of the {product_name} is stunning."],
            "neutral": [f"The {product_name} performs as expected, nothing extraordinary.", f"The {product_name} is decent for its price range."],
            "negative": [f"The {product_name} overheats easily.", f"The user interface of the {product_name} is confusing."]
        },
        "Clothing": {
            "positive": [f"The fabric of the {product_name} is so comfortable.", f"The {product_name} fits perfectly."],
            "neutral": [f"The {product_name} is okay, but the color is slightly off from the picture.", f"The {product_name} is decent, but not very durable."],
            "negative": [f"The sizing of the {product_name} is completely off.", f"The {product_name} started falling apart after just one wash."]
        },
    }
    
    general_comments = {
        "positive": [
            f"Overall, I'm extremely satisfied with my {product_name}.",
            f"I would definitely recommend the {product_name} to others.",
            f"The quality of the {product_name} exceeded my expectations.",
        ],
        "neutral": [
            f"The {product_name} is decent, but there's room for improvement.",
            f"My experience with the {product_name} was mixed.",
            f"The {product_name} is okay for its price, but nothing special.",
        ],
        "negative": [
            f"I'm very disappointed with the {product_name}.",
            f"I wouldn't recommend the {product_name} to anyone.",
            f"The {product_name} fell short of my expectations in every way.",
        ]
    }
    
    if product_category in category_specific_comments:
        comments = category_specific_comments[product_category][sentiment] + general_comments[sentiment]
    else:
        comments = general_comments[sentiment]
    
    num_sentences = min(random.randint(2, 4), len(comments))
    review = " ".join(random.sample(comments, num_sentences))
    
    sentiment_words = random.sample(sentiment_vocab[sentiment], 2)
    review += f" In summary, this product is {sentiment_words[0]} and the overall experience was {sentiment_words[1]}."
    
    return review, sentiment

# Generate the combined dataset
combined_data = []
start_date = datetime.now() - timedelta(days=365*2)
product_categories = ["Electronics", "Clothing", "Home & Garden", "Books", "Toys", "Beauty", "Sports"]
feedback_categories = ["Product Quality", "Customer Service", "Pricing", "Delivery", "User Experience"]

for i in range(1, num_records + 1):
    feedback_date = add_noise_to_date(fake.date_between(start_date=start_date, end_date='today'))
    product_name = fake.word().title() + " " + random.choice(["Pro", "X", "Plus", "Max"])
    product_category = random.choice(product_categories)
    review_text, sentiment = generate_sentiment_based_review(product_name, product_category)
    
    record = {
        "FeedbackID": i,
        "CustomerID": i,
        "CategoryID": i,
        "OrderID": i,
        "ProductID": i,
        "ReviewID": i,
        "FirstName": fake.first_name(),
        "LastName": fake.last_name(),
        "Email": fake.email(),
        "PhoneNumber": fake.phone_number(),
        "Gender": random.choice(["Male", "Female", "Other"]),
        "Age": generate_realistic_age(),
        "Location": f"{fake.city()}, {fake.country()}",
        "CategoryName": random.choice(feedback_categories),
        "CategoryNumber":random.randint(1, 5),
        "FeedbackDate": feedback_date,
        "Year": feedback_date.year,
        "Month": feedback_date.month,
        "Day": feedback_date.day,
        "Week": feedback_date.isocalendar()[1],
        "Quarter": (feedback_date.month - 1) // 3 + 1,
        "OrderDate": add_noise_to_date(fake.date_between(start_date=start_date, end_date='today')),
        "OrderStatus": random.choice(["Shipped", "Delivered", "Cancelled", "Pending", "Returned"]),
        "TotalAmount": generate_realistic_price(250, volatility=0.5),
        "ProductName": product_name,
        "ProductCategory": product_category,
        "ProductPrice": generate_realistic_price(random.uniform(10.0, 1000.0)),
        "ReviewText": review_text,
        "Sentiment": sentiment,
        "Rating": generate_realistic_rating(sentiment)
    }
    combined_data.append(record)

# Create DataFrame and save to CSV
df_combined = pd.DataFrame(combined_data)
df_combined.to_csv("feedback_data.csv", index=False)

print("Combined dataset has been written to 'feedback_data.csv'")