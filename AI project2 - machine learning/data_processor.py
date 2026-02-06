import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(dataset_path):
    # Load the news dataset from a parquet file
    news_dataframe = pd.read_parquet(dataset_path) 
    
    print(f"Total rows loaded: {len(news_dataframe)}") 
    
    # Text cleaning: Convert to lowercase and strip whitespace using pandas
    news_dataframe['text'] = news_dataframe['text'].str.lower().str.strip()
    
    # Feature extraction using TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    features_matrix = tfidf_vectorizer.fit_transform(news_dataframe['text'])
    labels_vector = news_dataframe['label']
    
    # Split the dataset into 80% training and 20% testing
    split_results = train_test_split(features_matrix, labels_vector, test_size=0.2, random_state=42)
    
    return split_results, tfidf_vectorizer