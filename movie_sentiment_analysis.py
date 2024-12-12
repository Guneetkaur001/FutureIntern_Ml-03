import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)

def create_sample_reviews():
   
    reviews = [
        "This movie was fantastic! Great acting and amazing plot.",
        "Terrible waste of time. Poor acting and boring story.",
        "I loved every minute of this film, definitely recommended!",
        "Worst movie I've ever seen. Complete disaster.",
        "Excellent performance by the entire cast. A masterpiece!",
        "Don't waste your money. Nothing makes sense.",
        "Beautiful cinematography and compelling storyline.",
        "Disappointing plot with mediocre acting.",
        "A perfect blend of humor and drama. Must watch!",
        "Predictable story with awful dialogue.",
        "The special effects were incredible, loved it!",
        "The worst screenplay I've ever witnessed.",
        "Brilliant direction and outstanding performances.",
        "Completely missed the mark, terrible execution.",
        "A cinematic masterpiece that will be remembered.",
        "Save your time and money, absolutely terrible.",
        "Engaging story with unexpected twists.",
        "Poor character development and weak plot.",
        "One of the best films of the year!",
        "A complete waste of potential, very disappointing."
    ]
    
    sentiments = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

def simple_preprocess(text):
  
    text = text.lower()
 
    text = re.sub(r'[^a-z\s]', '', text)

    text = ' '.join(text.split())
    
    return text

def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
 
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

def plot_feature_importance(vectorizer, classifier, n_top_features=10):
  
    feature_names = vectorizer.get_feature_names_out()
    feature_importance = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
   
    top_indices = np.argsort(feature_importance)[-n_top_features:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(n_top_features), feature_importance[top_indices])
    plt.yticks(range(n_top_features), [feature_names[i] for i in top_indices])
    plt.xlabel('Importance Score')
    plt.title('Top Important Words for Sentiment Classification')
    plt.tight_layout()
    plt.show()

def main():
    print("Loading and preparing data...")
    data = create_sample_reviews()
    print(f"Dataset shape: {data.shape}")

    print("Preprocessing text data...")
    data['processed_review'] = data['review'].apply(simple_preprocess)

    X = data['processed_review']
    y = data['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Converting text to TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print("Training Naive Bayes classifier...")
    classifier = MultinomialNB()
    classifier.fit(X_train_tfidf, y_train)
   
    y_pred = classifier.predict(X_test_tfidf)
   
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))
    
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.2f}")
  
    plot_confusion_matrix(y_test, y_pred)
  
    plot_feature_importance(vectorizer, classifier)
   
    print("\nTesting with new reviews:")
    example_reviews = [
        "This movie was absolutely amazing!",
        "I regret watching this terrible film",
        "An interesting movie with good acting",
        "The special effects were outstanding",
        "The plot made no sense at all"
    ]
   
    processed_examples = [simple_preprocess(review) for review in example_reviews]
    example_tfidf = vectorizer.transform(processed_examples)
    predictions = classifier.predict(example_tfidf)
    probabilities = classifier.predict_proba(example_tfidf)
    
    print("\nPrediction Results:")
    for review, prediction, proba in zip(example_reviews, predictions, probabilities):
        sentiment = "Positive" if prediction == 1 else "Negative"
        confidence = proba.max() * 100
        print(f"\nReview: {review}")
        print(f"Predicted sentiment: {sentiment} (Confidence: {confidence:.2f}%)")

if __name__ == "__main__":
    main()