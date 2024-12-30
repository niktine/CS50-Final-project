import numpy as np
from joblib import load
import re
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')


try:
    tfidf = load("tf.joblib")
    logreg = load("logreg.joblib")
    print("Models loaded successfully!")
except FileNotFoundError as e:
    print(f"Model file not found: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    exit()



def main():
    
    try:
        paragraph = input("Please enter your paragraph: ")
        print("\n\nOriginal Text:", paragraph)

        paragraph_cleaned = preprocess_text(paragraph)
        print("\n\nCleaned Text:", paragraph_cleaned)

        paragraph_lemmatized = p_lemmatizer(paragraph_cleaned)
        print("\n\nLemmatized Text:", paragraph_lemmatized)

        predict_and_explain_instance(paragraph_lemmatized)
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")



def preprocess_text(text):
    """Clean and preprocess the input text."""
    try:
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        text = re.sub(r'http\S+|www\S+|https\S+', '', text).strip()  # Remove URLs and trim spaces
        return text
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise



def p_lemmatizer(text):
    """Lemmatize the input text."""
    try:
        lemmatizer = WordNetLemmatizer()
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    except Exception as e:
        print(f"Error during lemmatization: {e}")
        raise


def predict_and_explain_instance(cleaned_text):
    """Predict and explain the classification for the given text."""
    try:

        cleaned_text_vectorized = tfidf.transform([cleaned_text])
        probabilities_logreg = logreg.predict_proba(cleaned_text_vectorized)[0]
        predicted_label_logreg = np.argmax(probabilities_logreg)

        explainer = LimeTextExplainer(class_names=['Non-Suicide', 'Suicide'])
        exp = explainer.explain_instance(
            cleaned_text,
            classifier_fn=lambda texts: logreg.predict_proba(tfidf.transform(texts)),
            num_features=10)

        print(f"\nExplanation of the model's prediction: {'Suicide' if predicted_label_logreg == 1 else 'Not Suicide'}")
        exp.save_to_file("prediction_result.html")
        return predicted_label_logreg
    except Exception as e:
        print(f"Error during prediction and explanation: {e}")
        raise

if __name__ == "__main__":
    main()
