import pandas as pd
import numpy as np
import urllib.request
import zipfile
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import sys
import tkinter as tk
from tkinter import filedialog

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_FILENAME = "SMSSpamCollection"
ZIP_FILENAME = "sms_spam_collection.zip"
MODEL_FILENAME = "spam_model.pkl"

def download_and_prepare_data():
    if not os.path.exists(DATA_FILENAME):
        print(f"Downloading dataset from {DATA_URL}...")
        try:
            urllib.request.urlretrieve(DATA_URL, ZIP_FILENAME)
            with zipfile.ZipFile(ZIP_FILENAME, 'r') as zip_ref:
                zip_ref.extractall(".")
            print("Download and extraction complete.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            print("Creating a small dummy dataset for demonstration purposes instead.")
            create_dummy_dataset()
    else:
        print("Dataset found locally.")

def create_dummy_dataset():
    dummy_data = """ham\tGo until jurong point, crazy.. Available only in bugis n great world la e buffet...
ham\tOk lar... Joking wif u oni...
spam\tFree entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham\tU dun say so early hor... U c already then say...
ham\tNah I don't think he goes to usf, he lives around here though
spam\tFreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv
ham\tEven my brother is not like to speak with me. They treat me like aids patent.
ham\tAs per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune
spam\tWINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.
spam\tHad your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030"""
    with open(DATA_FILENAME, "w", encoding="utf-8") as f:
        f.write(dummy_data)

def load_data():
    df = pd.read_csv(DATA_FILENAME, sep='\t', header=None, names=['label', 'message'], encoding='utf-8', on_bad_lines='skip')
    df['label'] = df['label'].map({'ham': 'not spam', 'spam': 'spam'})
    return df

def train_model(X_train, y_train):
    print("Training new model with N-Grams...")
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))), 
        ('clf', MultinomialNB(alpha=0.1)),
    ])
    
    text_clf.fit(X_train, y_train)
    
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(text_clf, f)
    print(f"Model saved to {MODEL_FILENAME}")
    
    return text_clf

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    print("\n" + "-"*30)
    print(" MODEL EVALUATION ")
    print("-" * 30)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

def predict_message(model, message):
    prediction = model.predict([message])[0]
    proba = model.predict_proba([message])[0]
    confidence = max(proba) * 100
    return prediction, confidence

def batch_predict_from_file(model):
    print("\n--- Batch Prediction Mode ---")
    print("This mode reads a file of messages and saves the predictions to 'spam_predictions.csv'.")
    
    print("Tip: Press ENTER without typing to open a File Browser window.")
    input_path = input("Enter the path to your file: ").strip()
    
    input_path = input_path.strip('"').strip("'")
    
    if input_path == "":
        try:
            root = tk.Tk()
            root.withdraw() 
            root.attributes('-topmost', True)
            
            print("waiting for file selection...")
            input_path = filedialog.askopenfilename(
                title="Select a file to analyze",
                filetypes=[("CSV and Text", "*.csv *.txt"), ("All Files", "*.*")]
            )
            root.destroy()
            
            if not input_path:
                print("No file selected.")
                return
            print(f"Selected: {input_path}")
            
        except Exception as e:
            print(f"\nCould not open File Browser (Error: {e})")
            print("Please type the path manually below:")
            input_path = input("Enter the path to your file: ").strip()

    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' was not found.")
        return

    try:
        df = None
        msg_col = None
        
        if input_path.endswith('.csv'):
            print("Reading CSV file...")
            try:
                df = pd.read_csv(input_path, encoding='utf-8')
            except UnicodeDecodeError:
                print("UTF-8 encoding failed. Trying 'latin1'...")
                try:
                    df = pd.read_csv(input_path, encoding='latin1')
                except UnicodeDecodeError:
                    print("Latin1 failed. Trying 'cp1252'...")
                    df = pd.read_csv(input_path, encoding='cp1252')
            
            possible_cols = ['message', 'text', 'sms', 'content', 'body', 'v2', 'email']
            msg_col = next((col for col in possible_cols if col in df.columns), None)
            
            if not msg_col:
                msg_col = df.columns[0]
                print(f"Warning: Could not identify message column name. Using the first column: '{msg_col}'.")
            
            messages = df[msg_col].astype(str)
        else:
            print("Reading Text file (assuming one message per line)...")
            messages = []
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    messages = [line.strip() for line in f.readlines() if line.strip()]
            except UnicodeDecodeError:
                print("UTF-8 encoding failed. Trying 'latin1'...")
                with open(input_path, 'r', encoding='latin1') as f:
                    messages = [line.strip() for line in f.readlines() if line.strip()]
            
            df = pd.DataFrame({'message': messages})
            msg_col = 'message'
            messages = df['message']

        print(f"Analyzing {len(messages)} messages...")
        
        predictions = model.predict(messages)
        probabilities = model.predict_proba(messages)
        confidences = [max(prob) * 100 for prob in probabilities]
        
        df['Predicted_Label'] = predictions
        df['Confidence_Score'] = [f"{c:.2f}%" for c in confidences]
        
        output_filename = "spam_predictions.csv"
        df.to_csv(output_filename, index=False)
        
        print(f"\nSUCCESS! Predictions saved to '{output_filename}'")
        print("\n--- Preview of Results ---")
        print(df[[msg_col, 'Predicted_Label', 'Confidence_Score']].head())
        
    except Exception as e:
        print(f"Error processing file: {e}")

def show_top_features(model, n=15):
    vectorizer = model.named_steps['tfidf']
    classifier = model.named_steps['clf']
    
    get_feature_names = getattr(vectorizer, 'get_feature_names_out', vectorizer.get_feature_names)
    feature_names = get_feature_names()
    
    print("\n" + "="*60)
    print(" MODEL INSIGHTS: TOP WORDS LEARNT ")
    print("="*60)
    
    for i, class_label in enumerate(classifier.classes_):
        top_indices = np.argsort(classifier.feature_log_prob_[i])[-n:][::-1]
        top_words = [feature_names[j] for j in top_indices]
        
        print(f"\nTop indicators for '{class_label}':")
        print(", ".join(top_words))

def main():
    print("Initializing Advanced Spam Detection System...")
    
    download_and_prepare_data()
    
    model = None
    if os.path.exists(MODEL_FILENAME):
        try:
            print(f"Loading saved model from {MODEL_FILENAME}...")
            with open(MODEL_FILENAME, 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"Could not load model: {e}")
    
    if model is None:
        print("Loading data for training...")
        df = load_data()
        print(f"Loaded {len(df)} messages.")
        print(f"Class distribution:\n{df['label'].value_counts()}")
        
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        
        try:
            show_top_features(model)
        except Exception as e:
            print(f"Could not extract features: {e}")
    else:
        print("Model loaded successfully. Skipping re-training.")

    while True:
        print("\n" + "="*60)
        print(" MAIN MENU ")
        print("="*60)
        print("1. Interactive Mode (Type messages manually)")
        print("2. Batch Mode (Import file & Export results)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\n--- Interactive Mode (Type 'back' to return to menu) ---")
            while True:
                user_input = input("\nEnter message: ")
                if user_input.lower() in ['exit', 'quit', 'back']:
                    break
                if not user_input.strip():
                    continue
                    
                prediction, confidence = predict_message(model, user_input)
                
                if prediction == 'spam':
                    print(f"Result: \033[91mSPAM DETECTED\033[0m")
                    print(f"Confidence: {confidence:.2f}%")
                else:
                    print(f"Result: \033[92mNOT SPAM\033[0m")
                    print(f"Confidence: {confidence:.2f}%")
                    
        elif choice == '2':
            batch_predict_from_file(model)
            
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    main()
