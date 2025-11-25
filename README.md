# SPAM-DETECTION-SYSTEM
Spam Detection System (Machine Learning + Python)

This project is a simple but functional spam classifier built using Python.
It uses classic text-processing techniques and a machine-learning model to classify SMS/email-like messages as Spam or Ham (Not Spam).
Both the dataset and the model script are included:
spam.csv — training data
spamdetection.py — model training + prediction script
🚀 Features
Text preprocessing (cleaning, tokenization, etc.
TF-IDF vectorization
Machine Learning model (e.g., Naive Bayes / Logistic Regression depending on your script)
Predicts whether a message is spam or not
Loads dataset from CSV
Outputs accuracy and predictions
📂 Project Structure
├── spam.csv               # Dataset
├── spamdetection.py       # Main ML script
└── README.md              # Project documentation
⚙️ How It Works 
Loads the dataset (spam.csv)
Cleans and preprocesses the text
Splits into train/test sets
Converts text to vectors using TF-IDF
Trains the ML model
Evaluates accuracy
Allows testing on custom input text

📦 Requirements
Make sure these are installed:
pip install pandas numpy scikit-learn
If your script uses something extra (like NLTK), add it. Don’t play guessing games—add everything needed.
▶️ How to Run the Project
python spamdetection.py
If you want to test custom input, modify the code block inside the script where prediction happens.
📊 Model Accuracy

The accuracy depends on your preprocessing and model choice.
If you're getting anything below 90%, your preprocessing is weak or you're using a lousy model.
Tune with:
Better text cleaning
More balanced train-test spli
Trying Logistic Regression or SVM instead of naive models
📝 Dataset Information
The dataset contains labeled messages:
ham → not spam
spam → spam text

🙌 Author

Aditya Jain
