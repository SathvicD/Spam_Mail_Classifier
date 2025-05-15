import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv('spam_emails.csv')

# Features and labels
X = data['text']
y = data['label']

# Split data (optional, for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)

# Train model
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Function to classify new email
def classify_email(text):
    text_counts = vectorizer.transform([text])
    prediction = clf.predict(text_counts)
    return prediction[0]

# Main loop to get input and output
if __name__ == '__main__':
    print("Spam Email Classifier (Type 'exit' to quit)")
    while True:
        email = input("Enter email text: ")
        if email.lower() == 'exit':
            print("Exiting...")
            break
        result = classify_email(email)
        print(f"Result: {result.upper()}\n")