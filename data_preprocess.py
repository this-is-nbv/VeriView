import pandas as pd
import numpy as np
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, render_template, request, jsonify
from torch.utils.data import DataLoader, TensorDataset
import os

# Initialize Flask app
app = Flask(__name__)

UPLOAD_FOLDER = 'datasets'
MODEL_FOLDER = 'models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # Ensure this is set
app.config['MODEL_FOLDER'] = MODEL_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class MIANA(nn.Module):
    def __init__(self, input_size=300, hidden_size=150, dropout=0.5):
        super(MIANA, self).__init__()
        self.bi_lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.bi_lstm(x.unsqueeze(1))
        lstm_out = self.dropout(lstm_out)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_sum = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(weighted_sum)
        return self.sigmoid(output)

def preprocess_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return tokens

def train_category_model(category, dataset_path):
    df = pd.read_csv(dataset_path)
    df['validity'] = df['validity'].astype(int)
    df['cleaned_reviews'] = df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_reviews'], df['validity'], test_size=0.2, stratify=df['validity'], random_state=42
    )

    w2v_path = f"{MODEL_FOLDER}/w2v_{category}.bin"
    miana_path = f"{MODEL_FOLDER}/miana_{category}.pth"

    if os.path.exists(w2v_path):
        w2v_model = Word2Vec.load(w2v_path)
    else:
        w2v_model = Word2Vec(sentences=X_train, vector_size=300, window=5, min_count=2, workers=4, sg=1, epochs=10)
        w2v_model.save(w2v_path)

    def review_to_vector(review, model):
        vectors = [model.wv[word] for word in review if word in model.wv]
        return np.mean(vectors, axis=0) if vectors else np.full(300, -1)

    X_train_vecs = np.array([review_to_vector(review, w2v_model) for review in X_train])
    X_test_vecs = np.array([review_to_vector(review, w2v_model) for review in X_test])

    X_train_tensor = torch.tensor(X_train_vecs, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_vecs, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

    model = MIANA()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCELoss()

    for epoch in range(20):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch).squeeze(1)
            loss = criterion(predictions, y_batch.squeeze(1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[{category}] Epoch {epoch+1}, Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), miana_path)
def load_model(category):
    """Load the trained MIANA model for the given category from the models folder."""
    model_path = os.path.join(MODEL_FOLDER, f"miana_{category}.pth")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found for category '{category}': {model_path}")
        return None  # Return None if model does not exist

    try:
        model = MIANA()  # Initialize model architecture
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        print(f"✅ Successfully loaded model for category '{category}'")
        return model
    except Exception as e:
        print(f"❌ Error loading model for category '{category}': {str(e)}")
        return None


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or 'category' not in request.form:
        return jsonify({'error': 'Missing file or category'}), 400

    file = request.files['file']
    category = request.form['category'].strip()

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Ensure category is valid
    if not category:
        return jsonify({'error': 'Category name is required'}), 400

    dataset_filename = f"{category}.csv"
    dataset_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_filename)

    try:
        file.save(dataset_path)
        return jsonify({'message': f'Dataset for {category} uploaded successfully', 'file_path': dataset_path})
    except Exception as e:
        return jsonify({'error': f'Failed to save file: {str(e)}'}), 500

# Flask API for predictions
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detector')
def detector():
    return render_template('detector.html')  

@app.route('/admin_login')
def admin_login():
    return render_template('admin_login.html')  

@app.route('/admin')
def admin():
    return render_template('admin.html') 


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('review', '')
        category = data.get('category', '')

        if not review_text.strip() or len(word_tokenize(review_text)) < 3:
            return jsonify({'fake_review_probability': 1.0, 'is_fake': True})

        w2v_path = f"{MODEL_FOLDER}/w2v_{category}.bin"
        miana_path = f"{MODEL_FOLDER}/miana_{category}.pth"

        if not os.path.exists(w2v_path) or not os.path.exists(miana_path):
            return jsonify({'error': 'Model not found for selected category'}), 400

        w2v_model = Word2Vec.load(w2v_path)

        def review_to_vector(review, model):
            vectors = [model.wv[word] for word in review if word in model.wv]
            return np.mean(vectors, axis=0) if vectors else np.full(300, -1)

        tokens = preprocess_text(review_text)
        vector = review_to_vector(tokens, w2v_model)
        tensor_input = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

        model = load_model(category)
        model.load_state_dict(torch.load(miana_path))
        model.eval()

        with torch.no_grad():
            probability = model(tensor_input).item()
        predicted_label = 1 if probability > 0.5 else 0

        return jsonify({'fake_review_probability': probability, 'is_fake': bool(predicted_label)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
def train_all_models(cat):
    """Train models for all dataset files in the 'datasets' folder that match the category."""
    if not os.path.exists(UPLOAD_FOLDER):
        print("No dataset folder found!")
        return
    
    dataset_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]

    if not dataset_files:
        print("No dataset files found!")
        return

    for dataset_file in dataset_files:
        category = dataset_file.replace('.csv', '')  # Extract category from filename
        dataset_path = os.path.join(UPLOAD_FOLDER, dataset_file)
        try:
            # Check if the category matches the one passed in the 'cat' parameter
            if category == cat:
                print(f"Training model for category: {category}")
                train_category_model(category, dataset_path)
                print(f"✅ Model for '{category}' trained successfully and saved in '{MODEL_FOLDER}'")
        except Exception as e:
            print(f"❌ Failed to train model for '{category}': {str(e)}")

@app.route('/train_models', methods=['POST'])
def train_models():
    try:
        data = request.get_json()
        cat = data.get('category')  # Extract category
        if cat is None:
            return jsonify({'error': 'Category is required'}), 400
        train_all_models(cat)
        return jsonify({'message': 'Training started successfully!'})
    except Exception as e:
        return jsonify({'error': f'Training failed: {str(e)}'}), 500


# Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
