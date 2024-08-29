import os
import streamlit as st
import pickle
import numpy as np
from utils import preprocess_text,lemmatize_text
from nltk.stem import PorterStemmer,WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load your pre-trained models
pickle_folder_path = 'D:\\Saravanesh Personal\\Guvi\\Capstone Projects\\NLP\\pickle_files'

# next_word_model = pickle.load(open('next_word_model.pkl', 'rb'))


# Function for Sentiment Analysis
def predict_sentiment(text):
    # Loading the pickled model
    sentiment_model = tf.keras.models.load_model(os.path.join(pickle_folder_path, 'sentiment_analysis_iter2_lstm_model2.keras'))
    senti_tokenizer = pickle.load(open(os.path.join(pickle_folder_path,'sentiment_analysis_iter2_tokenizer.pkl'), 'rb'))  
    # Preprocess the input text as needed (e.g., tokenization, padding)
    text=preprocess_text(text)
    # Lemmatize the text 
    lemmatizer = WordNetLemmatizer()
    text=lemmatize_text(text,lemmatizer)
    # Tokenization
    sequences = senti_tokenizer.texts_to_sequences([text])
    # print("The sequences after applying the senti_tokenizer :",sequences)
    # Padding sequences
    max_sequence_length = 34 #This was the max_length in the train data so reusing the same. 
    X_pad = pad_sequences(sequences, maxlen=max_sequence_length)
    # print("The X_pad after padding the sequences :", X_pad)
    # Model Prediction 
    prediction = sentiment_model.predict([X_pad])
    # print("The prediction after the model.predict :",prediction)
    # Convert probabilities to class labels (0 or 1)
    prediction = (prediction > 0.5).astype(int)

    return "Positive" if prediction == 1 else "Negative"

# Function for Next Word Prediction
def predict_next_word(text):
    # Loading the pickled model
    word_predict_model = tf.keras.models.load_model(os.path.join(pickle_folder_path, 'next_word_prediction_model.keras'))
    word_predict_tokenizer = pickle.load(open(os.path.join(pickle_folder_path,'word_prediction_tokenizer.pkl'), 'rb'))
    max_sequence_len=40

    # Prepare the text for prediction
    token_list = word_predict_tokenizer.texts_to_sequences([text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')  

    # Predict the next word
    predicted_probs = word_predict_model.predict(token_list)
    predicted_word_index = np.argmax(predicted_probs)
    predicted_word = word_predict_tokenizer.index_word.get(predicted_word_index, '')

    return predicted_word

# Streamlit App
def main():
    # Initialize session state for storing option and user input
    if 'option' not in st.session_state:
        st.session_state.option = "Home"
    if 'user_input_sentiment' not in st.session_state:
        st.session_state.user_input_sentiment = ""
    if 'user_input_next_word' not in st.session_state:
        st.session_state.user_input_next_word = ""

    st.sidebar.title("NLP Use Cases")
    st.session_state.option = st.sidebar.selectbox("Choose a Use Case", ["Home", "Sentiment Analysis", "Next Word Prediction"], index=["Home", "Sentiment Analysis", "Next Word Prediction"].index(st.session_state.option))
    
    if st.session_state.option == "Home":
        st.title("Natural Language Processing - Project Overview")
        st.markdown("## Sentiment Analysis")
        st.write("""
        This use case leverages the Sentiment140 dataset, which contains 1.6 million tweets from the Twitter API. The model has been trained to predict whether a given sentence is positive or negative.
        """)
        st.markdown("## Next word Prediction")
        st.write("""
        This use case predicts the next word in a sentence based on a small corpus related to pizzas. RNN and LSTM architectures have been used to build this model.
        """)

    elif st.session_state.option == "Sentiment Analysis":
        st.title("Sentiment Analysis based on Twitter API data")
        st.session_state.user_input_sentiment = st.text_input("Enter a sentence to predict sentiment", value=st.session_state.user_input_sentiment)
        
        if st.button("Predict Sentiment"):
            if st.session_state.user_input_sentiment:
                prediction = predict_sentiment(st.session_state.user_input_sentiment)
                st.write(f"The sentiment of the sentence is: **{prediction}**")
            else:
                st.write("Please enter a sentence.")

    elif st.session_state.option == "Next Word Prediction":
        st.title("Next Word Prediction - Corpus Used: Pizzas")
        st.session_state.user_input_next_word = st.text_input("Enter a sentence to predict the next word", value=st.session_state.user_input_next_word)
        
        if st.button("Predict Next Word"):
            if st.session_state.user_input_next_word:
                predicted_word = predict_next_word(st.session_state.user_input_next_word)
                st.write(f"The next word prediction is: **{predicted_word}**")
            else:
                st.write("Please enter a sentence.")

    
if __name__ == "__main__":
    main()
