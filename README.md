# Sentiment Analysis and Next word Prediction

#### Problem Statement:
#### As part of this NLP Project we have worked on two use cases 1. Twitter Feed Sentiment Analyis and 2. Next word Prediction

#### 1. Sentiment Analysis : A huge dataset consisting of 1.6 Million tweets has been used for the training purpose. We have applied various Text Preprocessing techniques like lower casing, special characters removal, contractions, stop words removal, lemmatization, Tokenization and finally used the Google's Word2vec model to convert them to vectors. Here we have made use the Deep Learning networks like RNN & LSTM to build the models. 

#### 2. Next Word Prediction : For this use case we have used a small corpus/document which consists of details pertaining to the history and making of Pizzas. The preprocessing techniques for the Sentiment Analysis has been applied even for this. Even here we have made use of the RNN & LSTM architectures in order to perform the predictions.  

### Setting up the conda environment 
```conda create -p env python==3.10```

### Activate the conda environment
```conda activate env\```

### Install all the requirements 
```pip install -r requirements.txt```

### Twitter Sentiment Analysis 
#### Training - Execute the notebook  
#### Model Training would be completed and the following pickle files would be generated, store them in local and use those pickle files for Model Testing purpose.

### Model Testing 
### Run the Streamlit app, pass the required inputs and click on Predict
### In order to test the application
```streamlit run app.py```

