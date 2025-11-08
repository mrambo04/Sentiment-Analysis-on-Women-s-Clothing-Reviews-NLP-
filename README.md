# Sentiment-Analysis-on-Women-s-Clothing-Reviews-NLP-
This project applies natural language processing (NLP) techniques to analyze customer reviews for women’s clothing products. The goal is to classify sentiment (positive, neutral, negative), extract key insights and assist businesses in understanding what drives customer opinion.
# 💬 Sentiment Analysis on Women’s Clothing Reviews (NLP)

## 📘 Overview  
This project applies natural language processing (NLP) techniques to analyze customer reviews for women’s clothing products. The goal is to classify sentiment (positive, neutral, negative), extract key insights and assist businesses in understanding what drives customer opinion.

## 🎯 Objective  
To build a text-analysis pipeline that:  
- Processes raw review text, cleans and transforms it  
- Classifies review sentiment using machine learning  
- Provides actionable insights on customer feedback for product and service improvement

## 🧰 Tools & Technologies  
Python • Pandas • NumPy • NLTK / spaCy • Scikit-Learn • Matplotlib / Seaborn • Jupyter Notebook

## 🧮 Approach  
1. **Data Ingestion & Cleaning** – Load the reviews dataset, handle missing values, and clean text (lowercasing, punctuation removal, stopwords)  
2. **Exploratory Text Analysis** – Generate word clouds, frequency plots, and sentiment distributions  
3. **Feature Engineering** – Tokenize, lemmatize/stem words, vectorize using TF-IDF or word embeddings  
4. **Model Building** – Train classifiers (e.g., Logistic Regression, Naive Bayes, SVM) to predict sentiment categories  
5. **Evaluation** – Use metrics such as Accuracy, Precision, Recall, F1-Score, Confusion Matrix  
6. **Insights & Visualization** – Highlight key themes driving positive and negative sentiment, provide actionable takeaways  

## 📈 Key Results  
- Achieved **Accuracy** on the test set    
- Word-cloud analysis revealed that terms like *“fit”, “quality”, “size”* dominate positive sentiment, while *“delay”, “return”, “size”* dominate negative sentiment  
- Business insight: Review length and ratings correlate strongly with sentiment score; distinct clusters of “fit issues” and “shipping delays” emerged  

## 📂 Dataset  
[https://www.kaggle.com/code/rambabubevara/womens-clothing-comments-sentiment-analasys]   

## 🚀 Usage  
```bash
# Clone repository
git clone https://github.com/mrambo04/Sentiment-Analysis-on-Women-s-Clothing-Reviews-NLP-.git
cd Sentiment-Analysis-on-Women-s-Clothing-Reviews-NLP-

# (Optional) Create virtual environment & install dependencies
pip install -r requirements.txt

# Run notebook or script
jupyter notebook Sentiment_Analysis_Womens_Clothing.ipynb
