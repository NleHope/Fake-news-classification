# Fake-news-classification with Machine Learning and Deep Learning
Problem Statement
The rise of fake news and propaganda has created severe risks: misinformation during crises, erosion of public trust, political manipulation, and polarization. This project tackles fake news detection using Natural Language Processing (NLP) and machine learning models. The challenge lies in the subtle linguistic mimicry of legitimate news by fake articles and the dynamic nature of misinformation.

Dataset
True Articles: From reputable media (e.g., Reuters, NYT, Washington Post)

File: MisinfoSuperset_TRUE.csv

Fake Articles: From misinformation or propaganda sources

File: MisinfoSuperset_FAKE.csv

Datasets available at: Google Drive Link

Each article is labeled:

1 for True

0 for Fake

Project Workflow
This project follows the complete machine learning pipeline:

Data Preprocessing:
Load, merge, and clean the dataset.

Exploratory Data Analysis (EDA):
Analyze text distributions, class balance, and word usage patterns.

Feature Engineering:
Use NLP techniques like TF-IDF or word embeddings for vectorization.

Model Building:
Train multiple models such as:

Logistic Regression

Naive Bayes

SVM

Random Forest

(Optional) LSTM/BERT-based deep learning models

Hyperparameter Tuning:
Use Grid Search or other methods for model optimization.

Evaluation:
Metrics used include:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Conclusion:
Summary of findings, strengths/limitations of the model, and future work.

🛠️ Technologies Used
Python

Pandas, NumPy

Scikit-learn

Matplotlib / Seaborn

NLTK / SpaCy

(Optional) TensorFlow / PyTorch

👥 Team
Project completed by a team of 3 members for the final course assignment.
