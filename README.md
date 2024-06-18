

# Natural Language Processing (Bag Of Words)

## Logos
![Natural Language Processing](https://img.shields.io/badge/Natural_Language_Processing-purple.svg)
![Bag of Words](https://img.shields.io/badge/Bag_of_Words-red.svg)
![Sentiment Analysis](https://img.shields.io/badge/Sentiment_Analysis-blue.svg)
![Naive Bayes](https://img.shields.io/badge/Naive_Bayes-yellow.svg)
![Yuvraj Singh Chowdhary](https://img.shields.io/badge/Yuvraj%20Singh%20Chowdhary-orange.svg)




## Project Overview

This project involves performing sentiment analysis on restaurant reviews using Natural Language Processing (NLP) techniques. The dataset is in TSV (Tab-Separated Values) format. The steps include data cleaning, creating a Bag of Words model, training a Naive Bayes classifier, and evaluating the model's performance.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Bag of Words Model](#bag-of-words-model)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Owner](#owner)

## Introduction

Sentiment analysis is a key application of NLP that involves determining the sentiment expressed in text data. In this project, we analyze restaurant reviews to classify them as positive or negative.

## Installation

To run this project, you need to have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- NLTK
- scikit-learn

You can install the required libraries using pip:
```sh
pip install numpy pandas matplotlib nltk scikit-learn
```

## Dataset

The dataset used in this project is `Restaurant_Reviews.tsv`, which contains 1,000 reviews with corresponding sentiments (positive or negative).

## Data Preprocessing

### Cleaning Text

1. **Remove Non-letter Characters:** Using regular expressions to keep only letters.
2. **Convert to Lowercase:** Standardize text to lowercase.
3. **Tokenization:** Split text into individual words.
4. **Remove Stopwords:** Remove common words using NLTK's stopwords list.
5. **Stemming:** Reduce words to their root form using the PorterStemmer.

### Example Code:
```python
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
```

## Bag of Words Model

Convert the cleaned text into numerical data using the Bag of Words model with `CountVectorizer`.

### Example Code:
```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
```

## Model Training

Split the dataset into training and testing sets and train a Naive Bayes classifier.

### Example Code:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

## Evaluation

Predict sentiments for the test set and evaluate the model using a confusion matrix and accuracy score.

### Example Code:
```python
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy:', accuracy_score(y_test, y_pred))
```

## Results

The model's performance is measured by its accuracy and the confusion matrix, which indicates how many reviews were correctly or incorrectly classified.

## Contributing

If you wish to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Owner

- **Name:** Yuvraj Singh Chowdhary
- **LinkedIn:** [Yuvraj Singh Chowdhary](https://www.linkedin.com/in/yuvraj-singh-chowdhary/)
- **GitHub:** [chowdhary19](https://github.com/chowdhary19)

---

