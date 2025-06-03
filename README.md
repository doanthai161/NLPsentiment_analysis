
## INTRODUTION
This project implements a sentiment analysis model using various machine learning algorithms. The model is trained on a labeled dataset of text samples, where the goal is to classify the sentiment expressed in the text as positive, negative, or neutral. 

## Requirements
To run this code, you need the following Python libraries:

numpy \
pandas \
matplotlib \
nltk \
scikit-learn \
lightgbm
## Installation
You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib nltk scikit-learn lightgbm
```


## Data

The code expects two CSV files:

* Train.csv: The training dataset, containing text samples and their corresponding sentiment labels. 
* Test.csv: The testing dataset, containing text samples to evaluate the model. 

Make sure the CSV files are structured with at least the following columns:

* text: The text to be analyzed. 
* label: The sentiment label associated with each text sample.


## Usage

1. Load the data: The script loads the training and testing datasets from specified CSV files.
2. Data Cleaning: It preprocesses the text data, which includes:
* Converting text to lowercase
* Removing Twitter handles and URLs
* Eliminating punctuation and stopwords
* Tokenizing the text
3. Lemmatization: The text is lemmatized to reduce words to their base or root form.
4. Feature Extraction: The TF-IDF vectorizer is used to convert text data into numerical format.
5. Dimensionality Reduction: Truncated SVD is applied to reduce the feature space.
Run the script in your Python environment to see the performance of different classifiers.


## Functions
#cleaning(text)
* Cleans the input text by converting it to lowercase, removing unnecessary characters, and eliminating stopwords. 

#lemm(data)
* Lemmatizes the words in the input text data, simplifying the text for better analysis.
This command processes the input_video.mp4, runs the detection using the specified configuration and checkpoint, and outputs the results to output_video.mp4, while also displaying the video in a window.
## Modeling
The following classifiers are used to train and evaluate the model:

1. Logistic Regression
2. Gaussian Naive Bayes
3. Support Vector Machine (SVM)
4. LightGBM
5. Random Forest Classifier

## Evaluation
The performance of each model is evaluated using accuracy and confusion matrix. After training, the accuracy of each model is printed:
```bash
print('logitics ', accuracy_score(y_lr_pred, y_test))
print('gaussion ', accuracy_score(y_test, y_gs_pred))
print('svm ', accuracy_score(y_svm_pred, y_test))
print(accuracy_score(preds, y_test))
print('forest', accuracy_score(y_test, y_rdfr_pred))
print('forest', confusion_matrix(y_test, y_rdfr_pred))
```
