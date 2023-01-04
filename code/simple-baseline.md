To run the code, first you need to save the train/dev/test data in google drive under the path "/content/drive/MyDrive/".
For example, the path for train data is: "/content/drive/MyDrive/df_train.csv".

The link for the three .cvs files are:
train: https://drive.google.com/file/d/1C9A6Z5ZTe0IrqRoyNO_4yJtqxU4F-zbU/view?usp=sharing
dev: https://drive.google.com/file/d/1-9tBb8nelFohHN_DSyTyBPKyC0Tn-YeQ/view?usp=sharing
test: https://drive.google.com/file/d/1-5hSyZC9r8lyfHT5CXQJ5glza5eOyX1y/view?usp=sharing

If the file is running in colab or Jupyter Notebook, the file can be run using "Run ALL".

We use bag-of-words as features and implemented a logistic regression as the simple baseline model. On the development data set, we tuned the hyper parameter C, which represents the inverse of regularization strength. We tried values ranging from 0.1 to 3.0, and C = 2.5 gives the best prediction on development data. The best model has an accuracy of 71.07% on training data and 64.20% on development data.

Samples on development data:
- "two English women" has a sentiment label of 2, and the prediction is also 2.
- "becomes claustrophobic" has a sentiment label of 1, and the prediction is also 1.
- "the script , which has a handful of smart jokes" has a sentiment label of 3, and the prediction is also 3.
- "a little too familiar" has a sentiment label of 1, and the prediction gives 2.

