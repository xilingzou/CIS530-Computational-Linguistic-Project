To run the code, first you need to save the train/dev/test data in google drive under the path "/content/drive/MyDrive/".
For example, the path for train data is: "/content/drive/MyDrive/df_train.csv".

The link for the three .cvs files are:
train: https://drive.google.com/file/d/1C9A6Z5ZTe0IrqRoyNO_4yJtqxU4F-zbU/view?usp=sharing
dev: https://drive.google.com/file/d/1-9tBb8nelFohHN_DSyTyBPKyC0Tn-YeQ/view?usp=sharing
test: https://drive.google.com/file/d/1-5hSyZC9r8lyfHT5CXQJ5glza5eOyX1y/view?usp=sharing

If the file is running in colab or Jupyter Notebook, the file can be run using "Run ALL".
You can modify the parameters in the main function to try different add_on or whether to use weighted Loss function or unweighted loss function.

max_epoch = 31
train_loss_ = []
test_loss_ = []
model = CustomBERTModel()
model.cuda()

pred_list = []
preds=[]
for epoch in range(max_epoch):
    train_ = train_model(model, epoch, add_on="lstm", weighted = False) # modify this
    train_loss_.append(train_)
    preds, test_ = valid_model(model, add_on="lstm", weighted = False) # modify this
    pred_list.append(preds)
    print("Epoch: {}, Training Loss: {}, Validation Loss: {}".format(epoch, train_, test_))

v=[]
idx=[]
for i in preds:
    for j in i:
        y=j.to('cpu').detach().numpy().copy()
        v.append(y)
        idx.append(np.argmax(y))

cf_matrix = confusion_matrix(test.Sentiment.values, idx)
print("Confusion Matrix\n")
sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual',fontsize=12)
plt.show()

lr_report = classification_report(test.Sentiment.values, idx)

print("\nClassification Metrics\n")
print(lr_report)

