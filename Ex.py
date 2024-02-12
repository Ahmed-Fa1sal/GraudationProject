import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

data_dict = pickle.load(open('./dataOfTen.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Load the validation dataset from the provided file
validation_data_dict = pickle.load(open('./vald.pickle', 'rb'))

x_val = np.asarray(validation_data_dict['data'])
y_val = np.asarray(validation_data_dict['labels'])

# Train the model using the training set
model = RandomForestClassifier()
model.fit(data, labels)

# Make predictions on the validation set
y_val_predict = model.predict(x_val)

# Calculate validation metrics
accuracy_val = accuracy_score(y_val, y_val_predict)
precision_val = precision_score(y_val, y_val_predict, average='weighted')
recall_val = recall_score(y_val, y_val_predict, average='weighted')
f1_val = f1_score(y_val, y_val_predict, average='weighted')

# Print validation metrics
print('Validation Accuracy: {:.2f}%'.format(accuracy_val * 100))
print('Validation Precision: {:.2f}'.format(precision_val))
print('Validation Recall: {:.2f}'.format(recall_val))
print('Validation F1 Score: {:.2f}'.format(f1_val))

# Calculate confusion matrix for validation data
conf_matrix_val = confusion_matrix(y_val, y_val_predict)

# Save validation metrics in separate text files
with open('validation_accuracy.txt', 'w') as f:
    f.write(f'Validation Accuracy: {accuracy_val:.2f}%\n')

with open('validation_precision.txt', 'w') as f:
    f.write(f'Validation Precision: {precision_val:.2f}\n')

with open('validation_recall.txt', 'w') as f:
    f.write(f'Validation Recall: {recall_val:.2f}\n')

with open('validation_f1_score.txt', 'w') as f:
    f.write(f'Validation F1 Score: {f1_val:.2f}\n')

# Visualizing the confusion matrix for validation data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Validation)')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix for validation data as a PDF file
plt.savefig('confusion_matrix_validation.pdf')

# Save accuracy, recall, f1-score, and precision as single dots in separate PDF files
plt.figure(figsize=(4, 4))
plt.scatter(1, accuracy_val, color='blue')
plt.title('Validation Accuracy')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.xticks([])
plt.savefig('validation_accuracy.pdf')

plt.figure(figsize=(4, 4))
plt.scatter(1, precision_val, color='green')
plt.title('Validation Precision')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.xticks([])
plt.savefig('validation_precision.pdf')

plt.figure(figsize=(4, 4))
plt.scatter(1, recall_val, color='red')
plt.title('Validation Recall')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.xticks([])
plt.savefig('validation_recall.pdf')

plt.figure(figsize=(4, 4))
plt.scatter(1, f1_val, color='orange')
plt.title('Validation F1-Score')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xlim(0, 2)
plt.xticks([])
plt.savefig('validation_f1_score.pdf')

# Show the plots
plt.show()
