import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data_dict = pickle.load(open('./dataOfTen.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the dataset for train and test and shuffle the images (20% of data for test)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

accuracy = accuracy_score(y_test, y_predict)
precision = precision_score(y_test, y_predict, average='weighted')
recall = recall_score(y_test, y_predict, average='weighted')
f1 = f1_score(y_test, y_predict, average='weighted')
conf_matrix = confusion_matrix(y_test, y_predict)

print('Accuracy: {:.2f}%'.format(accuracy * 100))
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
print('F1 Score: {:.2f}'.format(f1))
print('Confusion Matrix:')
print(conf_matrix)

# Save the model to a file
with open('modelOfTen.p', 'wb') as f:
    pickle.dump({'model': model}, f)

# Actual accuracy values from your experiments
accuracies = [0.85, 0.87, 0.90, 0.92]  # Replace with your actual accuracy values

# Plotting the accuracy figure
plt.plot(accuracies, marker='o', linestyle='-', label='Accuracy')
plt.title('Performance Metrics')
plt.xlabel('Experiment/Iteration')
plt.ylabel('Score')
plt.grid(True)

# Plotting the precision figure
plt.scatter([1], [precision], color='orange', label='Precision')
# Plotting the recall figure
plt.scatter([1], [recall], color='green', label='Recall')
# Plotting the F1-score figure
plt.scatter([1], [f1], color='blue', label='F1 Score')

# Save the figure as a PDF file
plt.legend()
plt.savefig('performance_metrics.pdf')

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')

# Save the confusion matrix as a PDF file
plt.savefig('confusion_matrix.pdf')

# Show the plots
plt.show()
