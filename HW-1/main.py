import numpy as np
import matplotlib.pyplot as plt
import keras
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

plt.imshow(train_images[0], cmap='gray')
plt.title(f"Label: {train_labels[0]}")
plt.show()

# decrease value to number between 0-1 for increase performance
train_images_flat = train_images.reshape(train_images.shape[0], 28 * 28) / 255.0
test_images_flat = test_images.reshape(test_images.shape[0], 28 * 28) / 255.0
 

# split train,validation
train_images_new, val_images, train_labels_new, val_labels = train_test_split(
    train_images_flat, train_labels, test_size=0.1, random_state=42
)

# بخش ث
# پیاده سازی کلاس
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # برحسب فاصله اقلیدسی
    def predict_euclidean(self, X, batch_size=100):
        n_samples = X.shape[0]
        predicted_labels = []

        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]

            # Efficient Euclidean distance calculation using broadcasting
            distances = np.sum((batch[:, np.newaxis] - self.X_train) ** 2, axis=2)
            
            # Get indices of the k nearest neighbors
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_indices]

            # Determine the most common label for each sample in the batch
            batch_predictions = [Counter(row).most_common(1)[0][0] for row in k_nearest_labels]
            predicted_labels.extend(batch_predictions)

        return np.array(predicted_labels)
    

    # برحسب فاصله منهتنی
    def predict_manhatan(self, X, batch_size=100):
        n_samples = X.shape[0]
        predicted_labels = []

        for i in range(0, n_samples, batch_size):
            batch = X[i:i + batch_size]

            # Efficient Manhattan distance calculation using broadcasting
            distances = np.sum(np.abs(batch[:, np.newaxis] - self.X_train), axis=2)
            
            # Get indices of the k nearest neighbors
            k_indices = np.argpartition(distances, self.k, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_indices]

            # Determine the most common label for each sample in the batch
            batch_predictions = [Counter(row).most_common(1)[0][0] for row in k_nearest_labels]
            predicted_labels.extend(batch_predictions)

        return np.array(predicted_labels)

# بخش ج
# پیاده سازی و تعریف تابع میزان دقت
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

k_list=[3,5,7,9,11,15]
best_accuracy=0
best_k=0

for k in k_list:
    model = KNN(k=k)
    model.fit(train_images_new, train_labels_new)

    predictions = model.predict_euclidean(val_images, batch_size=100)

    acc = accuracy(val_labels, predictions)

    if acc>best_accuracy:
        best_accuracy=acc
        best_k=k

print(f"Validation Accuracy: {best_accuracy:.2f}\nbest K:{best_k}")


# بخش چ
# پیاده سازی ترین روی کل تست و ولیدیشن و در نهایت انجام تست
# az behtarin K ke entekhab kardim estefade mikonim(فینگلیش نوشتم چون ترتیب به هم میریخت)
model = KNN(k=best_k) 

# از ابتدا داده های تست و ولیدیشن را کامل داشتیم و نیازی نبود ترکیب کنیم
model.fit(train_images, train_labels)

# test ba estefade az fasele oghlidosi
predict_test_euclidean=model.predict_euclidean(test_images,batch_size=100)
train_euclidean_accuracy=accuracy(test_labels,predict_test_euclidean)
print(f"train_euclidean_accuracy : {train_euclidean_accuracy}")

# test ba estefade az faseleh manhatani
predict_test_manhatan=model.predict_manhatan(test_images,batch_size=100)
train_manhatan_accuracy=accuracy(test_labels,predict_test_euclidean)
print(f"train_manhatan_accuracy : {train_manhatan_accuracy}")


# بخش ح
knn = KNeighborsClassifier(n_neighbors=best_k)

# Fit the model
knn.fit(train_images, train_labels)

# Predict on the test set
y_pred = knn.predict(test_images)

# Calculate accuracy
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy:.2f}")



