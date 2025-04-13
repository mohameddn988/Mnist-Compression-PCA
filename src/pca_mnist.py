import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Flatten the images: 28x28 to 784
x_train_flat = x_train.reshape(x_train.shape[0], -1) / 255.0
x_test_flat = x_test.reshape(x_test.shape[0], -1) / 255.0

print("\n--- Accuracy vs PCA Compression ---")
for n in [16, 32, 64, 128, 256]:
    pca = PCA(n_components=n)
    x_train_pca = pca.fit_transform(x_train_flat)
    x_test_pca = pca.transform(x_test_flat)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_pca, y_train)
    acc = clf.score(x_test_pca, y_test)

    print(f"PCA({n}) → Accuracy: {acc:.4f}")


# Visualize some digit images
def showDigits(images, title, n=10):
    plt.figure(figsize=(n, 1.5))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()

# Apply PCA
nComponents = 64
pca = PCA(n_components=nComponents)
x_train_pca = pca.fit_transform(x_train_flat)
x_test_pca = pca.transform(x_test_flat)

print(f"Original shape: {x_train_flat.shape}")
print(f"Compressed shape: {x_train_pca.shape}")

# Reconstruct the images
x_train_reconstructed = pca.inverse_transform(x_train_pca)
x_test_reconstructed = pca.inverse_transform(x_test_pca)

# Reshape the reconstructed images to 28x28
x_train_reconstructed_images = x_train_reconstructed.reshape((-1, 28, 28))
x_test_reconstructed_images = x_test_reconstructed.reshape((-1, 28, 28))

# Show original and reconstructed images side by side
random_indices = random.sample(range(len(x_test)), 10)

plt.figure(figsize=(10, 4))
for i, idx in enumerate(random_indices):
    # Original images on the top row
    ax1 = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[idx], cmap='gray')
    ax1.axis('off')

    # Reconstructed images on the bottom row
    ax2 = plt.subplot(2, 10, i + 11)
    plt.imshow(x_test_reconstructed_images[idx], cmap='gray')
    ax2.axis('off')

plt.suptitle(f"Original and Reconstructed Images from {nComponents} PCA Components", fontsize=16)
plt.tight_layout()
plt.show()

# Train a classifier on PCA components
clf_original = LogisticRegression(max_iter=3000)
clf_original.fit(x_train_flat, y_train)
y_pred_original = clf_original.predict(x_test_flat)

clf_pca = LogisticRegression(max_iter=3000)
clf_pca.fit(x_train_pca, y_train)
y_pred_pca = clf_pca.predict(x_test_pca)

# Evaluate the classifiers
accuracy_original = accuracy_score(y_test, y_pred_original)
accuracy_pca = accuracy_score(y_test, y_pred_pca)

explained = pca.explained_variance_ratio_
print(f"Explained variance (total): {np.sum(explained):.4f}")


print(f"Accuracy with original features: {accuracy_original:.4f}")
print(f"Accuracy on PCA-compressed data ({nComponents} components): {accuracy_pca:.4f}")
