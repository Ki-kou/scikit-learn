from sklearn import datasets
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

digits = datasets.load_digits()
num = len(digits.data)
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:10]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)
plt.show()