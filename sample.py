from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.metrics import accuracy_score

digits = datasets.load_digits()
x = digits.images
y = digits.target
#二次元配列を一次元配列に変換
x = x.reshape((-1, 64))

# データを学習用とテスト用に分割する
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# データを学習
clf = svm.LinearSVC()
clf.fit(x_train, y_train)

# 予測して精度を確認する
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
