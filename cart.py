# 使用SVM进行MNIST手写数字分类
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score


# 加载数据
digits = load_digits()
data = digits.data
# 数据探索
print(data.shape)
# 查看第一幅图像
print(digits.images[8])
# 第一幅图像代表的数字含义
print(digits.target[8])
# 将第一幅图像显示出来
plt.gray()
plt.imshow(digits.images[8])
plt.show()

# 分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25, random_state=33)


# 采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)


clf = tree.DecisionTreeClassifier()
clf.fit(train_ss_x, train_y)
prediction = clf.predict(test_ss_x)
print('准确率: %0.4lf' % accuracy_score(prediction, test_y))

