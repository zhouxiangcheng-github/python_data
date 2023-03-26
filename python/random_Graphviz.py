from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import export_graphviz
import pydotplus
import os
import  numpy as np



iris = datasets.load_iris()

x = iris.data
y = iris.target
print(iris.feature_names,"y:",y)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.9)
clf = RandomForestClassifier(n_estimators=2, bootstrap = True,max_features = 'sqrt')
clf.fit(x_train,y_train)
n = 0
y = 0
for i in range(len(y_test)):
    y_predict = -1
    y_predict = clf.predict(x_test[i].reshape(1, -1))
    y_predict_prob = np.max(clf.predict_proba(x_test[i].reshape(1, -1)))
    #print("真实结果：", y_test)
    #time.sleep(0.5)
    if(y_predict == y_test[i]):
        y += 1
        print("应为:",y_test[i],"预测为：",y_predict,"",y_predict_prob)
        #print("y_predict:",np.max(clf.predict_proba(x_test[i].reshape(1, -1))))
    else:
        n += 1
        print("应为：", y_test[i],"预测为：",y_predict)
print("测试样本个数:", len(y_test))
print("预测准确个数：", y)
print("预测错误个数：", n)


for idx, estimator in enumerate(clf.estimators_):
    # 导出dot文件
    export_graphviz(estimator,
                    out_file='tree{}.dot'.format(idx),
                    feature_names=iris.feature_names,
                    class_names=iris.target_names,
                    rounded=True,
                    proportion=False,
                    precision=2,
                    filled=True)
    # 转换为png文件
    os.system('dot -Tpng tree{}.dot -o tree{}.png'.format(idx, idx))


# data对应了样本的4个特征，150行4列
print(iris.data.shape)


# 显示样本特征的前5行
print(iris.data[:5])


# target对应了样本的类别（目标属性），150行1列
print(iris.target.shape)

# 显示所有样本的目标属性
print(iris.target)










