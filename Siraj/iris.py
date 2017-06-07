import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()

dataset = iris.data
target = iris.target

names = { 
    '0': iris.feature_names[0],
    '1': iris.feature_names[1],
    '2': iris.feature_names[2],
    '3': iris.feature_names[3]}

classifier = skflow.LinearClassifier(n_classes = 3, feature_columns = names)
                                     
#feature_columns=skflow.infer_real_valued_columns_from_input(iris.data))

classifier.fit(dataset, target)

score = metrics.accuracy_score(target, classifier.predict(dataset))

print('Accuracy : ' + score)