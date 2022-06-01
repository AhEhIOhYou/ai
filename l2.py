import numpy as np

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from collections import defaultdict
from sklearn.naive_bayes import GaussianNB

class NaiveBayes(object):

    def __init__(self):
        self.class_frequency = defaultdict(lambda: 0)
        self.parameter_frequency = defaultdict(lambda: 0)

    def predict_class(self, X):
        p = []
        for item in self.class_frequency.keys():
            l = item
            x = X
            res = self.class_freq(x, l)
            p.append((res, l))
        # return max(p, key=lambda i: i[0])[1]
        return min(p, key=lambda i: i[0])[1]

    def class_freq(self, X, clss):
        # Произведение веротяностей или Сумма логарифмов

        frequency = - np.log(self.class_frequency[clss])
        for parameter in X:
            a = 10 ** (-7)
            frequency += - np.log(self.parameter_frequency.get((parameter, clss), a))

        # frequency = self.class_frequency[clss]
        # for parameter in X:
        #     a = 1
        #     frequency *= self.parameter_frequency.get((parameter, clss), a)

        return frequency

    def fit(self, X, y):
        # Вычисление частот повторений классов и характеристик у каждого класса
        for parameters, class_label in zip(X, y):
            self.class_frequency[class_label] += 1
            for parameter in parameters:
                self.parameter_frequency[(parameter, class_label)] += 1

        # Нормализация к диапазону [0,1]
        for item in self.class_frequency:
            self.class_frequency[item] /= len(X)

        # Подсчет веротяностей p(x|l) для каждого параметра в его классе
        for parameter, class_label in self.parameter_frequency:
            x = self.parameter_frequency[(parameter, class_label)]
            l = self.class_frequency[class_label]
            a = x / l
            self.parameter_frequency[(parameter, class_label)] = a

        return self


# Загрузка ирисок
# [5.1 3.5 1.4 0.2], [..] - 4 характерстики
# [0 0 0 .. 1 1 1 .. 2 2 2 ..] - 3 типа ирисок
iris_data = load_iris()
X = iris_data.data
y = iris_data.target

# Разбивка на тренировку/тест
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Обучение модели
model = NaiveBayes().fit(X_train, y_train)

# Получение предсказаний для тестовой выборки
predictions = list()
for x in X_test:
    predictions.append(model.predict_class(x))


m = GaussianNB()
m.fit(X_train, y_train)
print('Точность по sklearn: %s%%' % (m.score(X_test, y_test) * 100))

print('Точность: %s%%' % (accuracy_score(predictions, y_test) * 100))