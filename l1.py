import numpy as np
import pandas as pd
import sklearn.linear_model
import sklearn.neighbors

# Подготовка данных
path = "dataset/"
oecd_bli = pd.read_csv(path + "oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv(path + "gdp_per_capita.csv",
                             thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

# Подготовка массивов с данными
def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"] == "TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita, left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

# Метод ближайших соседей
# X, y - обучающие примеры, X_new - ВВП, к - количество обучающих примеров (соседей)
def my_KNeighborsRegressor(X, Y, x_new, k):

    arr_length = np.zeros((len(X), 2), dtype=int)

    # Формирование массива с растояниями между X_new и набором
    for i in range(len(X)):
        arr_length[i][0] = abs(x_new - X[i][0])
        # соотношение по y
        arr_length[i][1] = i

    # Сортировка по первому столбцу
    arr_length = sorted(arr_length, key=lambda x: x[0])
    res_arr = np.zeros(k)

    # Выборка k штук
    for i in range(k):
        res_arr[i] = Y[(arr_length[i][1])]

    return np.mean(res_arr)

# Метод простой линеной регрессии
# x, y - обучающие примеры, X_new - ВВП,
def my_linear_regression(x, y, x_new):
    # Минимизация
    n = np.size(x)
    # Подсчет среднего
    average_x = np.mean(X)
    average_y = np.mean(y)

    # Подсчет отклонение по x и y
    ss_xx = np.sum(x * x) - n * average_x * average_x
    ss_xy = np.sum(x * y) - n * average_x * average_y

    # Подсчет коэффицентов
    b_1 = ss_xy / ss_xx
    b_0 = average_y - b_1 * average_x

    return b_0 + b_1 * x_new

# Подготовить данные
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
# X - ВВП, y - уровень довольствия
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

# Выработать прогноз для Кипра, его ВВП -
X_new = [[22587]]

model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
model.fit(X, y)
print("Метод соседей")
print("Уровень довольствия по sklearn:", model.predict(X_new)[0][0])
print("По моей реализации:", my_KNeighborsRegressor(X, y, X_new[0][0], 5))

print("Метод простой линейной регрессии")
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)
print("Уровень довольствия по sklearn:", model.predict(X_new)[0][0])
print("По моей реализации:", my_linear_regression(X, y, X_new[0][0]))