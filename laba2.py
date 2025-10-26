import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report
import warnings

# Отключение предупреждений для более чистого вывода
warnings.filterwarnings('ignore')

# --- 1. Загрузка данных и первичный анализ ---
print("--- 1. Загрузка данных и первичный анализ ---")
try:
    df = pd.read_csv('train.csv') # Предполагается, что файл называется train.csv
except FileNotFoundError:
    print("Ошибка: Файл 'train.csv' не найден. Убедитесь, что файл находится в той же директории, что и скрипт.")
    exit()

print("Первые 5 строк датасесета:")
print(df.head())
print("\nИнформация о датасете:")
df.info()
print("\nКоличество пропущенных значений:")
print(df.isnull().sum())
print("\nСтатистика по числовым признакам:")
print(df.describe())

# --- 2. Предварительная обработка данных (Preprocessing) ---
print("\n--- 2. Предварительная обработка данных ---")

# Копируем датасет для обработки
df_processed = df.copy()

# Обработка признака 'Cabin'
# Разбиваем Cabin на Deck, Num, Side
# Важно: если Cabin NaN, то str.split() вернет NaN для всех трех новых столбцов.
df_processed[['Deck', 'CabinNum', 'Side']] = df_processed['Cabin'].str.split('/', expand=True)
df_processed.drop('Cabin', axis=1, inplace=True)


for col in ['HomePlanet', 'Destination', 'Deck', 'Side', 'CryoSleep', 'VIP']:
    if col in df_processed.columns: # Убедимся, что столбец существует
        # Для булевых (True/False/NaN) и категориальных (строк) мода - хороший выбор.
        # mode()[0] берет первое (и обычно единственное) модальное значение.
        if df_processed[col].isnull().any(): # Заполняем только если есть NaN
            mode_val = df_processed[col].mode()[0]
            df_processed[col].fillna(mode_val, inplace=True)

# Теперь преобразуем булевы признаки в 0/1, так как NaN уже заполнены
# 'Transported' - целевая переменная, также преобразуем ее
df_processed['Transported'] = df_processed['Transported'].astype(int)
df_processed['CryoSleep'] = df_processed['CryoSleep'].astype(int)
df_processed['VIP'] = df_processed['VIP'].astype(int)

# Обработка пропущенных значений для числовых признаков (стратегия: медиана)
for col in ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age', 'CabinNum']:
    if col in df_processed.columns: # Проверяем, что столбец существует
        # CabinNum после str.split может быть 'object' и содержать NaN.
        # Сначала преобразуем его в числовой тип (float), при этом нечисловые значения (если такие есть, кроме NaN) станут NaN.
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        # Затем заполняем NaN медианой, только если они есть
        if df_processed[col].isnull().any():
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)


print("\nКоличество пропущенных значений после обработки:")
print(df_processed.isnull().sum())

# Определение числовых и категориальных признаков для ColumnTransformer
# После обработки CabinNum стал числовым, CryoSleep и VIP стали int (числовыми)
# Убедитесь, что все столбцы, которые здесь указаны, существуют в df_processed
numerical_features = [
    'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck',
    'CabinNum', 'CryoSleep', 'VIP'
]
categorical_features = ['HomePlanet', 'Destination', 'Deck', 'Side']

# Удаляем признаки, которые не используются для обучения
# Исключаем 'Name' и 'PassengerId' так как они не являются признаками для модели
df_processed.drop(['Name', 'PassengerId'], axis=1, inplace=True, errors='ignore') # errors='ignore' если их уже нет

# Очищаем списки признаков, чтобы они содержали только существующие столбцы
numerical_features = [f for f in numerical_features if f in df_processed.columns]
categorical_features = [f for f in categorical_features if f in df_processed.columns]


# Создание ColumnTransformer для автоматической предобработки
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough' # Оставить остальные столбцы как есть (их нет после очистки)
)

# --- 3. Задание 1: Разделение датасета на обучающую и тестовую выборки ---
print("\n--- 3. Задание 1: Разделение датасета ---")

# Для классификации
X_clf = df_processed.drop('Transported', axis=1)
y_clf = df_processed['Transported']

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
)
print(f"Размер обучающей выборки для классификации: {X_clf_train.shape}")
print(f"Размер тестовой выборки для классификации: {X_clf_test.shape}")

# Для регрессии (будем предсказывать RoomService)
# Исключаем RoomService из признаков X для регрессии
X_reg = df_processed.drop(['RoomService'], axis=1) # RoomService - целевая переменная для регрессии
y_reg = df_processed['RoomService']

# Удаляем 'Transported' из X_reg, так как это целевая переменная классификации
if 'Transported' in X_reg.columns:
    X_reg = X_reg.drop('Transported', axis=1)


X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
print(f"Размер обучающей выборки для регрессии: {X_reg_train.shape}")
print(f"Размер тестовой выборки для регрессии: {X_reg_test.shape}")

# --- 4. Задание 2 и 3: Задача регрессии (предсказание RoomService) ---
print("\n--- 4. Задание 2 и 3: Задача регрессии (RoomService) ---")

# Создаем Pipeline для регрессии
# Обновляем списки признаков для preprocessor_reg на основе X_reg
numerical_features_reg = [f for f in numerical_features if f != 'RoomService' and f in X_reg.columns]
categorical_features_reg = [f for f in categorical_features if f in X_reg.columns]

preprocessor_reg = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features_reg),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features_reg)
    ],
    remainder='passthrough'
)

# Линейная регрессия
reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg),
                               ('regressor', LinearRegression())])
reg_pipeline.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_pipeline.predict(X_reg_test)

print("\nОценка простой Линейной регрессии:")
print(f"MAE: {mean_absolute_error(y_reg_test, y_reg_pred):.2f}")
print(f"MSE: {mean_squared_error(y_reg_test, y_reg_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)):.2f}")
print(f"R2: {r2_score(y_reg_test, y_reg_pred):.2f}")

# --- Улучшение регрессионной модели: Полиномиальная регрессия с регуляризацией Ridge ---
#print("\n--- Улучшение регрессионной модели: Полиномиальная регрессия с Ridge ---")

# Добавим PolynomialFeatures в пайплайн
poly_reg_pipeline = Pipeline(steps=[('preprocessor', preprocessor_reg),
                                    ('poly', PolynomialFeatures(degree=2, include_bias=False)), # степень 2
                                    ('scaler', StandardScaler(with_mean=False)), # Масштабируем после полиномиальных признаков
                                    ('regressor', Ridge())]) # Используем Ridge для регуляризации

# Подбор гиперпараметров для Ridge (alpha) и PolynomialFeatures (degree)
# Из-за ColumnTransformer и Pipeline, имена параметров будут выглядеть как 'step__parameter'
param_grid_reg = {
    'poly__degree': [1, 2, 3], # 1 - это линейная регрессия
    'regressor__alpha': [0.1, 1.0, 10.0]
}

grid_search_reg = GridSearchCV(poly_reg_pipeline, param_grid_reg, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search_reg.fit(X_reg_train, y_reg_train)

best_reg_model = grid_search_reg.best_estimator_
y_reg_pred_tuned = best_reg_model.predict(X_reg_test)


# Визуализация результатов регрессии (для одного признака, если возможно)
# Для многомерных данных сложно визуализировать
plt.figure(figsize=(10, 6))
plt.scatter(y_reg_test, y_reg_pred_tuned, alpha=0.3)
plt.plot([y_reg_test.min(), y_reg_test.max()], [y_reg_test.min(), y_reg_test.max()], 'r--')
plt.xlabel("Истинные значения RoomService")
plt.ylabel("Предсказанные значения RoomService")
plt.title("Истинные vs Предсказанные значения RoomService (Улучшенная Регрессия)")
plt.show()


# --- 5. Задание 4 и 5: Задача классификации (предсказание Transported) ---
print("\n--- 5. Задание 4 и 5: Задача классификации (Transported) ---")

# Создаем Pipeline для классификации
# Используем ранее определенный `preprocessor` для классификации
clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(random_state=42))])

clf_pipeline.fit(X_clf_train, y_clf_train)
y_clf_pred = clf_pipeline.predict(X_clf_test)

print("\nОценка :")
print(f"Accuracy: {accuracy_score(y_clf_test, y_clf_pred):.2f}")
print("\nМатрица ошибок:")
cm = confusion_matrix(y_clf_test, y_clf_pred)
print(cm)
print("\nОтчет по классификации:")
print(classification_report(y_clf_test, y_clf_pred))

# Визуализация матрицы ошибок
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Не Transported (0)', 'Transported (1)'],
            yticklabels=['Не Transported (0)', 'Transported (1)'])
plt.title('Матрица ошибок (Логистическая регрессия)')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.show()


# --- Улучшение классификационной модели: Логистическая регрессия с подбором гиперпараметров ---
#print("\n--- Улучшение классификационной модели: Логистическая регрессия с подбором гиперпараметров ---")

# Подбор гиперпараметров для LogisticRegression (C, penalty, solver)
param_grid_clf = {
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l1', 'l2'], # 'l1' требует solver='liblinear' или 'saga'
    'classifier__solver': ['liblinear', 'saga'],
    'classifier__class_weight': [None, 'balanced']
}

# Используем тот же ColumnTransformer (preprocessor) для классификации
tuned_clf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(random_state=42, max_iter=2000))]) # Увеличено max_iter для надежности

grid_search_clf = GridSearchCV(tuned_clf_pipeline, param_grid_clf, cv=5, scoring='f1', n_jobs=-1) # F1-score для классификации
grid_search_clf.fit(X_clf_train, y_clf_train)

best_clf_model = grid_search_clf.best_estimator_
y_clf_pred_tuned = best_clf_model.predict(X_clf_test)


