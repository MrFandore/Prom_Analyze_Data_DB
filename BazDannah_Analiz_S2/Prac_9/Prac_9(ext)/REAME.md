C:\PyProjects\PythonProject\BazDannah_Analiz_S2\.venv\Scripts\python.exe "C:\PyProjects\PythonProject\BazDannah_Analiz_S2\Prac_9\Prac_9(ext)\main.py" 
Исходное количество строк: 119734
Пропуски в size: 0
После очистки: 88342
Распределение классов после SMOTE:
size_encoded
1.0    23769
0.0    23769
3.0    23769
2.0    23769
Name: count, dtype: int64
Fitting 3 folds for each of 54 candidates, totalling 162 fits
Лучшие параметры XGBoost: {'learning_rate': 0.2, 'max_depth': 7, 'n_estimators': 300, 'subsample': 1.0}

XGBoost (optimized):
Accuracy: 0.5311
              precision    recall  f1-score   support

           S       0.62      0.73      0.67      4385
           M       0.53      0.43      0.47      5943
           L       0.35      0.40      0.37      3517
          XL       0.60      0.59      0.59      3824

    accuracy                           0.53     17669
   macro avg       0.53      0.54      0.53     17669
weighted avg       0.53      0.53      0.53     17669


Random Forest:
Accuracy: 0.5284
              precision    recall  f1-score   support

           S       0.61      0.72      0.66      4385
           M       0.53      0.41      0.46      5943
           L       0.35      0.41      0.38      3517
          XL       0.60      0.60      0.60      3824

    accuracy                           0.53     17669
   macro avg       0.52      0.53      0.53     17669
weighted avg       0.53      0.53      0.53     17669


Gradient Boosting:
Accuracy: 0.5316
              precision    recall  f1-score   support

           S       0.62      0.73      0.67      4385
           M       0.53      0.42      0.47      5943
           L       0.35      0.41      0.38      3517
          XL       0.60      0.59      0.60      3824

    accuracy                           0.53     17669
   macro avg       0.53      0.54      0.53     17669
weighted avg       0.53      0.53      0.53     17669


Stacking Classifier Accuracy: 0.5307
              precision    recall  f1-score   support

           S       0.62      0.73      0.67      4385
           M       0.54      0.40      0.46      5943
           L       0.35      0.44      0.39      3517
          XL       0.61      0.58      0.59      3824

    accuracy                           0.53     17669
   macro avg       0.53      0.54      0.53     17669
weighted avg       0.54      0.53      0.53     17669


Достигнутая точность: 0.5311
Цель не достигнута. Возможные улучшения:
- Добавить больше признаков (например, произведение возраста и роста)
- Использовать LightGBM или CatBoost
- Применить нейросетевой подход (MLPClassifier)

Process finished with exit code 0
