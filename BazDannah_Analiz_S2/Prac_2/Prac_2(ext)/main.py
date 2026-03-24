import pandas as pd

# Проверка содержимого файла
try:
    with open('movies_metadata.csv', 'r', encoding='utf-8-sig') as f:
        print("Первая строка:", repr(f.readline()))
        print("Вторая строка:", repr(f.readline()))
        print("Третья строка:", repr(f.readline()))
except FileNotFoundError:
    print("Файл не найден")