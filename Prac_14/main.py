import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import nltk

# Автоматическая загрузка необходимых ресурсов NLTK
def download_nltk_resources():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            if res == 'stopwords':
                nltk.data.find(f'corpora/{res}')
            else:
                nltk.data.find(f'tokenizers/{res}')
        except LookupError:
            print(f"Скачивание ресурса NLTK: {res}")
            nltk.download(res, quiet=True)


download_nltk_resources()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

# Загрузка стоп-слов
stop_words_ru = set(stopwords.words('russian'))
# Добавим свои стоп-слова
extra_stop_words = {
    'работа', 'вакансия', 'компания', 'должность', 'сотрудник', 'кандидат',
    'обязанности', 'требования', 'условия', 'навыки', 'опыт', 'знание', 'умение',
    'работать', 'быть', 'мочь', 'делать', 'также', 'более', 'менее', 'очень',
    'например', 'который', 'этот', 'свой', 'ваш', 'наш', 'год', 'человек',
    'можно', 'нужно', 'являться', 'становиться', 'иметь', 'находиться'
}
stop_words_ru.update(extra_stop_words)
stop_words_en = set(stopwords.words('english'))
STOP_WORDS = stop_words_ru.union(stop_words_en)

morph = pymorphy3.MorphAnalyzer()

# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
print("=" * 60)
print("ЗАДАНИЕ 1: Анализ описаний вакансий")
print("=" * 60)

file_path = 'jobs.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Загружен файл {file_path}, размер: {df.shape}")
except FileNotFoundError:
    print(f"Файл {file_path} не найден. Создаю синтетические данные.")
    data = {
        'job_title': ['Аналитик', 'Аналитик', 'Программист', 'Программист', 'Data Engineer', 'Data Engineer'],
        'description': [
            'Проведение анализа данных. Построение дашбордов. Работа с SQL. Знание Python. Визуализация.',
            'Сбор требований, анализ бизнес-процессов, создание отчетов в Tableau.',
            'Разработка на Python, знание Django, работа с базами данных, написание тестов.',
            'Java, Spring, микросервисы, Docker, Kafka.',
            'Создание ETL-пайплайнов, работа с Hadoop, Spark, Airflow, оптимизация запросов.',
            'Data warehousing, моделирование данных, работа с ClickHouse, dbt.'
        ]
    }
    df = pd.DataFrame(data)
    print("Созданы синтетические данные. Для реального анализа загрузите jobs.csv.")

print("\nСтолбцы датасета:", df.columns.tolist())
# Определим столбцы с названием профессии и описанием
title_col = None
desc_col = None
for col in df.columns:
    if 'title' in col.lower() or 'job' in col.lower() or 'position' in col.lower():
        title_col = col
    if 'desc' in col.lower() or 'text' in col.lower() or 'requirements' in col.lower():
        desc_col = col
if title_col is None:
    title_col = df.select_dtypes(include=['object']).columns[0]
if desc_col is None or desc_col == title_col:
    desc_col = df.select_dtypes(include=['object']).columns[1]
print(f"Используем: профессия='{title_col}', описание='{desc_col}'")

# ==================== 2. ВЫБОР ПРОФЕССИЙ ====================
professions = ['Аналитик', 'Программист', 'Data Engineer']
df[title_col] = df[title_col].astype(str).str.lower()
prof_lower = [p.lower() for p in professions]
df_filtered = df[df[title_col].str.contains('|'.join(prof_lower), na=False, case=False)]
if df_filtered.empty:
    print("Нет вакансий для выбранных профессий. Доступные:", df[title_col].unique())
    exit(1)
print(f"\nОтобрано вакансий: {len(df_filtered)}")
print(df_filtered[title_col].value_counts())


# ==================== 3. ФУНКЦИИ ОЧИСТКИ ====================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zа-яё\s]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_and_lemmatize(text: str, keep_pos=None):
    tokens = word_tokenize(text, language='russian')
    lemmas = []
    for token in tokens:
        if len(token) < 3 or token in STOP_WORDS:
            continue
        parsed = morph.parse(token)[0]
        if keep_pos is not None and parsed.tag.POS not in keep_pos:
            continue
        lemma = parsed.normal_form
        if len(lemma) >= 3 and lemma not in STOP_WORDS:
            lemmas.append(lemma)
    return lemmas


# ==================== 4. ОБРАБОТКА ====================
prof_lemmas = {p: [] for p in professions}
for p in professions:
    p_low = p.lower()
    mask = df_filtered[title_col].str.contains(p_low, na=False)
    desc_list = df_filtered.loc[mask, desc_col].astype(str).tolist()
    print(f"\nОбработка {p} ({len(desc_list)} описаний)")
    all_lemmas = []
    for desc in desc_list:
        all_lemmas.extend(tokenize_and_lemmatize(clean_text(desc)))
    prof_lemmas[p] = all_lemmas
    print(f"  Всего лемм: {len(all_lemmas)}, уникальных: {len(set(all_lemmas))}")

# ==================== 5. ВИЗУАЛИЗАЦИЯ ====================
n_profs = len(professions)
fig, axes = plt.subplots(n_profs, 2, figsize=(14, 5 * n_profs))
if n_profs == 1:
    axes = [axes]

for i, prof in enumerate(professions):
    lemmas = prof_lemmas[prof]
    if not lemmas:
        continue
    freq = Counter(lemmas).most_common(20)
    words, counts = zip(*freq) if freq else ([], [])
    ax_bar = axes[i][0] if n_profs > 1 else axes[i][0]
    ax_bar.barh(words, counts, color='steelblue')
    ax_bar.set_title(f'Топ-20: {prof}')
    ax_bar.invert_yaxis()

    ax_cloud = axes[i][1] if n_profs > 1 else axes[i][1]
    if freq:
        wc = WordCloud(width=800, height=400, background_color='white',
                       colormap='viridis', stopwords=STOP_WORDS).generate_from_frequencies(dict(freq))
        ax_cloud.imshow(wc, interpolation='bilinear')
        ax_cloud.axis('off')
        ax_cloud.set_title(f'Облако: {prof}')
    else:
        ax_cloud.text(0.5, 0.5, 'Нет данных', ha='center')
        ax_cloud.axis('off')
plt.tight_layout()
plt.show()

# ==================== 6. СРАВНЕНИЕ ====================
print("\n" + "=" * 60)
print("СРАВНЕНИЕ ПРОФЕССИЙ")
top_sets = {}
for prof in professions:
    lemmas = prof_lemmas[prof]
    if lemmas:
        top_sets[prof] = set([w for w, _ in Counter(lemmas).most_common(20)])

if top_sets:
    common = set.intersection(*top_sets.values())
    print("\nОбщие слова:", ", ".join(sorted(common)[:10]) if common else "Нет общих")
    for prof in professions:
        if prof not in top_sets:
            continue
        other = set.union(*[top_sets[p] for p in professions if p != prof and p in top_sets])
        unique = top_sets[prof] - other
        print(f"\nСпецифичные для {prof}:", ", ".join(sorted(unique)[:15]) if unique else "Нет")
else:
    print("Недостаточно данных для сравнения.")

print("\nВЫВОДЫ: Использованы профессии:", ", ".join(professions))

