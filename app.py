#Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings("ignore")


import streamlit as st

import numpy as np

import pandas as pd

import lightfm as lf

import nmslib

import pickle

import scipy.sparse as sparse

import plotly.express as px

@st.cache

def read_files(folder_name='data'):

    """

    Функция для чтения файлов.

    Возвращает два DataFrame с рейтингами и характеристиками книг.

    """

    ratings = pd.read_csv(folder_name + '/ratings.csv')

    books = pd.read_csv(folder_name + '/books.csv')

    return ratings, books


def make_mappers(books):
    
    """
    
    Функция для создания отображения id в title и authors.
    
    Возвращает два словаря:
    
    * Ключи первого словаря — идентификаторы книг, значения — их названия.
    
    * Ключи второго словаря — идентификаторы книг, значения — их авторы.
    
    """
    
    name_mapper = dict(zip(books.book_id, books.title))
    
    author_mapper = dict(zip(books.book_id, books.authors))


    
    return name_mapper, author_mapper

def load_embeddings(file_name='data/item_embeddings.pkl'):
    
    """
    
    Функция для загрузки векторных представлений.
    
    Возвращает прочитанные эмбеддинги книг и индекс (граф) для поиска похожих книг.
    
    """
    
    with open(file_name, 'rb') as f:
    
        item_embeddings = pickle.load(f)


    
    # Тут мы используем nmslib, чтобы создать быстрый knn
    
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    
    nms_idx.addDataPointBatch(item_embeddings)
    
    nms_idx.createIndex(print_progress=True)
    
    return item_embeddings, nms_idx


def nearest_books_nms(book_id, index, n=10):
    
    """
    
    Функция для поиска ближайших соседей, возвращает построенный индекс.
    
    Возвращает n наиболее похожих книг и расстояние до них.
    
    """
    
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    
    return nn

def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    
    """
    
    Функция для составления таблицы из рекомендованных книг.
    
    Возвращает DataFrame со столбцами:
    
    * book_name — название книги;
    
    * book_author — автор книги;
    
    * distance — значение метрики расстояния до книги.
    
    """
    
    names = []
    
    authors = []
    
    #Для каждого индекса книги находим её название и автора
    
    #Результаты добавляем в списки
    
    for idx in ids:
    
        names.append(name_mapper[idx])
    
        authors.append(author_mapper[idx])
    
    #Составляем DataFrame
    
    recomendation_df = pd.DataFrame({'book_name': names, 'book_author': authors, 'distance': distances})
    
    return recomendation_df


#Загружаем данные

ratings, books = read_files(folder_name='data')

#Создаём словари для сопоставления id книг и их названий/авторов

name_mapper, author_mapper = make_mappers(books)

#Загружаем эмбеддинги и создаём индекс для поиска

item_embeddings, nms_idx = load_embeddings()