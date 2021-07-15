import numpy as np
import pandas as pd
import os
from peewee import *

from model.book import Book
from model.comment import Comment


def load_dataset():
    ratings_filename = '../book_data/bx_book_ratings.csv'
    book_filename = '../book_data/book_info.csv'
    data = pd.read_csv(ratings_filename, header=None, names=['userid', 'bookid', 'score'])

    col = ['bookid', 'book_category_id', 'store_id', 'Name', 'outline', 'detail', 'press', 'author', 'Publish_date',
           'size', 'version', 'translator', 'isbn', 'price', 'pages', 'catalog', 'market_price', 'member_price',
           'deal_mount', 'look_mount', 'discount', 'image_url', 'store_mount', 'store_time', 'pack_style', 'is_shelf',
           'cname', 'description', 'cata', 'content']
    books = pd.read_csv(book_filename, header=None, names=col, usecols=['bookid', 'Name'], low_memory=False)
    print(data)
    print(books)
    return data, books


def load_data_from_db():
    list = []
    rate = Comment.select(Comment)
    rates = pd.DataFrame(columns=['userid', 'bookid', 'score'])
    for r in rate:
        rates = rates.append({'userid': r.from_id, 'bookid': r.book_id, 'score': r.review_rank}, ignore_index=True)
        list.append(r.book_id)

    book = Book.select(Book)
    books = pd.DataFrame(columns=['bookid', 'Name'])
    for b in book:
        if b.id in list:
            books = books.append({'bookid': b.id, 'Name': b.book_name}, ignore_index=True)

    return rates, books


def convert(data, num_books):
    '''
    Convert the data from RAW csv to a single vector for every user
    This vector has ids of rated books by a user
    '''
    users = data['userid'].unique()
    data = data.values
    N = data.shape[0]
    book_arr = []
    rating_arr = []
    i = 0
    index = 0

    for id in users:
        book_ids = []
        user_rating = []
        while index < N and data[index][0] == id:
            book_ids.append(data[index][1])
            user_rating.append(data[index][2] / 5)
            # arr[i, data[index][1]] = data[index][2]/5
            index += 1

        book_arr.append(list(map(int, book_ids)))
        rating_arr.append(user_rating)
        i += 1

    book_arr = np.array(book_arr, dtype=object)
    rating_arr = np.array(rating_arr, dtype=object)

    return book_arr, rating_arr


def pre_process(data, books):
    '''
    Parameters
    ----------
    data : ratings dataframe
    books : books dataframe

    Returns
    -------
    book_arr : preprocessed reviews list
    ratings_arr : preprocessed ratings list
    d : book-cat_id to book-id dict
    books : processed book dataframe
    '''
    data['bookid'] = data['bookid'].astype('category')
    d = dict(enumerate(data['bookid'].cat.categories))
    data['bookid'] = data['bookid'].cat.codes
    books = books.set_index('bookid')
    num_books = len(books)
    print("Number of books : ", num_books)
    book_arr, ratings_arr = convert(data, num_books)

    return book_arr, ratings_arr, d, books

#
# if __name__ == '__main__':
#     data, books = load_dataset()
#     pre_process(data, books)
