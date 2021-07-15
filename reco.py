import pickle
from recommend.load_data import *
from recommend.recommend import recommend


def recommend(self, user_id):
    data, books = load_data_from_db()
    print(data)
    # data, books = load_dataset()
    book_arr, ratings_arr, d, books = pre_process(data, books)
    num_books = len(books)

    rbm_model = open('RBM.pkl', 'rb')
    rbm = pickle.load(rbm_model)

    inds = recommend(rbm, user_id, data, num_books)
    reco_list = []
    for x in inds:
        reco_list.append(x.item())
    return reco_list
