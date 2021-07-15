import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import time
import math
from load_data import *
from RBM import RBM
from recommend import *

# load data

data, books = load_data_from_db()
print(data)
# data, books = load_dataset()
book_arr, ratings_arr, d, books = pre_process(data, books)
num_books = len(books)
print(d)
# train-test split 80:20

book_train, book_test, ratings_train, ratings_test = train_test_split(book_arr, ratings_arr, test_size=0.2,
                                                                        shuffle=True)

print("Number of total users : ", book_arr.shape[0])
print("Number of users for training : ", book_train.shape[0])
print("Number of users for testing : ", book_test.shape[0])

# Empty CUDA cache and create RBM object on GPU

torch.cuda.empty_cache()
n_visible = num_books
n_hidden = 1024
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
rbm = RBM(n_visible, n_hidden, device, lr=0.001)
print("Number of visible nodes = ", n_visible)
print("Number of hidden nodes = ", n_hidden)
print("------------------------------------------------------")

# Training Phase

print("\nTraining Started \n")
start_time = time.time()
epochs = 50
batch_size = 128
N = book_train.shape[0]
num_batches = math.ceil(N / batch_size)
loss_fn = nn.MSELoss()
for epoch in range(epochs):
    loss = 0.0
    i = 0
    while i < N:
        m_data = book_train[i: i + batch_size]
        r_data = ratings_train[i: i + batch_size]
        i += batch_size
        input = np.zeros((m_data.shape[0], num_books))
        for ind in range(m_data.shape[0]):
            # print(len(m_data[ind]), len(r_data[ind]))
            input[ind, m_data[ind]] = r_data[ind]
        input = torch.Tensor(input)
        out = rbm.cont_div(input)
        out = out.to(device)
        input = input.to(device)
        loss += loss_fn(input, out).item()
        print("process: %s" %(100*i/N) )

    print("Epoch %s => Loss = %s" % (epoch + 1, loss / num_batches))

print("Training time = %s" % (time.time() - start_time))
print("------------------------------------------------------")

# Testing Phase

print("\nTesting Started")
batch_size = 128
N = book_test.shape[0]
num_batches = math.ceil(N / batch_size)
loss_fn = nn.MSELoss()
loss = 0.0
i = 0
while i < N:
    m_data = book_test[i: i + batch_size]
    r_data = ratings_test[i: i + batch_size]
    i += batch_size
    input = np.zeros((m_data.shape[0], num_books))
    for ind in range(m_data.shape[0]):
        # print(len(m_data[ind]), len(r_data[ind]))
        input[ind, m_data[ind]] = r_data[ind]
    input = torch.Tensor(input)
    out = rbm.infer(input)
    out = out.to(device)
    input = input.to(device)
    loss += loss_fn(input, out).item()

print("\nTest Loss = %s\n" % (loss / num_batches))
print("------------------------------------------------------")

# Save Model and recommend books

path = 'RBM.pkl'
with open(path, 'wb') as output:
    pickle.dump(rbm, output)

with open("book-id-dict.pkl", 'wb') as output:
    pickle.dump(d, output)

rbm_model = open('RBM.pkl', 'rb')
rbm = pickle.load(rbm_model)
print("\nRBM pickle file saved at : %s\n" % (os.path.abspath('RBM.pkl')))
print("------------------------------------------------------")
user_id = 41
inds = recommend(rbm, user_id, data, num_books)
for x in inds:
    book_index = x.item()
    # if x.item() > 780:

    print(d[book_index])
# rec_books = [books.loc[d[x.item()]]['Name'] for x in inds]
# print("\nRecommended books for User-id %s : \n" % (user_id))
# for book in rec_books:
#     print(book)
