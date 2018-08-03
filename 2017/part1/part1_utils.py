from __future__ import division
import numpy as np

import matplotlib 
import matplotlib.pyplot as plt

import copy, itertools

import csv

def check_answer(expected, actual):
    if expected != actual:
        print "Error!"
        print "Expected:\t{}\tGot:\t{}".format(expected, actual)
        return False
    return True

def test_hypothesis(student_fn):
    x_vals = np.random.rand(100)
    w_vals = np.random.rand(100)
    w0_vals = np.random.rand(100)
    y_vals = x_vals * w_vals + w0_vals
    for i in range(100):
        y_student = student_fn(w_vals[i], w0_vals[i], x_vals[i])
        if not check_answer(y_vals[i], y_student):
            return False
    return True

def test_quadLoss(student_fn):
    y_true = np.random.rand(100)
    y_guess = np.random.rand(100)
    L = np.power(y_true - y_guess, 2)
    for i in range(100):
        l_student = student_fn(y_guess[i], y_true[i])
        if not check_answer(L[i], l_student): return False
    return True

def test_datasetLoss1D(student_fn):
    x_vals = np.random.rand(100)
    w_vals = np.random.rand(100)
    w0_vals = np.random.rand(100)
    y_vals = x_vals * w_vals[0] + w0_vals[0]

    for i in range(100):
        w, w0 = w_vals[i], w0_vals[i]
        l_student = student_fn(w, w0, x_vals, y_vals)
        y_guess = w*x_vals + w0
        l_i = np.sum(np.power(y_guess - y_vals, 2))
        if not check_answer(l_i, l_student): return False
    return True

N_test = 100
N_train = 50

def get_quad_prods(x):
    x_quad = copy.copy(x)
    for a, b in itertools.product(x, x):
        x_quad.append(a * b)
    return x_quad

def _load_ee_data():
    data = ['X{}'.format(i) for i in range(1, 9)]
    with open("energy_efficiency_data.csv", "r") as f:
        x_vals = []
        y_vals = []
        freader = csv.DictReader(f)
        for row in freader:
            x = []
            for l in data:
                x.append(float(row[l]))
            x = get_quad_prods(x)
            y = float(row['Y2'])
            x_vals.append(np.array(x))
            y_vals.append(y)
    np.random.seed(0)
    inds = range(len(y_vals))
    np.random.shuffle(inds)
    x_vals_shuffled = [x_vals[i] for i in inds]
    y_vals_shuffled = [y_vals[i] for i in inds]
    return x_vals_shuffled, y_vals_shuffled

def load_ee_training():
    x, y = _load_ee_data()
    return x[N_test:N_test+N_train], y[N_test:N_test+N_train]

def ee_evaluate(w):
    def predict(x):
        return np.dot(w, np.r_[1, x])
    x_vals, y_vals = _load_ee_data()
    x_test, y_test = x_vals[:N_test], y_vals[:N_test]
    l = 0
    for x, y in zip(x_test, y_test):
        #print predict(x), y, np.power(predict(x) - y, 2)
        l += np.power(predict(x) - y, 2) / len(x_test)
    return l

def _load_death_rate_data():
    data = ['A{}'.format(i) for i in range(1, 16)]
    labels = ['B']
    with open("death_rate_data.csv", "r") as f:
        x_vals = []
        y_vals = []
        freader = csv.DictReader(f)
        for row in freader:
            x = [float(row[l]) for l in data]
            y = float(row['B'])
            x_vals.append(np.array(x))
            y_vals.append(y)
    np.random.seed(4)
    inds = range(len(y_vals))
    np.random.shuffle(inds)
    x_vals_shuffled = [x_vals[i] for i in inds]
    y_vals_shuffled = [y_vals[i] for i in inds]
    return x_vals_shuffled, y_vals_shuffled

def load_death_rate_training():
    x_vals, y_vals = _load_death_rate_data()
    return x_vals[N_test:], y_vals[N_test:]

def death_rate_evaluate(w):
    def predict(x):
        return np.dot(w, np.r_[1, x])
    x_vals, y_vals = _load_death_rate_data()
    x_test, y_test = x_vals[:N_test], y_vals[:N_test]
    l = 0
    for x, y in zip(x_test, y_test):
        print predict(x), y, np.power(predict(x) - y, 2)
        l += np.power(predict(x) - y, 2) / len(x_test)
    return l
    

def _load_forest_data():
    data = ['X', 'Y', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    labels = ['area']
    with open("forestfires.csv", "r") as f:
        x_vals = []
        y_vals = []
        freader = csv.DictReader(f)
        for row in freader:
            if row['month'] not in ['aug', 'jul']:
                continue
            x = [float(row[l]) for l in data]
            y = np.log(float(row['area']) + 1e-8)
            x_vals.append(np.array(x))
            y_vals.append(y)
    np.random.seed(0)
    inds = range(len(y_vals))
    np.random.shuffle(inds)
    x_vals_shuffled = [x_vals[i] for i in inds]
    y_vals_shuffled = [y_vals[i] for i in inds]
    return x_vals_shuffled, y_vals_shuffled

def load_forest_data():
    x, y = _load_forest_data()
    return x[N_test:], y[N_test:]

def forest_test_evaluate(w):
    def predict(x):
        return np.dot(w, np.r_[1, x])
    x, y = _load_forest_data()
    x_test, y_test = x[:N_test], y[:N_test]
    l = 0
    for x, y in zip(x_test, y_test):
        l += np.power(predict(x) - y, 2) / len(x_test)
    return l
    

x_min, x_max = (0, 1)

def plot_1D(w, w0, x_vals=None, y_vals=None, x_lim=None):
    def f(x):
        return w*x + w0
    plot_f(f, x_vals, y_vals, x_lim)

def plot_f(f, x_vals=None, y_vals=None,
           x_lim=None, ax=None, color='b'):
    if x_lim is None:
        x_lim = (x_min, x_max)
    dx = np.linspace(x_lim[0], x_lim[1], 500)
    y = []
    for x in dx:
        y.append(f(x))
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    plt.sca(ax)
    plt.plot(dx, y, color=color)
    if x_vals is not None:
        plt.scatter(x_vals, y_vals)
    
def gen_scatter_data():
    for i in range(3):
        w = 5 * np.random.normal()
        w0 = np.random.normal()
        print w, w0
        x_vals = np.random.rand(100)
        e_vals = np.random.normal(size = 100)
        y_vals = w * x_vals + w0 + e_vals
        np.save('x_{}.npy'.format(i), x_vals)
        np.save('y_{}.npy'.format(i), y_vals)
