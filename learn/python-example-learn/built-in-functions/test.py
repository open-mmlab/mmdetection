import sys
import time

class Student():
    def __init__(self, id, name):
        self.id = id
        self.name = name

    def __repr__(self):
        return 'id = ' + self.id + ', name = ' + self.name

    def __call__(self) -> str:
        print('I can be called')
        print(f'my name is {self.name}')
        return "test"

    @classmethod
    def f(cls):
        print(cls)

xiaoming = Student(id='001', name='xiaoming')

class undergraduate(Student):
    def studyClass(self):
        pass
    def attendActivity(self):
        pass

# lst = [1, 3, 5]
# for i in iter(lst):
#     print(i)

class TestIter(object):
    def __init__(self):
        self.l = [1, 3, 2, 3, 4, 5]
        self.i = iter(self.l)

    def __call__(self, *args, **kwargs):
        item = next(self.i)
        print("__call__ is called, fowhich would return", item)
        return item

    def __iter__(self):
        print("__iter__ is called!")
        return iter(self.i)

# t = TestIter()
# t()
#
# for e in TestIter():
#     print(e)

# fo = open('../../test.py', mode='r', encoding='utf-8')
# # print(fo.read())

class C:
    def __init__(self):
        self._x = None

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @x.deleter
    def x(self):
        del self._x

a = [1,4,2,3,1]



a = [{'name':'xiaoming', 'age':18, 'gender':'male'}
     , {'name': 'xiaohong', 'age':20, 'gender':'femal'}]
# print(sorted(a, key=lambda x:x['age'], reverse=False))

x = [3, 2, 1]
y = [4, 5, 6]
# print(list(zip(y, x)))

a = range(5)
b = list('abcde')
c = [str(y) + str(x) for x, y in zip(a, b)]
# print(c)

def excepter(f):
    i = 0
    t1 = time.time()

    def wrapper():
        try:
            f()
        except Exception as e:
            nonlocal i
            i += 1
            print(f'{e.args[0]}: {i}')
            t2 = time.time()
            if i == n:
                print(f'spending time: {round(t2-t1, 2)}')
        return wrapper

from operator import *

def calculator(a, b, k):
    return {
        '+': add,
        '-': sub,
        '*': mul,
        '/': truediv,
        '**': pow
    }[k](a, b)
a = calculator(1, 2, '+')
# print(a)

def add_or_sub(a, b, oper):
    return (add if oper == '+' else sub)(a, b)

def score_mean(lst):
    lst.sort()
    lst2 = lst[1:len(lst)-1]
    return round((sum(lst2)/len(lst2)), 1)

lst = [9.19, 0.82, 4.3, 45]
# print(score_mean(lst))

# for i in range(1, 10):
#     for j in range(1, i+1):
#         print('%d*%d=%d' % (j, i, j*i), end="\t")
#     print()

array = [[[1, 2, 3], [4, 5]]]

from collections.abc import *

def flatten(lst, out_lst=None):
    if out_lst is None:
        out_lst = []
    for i in lst:
        if isinstance(i, Iterable):
            flatten(i, out_lst)
        else:
            out_lst.append(i)

    return out_lst

# print(flatten(array)

import numpy as np

b = np.array([[1, 2, 3], [4, 5]])

from math import ceil

def divide(lst, size):
    if size <= 0:
        return [lst]
    return [lst[i * size:(i+1)*size] for i in range(0, ceil(len(lst) / size))]

def filter_false(lst):
    return list(filter(bool, lst))

def max_length(*lst):
    return max(*lst, key=lambda v:len(v))

r = max_length([1, 2, 3], [4, 5, 6, 7], [8])

def top1(lst):
    return max(lst, default='blank is None!', key=lambda v:lst.count(v))

lst = [1, 3, 3, 2, 1, 1, 2]
r = top1(lst)

def max_lists(*lst):
    return max(max(*lst, key=lambda v:max(v)))

r = max_lists([1, 2, 3], [6, 7, 8], [4, 5])

def has_duplicate(lst):
    return len(lst) == len(set(lst))
x = [1, 1, 2, 2, 3]
y = [1, 2, 3, 1, 1]

def reverse(lst):
    return lst[::-1]
r = reverse([1, -2, 3, 4, 1, 2])

def rang(start, stop, n):
    start, stop, n = float('%.2f' % start), float('%.2f' % stop), int('%.d' % n)
    step = (stop - start) / n
    lst = [start]
    while n > 0:
        start, n = start + step, n-1
        lst.append(round((start), 2))
    return lst


def bif_by(lst, f):
    return [[x for x in lst if f(x)], [x for x in lst if not f(x)]]

records = [25, 89, 31, 34]

lst1 = [1, 2, 3, 4, 5, 6]
lst2 = [3, 4, 5, 6, 3, 2]
a = list(map(lambda x, y:x*y+1, lst1, lst2))

def max_pairs(dic):
    if len(dic) == 0:
        return dic
    max_val = max(map(lambda v:v[1], dic.items()))
    return [item for item in dic.items() if item[1] == max_val]


r = max_pairs({'a': -10, 'b': 5})

# def merge_dict(dict1, dict2):
#     print(**dict1)
#     print(**dict2)
#     return {**dict1, **dict2}
#
# print(merge_dict({'a': 1, 'b': 2}, {'c': 3}))

from random import shuffle, randint

lst = [randint(0, 50) for _ in range(100)]
shuffle(lst)

from random import uniform

a = [(uniform(0, 10), uniform(0, 10)) for _ in range(10)]

from random import gauss
x = range(10)
y = [2*xi+1+gauss(0, 1) for xi in x]
points = list(zip(x, y))

from itertools import chain

a = [1, 2, 3, 4, 5, 6]
b = (2, 4, 5)

def f():
    print('i\'m f')

def g():
    print('i\'m g')

print([f, g][1]())

def f(a, *b, c=10, **d):
    print(f'a:{a}, b:{b}, c:{c}, d:{d}')

