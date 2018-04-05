from __future__ import division
from collections import defaultdict
import math
import sys
import random

param = defaultdict(list)

train_file = tuple(open(sys.argv[1],"r"))
test_file = tuple(open(sys.argv[2], "r"))

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]]
with open('train1.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[7]]
with open('train2.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[4], sys.argv[6], sys.argv[7]]
with open('train3.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[3], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train4.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

f_name = [sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]]
with open('train5.txt', 'w') as out_file:
    for f in f_name:
        with open(f) as in_file:
            for line in in_file:
                out_file.write(line)

t_file1 = tuple(open("train1.txt","r"))
t_file2 = tuple(open("train2.txt","r"))
t_file3 = tuple(open("train3.txt","r"))
t_file4 = tuple(open("train4.txt","r"))
t_file5 = tuple(open("train5.txt","r"))

split_file1 = tuple(open(sys.argv[3], "r"))
split_file2 = tuple(open(sys.argv[4], "r"))
split_file3 = tuple(open(sys.argv[5], "r"))
split_file4 = tuple(open(sys.argv[6], "r"))
split_file5 = tuple(open(sys.argv[7], "r"))

def svm(t_file, weight, bias, learning_rate, C):
    line_array = random.sample(t_file, len(t_file))
    for t in range(len(t_file)):
        line = line_array[t]
        index = []
        line = line.strip("\n").split()
        output = line[0]
        output = int(output)
        for j in range(len(line)):
            if j != 0:
                Index_Value = line[j].split(":")
                k = int(Index_Value[0])
                index.append(k-1)
        Sum = 0
        for i in range(len(index)):
            Sum = Sum + (weight[i] * index[i])
        res_sum = Sum + bias
        for i in range(len(weight)):
            weight[i] = (1 - learning_rate) * weight[i]
        bias = ((1 - learning_rate) * bias)
        if (output * res_sum) <= 1:
            y = float(learning_rate * C * output * bias)
            bias = bias + y
            for w in range(len(index)):
                x = float(learning_rate * C * output * index[w])
                weight[w] = weight[w] + x
    return (weight, bias)

def dev(e_file, weight, bias):
    accuracy = 0.0
    instances = 0.0
    for s in range(len(e_file)):
        lineD = e_file[s]
        indexD = []
        instances = instances + 1
        lineD = lineD.strip("\n").split()
        outputD = lineD[0]
        outputD = int(outputD)
        for j in range(len(lineD)):
            if j != 0:
                Index_Value = lineD[j].split(":")
                k = int(Index_Value[0])
                indexD.append(k-1)
        SumDev = 0
        for i in range(len(indexD)):
            SumDev = SumDev + (weight[i] * indexD[i])
        res_sum = SumDev + bias
        if res_sum <= 0:
            r = -1
        else:
            r = 1
        if r == outputD:
            accuracy = accuracy + 1
    return (accuracy/instances) * 100

def main():
    weight = []
    bias = random.uniform(-0.01, 0.01)
    for i in range(68000):
        z = random.uniform(-0.01, 0.01)
        weight.append(z)
    l_r = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    c_r = [10, 1, 0.1, 0.01, 0.001, 0.0001]
    for l in l_r:
        old_l = l
        for c in c_r:
            temp_accuracy = 0.0
            for e in range(10):
                den = e + 1
                l = old_l/den
                weight, bias = svm(t_file1, weight, bias, l, c)
                a = dev(split_file5, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = svm(t_file2, weight, bias, l, c)
                a = dev(split_file4, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = svm(t_file3, weight, bias, l, c)
                a = dev(split_file3, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = svm(t_file4, weight, bias, l, c)
                a = dev(split_file2, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                weight, bias = svm(t_file5, weight, bias, l, c)
                a = dev(split_file1, weight, bias)
                if a > temp_accuracy:
                    temp_accuracy = a
                if e == 9:
                    r = str(l)
                    m = str(c)
                    key = r + ":" + m
                    param[key] = temp_accuracy
    best = max(param, key=param.get)
    b = best.split(":")
    print "Best hyperparamerter: Learning Rate is ", b[0], "Tradeoff is ", b[1]
    print "Best hyperparamerter's Accuracy: ", param[best]
    b[0] = float(b[0])
    b[1] = float(b[1])
    weight, bias = svm(train_file, weight, bias, b[0], b[1])
    a = dev(train_file, weight, bias)
    print "Training Set Accuracy: ", a
    a = dev(test_file, weight, bias)
    print "Test Set Accuracy: ", a

if __name__ == '__main__':
    main()
