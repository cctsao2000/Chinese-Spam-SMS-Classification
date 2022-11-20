from ast import While
import numpy as np 
import pandas as pd 

def get_data(data_file_path):
    
    lines = []    
    labels = []
    
    for line in open(data_file_path, "r", encoding="utf-8"):

        arr = line.rstrip().split("\t")
        if len(arr) < 3:
            continue
        
        if int(arr[0]) == 1:
            label = 1
        elif int(arr[0]) == 0 or int(arr[0]) == -1:
            label = 0
        else:
            continue
        labels.append(label)
        
        line = arr[2].split()
        lines.append(line)
        
    return lines, labels


def create_vocab_dict(data_lines):
    vocab_dict = {}
    for data_line in data_lines:
        for word in data_line:
            if word in vocab_dict:
                vocab_dict[word] += 1
            else:
                vocab_dict[word] = 1
    return vocab_dict

def BOW_feature(vocab_list, input_line):
    return_vec = np.zeros(len(vocab_list), )
    for word in input_line:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


if __name__ == "__main__":

    train_file_path = "mudou_spam_train.txt"
    test_file_path = "mudou_spam_test.txt"

    train_data, train_label = get_data(train_file_path)
    test_data, test_label = get_data(test_file_path)


    vocab_dict = create_vocab_dict(train_data)
    sorted_vocab_list = sorted(vocab_dict.items(), key=lambda d:d[1], reverse=True)
    min_freq = 5
    vocab_list = [v[0] for v in sorted_vocab_list if int(v[1]) > min_freq]
   
    train_X = []
    i=0
    for one_msg in train_data:
        i+=1
        train_X.append(BOW_feature(vocab_list, one_msg))
        print(i)


    test_X = []
    j=0
    for one_msg in test_data:
        j+=1
        test_X.append(BOW_feature(vocab_list, one_msg))
        print(j)

    train_X = np.array(train_X)
    test_X = np.array(test_X)
    train_label = np.array(train_label)
    test_label = np.array(test_label)
    with open('train_X.npy', 'wb') as tr_X:
        np.save(tr_X, train_X)
    with open('test_X.npy', 'wb') as te_X:
        np.save(te_X, test_X)
    with open('train_label.npy', 'wb') as tr_y:
        np.save(tr_y, train_label)
    with open('test_label.npy', 'wb') as te_y:
        np.save(te_y, test_label)

