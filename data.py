import os
import ipdb
from collections import namedtuple
from os import listdir
from os.path import isfile, join
import ipdb

Sentence = namedtuple('Sentence', ['index', 'color', 'string'] )

def get_folder_items(path):
    return [f for f in listdir(path) if isfile(join(path, f))]

def read_vector_file(path):
    with open(path,'r') as f:
        return [ float(l.strip()) for l in f.readlines()]

def read_vector_file_test(path):
    with open(path, 'r') as f:
        return [  l.strip().split(",") for l in f.readlines()] 
        
def get_corpusindex():
    index = {}
    with open("/home/user/data/rugstk/data/munroecorpus/corpusindex.txt", 'r') as f:        
        for l in f.readlines():
            name, file_ = l.strip().split(",") 
            index[file_] = name 

    return index

def read_train_data(path):
    corpusindex = get_corpusindex()
    items = get_folder_items(path)
    items_index = list(set([i.split(".")[0] for i in items if i.split(".")[0] != '']))

    data = [] 
    for i in items_index:
        for h,s,v in zip (read_vector_file(join(path,i+".h_train")),
                          read_vector_file(join(path,i+".s_train")),
                          read_vector_file(join(path,i+".v_train")) 
                         ):
            data.append((i, (h*360, s*100, v*100), corpusindex[i]))

    descriptions = [Sentence(i, color,string) 
            for i, (_ , color, string) in enumerate(data)]

    return descriptions


def read_train_data_char(path):
    corpusindex = get_corpusindex()
    items = get_folder_items(path)
    items_index = list(set([i.split(".")[0] for i in items if i.split(".")[0] != '']))

    data = [] 
    for i in items_index:
        for h,s,v in zip (read_vector_file(join(path,i+".h_train")),
                          read_vector_file(join(path,i+".s_train")),
                          read_vector_file(join(path,i+".v_train")) 
                         ):
            data.append((i, (h*360, s*100, v*100), list(corpusindex[i])))

    descriptions = [Sentence(i, color,string) 
            for i, (_ , color, string) in enumerate(data)]

    return descriptions



def read_test_data(path):
    corpusindex = get_corpusindex()
    items = get_folder_items(path)
    items_index = list(set([ i.split(".")[0] for i in items if i.split(".")[0] != '']))

    data = []
    for i in items_index:
        for  hsv  in read_vector_file_test(join(path,i+".test")):
            
            data.append((i, (  float(hsv[0])*360, float(hsv[1])*100, float(hsv[2])*100), corpusindex[i] ))

    descriptions = [Sentence(i, color, string) for i, (_, color, string) in enumerate(data)]

    return descriptions

def read_test_data_char(path):
    corpusindex = get_corpusindex()
    items = get_folder_items(path)
    items_index = list(set([ i.split(".")[0] for i in items if i.split(".")[0] != '']))

    data = []
    for i in items_index:
        for  hsv  in read_vector_file_test(join(path,i+".test")):
            
            data.append((i, (  float(hsv[0])*360, float(hsv[1])*100, float(hsv[2])*100), list(corpusindex[i]) ))

    descriptions = [Sentence(i, color, string) for i, (_, color, string) in enumerate(data)]

    return descriptions

