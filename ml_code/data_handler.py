'''
Purpose:
    - Wrangle data into usable format
    - Accept arguments about data types

    - Two targets of interaction:
        - tranco: consist of top-sites that are hardened against manipulation
        - phish_tanks.csv: consists of known phishing examples

'''
import numpy as np

from encoding_handler import WordEncoder

def return_positives():
    '''
    Returns dataset consisting of only positive samples (IS a phishing link)
    
    Accesses: phish_tank.csv

    Use Case: Create an anomaly detection algorithm (the anomaly is a benign sample)

    Data Format:
        - ID, URL, Phish_Detail_URL, Submission Time, Verified, Verification Time, Online, Target
    '''

    # assumes that phish_tank is in the same directory as this py file
    file = open('phish_tank.csv', 'r')

    x = []
    y = [] # dummy variable

    for i, line in enumerate(file):
        if i == 0:
            continue

        data = line.strip().split(',')
        x.append(data[1].strip().lower())

    return x, y 


def return_negatives():
    '''
    Returns dataset consisting of only negative samples (IS NOT a phishing link)

    Accesses: tranco/
    
    Use Case: Create an anomaly detection algorithm (the anomaly is a phishing link)
    '''
    file = open('tranco.csv', 'r')

    x, y = [], []

    for i, line in enumerate(file):
        data = line.strip().split(',')
        x.append(data[1].strip().lower())

    return x, y

def return_all():
    '''
    Returns dataset consisting of both positive and negative samples

    Accesses: tranco/ and phish_tank.csv

    Use Case: Create a binary classifier (1 is phishing and 0 is not phishing)

    '''

    x_neg, y_neg = return_negatives()
    x_pos, y_pos = return_positives()

    x = x_neg + x_pos
    y = y_neg + y_pos 

    return x, y

def generate_data(data_type="positive", encoding="char_encoding", *args, **kwargs):
    '''
    Loads, encodes, and returns dataset x/y as torch tensor


    '''
    func_call = None
    encoder = kwargs['encoder_obj'] if 'encoder_obj' in kwargs else WordEncoder()

    if data_type == "positive":
        func_call = return_positives
    elif data_type == "negative":
        func_call = return_negatives
    else:
        func_call = return_all

    x, y = func_call()
    
    if encoding == "char_encoding":
        x = encoder.encode_char(x)
    else:
        x = x

    return x, y, encoder