'''email_preprocessor.py
Preprocess Enron email dataset into features for use in supervised learning algorithms
YOUR NAME HERE
CS 251 Data Analysis Visualization, Spring 2020
'''
import re
import os
import numpy as np
import time
import pandas as pd
import collections
import itertools

def tokenize_words(text):
    '''Transforms an email into a list of words.

    Parameters:
    -----------
    text: str. Sentence of text.

    Returns:
    -----------
    Python list of str. Words in the sentence `text`.

    This method is pre-filled for you (shouldn't require modification).
    '''
    # Define words as lowercase text with at least one alphabetic letter
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())





def read_and_count_file_helper(included_word,word_freq, pad_by = 0):
    '''
    helper function for read and count file
    returns the count of eveyword in the file in included words plus 1 and then just one for all the included
    words not in the file
    '''
    if included_word in word_freq.keys():
        return word_freq[included_word]+pad_by
    else:
        return pad_by


#helper function to read in and tokenize the file the get the word counts of the words included in the file
def read_and_count_file(file_path,included_words, debug = False):

    #open the file
    with open(file_path,'r') as text_file:
        text_in_file = text_file.read()
        tokenized_text = tokenize_words(text_in_file)
        tokenized_text = list(filter(lambda x: x in included_words, tokenized_text))

        #get the word count of each text
        word_freq = dict(collections.Counter(tokenized_text))
        word_counts = list(map(lambda x:read_and_count_file_helper(x,word_freq), iter(included_words)))

        #for debugging
        if debug:
            print(f'Word Count Adjusted\n {included_words} \n {word_counts} \n')
    return word_counts



#helper function to read in and tokenize it file
def read_and_tokenize_file(file_path,included_words = []):

    #open the file
    with open(file_path,'r') as text_file:
        text_in_file = text_file.read()
        tokenized_text = tokenize_words(text_in_file)
        if included_words:
            tokenized_text = list(filter(lambda x: x in included_words, tokenized_text))
    return tokenized_text



def count_words(email_path='data/enron', multi_thread = False, testTime = False):
    '''Determine the count of each word in the entire dataset (across all emails)

    Parameters:
    -----------
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_emails: int. Total number of emails in the dataset.

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use the `tokenize_words` function above to chunk it into a list of words.
    - Update the counts of each word in the dictionary.

    Hints:
    - Check out Python functions in the os and os.path modules for walking the directory structure.
    '''
    #for debugging
    if testTime:
        start_time = time.time()




    #load in all the files
    if multi_thread:
        #TODO Implement Multi Threading
        text_from_files = []
    else:
        # get all the file-paths in the directory from email_path
        all_file_paths = []

        # dictionary holding every directory with its sub directories
        dir_sub_dirs = {}
        # boolean for into sub directories
        insub_dirs = False
        for path, subdirs, files in os.walk(email_path):
            if files and insub_dirs:
                file_paths_in_dir = list(map(lambda x: f'{path}/{x}', files))
                # if path in sub_dirs.values()[:][:]:
                #     file_paths_in_dir = list(map(lambda ))
                all_file_paths += file_paths_in_dir
            dir_sub_dirs[path] = subdirs
            insub_dirs = True
        # set the number of emails
        num_emails = len(all_file_paths)
        text_from_files = list(map(lambda x: (read_and_tokenize_file(x)),all_file_paths))

        flattened_text = list(itertools.chain.from_iterable(iter(text_from_files)))
        word_freq = dict(collections.Counter(flattened_text).most_common())

    # for debugging
    if testTime:
        print(f'Done with count_words()!! It took {start_time-time.time()} seconds to run')
    return (word_freq, num_emails)

def find_top_words(word_freq, num_features=200):
    '''Given the dictionary of the words that appear in the dataset and their respective counts,
    compile a list of the top `num_features` words and their respective counts.

    Parameters:
    -----------
    word_freq: Python dictionary. Maps words (keys) to their counts (values) across the dataset.
    num_features: int. Number of top words to select.

    Returns:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    counts: Python list. Counts of the `num_features` words in high-to-low count order.
    '''

    #TODO why is it not returning a list but dict keys and dict values item
    top_words = list(word_freq.keys())[:num_features]
    counts = list(word_freq.values())[:num_features]
    return top_words, counts
def make_feature_vectors(top_words_to_find, num_emails=None, email_path='data/enron', multi_thread = False, testTime = False):
    '''Count the occurance of the top W (`num_features`) words in each individual email, turn into
    a feature vector of counts.

    Parameters:
    -----------
    top_words: Python list. Top `num_features` words in high-to-low count order.
    num_emails: int. Total number of emails in the dataset.
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    feats. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)

    TODO:
    - Descend into the dataset base directory -> class folders -> individual emails.
    - Read each email file as a string.
    - Use `tokenize_words` to chunk it into a list of words.
    - Count the occurance of each word, ONLY THOSE THAT APPEAR IN `top_words`.

    HINTS:
    - Start with your code in `count_words` and modify as needed.
    '''
    # for debugging
    if testTime:
        start_time = time.time()

    # load in all the files
    if multi_thread:
        # TODO Implement Multi Threading
        text_from_files = []
    else:

        # get all the file-paths in the directory from email_path and the number of the category they are in
        all_file_paths = []

        #int to hold what category (path ie directory) each file is in
        category_counter = 0

        # boolean for into sub directories
        insub_dirs = False
        for path, subdirs, files in os.walk(email_path):
            if files and insub_dirs:

                file_paths_in_dir = list(map(lambda x: (category_counter,f'{path}/{x}'), files))
                all_file_paths += file_paths_in_dir
                category_counter += 1
            insub_dirs = True

        #set the number of emails
        num_emails = len(all_file_paths)

        text_from_files_count = list(map((lambda x:(x[0],read_and_count_file(x[1],top_words_to_find))), all_file_paths))
        y, feats = zip(*text_from_files_count)


    # for debugging
    if testTime:
        print(f'Done with count_words()!! It took {start_time - time.time()} seconds to run')

    #make arrays
    y = np.array(y)
    feats = np.array(feats)

    return feats, y


def make_train_test_sets(features, y, test_prop=0.2, shuffle=True, testTime = False):
    '''Divide up the dataset `features` into subsets ("splits") for training and testing. The size
    of each split is determined by `test_prop`.

    Parameters:
    -----------
    features. ndarray. shape=(num_emails, num_features).
        Vector of word counts from the `top_words` list for each email.
    y. ndarray of nonnegative ints. shape=(num_emails,).
        Class index for each email (spam/ham)
    test_prop: float. Value between 0 and 1. What proportion of the dataset samples should we use
        for the test set? e.g. 0.2 means 20% of samples are used for the test set, the remaining
        80% are used in training.
    shuffle: boolean. Should we shuffle the data before splitting it into train/test sets?

    Returns:
    -----------
    x_train: ndarray. shape=(num_train_samps, num_features).
        Training dataset
    y_train: ndarray. shape=(num_train_samps,).
        Class values for the training set
    inds_train: ndarray. shape=(num_train_samps,).
        The index of each training set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    x_test: ndarray. shape=(num_test_samps, num_features).
        Test dataset
    y_test:ndarray. shape=(num_test_samps,).
        Class values for the test set
    inds_test: ndarray. shape=(num_test_samps,).
        The index of each test set email in the original unshuffled dataset.
        For example: if we have originally N=5 emails in the dataset, their indices are
        [0, 1, 2, 3, 4]. Then we shuffle the data. The indices are now [4, 0, 3, 2, 1]
        let's say we put the 1st 3 samples in the training set and the remaining
        ones in the test set. inds_train = [4, 0, 3] and inds_test = [2, 1].
    '''
    # for debugging
    if testTime:
        start_time = time.time()

    inds = np.arange(y.size)
    if shuffle:
        features = features.copy()
        y = y.copy()

        inds = np.arange(y.size)
        np.random.shuffle(inds)
        features = features[inds]
        y = y[inds]

    # Your code here:

    #get test and train size
    test_size = int(y.size * test_prop)
    train_size = y.size - test_size
    split_inds = np.arange(y.size)

    test_split_inds = list(np.random.choice(split_inds,test_size,replace=False))
    train_split_inds = list(filter(lambda x: x not in test_split_inds, split_inds))

    x_train = features[train_split_inds]
    y_train = y[train_split_inds]
    inds_train = inds[train_split_inds]

    x_test = features[test_split_inds]
    y_test = y[test_split_inds]
    inds_test = inds[test_split_inds]

    # for debugging
    if testTime:
        print(f'Done with count_words()!! It took {start_time - time.time()} seconds to run')

    return x_train,y_train,inds_train,x_test,y_test,inds_test

def retrieve_emails(inds, email_path='data/enron'):
    '''Obtain the text of emails at the indices `inds` in the dataset.

    Parameters:
    -----------
    inds: ndarray of nonnegative ints. shape=(num_inds,).
        The number of ints is user-selected and indices are counted from 0 to num_emails-1
        (counting does NOT reset when switching to emails of another class).
    email_path: str. Relative path to the email dataset base folder.

    Returns:
    -----------
    Python list of str. len = num_inds = len(inds).
        Strings of entire raw emails at the indices in `inds`
    '''
    # get all the file-paths in the directory from email_path
    all_file_paths = []

    #boolean for into sub directories
    insub_dirs = False
    for path, subdirs, files in os.walk(email_path):
        if files and insub_dirs:
            file_paths_in_dir = list(map(lambda x: f'{path}/{x}', files))
            # if path in sub_dirs.values()[:][:]:
            #     file_paths_in_dir = list(map(lambda ))

            all_file_paths += file_paths_in_dir
        insub_dirs =True
    # get selected email paths
    select_emails = list(filter(lambda x: x[0] in inds, enumerate(all_file_paths)))
    # set the number of emails
    num_emails = len(all_file_paths)
    text_from_files = list(map(lambda x: read_and_tokenize_file(x[1]), select_emails))
    return text_from_files
