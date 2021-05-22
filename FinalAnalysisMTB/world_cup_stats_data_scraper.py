'''
world_cup_stats_data_scraper.py

This is a script to scrape all the data from http://worldcup.eliotjackson.com using request-html

This is done in a script because request-html does not work properly in a jupyter notebook

'''

import os
import random
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import pandas as pd

#imports for the tools I created
import rbf_net
import knn_P7 as knn
import kmeansGPU as kmeans
import linear_regression_gpu as lineaer_regeression
import naive_bayes_multinomial_P7 as naive_bayes

# importing libraries for scrapping webdata
import requests
from bs4 import BeautifulSoup , NavigableString
import json
from requests_html import HTMLSession,HTML,AsyncHTMLSession
import time

# #create requests-html session
# session = HTMLSession()



def get_url_info(url, session_obj ,ret_request_obj = True, wait_time = 1.0):
    '''
    a function to get a dict of all the urls imprtant info
    or the entire request object fully rendered

    inputs:
        url (str): a string of the url to get all the links from

        ret_request_obj (bool): If true return the request object also

    returns:
        url_dict (Dict): python dict containg the url links and abs links along with url, url that was called,
                        and also the urls html file fully rendered

        or it returns
        url_return_obj (HTMLreturn object): url_request
    '''



    url_request = session_obj.get(url=url) # request url

    time.sleep(wait_time)

    try:
        url_request.html.render() # render the page
    except:
        print(f'Warning!! Url {url}  Can not have its HTML rendered')

    if ret_request_obj:
        return url_request
    else:
        # create URL dict
        url_dict = {}
        # set dict vals
        url_dict['url'] = url
        url_dict['HTML full'] = url_request.html.full_text
        url_dict['links'] = url_request.html.links
        url_dict['abs_links'] = url_request.html.absolute_links
        return url_dict






def main():
    # creating session obj
    session = HTMLSession()

    # Get API data
    # wc_api_url = 'http://worldcup.eliotjackson.com/api'
    # wc_api_req = get_url_info(wc_api_url)
    #
    # wc_api_riders_url = 'http://worldcup.eliotjackson.com/api/riders'
    # wc_api_riders_req = get_url_info(wc_api_riders_url)
    #
    # wc_api_results_url = 'http://worldcup.eliotjackson.com/api/results'
    # wc_api_results_req = get_url_info(wc_api_results_url)
    #
    # wc_api_races_url = 'http://worldcup.eliotjackson.com/api/races'
    # wc_api_races_req = get_url_info(wc_api_races_url)

    # wc_api_races_url_slash = 'http://worldcup.eliotjackson.com/api/races/'
    # wc_api_races_req_slash = get_url_info(wc_api_races_url_slash)

    # Getting the url data from the main web pages and not the api of http://worldcup.eliotjackson.com
    # wc_main_races_url = 'http://worldcup.eliotjackson.com/races/'
    # wc_main_races_req = get_url_info(wc_main_races_url)

    wc_main_races_2020_url = 'http://worldcup.eliotjackson.com/races/2020/'
    wc_main_races_2020_req = get_url_info(wc_main_races_2020_url, session)

    print(f'Done Getting API Requests')

if __name__ == '__main__':
    main()
#######################################################################################################################
#  Keep
#######################################################################################################################
#  Scratch
#######################################################################################################################

# # Results page from the main api top link
# wc_api_request = session.get(url='http://worldcup.eliotjackson.com/api')
# #render the page
# wc_api_request.html.render()
# #get links from http://worldcup.eliotjackson.com/api
# wc_api_links_list = [wc_api_request.html.links,wc_api_request.html.absolute_links]



#
# # Results page from main site (not API)
# race_requests_request = session.get(url='http://worldcup.eliotjackson.com/results')
# #render the page
# race_requests_request.html.render()
# race_requests_links = [race_requests_request.html.links,race_requests_request.html.absolute_links]



