import requests
from bs4 import BeautifulSoup
import unidecode
import re
import pickle
import pandas as pd
from time import time, sleep
from IPython.display import clear_output
from random import randint
from datetime import datetime, date
from selenium import webdriver
from selenium.common.exceptions import WebDriverException

import pdb

###############################################################################
########################### SCRAPING REALO WEBSITE ############################
###############################################################################
# Return parsed web page
def parsed_page(url):
    response = requests.get(url)

    # Send a warning if Response code isn't 200
    # if response.status_code != 200:
    #    warn(f'Request for url : {url} has code: {response.status_code}')
    content = response.content
    parser = BeautifulSoup(content, 'html.parser')
    return parser

def get_realo_cities_links(realo_cities_url):
    realo_base_url = "https://www.realo.be"

    # Parse realo's city's page
    realo_soup = parsed_page(realo_cities_url)

    realo_list = realo_soup.select(".icn-after-arrow-right")

    realo_dict = {}  # Dictionnary to save scraped cities and respective links
    for city in realo_list:
        city_name = city.get_text(strip=True)
        url_city = city.get('href')
        realo_dict[city_name] = realo_base_url + url_city

    return realo_dict


###############################################################################
############################# SCRAPING IMMOWEB ################################
###############################################################################


# Get Results page based on the type of good, the region and the desired page_num
def return_full_url(type_of_good, region, page_num=1):
    ''' Returns the immoweb url with regards to type_of_good, region and page_num

    :param type_of_good (string): 'appartement' or 'immeuble-de-rapport'
    :param region (string): one of the belgian provinces written in french
    :param page_num (int): page number of websearch
    :return:
    '''

    full_url = f"https://www.immoweb.be/fr/recherche/{type_of_good}/a-vendre/{region}"\
    f"/province?countries=BE&page={page_num}&orderBy=relevance"
    return full_url


def parsed_protected_page(url):
    ''' Returns the parsed page through BeautifulSoup with Chrome as webdriver

    :param url (string): url of that we want to parse
    :return: the parsed page through BeautifulSoup with Chrome as webdriver
    '''

    # Use the headless option to avoid opening a new browser window
    # source: https://towardsdatascience.com/web-scraping-with-selenium-d7b6d8d3265a
    options = webdriver.ChromeOptions()
    # options.add_argument("headless") # Selenium Webdriver was being detected afterwards with this option
    desired_capabilities = options.to_capabilities()
    driver = webdriver.Chrome(executable_path="/usr/local/bin/chromedriver", desired_capabilities=desired_capabilities)

    # Get Page with Url
    get_url = driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    return soup


def get_individual_urls(one_search_page_url, filter_opt=True):
    ''' Returns 3 lists, with the urls, the postcodes and cities

    :param one_search_page_url (string): url of 1 single search page
    :param filter_opt (boolean): if True => filter out the offers for several goods (for example 10 apts)
    :return: 4 lists, with the urls, the prices, the postcodes and cities
    '''

    # Parse the input url and get a Beautifulsoup object
    soup = parsed_protected_page(one_search_page_url)

    # Find the different urls of goods to potentially investigate
    urls = soup.select(".card__title-link")
    localities = soup.select(".card__information--locality")  # .get_text(strip=True).split()
    prices = soup.select(".card--result__price")

    if filter_opt:
        # Do they talk of multiple goods in one page?
        # Final urls to return
        final_urls = []
        price_list = []
        post_code_list = []
        city_list = []

        for url, price, locality in zip(urls, prices, localities):
            s = price.text
            newString = (s.encode('ascii', 'ignore')).decode("utf-8")
            if "-" not in newString:
                final_urls.append(url.get('href'))
                price_list.append(re.findall(r'[0-9]+', price.get_text(strip=True).split("€")[-2])[0])
                post_code_list.append(locality.get_text(strip=True).split(" ")[0])
                city_list.append(locality.get_text(strip=True).split(" ")[1])

    else:
        final_urls = [url.get('href') for url in urls]
        # Prices: regex as sometimes after splitting on €, there's still some text part
        price_list = [re.findall(r'[0-9]+', price.get_text(strip=True).split("€")[-2])[0] for price in prices]
        post_code_list = [locality.get_text(strip=True).split(" ")[0] for locality in localities]
        city_list = [locality.get_text(strip=True).split(" ")[1] for locality in localities]

    return final_urls, price_list,  post_code_list, city_list


def get_nb_pages(url):
    ''' Returns the total number of search pages resulting of the search query

    :param url (string):
    :return: the total number of search pages resulting of the search query
    '''

    # Parse the input url and get a Beautifulsoup object
    soup = parsed_protected_page(url)

    # Get the span containing the amount of pages
    pages = soup.select(".pagination span")

    numbers = []
    for page in pages:
        text = page.text
        if len(text) < 3:
            numbers.append(int(text))

    return max(numbers)


def string_preprocessing(feature_name):
    ''' Return preprocessed feature_name

    :param feature_name (string): name of feature that has to be preprocessed
    :return (string): preprocessed name of feature (feature_name)
    '''
    unaccented_string = unidecode.unidecode(feature_name)
    lower_case = unaccented_string.lower()
    no_comma = lower_case.replace(",","").replace("& ","")
    no_space = no_comma.replace(" ","_")
    no_prime = no_space.replace("'","_")
    return no_prime


def get_features_one_page(url, postcode, city, price, region, type_of_good):
    ''' Returns dictionnary with the different house features of current immoweb page

    :param url (string): url of property we want to scrape the features from
    :param postcode (string): postcode gathered through get_individual_urls()
    :param city (string): city gathered through get_individual_urls()
    :param price (string): price gathered through get_individual_urls()
    :param region (string): region know from  return_full_url
    :param type_of_good (string): 'appartement' or 'immeuble-de-rapport'
    :return (dict): dictionnary containing the different house features of the current immoweb page
    '''

    # Parse the input url and get a Beautifulsoup object
    soup = parsed_protected_page(url)

    # Dictionnary with the features
    features = {}

    # Postcode and cities were scraped on the general reseach page
    features['page_url'] = url
    features['postcode'] = postcode
    features['city'] = city
    features['type_of_good'] = type_of_good
    features['price'] = price
    features['region'] = region

    ## Get the different features
    # Get the Immoweb code
    try:
        immoweb_code = soup.select(".classified__information--immoweb-code")[0].get_text(strip=True).split(":")[-1].strip()
        features["immoweb_code"] = immoweb_code
    except:
        features["immoweb_code"] = ""

    # Get the description of good
    try:
        description = soup.select(".classified__description")[0].get_text(strip=True)
        features["description"] = description
    except:
        features["description"] = ""

    # Get the picture
    try:
        picture = soup.select(".classified-gallery__picture")[0].get("src")
        features["picture_url"] = picture
    except:
        features["picture_url"] = ""

    # Get feature names:
    name_features = soup.select(".classified-table__header")
    cleaned_name_features = [string_preprocessing(x.get_text(strip=True)) for x in name_features]

    # Get the feature values:
    value_features = soup.select(".classified-table__data")

    # Saving in a dictionary
    for key, value in zip(cleaned_name_features, value_features):
        if key == "prix":
            amount = value.select(".sr-only")[0].get_text(strip=True)
            features[key] = amount
        else:
            features[key] = value.get_text(strip=True)

    return features




def get_urls_from_searchpages(type_of_good, region, filter_opt=True):
    '''
    :param type_of_good (string): 'appartement' or 'immeuble-de-rapport'
    :param region (string): region that will be scraped
    :param filter_opt (boolean): True : Filter out the properties with several prices or prices ranges in title
    :return:
    '''

    # Global variables
    all_urls = []
    all_prices = []
    all_postcodes = []
    all_cities = []

    # Getting the total amount of search pages found with specific query
    full_url = return_full_url(type_of_good, region)
    nb_pages = get_nb_pages(full_url)

    for nb_page in range(1, nb_pages + 1):
        search_url = return_full_url(type_of_good, region, nb_page)
        # Per page getting urls and linked info
        try:
            urls, prices, postcodes, cities = get_individual_urls(search_url)

        # https://selenium-python.readthedocs.io/api.html
        except WebDriverException:
            pass
        # Updating the global lists
        all_urls += urls
        all_prices += prices
        all_postcodes += postcodes
        all_cities += cities

    return all_urls, all_prices, all_postcodes, all_cities


def scrape_features_from_search_urls(type_of_good, region, list_codes_already_scraped):
    ''' Returns the features scraped for 1 region

    :param type_of_good (string): 'appartement' or 'immeuble-de-rapport'
    :param region (string): region that will be scraped
    :param list_codes_already_scraped (list): list of codes of houses that have already been scraped before
        in order to avoid scraping them again.
    :return (list of dict): Returns a list of dictionnaries containing the features of individual properties
    '''


    # Get url of immoweb for first Search page
    full_url = return_full_url(type_of_good, region)
    # Get list of result urls of across the nb_pages
    urls, prices, postcodes, cities = get_urls_from_searchpages(type_of_good, region)
    ## Get the features of the different found urls
    feature_list = []

    request_num = 1  # counter
    for url, price, postcode, city in zip(urls, prices, postcodes, cities):
        try:
            # Retrieve the immoweb code from the url in order to check
            # if the proporty hasn't been scraped before
            code_from_url = url.split('/')[-1].split('?')[0]
            if code_from_url not in list_codes_already_scraped:
                features = get_features_one_page(url, postcode, city, \
                                                 price, region, type_of_good)
                feature_list.append(features)
        except WebDriverException:
            pass

        # Following the scraping progression
        request_num += 1

        # Adding random sleep periode of few seconds to avoid overloading the server / being banned
        if request_num % 20 == 0:
            sleep(randint(3, 10))
        print(f'REGION de {region} Request: {request_num}, Part already scraped: ',\
                round((request_num / len(urls)) * 100, 2), '%')
        clear_output(wait=True)
    return feature_list


def scraping_diff_regions(type_of_good, list_of_regions, list_codes_already_scraped):
    ''' Returns a list of dictionnaries with the individual house features

    :param type_of_good (string): 'appartement' or 'immeuble-de-rapport'
    :param list_of_regions (string): list of regions that have to be scraped
    :param list_immoweb_code_already_done (string): list of codes of houses that have already been scraped before
        in order to avoid scraping them again.
    :return (list): list of dictionnaries with the individual house features
    '''

    # List that will contain all the feature dictionnaries scraped
    features_list = []

    # Today's date for name of saved files
    date = "{:%Y%m%d}".format(datetime.now())
    # We'll measure how much time is needed to scrape the different regions
    start_time = time()

    for region in list_of_regions:
        start_region = time()
        region_feature_list = scrape_features_from_search_urls(type_of_good, region, list_codes_already_scraped)
        features_list+=region_feature_list
        # Computing the time need to scrape the full region
        elapsed_time = time() - start_region
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'The Scraping action for the region of {region} has taken {hours}h-{minutes}min')
        # Saving Checkpoints for each Region
        # Converting list of dictionaries to a Pandas DataFrame
        region_property_features_df = pd.DataFrame(region_feature_list)
        with open(f'Saved_Variables/{date}_{region}_{type_of_good.replace("-","_")}_features.pkl', 'wb') as f:
            pickle.dump(region_property_features_df, f)

    # Computing the amount of time taken by the scraping action
    end_time = time()
    scraping_time = (end_time - start_time)
    hours, rem = divmod(scraping_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f'The Total Scraping action has taken {hours}h-{minutes}min')

    # Converting the list of dictionnaries to a Pandas DataFrame
    property_features_df = pd.DataFrame(features_list)

    # Saving the Pandas DataFrame in Folder Saved_Variables
    with open(f'Saved_Variables/{date}_all_regions_{type_of_good.replace("-","_")}_features.pkl', 'wb') as f:
        pickle.dump(property_features_df, f)
    return features_list






