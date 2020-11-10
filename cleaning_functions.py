import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import unidecode
import math
import pickle


# Load file for translating postcode to longitudes and latitudes
code_2_lat_long = pd.read_csv('zipcode-belgium.csv', header=None)
code_2_lat_long.columns = ['postcode', 'city', 'long', 'lat']
code_2_lat_long['list_long_lat'] = code_2_lat_long.apply(lambda row: [row['long'], row['lat']], axis=1)
# Create dictionary with as Keys
code2coord = dict(zip(code_2_lat_long['postcode'], code_2_lat_long['list_long_lat']))


def clean_classe_en(class_en):
    ''' Returns a cleaned classe_energetique string

    :param class_en (string): string of classe_energetique for 1 sample
    :return: cleaned string of classe_energetique (removing '_', grouping types with '+')
    '''

    if '+' in class_en:
        # grouping A+ and A++ with A
        return class_en.split('+')[0]
    elif '_' in class_en:
        return class_en.split('_')[-1]
    elif 'Non' in class_en:
        # return unidecode.unidecode(class_en.lower())
        return 'not_given'
    else:
        return class_en


def quantiles_calc(df_new, col_name):
    ''' Returns a list with the quantiles for columns with name 'col_name'

    :param df_new (pd.DataFrame): a pandas DataFrame
    :param col_name (string): name of the columns, the quantiles will be computed
    :return (list): list of the quantiles computed on column col_name of DataFrame df_new
    '''
    quantiles = df_new[df_new[col_name].notnull()][col_name]\
                        .astype('int').quantile(q=[0.25,0.5,0.75])
    return list(quantiles)


def group_to_quantiles(x,quantiles):
    ''' Returns a number from 1 to 4 depending on appartenance to a specific quantile

    :param x (int): value needed to be set in one of the 4 quantiles
    :param quantiles (list): list of quantiles, the value will be compared to (typically returned by quantiles_calc()
    :return (int): quantile number associated to the given x value
    '''
    if x <= quantiles[0]:
        return 1
    elif x <= quantiles[1]:
        return 2
    elif x <= quantiles[2]:
        return 3
    else:
        return 4


def imputing_missing_val(df_new, df_type, list_col_names, strategy='most_frequent', na_imputer=None):
    ''' Returns: a DataFrame where the missing values of list_col_names have been imputed according to the given strategy

    :param df_new (pd.DataFrame): a DataFrame with some columns with missing values to be imputed
    :param df_type (string): "train" or "test", important as the imputer is fitted only on the train set
    :param list_col_names (string): list with column names that have to be imputed according to the same strategy
    :param strategy (string): desired strategy for imputation for examle: "most_frequent", "median", ...
    :param na_imputer (imputer): should be None if applied to train set and a fitter imputer if test set
    :return (DatFrame, Imputer): returns the updated DataFrame with imputed values and the Imputer used
    '''
    sub_df = df_new[list_col_names]

    # If we deal with the Train Set
    if df_type == "train":
        # Defining the Imputer
        imputer = SimpleImputer(strategy=strategy)
        imputed_sub_df = pd.DataFrame(imputer.fit_transform(sub_df))

    # in case, we deal with the Test Set
    else:
        imputer = na_imputer
        imputed_sub_df = pd.DataFrame(imputer.transform(sub_df))

    # Adding the column names back
    imputed_sub_df.columns = sub_df.columns

    df_new[list_col_names] = imputed_sub_df.values

    return df_new, imputer


# Cleaning Function for Buildings:
def cleaning_df(df, df_type='train', type_of_good='building', na_imputer_most_freq=None, na_imputer_median=None):
    ''' Returns a cleaned DataFrame

    :param df (pd.DataFrame): DataFrame to be cleaned
    :param df_type (string): type of DataFrame to be cleaned : "train" or "test"
    :param type_of_good (string): 'building' or 'apartment'
    :param na_imputer_most_freq (Imputer): None if df_type == "train" else imputer fitted on the train set
    :param na_imputer_median (Imputer): None if df_type == "train" else imputer fitted on the train set
    :return (pd.DataFrame): returns a cleaned DataFrame (cleaned string notations, no missing values, adapted datatypes)
    '''

    # First Column Selection
    if type_of_good == 'building':
        selected_cols = ['immoweb_code', 'price', 'region', 'postcode', 'description', 'type_de_zone_inondable',
                        'salles_de_bains', 'double_vitrage', 'chambres', 'revenu_cadastral', 'type_de_chauffage',
                        'surface_du_terrain', 'surface_habitable', 'etat_du_batiment', 'annee_de_construction',
                        'facades', 'classe_energetique']

        # List of columns for imputation : "most_frequent"
        list_col_names = ['salles_de_bains', 'etat_du_batiment', 'facades']

        # List of columns for imputation : "median"
        list_median_col = ['surface_habitable', 'annee_de_construction']

        # List of columns where missing values are filled with -1
        minus_one_cols = ['chambres', 'surface_du_terrain', 'revenu_cadastral']

        # List of columns to be converted to 'int'
        int_cols = ['surface_du_terrain', 'surface_habitable', 'chambres', 'revenu_cadastral', 'salles_de_bains',
                    'annee_de_construction', 'facades', 'price']


    elif type_of_good == 'apartment':
        selected_cols = ['immoweb_code', 'price', 'region', 'postcode', 'description', 'type_de_zone_inondable',
                         'salles_de_bains', 'double_vitrage', 'chambres', 'revenu_cadastral','type_de_chauffage',
                         'surface_habitable', 'etat_du_batiment', 'annee_de_construction', 'facades',
                         'classe_energetique', 'terrasse', 'salles_de_douche', 'toilettes', 'parkings_exterieurs',
                         'parkings_interieurs', 'cave', 'ascenseur', 'nombre_d_etages', 'etage']

        # List of columns for imputation : "most_frequent"
        list_col_names = ['salles_de_bains', 'toilettes', 'etat_du_batiment', 'facades', 'etage', 'nombre_d_etages']

        # List of columns for imputation : "median"
        list_median_col = ['surface_habitable', 'annee_de_construction']

        # List of columns where missing values are filled with -1
        minus_one_cols = ['chambres',  'revenu_cadastral']

        # List of columns to be converted to 'int'
        int_cols = ['surface_habitable', 'chambres', 'revenu_cadastral', 'salles_de_bains', 'salles_de_douche',
                    'toilettes', 'etage', 'nombre_d_etages', 'parkings_exterieurs', 'parkings_interieurs',
                    'annee_de_construction', 'facades', 'price']

    else:
        print('ERROR: type_of_good must be either "building" or "apartment')
        return

    # Checking that selected columns are in the given df
    try:
        df_new = df[selected_cols]
    except:
        # If not all preselected columns are in the given argument dataframe
        # The missing columns will be added filled with np.nan
        df_new = df.reindex(columns=selected_cols)


    ### CLEANING AND GROUPING SOME STRING EXPRESSIONS
    ## "type_de_zone_inondable" : 0 for non-flood zone and 1 otherwise
    # First convert the NaN values to 'Zone non inondable'
    df_new['type_de_zone_inondable'] = df_new['type_de_zone_inondable'].fillna('Zone non inondable')
    no_flood = ['Zone non inondable',
                'Bien immobilier situé tout ou en partie dans une zone riveraine délimitée']
    df_new['type_de_zone_inondable'] = df_new['type_de_zone_inondable'].apply(lambda x: \
                                                                                  0 if x in no_flood else 1)

    ## Revenu Cadastrale
    # clean first the non NaN values
    df_new['revenu_cadastral'] = df_new['revenu_cadastral'].apply(lambda x: x.split("€")[-2] \
        if isinstance(x, str) else x)

    ## Area to live [m2]
    df_new['surface_habitable'] = df_new['surface_habitable'].apply(lambda x: x.split('m²')[0] \
        if isinstance(x, str) else x)

    if type_of_good == 'building':
        ## Area of the ground [m2]
        df_new['surface_du_terrain'] = df_new['surface_du_terrain'].apply(lambda x: x.split('m²')[0] \
            if isinstance(x, str)  else x)

    # If dataset is Train set, we compute the quantiles
    # OTHERWISE if Test Set, we'll use the quantiles computed on the Train Set
    if df_type == 'train':
        qtl_cadastre = quantiles_calc(df_new, 'revenu_cadastral')

        if type_of_good == 'building':
            qtl_terrain = quantiles_calc(df_new, 'surface_du_terrain')

    df_new['revenu_cadastral'] = df_new['revenu_cadastral'].apply(lambda x: \
                                                                      group_to_quantiles(int(x), qtl_cadastre) \
                                                                          if isinstance(x, str) else x)
    if type_of_good == 'building':
        df_new['surface_du_terrain'] = df_new['surface_du_terrain'].apply(lambda x: \
                                                                          group_to_quantiles(int(x), qtl_terrain) \
                                                                              if isinstance(x, str) else x)

    ## etat_du_batiment
    # Group 'A rénover' and 'A restaurer' together as both are very similar terms
    df_new['etat_du_batiment'] = df_new['etat_du_batiment'].apply(lambda x: 'À rénover' if x == 'À restaurer' \
        else x)

    ## classe_energetique
    df_new['classe_energetique'] = df_new['classe_energetique'].apply(lambda x: clean_classe_en(x) if \
        isinstance(x, str) else x)

    ### FILLNA WTH CONSTANT VALUES
    ## value = -1
    df_new[minus_one_cols] = df_new[minus_one_cols].fillna(-1)

    ## value = 'not_given'
    not_given_cols = ['type_de_chauffage', 'classe_energetique', 'double_vitrage']
    df_new[not_given_cols] = df_new[not_given_cols].fillna('not_given')

    ## value = 0
    if type_of_good == 'apartment':
        zero_cols = ['terrasse', 'salles_de_douche', 'parkings_interieurs', 'parkings_exterieurs', 'cave', 'ascenseur']
        df_new[zero_cols] = df_new[zero_cols].fillna(0)

    ### GROUPING CATEGORIES
    ## Type of heating
    alternative = ['pellets', 'bois', 'solaire']
    df_new['type_de_chauffage'] = df_new['type_de_chauffage'].apply(lambda x: unidecode.unidecode(x.lower()) \
        if unidecode.unidecode(x.lower()) not in alternative else 'alternative')

    ### IMPUTATION STRATEGIES
    ## Apply imputation for facades, salles_de_bains, toilettes and etat_du_batiment
    df_new, imputer = imputing_missing_val(df_new, df_type, list_col_names, strategy='most_frequent', \
                                           na_imputer=na_imputer_most_freq)

    # Apply median imputation for the surface_habitable
    df_new, surf_imputer = imputing_missing_val(df_new, df_type, list_median_col, strategy='median', \
                                                na_imputer=na_imputer_median)

    ## TYPE CONVERSION
    df_new[int_cols] = df_new[int_cols].astype('int')

    return df_new


def encoding_df(cleaned_df, type="train", type_of_good='building'):
    ''' Returns: an DataFrame where Categorical variables are encoded

    :param cleaned_df (pd.DataFrame): DataFrame to be encoded
    :param type (string): 'train' or 'test'
    :param (string): 'building' or 'apartment'
    :return (pd.DataFrame): a DataFrame where the Categorical Features have been encoded
    '''
    ## double_vitrage
    cleaned_df['double_vitrage'] = cleaned_df['double_vitrage'].apply(lambda x: 1 if x == 'Oui' else 0)

    ## Label Encode etat_du_batiment
    keys = ['Excellent état', 'Fraîchement rénové', 'Bon', 'À rafraîchir', 'À rénover']
    values = [1, 2, 3, 4, 5]
    label_dict = dict(zip(keys, values))
    cleaned_df['etat_du_batiment'] = cleaned_df['etat_du_batiment'].apply(lambda x: label_dict[x] \
        if x in label_dict.keys() else 6)
    ## Label Encode classe_energetique
    keys_en = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'not_given']
    values_en = [1, 2, 3, 4, 5, 6, 7, -1]
    energy_dict = dict(zip(keys_en, values_en))
    cleaned_df['classe_energetique'] = cleaned_df['classe_energetique'].apply(lambda x: energy_dict[x] \
        if x in energy_dict.keys() else -1)
    # OHE type_de_chauffage
    OHE_type_de_chauffage = pd.get_dummies(cleaned_df["type_de_chauffage"], prefix='type_de_chauffage')
    cleaned_df = pd.concat([cleaned_df, OHE_type_de_chauffage], axis=1)
    cleaned_df.drop(['type_de_chauffage'], axis=1, inplace=True)

    ## Encoding the postcodes to longitude and latitude
    cleaned_df['long'] = cleaned_df['postcode'].apply(lambda x: code2coord[int(x)][0])
    cleaned_df['lat'] = cleaned_df['postcode'].apply(lambda x: code2coord[int(x)][1])

    if type_of_good == 'apartment':
        ## terrasse
        cleaned_df['terrasse'] = cleaned_df['terrasse'].apply(lambda x: 1 if x == 'Oui' else 0).astype('int')
        cleaned_df['cave'] = cleaned_df['cave'].apply(lambda x: 1 if x == 'Oui' else 0).astype('int')
        cleaned_df['ascenseur'] = cleaned_df['ascenseur'].apply(lambda x: 1 if x == 'Oui' else 0).astype('int')

    return cleaned_df



