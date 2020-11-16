import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from cleaning_functions import *


################################### PREPA UNIVARIATE PLOTS ###################################

def prepa_energy_class(df):
    df_no_null = df.copy()
    # Removing the goods where the energy class wasn't given or is null
    df_no_null = df_no_null[df_no_null["classe_energetique" ]!='Non communiqué']
    df_no_null = df_no_null[df_no_null["classe_energetique"].notnull()]

    # Cleaning notation as C_D, ...
    df_no_null['classe_energetique'] = df_no_null['classe_energetique'].apply(lambda x: clean_classe_en(x) if \
        isinstance(x, str) else x)

    # Ordered Categorical Type
    energy_classes = ['A' ,'B' ,'C' ,'D' ,'E' ,'F' ,'G']
    ordered_classes = pd.api.types.CategoricalDtype(ordered = True ,categories = energy_classes)
    df_no_null['classe_energetique'] = df_no_null['classe_energetique'].astype(ordered_classes)

    return df_no_null


def prepa_revenu_cadastral(df):
    df_no_null = df.copy()

    # Formatting the strings in the correct way (to be able to convert them afterwards to integers)
    df_no_null['revenu_cadastral'] = df_no_null['revenu_cadastral'].apply(lambda x: x.split("€")[-2] \
        if isinstance(x, str) else x)

    # Removing null_values
    df_no_null = df_no_null[df_no_null['revenu_cadastral'].notnull()]

    # Converting to integers
    df_no_null['revenu_cadastral'] = df_no_null['revenu_cadastral'].astype('int')

    return df_no_null


def prepa_surface(df, feature_name):
    df_no_null = df.copy()

    # Formatting the surface_habitable, in order to get the integer representing the area
    df_no_null[feature_name] = df_no_null[feature_name].apply(lambda x: x.split('m²')[0] \
        if isinstance(x, str) else x)

    # Filterning out the null value
    df_no_null = df_no_null[df_no_null[feature_name].notnull()]

    # Define as Integer
    df_no_null[feature_name] = df_no_null[feature_name].astype('int')

    return df_no_null


def prepa_construction_yr(df):
    df_no_null = df.copy()

    # Removing the null values
    df_no_null = df_no_null[df_no_null["annee_de_construction"].notnull()]

    # Changing the datatype
    df_no_null["annee_de_construction"] = df_no_null["annee_de_construction"].astype('int')

    # Help column for the groupby every 10 years
    df_no_null['decades'] = df_no_null['annee_de_construction'].apply(lambda x: (x // 10) * 10)

    # Grouping by decades
    df_grp_decades = df_no_null.groupby('decades').count()

    prop_series = (df_grp_decades['annee_de_construction'] / df_no_null.shape[0])

    return df_no_null, prop_series


def prepa_4_not_null(df, feature_name):
    df_no_null = df.copy()

    # Removing the null values
    df_no_null = df_no_null[df_no_null[feature_name].notnull()]

    # Changing the datatype
    df_no_null[feature_name] = df_no_null[feature_name].astype('int')

    return df_no_null


def prepa_rooms_etc(df, type_of_good):
    df_no_null = df.copy()
    rooms_no_null = df_no_null[df_no_null["chambres"].notnull()]
    rooms_no_null["chambres"] = rooms_no_null["chambres"].astype('int')
    bathrooms_no_null = df_no_null[df_no_null["salles_de_bains"].notnull()]
    bathrooms_no_null["salles_de_bains"] = bathrooms_no_null["salles_de_bains"].astype('int')

    if type_of_good == "apartment":
        toilets_no_null = df_no_null[df_no_null["toilettes"].notnull()]
        toilets_no_null["toilettes"] = toilets_no_null["toilettes"].astype('int')
        showers_no_null = df_no_null[df_no_null["salles_de_douche"].notnull()]
        showers_no_null["salles_de_douche"] = showers_no_null["salles_de_douche"].astype('int')

        return rooms_no_null, bathrooms_no_null, toilets_no_null, showers_no_null

    elif type_of_good == "building":
        return rooms_no_null, bathrooms_no_null


def prepa_other_features(df):
    df_no_null = df.copy()

    # Double windows : if na => 0
    df_no_null["double_vitrage"] = df_no_null["double_vitrage"].apply(lambda x: 1 if x == 'Oui' else 0)

    # Facades
    facades_no_null = df_no_null[df_no_null['facades'].notnull()]
    # State of the good
    state_no_null = df_no_null[df_no_null['etat_du_batiment'].notnull()]
    # Type of heating
    heating_no_null = df_no_null[df_no_null['type_de_chauffage'].notnull()]

    return df_no_null, facades_no_null, state_no_null, heating_no_null


def prepa_terrace_cellar_lift(df, features):
    df_no_null = df.copy()
    # Asumption missing values = no occurance
    for f in features:
        df_no_null[f] = df_no_null[f].apply((lambda x: 1 if x == 'Oui' else 0)).astype('int')

    return df_no_null


################################### UNIVARIATE PLOTS ###################################

def hist_1D(df, feature_name, type_of_good,xlabel,xlim, bins=None):
    fig, ax = plt.subplots(figsize=[8,6])
    plt.ticklabel_format(style='plain', axis='x')
    sns.distplot(df[feature_name], kde=False, bins=bins)
    plt.title(f'Distribution of the {feature_name} of the {type_of_good} dataset')
    plt.xlim(0,xlim)
    plt.xlabel(xlabel)
    plt.ylabel('Count');


def boxplot_1D_outliers(df, feature_name, type_of_good, xlabel):
    fig, ax = plt.subplots(1, 2, figsize=[20, 4])

    # Subplot 1: with the outliers
    # ax1 = fig.add_subplot(1,2,1)
    g1 = sns.boxplot(data=df, x=feature_name, color="skyblue", ax=ax[0], showfliers=True)
    # ax.set_xticklabels(ax1.get_xticklabels(), rotation=30)
    g1.ticklabel_format(style='plain', axis='x')
    g1.set(xlabel=xlabel)
    g1.set(title=f'Boxplot of {feature_name} for the {type_of_good}s')

    # Subplot 2: without the outliers
    # ax2 = fig.add_subplot(1,2,2)
    g2 = sns.boxplot(data=df, x=feature_name, color="skyblue", ax=ax[1], showfliers=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    g2.ticklabel_format(style='plain', axis='x')
    g2.set(xlabel=xlabel)
    g2.set(title=f'Boxplot of {feature_name} for the {type_of_good}s without outliers');

def count_plot_1D(df, feature_name, type_of_good):
    order = df[feature_name].value_counts().index
    fig, ax = plt.subplots(figsize=(10,4))
    ax = sns.countplot(x='region',data=df, order=order,color="skyblue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    plt.xlabel("")
    plt.title(f'Countplot on {feature_name} feature for the {type_of_good} dataset');

def proportion_barplot(df, feature_name, type_of_good, energy_class=False, ordered=True, size=[10, 4], \
                           tilt=30, ticks_scale=0.05, create_fig=True):
    if create_fig:
        plt.figure(figsize=size)

    # Preparing the relative frequencies
    n_points = df.shape[0]
    max_count = df[feature_name].value_counts().max()
    max_prop = (max_count / n_points)

    # generate the tick mark locations and nammes
    tick_props = np.arange(0, max_prop, ticks_scale)
    tick_names = ['{:0.2f}'.format(v) for v in tick_props]

    if energy_class == True:
        # Defining color palette explicitely for energy_class
        my_pal = {"A": "darkgreen", "B": "green", "C": "lightgreen", "D": "yellow", "E": "orange",
                      "F": "darkorange",
                      "G": "red"}
        sns.countplot(data=df, x=feature_name, palette=my_pal)
    else:
        if ordered:
            order = df[feature_name].value_counts().index
            ax = sns.countplot(data=df, x=feature_name, order=order, color="skyblue")
        else:
            ax = sns.countplot(data=df, x=feature_name, color="skyblue")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=tilt)
            plt.xlabel("")

    plt.yticks(tick_props * n_points, tick_names)
    plt.ylabel('proportion')
    plt.title(f'Proportions of each {feature_name} for the {type_of_good}');

def constr_yr_barplot(prop_series, type_of_good):
    fig, ax = plt.subplots(figsize=[15,5])
    prop_series.plot(kind='bar', color='skyblue')
    plt.xticks(rotation = 15)
    plt.xlabel('Decades')
    plt.ylabel('Proportion')
    plt.title(f'Barplot for the construction year of the {type_of_good} dataset');


def draw_room_repartition(list_df, type_of_good, size):
    plt.figure(figsize=size)

    if type_of_good == "building":
        features = ["chambres", "salles_de_bains"]
        n_rows = 1

    elif type_of_good == "apartment":
        features = ["chambres", "salles_de_bains", "toilettes", "salles_de_douche"]
        n_rows = 2

    for i, df_and_name in enumerate(zip(list_df, features)):
        plt.subplot(n_rows, 2, i + 1)
        proportion_barplot(df_and_name[0], df_and_name[1], type_of_good, energy_class=False, ordered=False, \
                               create_fig=False)

def draw_other_features(list_df, list_features, type_of_good, size=[20, 10], nrows=2, ncols=2):
    plt.figure(figsize=size)

    features = list_features

    for i, df_and_name in enumerate(zip(list_df, features)):
        plt.subplot(nrows, ncols, i + 1)
        proportion_barplot(df_and_name[0], df_and_name[1], type_of_good, energy_class=False, ordered=False, \
                               create_fig=False)