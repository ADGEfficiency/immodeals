import pickle
import pandas as pd
import json
import base64

## LOADING DATA
# Loading DataFrames
with open('../Saved_Variables/20201122_BUILDINGS_df_DASH.pkl', 'rb') as f:
    buildings = pickle.load(f)

with open('../Saved_Variables/20201122_FLATS_df_DASH.pkl', 'rb') as f:
    flats = pickle.load(f)

belgian_municipalities = json.load(open("communes-belges.geojson", 'r'))

with open('cp2ins.pkl', 'rb') as f:
    cp2ins = pickle.load(f)

with open('oldcp2ins.pkl', 'rb') as f:
    oldcp2ins = pickle.load(f)

# Adding type_of_good columns and joining the 2 dataframes
buildings['type_of_good'] = 'building'
flats['type_of_good'] = 'flat'
all_goods = pd.concat([buildings, flats], ignore_index=True)



## DATA PREPA
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

def prepa_energy_class(df):
    df_new = df.copy()
    # Removing the goods where the energy class wasn't given or is null
    df_new = df_new[df_new["classe_energetique" ]!='Non communiqué']

    # Cleaning notation as C_D, ...
    df_new['classe_energetique'] = df_new['classe_energetique'].apply(lambda x: clean_classe_en(x) if \
        isinstance(x, str) else x)

    # Ordered Categorical Type
    energy_classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    ordered_classes = pd.api.types.CategoricalDtype(ordered=True, categories=energy_classes)
    df_new['classe_energetique'] = df_new['classe_energetique'].astype(ordered_classes)

    return df_new

def prepa_surface(df, feature_name):
    df_new = df.copy()
    # Formatting the surface_habitable, in order to get the integer representing the area
    df_new[feature_name] = df_new[feature_name].apply(lambda x: x.split('m²')[0] \
        if isinstance(x, str) else x)
    df_new[feature_name] = df_new[feature_name].astype('int')
    return df_new

def prepa_revenu_cadastral(df):
    df_new = df.copy()
    # Formatting the strings in the correct way (to be able to convert them afterwards to integers)
    df_new['revenu_cadastral'] = df_new['revenu_cadastral'].apply(lambda x: x.split("€")[-2] \
        if isinstance(x, str) else x)
    # Converting to integers
    df_new['revenu_cadastral'] = df_new['revenu_cadastral'].astype('int')
    return df_new

def prepa_construction_yr(df, agg_type='count'):
    df_new = df.copy()

    # Removing the null values
    df_new = df_new[df_new["annee_de_construction"].notnull()]

    # Changing the datatype
    df_new["annee_de_construction"] = df_new["annee_de_construction"].astype('int')

    # Help column for the groupby every 10 years
    df_new['decades'] = df_new['annee_de_construction'].apply(lambda x: (x // 10) * 10)

    if agg_type == 'count':
        # Grouping by decades
        df_grp_decades = df_new.groupby('decades').count()
        prop_series = (df_grp_decades['annee_de_construction'] / df_new.shape[0])
        return df_new, prop_series

    elif agg_type == 'mean':
        df_grp_decades = df_new.groupby('decades').mean()
        return df_grp_decades

def prepa_state(df):
    state_classes = ['Excellent état', 'Fraîchement rénové', 'Bon', 'À rafraîchir', 'À rénover']
    ordered_classes = pd.api.types.CategoricalDtype(ordered=True, categories=state_classes)
    df['etat_du_batiment'] = df['etat_du_batiment'].astype(ordered_classes)
    return df


def prepa_3_regions(province):
    Flanders = ['anvers', 'limbourg', 'flandre-occidentale', 'flandre-orientale', 'brabant-flamand']
    Wallonia = ['hainaut', 'luxembourg', 'liege', 'namur', 'brabant-wallon']

    if province in Flanders:
        return 'Flanders'
    elif province in Wallonia:
        return 'Wallonia'
    else:
        return 'Brussels'

def prep_price_pr_sqrmeter(df):
    # Adding price/m2 feature :
    # Remove all missing values for the living area
    df = df[df['surface_habitable'].notnull()]
    # Prepare the living surface
    df = prepa_surface(df, 'surface_habitable')
    # Add a column : price_per_sqr_m
    df['price_per_sqr_m'] = df['price'].astype('int')/df['surface_habitable']
    df['price_per_sqr_m'] = df['price_per_sqr_m'].astype('int')
    return df

def fill_na_by_zeros(df, type_of_good):
    filled_df = df.copy()
    filled_df["double_vitrage"] = filled_df["double_vitrage"].apply(lambda x: 1 if x == 'Oui' else 0)
    if type_of_good == 'flats':
        col_fill = ['toilettes', 'salles_de_douche', 'parkings_exterieurs', 'parkings_interieurs']
        filled_df[col_fill] = filled_df[col_fill].fillna(0)

        col_2_number = ['terrasse', 'cave', 'ascenseur']
        for col in col_2_number:
            filled_df[col] = filled_df[col].apply(lambda x: 1 if x == 'Oui' else 0)
    return filled_df

def prepa_features_heatmap(type_of_good, missing_values, outliers, df, feature_x, feature_y, max_rooms, max_baths):
    features = [feature_x, feature_y]

    # Fill with 0 as asumption for certain features
    df = fill_na_by_zeros(df, type_of_good)

    if missing_values == 'without':
        df = df[df[feature_x].notnull()]
        df = df[df[feature_y].notnull()]

        for feature_name in features:
            if 'surface' in feature_name:
                df = prepa_surface(df, feature_name)
                df[feature_name] = df[feature_name].astype('int')
            elif feature_name == 'classe_energetique':
                df = prepa_energy_class(df)
            elif feature_name == 'etat_du_batiment':
                df = prepa_state(df)
            elif feature_name == 'chambres':
                df[feature_name] = df[feature_name].astype('int')
                df = df[df[feature_name] <= max_rooms]
            elif feature_name == 'salles_de_bains':
                df[feature_name] = df[feature_name].astype('int')
                df = df[df[feature_name] <= max_baths]
            elif feature_name == 'facades':
                df[feature_name] = df[feature_name].astype('int')
                df = df[df[feature_name] <= 4]

    return df

def prepa_geojson_file(belgian_municipalities):
    # Adding an id (comming from the properties), that will be used for the geomapping
    for feature in belgian_municipalities['features']:
        feature['id'] = feature['properties']['nsi']

    return belgian_municipalities

def convert_cp2ins(cp):
    if cp in cp2ins.keys():
        return cp2ins[cp]
    elif cp in oldcp2ins.keys():
        return oldcp2ins[cp]
    elif cp[:2] == '20':
        # after checking : '2020', '2018', '2030' not found but related to 2000
        return cp2ins['2000']
    elif cp == "3798":
        return "73075"
    else:
        return


def prepa_df_choropleth(df):
    new_df = df.copy()

    # adding price/m2
    new_df = prep_price_pr_sqrmeter(new_df)
    new_df['price'] = new_df['price'].astype('int')
    # grouping by group_feature
    grouped_df = new_df.groupby('postcode').median().reset_index()

    # adding the id column for the link with the geojson file
    grouped_df['id'] = grouped_df['postcode'].apply(lambda x: convert_cp2ins(x))

    return grouped_df

#def encode_image(image_file):
#    encoded = base64.b64encode(open(image_file, 'rb').read())
#    return 'data:image/png;base64,{}'.format(encoded.decode())


## GRAHP HELPER FUNCTIONS
def help_barchart(type_of_good, na_values, outliers_values):
    if type_of_good == 'buildings':
        df = buildings.copy()
    elif type_of_good == 'flats':
        df = flats.copy()

    if na_values == 'without':
        df = df[df['classe_energetique'].notnull()]
        df_new = prepa_energy_class(df)

    return df_new

def prepa_heating(df):
    df_new = df.copy()
    df_new['type_de_chauffage'] = df_new['type_de_chauffage'].fillna('not_given')
    return df_new


def help_histogram(type_of_good, na_values, outliers_values, feature):
    if type_of_good == 'buildings':
        df = buildings
    elif type_of_good == 'flats':
        df = flats

    feature_name = feature.split(" ")[0]
    if na_values == 'without':
        df = df[df[feature_name].notnull()]
        if 'surface' in feature_name:
            df = prepa_surface(df, feature_name)
        elif 'revenu' in feature_name:
            df = prepa_revenu_cadastral(df)

        df[feature_name] = df[feature_name].astype('int')

    return df


def help_yr_barchart(type_of_good):
    if type_of_good == 'buildings':
        df = buildings
    elif type_of_good == 'flats':
        df = flats

    _, yr_series = prepa_construction_yr(df)

    return yr_series

def help_boxplots_region_price(na_values, outliers_values):
    df = all_goods.copy()

    # Remove all missing values for the living area
    df = df[df['surface_habitable'].notnull()]
    # Prepare the living surface
    df = prepa_surface(df, 'surface_habitable')
    # Add a column : price_per_sqr_m
    df['price_per_sqr_m'] = df['price'].astype('int')/df['surface_habitable']
    df['price_per_sqr_m'] = df['price_per_sqr_m'].astype('int')


    if na_values == 'without':
        # Remove missing values for the current feature
        df = df[df['region'].notnull()]
    return df

def help_2D_boxplots(type_of_good, na_values, outliers_values, feature):
    if type_of_good == 'buildings':
        df = buildings
    elif type_of_good == 'flats':
        df = flats

    # Remove all missing values for the living area
    df = df[df['surface_habitable'].notnull()]
    # Prepare the living surface
    df = prepa_surface(df, 'surface_habitable')
    # Add a column : price_per_sqr_m
    df['price_per_sqr_m'] = df['price'].astype('int')/df['surface_habitable']
    df['price_per_sqr_m'] = df['price_per_sqr_m'].astype('int')


    if na_values == 'without':
        # Remove missing values for the current feature
        df = df[df[feature].notnull()]
        if feature == 'classe_energetique':
            df = prepa_energy_class(df)
        elif feature == 'etat_du_batiment':
            df = prepa_state(df)
        elif feature in ['chambres', 'salles_de_bains']:
            df[feature] = df[feature].astype('int')
            df = df[df[feature] < 20]

    return df


def help_scatter_and_countour_plots(type_of_good, feature):
    # Changing the limits of the axis with conditions
    if type_of_good == 'buildings':
        y_lim = 1200000

        if feature == 'revenu_cadastral':
            x_lim = 4500
        elif feature == 'surface_habitable':
            x_lim = 700
        else:
            x_lim = 3000

    elif type_of_good == 'flats':
        y_lim = 500000

        if feature == 'revenu_cadastral':
            x_lim = 2500
        elif feature == 'surface_habitable':
            x_lim = 250
        else:
            x_lim = 700

    return x_lim, y_lim

def help_heatmap(type_of_good, missing_values, outliers, feature_x, feature_y):
    if type_of_good == 'buildings':
        df = buildings
        max_rooms = 20
        max_baths = 12
    elif type_of_good == 'flats':
        df = flats
        max_rooms = 5
        max_baths = 5

    # Cleaning the feature name in case for ex: "toilettes (only for flats)""
    feature_x = feature_x.split(" ")[0]
    feature_y = feature_y.split(" ")[0]

    # Feature Engineering adding price per sqrmeters
    df = prep_price_pr_sqrmeter(df)

    # Prepa features for heatmap
    df = prepa_features_heatmap(type_of_good, missing_values, outliers, df, feature_x, feature_y, max_rooms, max_baths)

    return df

def help_choropleth(type_of_good, outliers):
    # NB: no choice of outliers as those are cleaned when computing the price/m2
    if type_of_good == 'buildings':
        df = buildings
    elif type_of_good == 'flats':
        df = flats

    geojson_file = prepa_geojson_file(belgian_municipalities)
    prepared_df = prepa_df_choropleth(df)

    return geojson_file, prepared_df



''' 

def data_2_options(bld_df, flat_df, type_of_good, na_values, outliers):
    col_na = []
    if type_of_good == 'buildings':
        df = bld_df.copy()
        col_na = ['salles_de_bains', 'chambres', 'revenu_cadastral', 'type_de_chauffage', 'surface_du_terrain', \
                 'surface_habitable', 'etat_du_batiment', 'facades', 'classe_energetique']
        col_fill = ['double_vitrage']

    elif type_of_good == 'flats':
        df = flat_df.copy()
        col_na = ['salles_de_bains', 'chambres', 'revenu_cadastral', 'type_de_chauffage', 'surface_habitable',\
                  'etat_du_batiment', 'facades', 'classe_energetique', 'toilettes', 'nombre_d_etages', 'etage']
        col_fill = ['double_vitrage', 'terrasse', 'salles_de_douche',\
                    'parkings_exterieurs', 'parkings_interieurs', 'cave', 'ascenseur']

    if na_values == 'without':
        df_na = df[df[col_na].notnull()]
        df_na = fill_na_values(df_na)

    elif na_values == 'cleaned':
        df_na = clean_data(df)



    if outliers == 'without':
        df_na = remove_outliers(df_na)

    return df_na

'''