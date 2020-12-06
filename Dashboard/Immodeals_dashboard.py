import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pickle
import json
import pandas as pd
from dash_helper_functions import *


## VARIABLES
# Loading DataFrames
with open('../Saved_Variables/20201122_BUILDINGS_df_DASH.pkl', 'rb') as f:
    buildings = pickle.load(f)

with open('../Saved_Variables/20201122_FLATS_df_DASH.pkl', 'rb') as f:
    flats = pickle.load(f)


hist_col = ['price', 'revenu_cadastral', 'surface_habitable', 'surface_du_terrain (only for buildings)']
dropbox_col = ['region', 'classe_energetique', 'etat_du_batiment', 'chambres', 'salles_de_bains']
contour_col = ['revenu_cadastral', 'surface_habitable', 'surface_du_terrain (only for buildings)']
heatmap_col = ['region', 'classe_energetique', 'chambres', 'salles_de_bains', 'facades', 'etat_du_batiment',
               'type_de_chauffage', 'toilettes (only for flats)', 'salles_de_douches (only for flats)',
               'terrasse (only for flats)', 'cave (only for flats)', 'ascenseur (only for flats)',
               'parkings_exterieurs (only for flats)', 'parkings_interieurs (only for flats)']



### DASH APPLICATION
app = dash.Dash()

################################################################################
############################## THE LAYOUTS
################################################################################
### MAIN DIV
app.layout = \
html.Div(className='globalDiv', children=[
    html.Div(className='mainDiv', children=[
        html.H1(children='Dashboard Immodeals'),
        ## LEFT DIV
        html.Div(className='leftDiv', children=['Left Side DIV 1/5 width',
            # Div for the selection of the dataset loading options
            html.Div(className='optionsDiv', children=[
                # Dropdown for the choice of the dataset
                html.Label('Choose type of data'),
                html.Div([
                    dcc.Dropdown(id='global_data',
                                 options=[
                                     {'label': 'Buildings', 'value': 'buildings'},
                                     {'label': 'Flats', 'value': 'flats'}
                                 ],
                                 value='buildings'
                                 )
                ], style={'height': 70,'padding':10}),

                # Radio Items for option selection for missing values
                html.Div(children=[
                    html.Label('Missing Values'),
                    html.Div(children=[
                        dcc.RadioItems(id='missing_values',
                                       options=[
                                           {'label': 'Without', 'value': 'without'},
                                           {'label': 'Cleaned', 'value': 'cleaned'}
                                       ],
                                       value='without'
                                       )
                    ], style={'padding-top': 10, 'padding-bottom': 20}),

                    # Radio Items for option selection regarding the outliers
                    html.Label('Outliers'),
                    html.Div(children=[
                        dcc.RadioItems(id='outliers',
                                       options=[
                                           {'label': 'With', 'value': 'with'},
                                           {'label': 'Without', 'value': 'without'}
                                       ],
                                       value='with'
                                       )
                    ], style={'padding-top': 10})
                ], style={'height':30, 'padding-left': 30}),

            ], style={'border':'2px black solid', 'padding':5, 'height': 300}),
            # Div for the horizontal barchart in function of the regions
            html.Div(className='sidebarDiv', children=[
                html.I('Amount of good/region'),
                dcc.Graph(id='region_barchart'),
                html.I('Repartition across energy classes'),
                dcc.Graph(id='en_class_barchart')
            ], style={'border': '2px black solid', 'padding':5})
        ], style={'border': '2px black solid', 'padding': 10,
                  'width': '17%', 'height': '98%', 'display': 'inline-block', 'vertical-align': 'top'}),

        ## RIGHT DIV
        html.Div(className='rightDiv', children=['Right Side DIV 4/5 width',
            # Div with the displays of how many lines are taken into account (function of initial selection)
            html.Div(className='displaysDiv', children=['Displays Div'
            ], style={'border': '2px black solid', 'padding': 5}),

            # Div with the univariate graphs: histogram/boxplots with possible selection of outliers
            html.Div(className='oneDimDiv', children=[
                html.Div(className='histBoxGraph', children=[
                    html.Div(children=[
                        html.Div(children=[
                            dcc.Dropdown(id='xhist',
                                    options=[{'label': i, 'value': i} for i in hist_col],
                                    value='price'
                         )
                        ],style={'width':'40%', 'display':'inline-block', 'padding-top':10}),

                        html.Div(children=[
                            html.I("Max range"),
                            dcc.Input(id='maxRange', value=900000,type='number')
                        ],style={'width':'28%', 'display':'inline-block', 'vertical-align':'top', 'padding-left':10,
                                 'padding-top': 5}),
                        html.Div(children=[
                            html.I("Size of Bins"),
                            dcc.Input(id='sizeBin', value=10000, type='number')
                        ], style={'width': '28%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-left': 10,
                                  'padding-top': 5})
                    ], style={}),

                    dcc.RadioItems(id='xhist_type',
                         options = [{'label': i, 'value': i} for i in ['linear', 'log']],
                         value = 'linear',
                         labelStyle={'display' : 'inline-block'}
                        ),

                    dcc.Graph(id='histogram'),
                    dcc.Graph(id='boxplot')
                ], style={'border': '2px black solid', 'padding': 5,
                          'width': '54%', 'display': 'inline-block'}),
                html.Div(className='outlierInfo', children=['Info Outliers',
                    html.Div([
                        #html.Img(id='hover-image', src='children', height=300)
                        html.Pre(id="hover-image", style={'padding': 15, 'whiteSpace': 'pre-line', 'height':300,
                                                          'whiteSpace': 'pre-line'})
                    ], style={'paddingTop': 10}),
                    html.Div(
                        html.Pre(id="hover-data", style={'padding': 15, 'whiteSpace': 'pre-line'}),
                    style={})






                ], style={'border': '2px black solid', 'padding': 5, 'width': '43%','display': 'inline-block',
                          'vertical-align': 'top'})

            ],style={'border':'2px black solid', 'padding':5}),

            # With Repartition over build years
            html.Div(children=[
                dcc.Graph(id='yearBarchart')
            ], style={}),

            # Div with the Bivariate barcharts and boxplots
            html.Div(className='twoDimDiv', children=[
                html.Div(children=[
                    dcc.Dropdown(id='xboxplot',
                                options=[{'label': i, 'value': i} for i in dropbox_col],
                                value='classe_energetique'
                                 ),
                ], style={'width': '20%'}),

                dcc.Graph(id='2D_boxplots')
            ], style={'border': '2px black solid', 'padding': 5}),

        ], style={'border': '2px black solid', 'padding':10, 'width':'79%', 'display':'inline-block'})

    ], style={'border': '2px black solid', 'padding':10, 'width':'98%'}),
    html.Div(className='lowerMainDiv', children=[
        # Div with Scatterplot
        html.Div(className='scatterDiv', children=[
            html.Div(children=[
                dcc.Dropdown(id='xcontour',
                             options=[{'label': i, 'value': i} for i in contour_col],
                             value='revenu_cadastral'
                             )
            ], style={'width': '20%'}),

            html.Div(children=[
                dcc.Graph(id='scatterPlot')
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div(children=[
                dcc.Graph(id='contourPlot')
            ], style={'width': '48%', 'display': 'inline-block'}),

        ], style={'border': '2px black solid', 'padding': 5}),

        html.Div(children=[
            dcc.Graph(id='regionPriceSqrMBoxplot')
        ], style={}),

        html.Div(className='mapsDiv', children=[
            html.Div(className='densityheatmapDiv', children=[
                html.Div(className='heatDropdownsDiv', children=[
                    html.Div(children=[
                        html.Label('Choose x feature'),
                        dcc.Dropdown(id='xheatmap',
                                     options=[{'label': i, 'value': i} for i in heatmap_col],
                                     value='region'
                                     ),
                    ], style={'display': 'inline-block', 'width': '48%', 'padding': 5}),
                    html.Div(children=[
                        html.Label('Choose y feature'),
                        dcc.Dropdown(id='yheatmap',
                                     options=[{'label': i, 'value': i} for i in heatmap_col],
                                     value='classe_energetique'
                                     ),
                    ], style={'display': 'inline-block', 'width': '48%', 'padding': 5})
                ], style={}),
                dcc.Graph(id='densityheatmapPlot')
            ], style={'display': 'inline-block', 'width': '48%', 'border': '2px black solid', 'vertical-align': 'top'}),

            html.Div(className='heatmapDiv', children=[
                dcc.Graph(id='heatmapPlot')

            ], style={'display': 'inline-block', 'width': '48%', 'border': '2px black solid', 'vertical-align': 'bottom'})
        ], style={'border': '2px black solid'}),

        html.Div(id='choroplethDiv', children=[
            dcc.Graph(id='choroplethGraph')
        ], style={})


####################
# chloropleth graphs



    ], style={})
], style={})

################################################################################
############################## THE CALLBACKS ##################################
################################################################################

## LEFT SIDE BAR
@app.callback(
    Output('region_barchart', 'figure'),
    [Input('global_data', 'value')]
)
# Left Bar Chart with the amount of goods/region
def update_region_barchart(type_of_good):
    if type_of_good == 'buildings':
        df = buildings['region'].value_counts(ascending=True)
    elif type_of_good == 'flats':
        df = flats['region'].value_counts(ascending=True)

    traces = [
        go.Bar(
            x=df.values,
            y=df.index,
            orientation='h'
        )
    ]
    return {
        'data': traces,
        'layout': go.Layout(
            yaxis={'tickangle': 45},
            margin={'l':100, 'r':5, 'b':70, 't':65, 'pad':4}
        )
    }

@app.callback(
    Output('en_class_barchart', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value')]
)
def update_energyclass_barchart(type_of_good, na_values, outliers_values):
    df = help_barchart(type_of_good, na_values, outliers_values)
    df_en = df.groupby('classe_energetique').count()['price']/df.shape[0]

    traces = [
        go.Bar(
            x=df_en.values,
            y=df_en.index,
            orientation='h'
        )
    ]
    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l':20, 'r':5, 'b':25, 't':5, 'pad':4}
        )
    }


## UNIVARIATE PLOTS
# Histogram
@app.callback(
    Output('histogram', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xhist','value'), Input('sizeBin', 'value'), Input('maxRange', 'value'),
     Input('xhist_type', 'value')]
)
def update_histogram(type_of_good, na_values, outliers_values, feature, binsize, maxrange, scale):
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    feature_name = feature.split(" ")[0]

    traces = [
        go.Histogram(
            x=df[feature_name],
            xbins=dict(start=df[feature_name].min(),end=maxrange, size=binsize)
        )
    ]
    return{
        'data': traces,
        'layout': go.Layout(
            margin={'l': 35, 'r': 25, 'b': 35, 't': 5, 'pad': 4},
            xaxis={'title': feature_name, 'type': scale}
        )
    }

# BOXPLOT
@app.callback(
    Output('boxplot', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xhist','value'), Input('sizeBin', 'value'), Input('maxRange', 'value'),
     Input('xhist_type', 'value')]
)
def update_boxplot(type_of_good, na_values, outliers_values, feature, binsize, maxrange, scale):
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    feature_name = feature.split(" ")[0]

    traces = [
        go.Box(
            x=df[feature_name],
            boxpoints='outliers'
        )
    ]
    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 5, 'r': 25, 'b': 35, 't': 30, 'pad': 4},
            xaxis={'title': feature_name, 'type': scale},
            height=200,
            hovermode='closest'
        )
    }

# Outlier info
@app.callback(Output('hover-image', 'children'),
              [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
               Input('xhist','value'), Input('boxplot', 'hoverData')])
def return_outlier_image(type_of_good,  na_values, outliers_values, feature, hoverData):
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    index = hoverData["points"][0]['pointIndex']
    if index == 0:
        return "No image found, Hover an outlier point on the Boxplot on the left"
    else:
        image_path=df.loc[index, 'picture_url']
        return f'Image url : {image_path}'

@app.callback(Output('hover-data', 'children'),
              [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
               Input('xhist','value'), Input('boxplot', 'hoverData')])
def return_outlier_description(type_of_good,  na_values, outliers_values, feature, hoverData):
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    index = hoverData["points"][0]['pointIndex']

    if index == 0:
        return "Hover an outlier point on the Boxplot on the left"
    else:
        return f'Price of the property is : {df.loc[index,"price"]}\n Description: \n {df.loc[index,"description"]}'

# BARCHART
@app.callback(
    Output('yearBarchart', 'figure'),
    [Input('global_data', 'value')]
)
def update_year_barchart(type_of_good):
    yr_series = help_yr_barchart(type_of_good)

    traces = [
        go.Bar(
            x=yr_series.index,
            y=yr_series.values
        )
    ]
    return{
        'data': traces,
        'layout': go.Layout(
            margin={'l': 35, 'r': 30, 'b': 50, 't': 25, 'pad': 4},
            height=250,
            xaxis={'tickangle': 45}
        )
    }

## BIVARIATE PLOTS
# BOXPLOTS comparison price/m2 per region
@app.callback(
    Output('regionPriceSqrMBoxplot', 'figure'),
    [Input('missing_values', 'value'), Input('outliers', 'value')]
)
def update_boxplot_region_price(na_values, outliers_values):
    df = help_boxplots_region_price(na_values, outliers_values)
    traces = []
    for type_good in df['type_of_good'].unique():
        df_plot = df[df['type_of_good']==type_good]
        traces.append(
            go.Box(
            x=df_plot['region'],
            y=df_plot['price_per_sqr_m'],
            boxpoints=False,
            name=type_good
            )
        )

    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 35, 'r': 25, 'b': 80, 't': 30, 'pad': 4},
            yaxis=go.layout.YAxis(
                title='€/m2',
                range=[0,20000],
                zeroline=False,
            ),
            boxmode='group',
            height=300
        )
    }

# 2D-BOXPLOTS
@app.callback(
    Output('2D_boxplots', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xboxplot','value')]
)
def update_2D_boxplots(type_of_good, na_values, outliers_values, feature):
    df = help_2D_boxplots(type_of_good, na_values, outliers_values, feature)
#    if feature in ['classe_energetique', 'etat_du_batiment']:
#        type='linear'
#    else:
#        type = 'category'

    traces = [
        go.Box(
            x=df[feature],
            y=df['price_per_sqr_m'],
            boxpoints=False
        )
    ]
    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 35, 'r': 25, 'b': 80, 't': 30, 'pad': 4},
 #           xaxis={'categoryorder':'array', 'categoryorder': {'classe_enegetique':['A', 'B', 'C', 'D', 'E', 'F', 'G'],
 #                                                             'etat_du_batiment': ['Excellent état', 'Fraîchement rénové',
 #                                                                                  'Bon', 'À rafraîchir', 'À rénover']}},
 #           xaxis={'type':type},
            yaxis={'title': '€/m2', 'range':[0,15000]},
            height=350
        )
    }

# SCATTER PLOT
@app.callback(
    Output('scatterPlot', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xcontour','value')]
)
def update_scatterplot(type_of_good, na_values, outliers_values, feature):
    feature = feature.split(" ")[0]
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    df['price'] = df['price'].astype('int')
    # Sampling for the scatterplot:
    df_scatter = df.sample(1000)
    x_scatter = df_scatter[feature]
    y_scatter = df_scatter['price']

    # Dividing in one of the 3 mains regions of Belgium
    df_scatter['main_region'] = df_scatter['region'].apply(lambda x: prepa_3_regions(x))

    x_lim, y_lim = help_scatter_and_countour_plots(type_of_good, feature)

    traces = []
    for main_region in df_scatter['main_region'].unique():
        df_plot = df_scatter[df_scatter['main_region']==main_region]
        traces.append(
            go.Scatter(
                x=df_plot[feature],
                y=df_plot['price'],
                name=main_region,
                mode='markers',
                opacity=0.5,
                marker={'size': 7}
            )
        )

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis={'range' : [0, x_lim]},
            yaxis={'range' : [0, y_lim]},
            margin={'l': 45, 'r': 10, 'b': 35, 't': 20, 'pad': 4}
        )
    }

# CONTOUR PLOT
@app.callback(
    Output('contourPlot', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xcontour', 'value')]
)
def update_countourplot(type_of_good, na_values, outliers_values, feature):
    feature = feature.split(" ")[0]
    df = help_histogram(type_of_good, na_values, outliers_values, feature)
    df['price'] = df['price'].astype('int')
    x_values = df[feature]
    y_values = df['price']

    x_lim, y_lim = help_scatter_and_countour_plots(type_of_good, feature)

    traces = [go.Histogram2dContour(
        x=x_values,
        y=y_values,
        colorscale='Blues',
        xaxis='x',
        yaxis='y'
        )
    ]

    return {
        'data': traces,
        'layout': go.Layout(
            xaxis = {'range':[0, x_lim]},
            yaxis = {'range':[0, y_lim]},
            margin={'l': 55, 'r': 10, 'b': 35, 't': 20, 'pad': 4}
        )
    }

# HEATMAPS
@app.callback(
    Output('densityheatmapPlot', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xheatmap', 'value'), Input('yheatmap', 'value')]
)
def update_density_heatmap(type_of_good, missing_values, outliers, feature_x, feature_y):
    df = help_heatmap(type_of_good, missing_values, outliers, feature_x, feature_y)
    feature_x = feature_x.split(" ")[0]
    feature_y = feature_y.split(" ")[0]

    traces = [go.Histogram2d(
                   x=df[feature_x],
                   y=df[feature_y]
                    )
    ]

    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 85, 'r': 10, 'b': 85, 't': 20, 'pad': 4}
        )
    }

@app.callback(
    Output('heatmapPlot', 'figure'),
    [Input('global_data', 'value'), Input('missing_values', 'value'), Input('outliers', 'value'),
     Input('xheatmap', 'value'), Input('yheatmap', 'value')]
)
def update_heatmap(type_of_good, missing_values, outliers, feature_x, feature_y):
    df = help_heatmap(type_of_good, missing_values, outliers, feature_x, feature_y)
    feature_x = feature_x.split(" ")[0]
    feature_y = feature_y.split(" ")[0]

    traces = [go.Heatmap(
                   z=df['price_per_sqr_m'],
                   x=df[feature_x],
                   y=df[feature_y],
                   hoverongaps=False
                    )
    ]

    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 85, 'r': 10, 'b': 85, 't': 20, 'pad': 4}
        )
    }


@app.callback(
    Output('choroplethGraph', 'figure'),
    [Input('global_data', 'value'), Input('outliers', 'value')]
)
def update_choropleth(type_of_good, outliers):
    geojson_file, prepared_df = help_choropleth(type_of_good, outliers)


    #fig = px.choropleth(prepared_df, locations='id', geojson=geojson_file, color='price_per_sqr_m',
    #                    scope='europe')
    #fig.update_geos(fitbounds="locations", visible=False)

    #return fig


    traces = [
        go.Choroplethmapbox(
            locations=prepared_df['id'].tolist(),  # Spatial coordinates
            geojson=geojson_file,
            z=prepared_df['price_per_sqr_m'].tolist(),  # Data to be color-coded
            #colorscale='Reds',
            colorbar_title="price/m2",
            visible=False
        )
    ]

    return {
        'data': traces,
        'layout': go.Layout(
            margin={'l': 85, 'r': 10, 'b': 85, 't': 20, 'pad': 4},
            geo_scope='europe',
            autosize=True
        )
    }




###################################################################################
if __name__ == '__main__':
    app.run_server(debug=True)