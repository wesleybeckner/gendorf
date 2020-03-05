# -*- coding: utf-8 -*-
import dash
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import datetime

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

df = pd.read_csv('data/products.csv')
descriptors = df.columns[:8]
stat_df = pd.read_csv('data/category_stats.csv')
old_products = df[descriptors].sum(axis=1).unique().shape[0]

def calculate_opportunity(sort='Worst', select=[0,10]):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
        new_df = pd.DataFrame()
        for index in range(select[0],select[1]):
            x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, x])
    else:
        local_df = stat_df
        new_df = df
        for index in range(select[0],select[1]):
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]

    new_EBIT = 1 / (new_df['Sales Quantity in KG'].sum() /
        df['Sales Quantity in KG'].sum()) * new_df['EBIT'].sum()
    EBIT_percent = (new_EBIT - df['EBIT'].sum()) / df['EBIT'].sum() * 100
    new_products = new_df[descriptors].sum(axis=1).unique().shape[0]
    product_percent_reduction = (old_products - new_products) / \
        old_products * 100

    return "${:.1f} M".format(new_EBIT/1e6), "{:.01f}%".format(EBIT_percent),\
           "{}".format(new_products), \
           "{:.01f}%".format(product_percent_reduction)

def make_violin_plot(sort='Worst', select=[0,10]):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    fig = go.Figure()
    for index in range(select[0],select[1]):
        x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
            local_df.iloc[index]['group'])]['EBIT']
        y = local_df.iloc[index]['descriptor'] + ': ' + df.loc[(df[local_df.iloc\
            [index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']]
        name = 'EBIT: {:.0f}, P{}'.format(x.median(),
            local_df.iloc[index]['group'])
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'EBIT by Product Descriptor',
                })
    return fig

# Describe the layout/ UI of the app
app.layout = html.Div([
    html.H4(["Product Characterization"]),
    html.Div([
        html.Div([
            html.H6(id='new-rev'), html.P('EBIT')
        ], className='mini_container',
           id='rev',
        ),
        html.Div([
            html.H6(id='new-rev-percent'), html.P('EBIT Increase')
        ], className='mini_container',
           id='rev-percent',
        ),
        html.Div([
            html.H6(id='new-products'), html.P('Number of Products')
        ], className='mini_container',
           id='products',
        ),
        html.Div([
            html.H6(id='new-products-percent'), html.P('Product Reduction')
        ], className='mini_container',
           id='products-percent',
        ),
    ], className='row container-display'
    ),
    html.Div([
        html.P('Sort product descriptors by selecting (best) products for'\
            'portfolio or eliminating (worst) products from portfolio'),
        dcc.RadioItems(
                    id='sort',
                    options=[{'label': i, 'value': i} for i in \
                            ['Best', 'Worst']],
                    value='Worst',
                    labelStyle={'display': 'inline-block'},
                    style={"margin-bottom": "10px"},),
        html.P('Number of Descriptors:', id='descriptor-number'),
        dcc.RangeSlider(
                    id='select',
                    min=0,
                    max=stat_df.shape[0],
                    step=1,
                    value=[0,10],
        ),
    ], className='mini_container'
    ),
    dcc.Graph(
                id='violin_plot',
                figure=make_violin_plot()),
    html.Div([
        html.H6(["Background"]),
        html.P('Median testing is performed for EBIT on 926 variables that '\
               'describe 2643 unique prodcuts. Variables with p-values below '\
               '.01 are used to select products for a hypothetical product '\
               'portfolio. Annualized EBIT is then calcualted based on the '\
               'production for 2019 (kg).')
    ], className='mini_container'
    ),
    ], className='pretty container'
    )

app.config.suppress_callback_exceptions = False

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1])

@app.callback(
    Output('violin_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value')]
)
def display_violin_plot(sort, select):
    return make_violin_plot(sort, select)

@app.callback(
    [Output('new-rev', 'children'),
     Output('new-rev-percent', 'children'),
     Output('new-products', 'children'),
     Output('new-products-percent', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value')]
)
def display_opportunity(sort, select):
    return calculate_opportunity(sort, select)

if __name__ == "__main__":
    app.run_server(debug=True)
