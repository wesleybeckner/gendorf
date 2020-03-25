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
import plotly.express as px

import pandas as pd
import numpy as np
import datetime
from itertools import cycle

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

df = pd.read_csv('data/products.csv')
descriptors = df.columns[:8]
production_df = df
production_df['product'] = production_df[descriptors[2:]].agg('-'.join, axis=1)
production_df = production_df.sort_values(['Product Family', 'EBIT'], ascending=False)

stat_df = pd.read_csv('data/category_stats.csv')
old_products = df[descriptors].sum(axis=1).unique().shape[0]

def calculate_opportunity(sort='Worst', select=[0,10], descriptors=None):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    if sort == 'Best':

        new_df = pd.DataFrame()
        for index in range(select[0],select[1]):
            x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            new_df = pd.concat([new_df, x])
    else:

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

def make_violin_plot(sort='Worst', select=[0,10], descriptors=None):

    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    fig = go.Figure()
    for index in range(select[0],select[1]):
        x = df.loc[(df[local_df.iloc[index]['descriptor']] == \
            local_df.iloc[index]['group'])]['EBIT']
        y = local_df.iloc[index]['descriptor'] + ': ' + df.loc[(df[local_df.iloc\
            [index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']]
        name = 'EBIT: {:.0f}, {}'.format(x.median(),
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

def make_sunburst_plot(clickData=None, toAdd=None, col=None, val=None):
    if clickData != None:
        col = clickData["points"][0]['x'].split(": ")[0]
        val = clickData["points"][0]['x'].split(": ")[1]
    elif col == None:
        col = 'Thickness Material A'
        val = '47'

    desc = list(descriptors[:-2])
    if col in desc:
        desc.remove(col)
    if toAdd != None:
        for item in toAdd:
            desc.append(item)
    test = production_df.loc[production_df[col] == val]
    fig = px.sunburst(test, path=desc[:], color='EBIT', title='{}: {}'.format(
        col, val),
        color_continuous_scale='RdBu')
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'EBIT, {}: {}'.format(col,val),
                })
    return fig

def make_ebit_plot(production_df, select=None, sort='Worst', descriptors=None):
    families = production_df['Product Family'].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    colors_cycle = cycle(colors)
    grey = ['#7f7f7f']
    color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(families, colors)}
    grey_dic =  {'{}'.format(i): '{}'.format('#7f7f7f') for i in families}
    fig = go.Figure()


    if select == None:
        for data in px.scatter(
                production_df,
                x='product',
                y='EBIT',
                color='Product Family',
                color_discrete_map=color_dic,
                opacity=1).data:
            fig.add_trace(
                data
            )

    elif select != None:
        color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(select, colors)}
        for data in px.scatter(
                production_df,
                x='product',
                y='EBIT',
                color='Product Family',

                color_discrete_map=color_dic,
                opacity=0.09).data:
            fig.add_trace(
                data
            )

        if sort == 'Best':
            local_df = stat_df.sort_values('score', ascending=False)
        elif sort == 'Worst':
            local_df = stat_df


        new_df = pd.DataFrame()
        if descriptors != None:
            local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
        for index in select:
            x = production_df.loc[(production_df[local_df.iloc[index]['descriptor']] == \
                local_df.iloc[index]['group'])]
            x['color'] = next(colors_cycle) # for line shapes
            new_df = pd.concat([new_df, x])
            new_df = new_df.reset_index(drop=True)
        for data in px.scatter(
                new_df,
                x='product',
                y='EBIT',
                color='Product Family',

                color_discrete_map=color_dic,
                opacity=1).data:
            fig.add_trace(
                data
            )
        shapes=[]

        for index, i in enumerate(new_df['product']):
            shapes.append({'type': 'line',
                           'xref': 'x',
                           'yref': 'y',
                           'x0': i,
                           'y0': -4e5,
                           'x1': i,
                           'y1': 4e5,
                           'line':dict(
                               dash="dot",
                               color=new_df['color'][index],)})
                               # color=color_dic[new_df['Product Family'][index]])})
        fig.update_layout(shapes=shapes)
    fig.update_layout({
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": 'EBIT by Product Family',
            "height": 750,
            })
    return fig

# Describe the layout/ UI of the app
app.layout = html.Div([
    html.H4(["Product Characterization"]),
    html.P('Median testing is performed for EBIT on 926 variables that '\
           'describe 2643 unique prodcuts. Variables with p-values below '\
           '.01 are used to select products for a hypothetical product '\
           'portfolio. Annualized EBIT is then calculated based on the '\
           'production for 2019 (kg). Selecting a product descriptor in the '\
           'violine plot will return a pareto of all the products made with '\
           'that descriptor in 2019.'),
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
        # html.P('Sort product descriptors by selecting (best) products for '\
        #     'portfolio or eliminating (worst) products from portfolio'),
        html.Div([
            html.P('Descriptors'),
            dcc.Dropdown(id='descriptor_dropdown',
                         options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'},
                                 {'label': 'Base Type', 'value': 'Base Type'},
                                 {'label': 'Additional Treatment', 'value': 'Additional Treatment'},
                                 {'label': 'Color', 'value': 'Color Group'},
                                 {'label': 'Product Group', 'value': 'Product Group'},
                                 {'label': 'Base Polymer', 'value': 'Base Polymer'},
                                 {'label': 'Product Family', 'value': 'Product Family'}],
                         value=['Thickness Material A',
                                'Width Material Attri', 'Base Type',
                                'Additional Treatment', 'Color Group',
                                'Product Group',
                                'Base Polymer', 'Product Family'],
                         multi=True,
                         className="dcc_control"),
            html.P('Number of Descriptors:', id='descriptor-number'),
            dcc.RangeSlider(
                        id='select',
                        min=0,
                        max=stat_df.shape[0],
                        step=1,
                        value=[0,10],
            ),
            html.P('Sort by:'),
            dcc.RadioItems(
                        id='sort',
                        options=[{'label': i, 'value': i} for i in \
                                ['Best', 'Worst']],
                        value='Best',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},),
                ], className='mini_container',
                    id='descriptorBlock',
                ),
        ], className='row container-display',
        ),
    html.Div([
        html.Div([
            dcc.Graph(
                        id='violin_plot',
                        figure=make_violin_plot()),
                ], className='mini_container',
                   id='violin',
                ),
        html.Div([
            dcc.Dropdown(id='length_width_dropdown',
                        options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                 {'label': 'Width', 'value': 'Width Material Attri'}],
                        value=['Width Material Attri'],
                        multi=True,
                        placeholder="Include in sunburst chart...",
                        className="dcc_control"),
            dcc.Graph(
                        id='sunburst_plot',
                        figure=make_sunburst_plot()),
                ], className='mini_container',
                   id='sunburst',
                ),
            ], className='row container-display',
            ),
    html.Div([
        html.P('Overlay Violin Data:'),
        daq.BooleanSwitch(
          id='daq-violin',
          on=True,
          style={"margin-bottom": "10px", "margin-left": "0px",
          'display': 'inline-block'}),
        dcc.Graph(
                    id='ebit_plot',
                    figure=make_ebit_plot(production_df)),
            ], className='mini_container',
            ),
    ], className='pretty container'
    )

app.config.suppress_callback_exceptions = False

@app.callback(
    Output('sunburst_plot', 'figure'),
    [Input('violin_plot', 'clickData'),
     Input('length_width_dropdown', 'value'),
     Input('sort', 'value'),
     Input('select', 'value'),
     Input('descriptor_dropdown', 'value')])
def display_sunburst_plot(clickData, toAdd, sort, select, descriptors):
    if sort == 'Best':
        local_df = stat_df.sort_values('score', ascending=False)
        local_df = local_df.reset_index(drop=True)
    else:
        local_df = stat_df
    if descriptors != None:
        local_df = local_df.loc[local_df['descriptor'].isin(descriptors)]
    col = local_df['descriptor'][select[0]]
    val = local_df['group'][select[0]]
    return make_sunburst_plot(clickData, toAdd, col, val)

# @app.callback(
#     Output('click-data', 'children'),
#     [Input('violin_plot', 'clickData')])
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)

@app.callback(
    [Output('select', 'max'),
    Output('select', 'value')],
    [Input('descriptor_dropdown', 'value')]
)
def update_descriptor_choices(descriptors):
    max_value = stat_df.loc[stat_df['descriptor'].isin(descriptors)].shape[0]
    value = min(10, max_value)
    return max_value, [0, value]

@app.callback(
    Output('descriptor-number', 'children'),
    [Input('select', 'value')]
)
def display_descriptor_number(select):
    return "Number of Descriptors: {}".format(select[1]-select[0])

@app.callback(
    Output('violin_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_violin_plot(sort, select, descriptors):
    return make_violin_plot(sort, select, descriptors)

@app.callback(
    Output('ebit_plot', 'figure'),
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value'),
    Input('daq-violin', 'on')]
)
def display_ebit_plot(sort, select, descriptors, switch):
    if switch == True:
        select = list(np.arange(select[0],select[1]))
        return make_ebit_plot(production_df, select, sort=sort, descriptors=descriptors)
    else:
        return make_ebit_plot(production_df)

@app.callback(
    [Output('new-rev', 'children'),
     Output('new-rev-percent', 'children'),
     Output('new-products', 'children'),
     Output('new-products-percent', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_opportunity(sort, select, descriptors):
    return calculate_opportunity(sort, select, descriptors)

if __name__ == "__main__":
    app.run_server(debug=True)
