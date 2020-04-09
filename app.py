# -*- coding: utf-8 -*-
import dash
import dash_auth
import json
import dash_core_components as dcc
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
from itertools import cycle

import pandas as pd
import numpy as np
import datetime

VALID_USERNAME_PASSWORD_PAIRS = {
    'gendorf': 'assessment'
}

app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)

auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS
)

server = app.server

opp = pd.read_csv('data/opportunity.csv', index_col=[0,1,2,3])
opportunity = pd.read_csv('data/days.csv', index_col=[0,1,2,3])
annual_operating = pd.read_csv('data/annual.csv', index_col=[0,1])
stats = pd.read_csv('data/scores.csv')
quantiles = np.arange(50,101,1)
quantiles = quantiles*.01
quantiles = np.round(quantiles, decimals=2)
dataset = opp.sort_index()
lines = opp.index.get_level_values(1).unique()
asset_metrics = ['Yield', 'Rate', 'Uptime']
groupby = ['Line', 'Product group']
oee = pd.read_csv('data/oee.csv')
oee['From Date/Time'] = pd.to_datetime(oee["From Date/Time"])
oee['To Date/Time'] = pd.to_datetime(oee["To Date/Time"])
oee["Run Time"] = pd.to_timedelta(oee["Run Time"])
oee = oee.loc[oee['Rate'] < 2500]
res = oee.groupby(groupby)[asset_metrics].quantile(quantiles)

df = pd.read_csv('data/products.csv')
descriptors = df.columns[:8]
production_df = df
production_df['product'] = production_df[descriptors[2:]].agg('-'.join, axis=1)
production_df = production_df.sort_values(['Product Family', 'EBIT'],
                                          ascending=False)

stat_df = pd.read_csv('data/category_stats.csv')
old_products = df[descriptors].sum(axis=1).unique().shape[0]
weight_match = pd.read_csv('data/weight_match.csv')

def make_bubble_chart(x='EBITDA per Hr Rank', y='Adjusted EBITDA', color='Line',
                      size='Net Sales Quantity in KG'):

    fig = px.scatter(weight_match, x=x, y=y, color=color, size=size)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                # "title": 'EBIT by Product Descriptor',
                })

    return fig

def calculate_margin_opportunity(sort='Worst', select=[0,10], descriptors=None):
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
        new_df = new_df.drop_duplicates()
    else:

        new_df = df
        for index in range(select[0],select[1]):
            new_df = new_df.loc[~(new_df[local_df.iloc[index]['descriptor']] ==\
                    local_df.iloc[index]['group'])]

    new_EBITDA = new_df['Adjusted EBITDA'].sum()
    EBITDA_percent = new_EBITDA / df['Adjusted EBITDA'].sum() * 100

    new_products = new_df[descriptors].sum(axis=1).unique().shape[0]

    product_percent_reduction = (new_products) / \
        old_products * 100

    new_kg = new_df['Sales Quantity in KG'].sum()
    old_kg = df['Sales Quantity in KG'].sum()
    kg_percent = new_kg / old_kg * 100

    return "€{:.1f} M of €{:.1f} M ({:.1f}%)".format(new_EBITDA/1e6,
                df['Adjusted EBITDA'].sum()/1e6, EBITDA_percent), \
            "{} of {} Products ({:.1f}%)".format(new_products,old_products,
                product_percent_reduction),\
            "{:.1f} M of {:.1f} M kg ({:.1f}%)".format(new_kg/1e6, old_kg/1e6,
                kg_percent)

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
            local_df.iloc[index]['group'])]['Adjusted EBITDA']
        y = local_df.iloc[index]['descriptor'] + ': ' + df.loc[(df[local_df\
            .iloc[index]['descriptor']] == local_df.iloc[index]['group'])]\
            [local_df.iloc[index]['descriptor']]
        name = '€ {:.0f}'.format(x.median())
        fig.add_trace(go.Violin(x=y,
                                y=x,
                                name=name,
                                box_visible=True,
                                meanline_visible=True))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": 'Adjusted EBITDA by Product Descriptor (Median in Legend)',
                "yaxis.title": "EBITDA (€)",
                "height": 400,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
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
    fig = px.sunburst(test, path=desc[:], color='Adjusted EBITDA', title='{}: {}'.format(
        col, val),
        color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "title": '(Select in Violin) {}: {}'.format(col,val),
                "paper_bgcolor": "#F9F9F9",
                "height": 400,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
                })
    return fig

def make_ebit_plot(production_df, select=None, sort='Worst', descriptors=None):
    families = production_df['Product Family'].unique()
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3',\
              '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    colors_cycle = cycle(colors)
    grey = ['#7f7f7f']
    color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(families,
                                                                 colors)}
    grey_dic =  {'{}'.format(i): '{}'.format('#7f7f7f') for i in families}
    fig = go.Figure()


    if select == None:
        for data in px.scatter(
                production_df,
                x='product',
                y='Adjusted EBITDA',
                color='Product Family',
                color_discrete_map=color_dic,
                opacity=1).data:
            fig.add_trace(
                data
            )

    elif select != None:
        color_dic = {'{}'.format(i): '{}'.format(j) for i, j  in zip(select,
                                                                     colors)}
        for data in px.scatter(
                production_df,
                x='product',
                y='Adjusted EBITDA',
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
            x = production_df.loc[(production_df[local_df.iloc[index]\
                ['descriptor']] == local_df.iloc[index]['group'])]
            x['color'] = next(colors_cycle) # for line shapes
            new_df = pd.concat([new_df, x])
            new_df = new_df.reset_index(drop=True)
        for data in px.scatter(
                new_df,
                x='product',
                y='Adjusted EBITDA',
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
        fig.update_layout(shapes=shapes)
    fig.update_layout({
            "plot_bgcolor": "#F9F9F9",
            "paper_bgcolor": "#F9F9F9",
            "title": 'Adjusted EBITDA by Product Family',
            "yaxis.title": "EBITDA (€)",
            "height": 500,
            "margin": dict(
                   l=0,
                   r=0,
                   b=0,
                   t=30,
                   pad=4
),
            "xaxis.tickfont.size": 8,
            # "font":dict(
            #     size=8,
            # ),
            })
    return fig

def calculate_overlap(lines=['E27', 'E26']):
    path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment']

    line1 = oee.loc[oee['Line'].isin([lines[0]])].groupby(path)\
                    ['Quantity Good'].sum()
    line2 = oee.loc[oee['Line'].isin([lines[1]])].groupby(path)\
                    ['Quantity Good'].sum()

    set1 = set(line1.index)
    set2 = set(line2.index)

    both = set1.intersection(set2)
    unique = set1.union(set2) - both

    kg_overlap = (line1.loc[list(both)].sum() + line2.loc[list(both)].sum()) /\
    (line1.sum() + line2.sum())
    return kg_overlap*100

def make_product_sunburst(lines=['E27', 'E26']):
    fig = px.sunburst(oee.loc[oee['Line'].isin(lines)],
        path=['Product group', 'Polymer', 'Base Type', 'Additional Treatment',\
                'Line'],
        color='Line')
    overlap = calculate_overlap(lines)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 500,
                 "margin": dict(
                        l=0,
                        r=0,
                        b=0,
                        t=30,
                        pad=4
    ),
                 "title": "Product Overlap {:.1f}%: {}, {}".format(overlap,
                                                            lines[0], lines[1]),
     })
    return fig

def make_metric_plot(line='K40', pareto='Product', marginal='histogram'):
    plot = oee.loc[oee['Line'] == line]
    plot = plot.sort_values('Thickness Material A')
    plot['Thickness Material A'] = pd.to_numeric(plot['Thickness Material A'])
    if marginal == 'none':
        fig = px.density_contour(plot, x='Rate', y='Yield',
                     color=pareto)
    else:
        fig = px.density_contour(plot, x='Rate', y='Yield',
                 color=pareto, marginal_x=marginal, marginal_y=marginal)
    fig.update_layout({
                 "plot_bgcolor": "#F9F9F9",
                 "paper_bgcolor": "#F9F9F9",
                 "height": 750,
                 "title": "{}, Pareto by {}".format(line, pareto),
     })
    return fig

def make_utilization_plot():
    downdays = pd.DataFrame(oee.groupby('Line')['Uptime'].sum().sort_values()/24)
    downdays.columns = ['Unutilized Days, 2019']
    fig = px.bar(downdays, y=downdays.index, x='Unutilized Days, 2019',
           orientation='h', color=downdays.index)
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": "Utilization, All Lines (Note: data did not "\
                        "distinguish between downtime and utilization)",
                "height": 400,
    })
    return fig

def find_quantile(to_remove_line='E26', to_add_line='E27',
                  metrics=['Rate', 'Yield', 'Uptime'],
                  uptime=None):
    if type(to_add_line) == str:
        to_add_line = [to_add_line]
    to_remove_kg = annual_operating.loc[to_remove_line]['Quantity Good']
    to_remove_rates = res.loc[to_remove_line].unstack()['Rate']
    to_remove_yields = res.loc[to_remove_line].unstack()['Yield']
    target_days_needed = pd.DataFrame(to_remove_kg).values / to_remove_yields\
                            / to_remove_rates / 24
    target_days_needed = target_days_needed.T
    target_days_needed['Total'] = target_days_needed.sum(axis=1)

    target_data = opportunity.loc['Additional Days'].loc[to_add_line].unstack()\
                    [metrics].sum(axis=1)
    target = pd.DataFrame(target_data).unstack().T.loc[0]

    if uptime != None:
        target[to_add_line] = target[to_add_line] + uptime

    final = pd.merge(target_days_needed, target, left_index=True, right_index=True)
    quantile = (abs(final[to_add_line].sum(axis=1) - final['Total'])).idxmin()
    return quantile, final.iloc[:-1]

def make_consolidate_plot(remove='E26', add='E27',
                          metrics=['Rate', 'Yield', 'Uptime'],
                          uptime=None):
    quantile, final = find_quantile(remove, add, metrics, uptime)
    fig = go.Figure(data=[
    go.Bar(name='Days Available', x=final.index, y=final[add]),
    go.Bar(name='Days Needed', x=final.index, y=final['Total'])
    ])
    if uptime != None:
        title = "Quantile-Performance Target: {} + {} Uptime Days"\
            .format(quantile, uptime)
    else:
        title = "Quantile-Performance Target: {}".format(quantile)
    # Change the bar mode
    fig.update_layout(barmode='group',
                  yaxis=dict(title="Days"),
                   xaxis=dict(title="Quantile"))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "title": title,
                "margin": dict(
                       l=0,
                       r=0,
                       b=0,
                       t=30,
                       pad=4
   ),
    })
    return fig

def pareto_product_family(quantile=0.9, clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = opportunity.reorder_levels([0,2,1,3]).sort_index().\
            loc['Additional Days', quantile, line]
    total = data.sum().sum()
    cols = data.columns
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col],
        )
        )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="group",
            yaxis_type="category",
            yaxis=dict(title="Product Group"),
            xaxis=dict(title="Days"),
            title="{}: {:.1f} days of opportunity".format(line,total),
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_days_plot(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index()\
                .loc['Additional Days', quantile].groupby('Line').sum()
    cols = ['Rate', 'Yield', 'Uptime']
    data['Total'] = data.sum(axis=1)
    data = data.sort_values(by='Total')
    bar_fig = []
    for col in cols:
        bar_fig.append(
        go.Bar(
        name=col,
        orientation="h",
        y=[str(i) for i in data.index],
        x=data[col],
        customdata=[col]
        )
    )

    figure = go.Figure(
        data=bar_fig,
        layout=dict(
            barmode="stack",
            yaxis_type="category",
            yaxis=dict(title="Line"),
            xaxis=dict(title="Days"),
            title="Annualized Opportunity",
            plot_bgcolor="#F9F9F9",
            paper_bgcolor="#F9F9F9"
        )
    )
    figure.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return figure

def make_culprits():
    fig = px.bar(stats, x='group', y='score', color='metric',
        barmode='group')
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
                "xaxis.title": "Contingency Table Score",
    })
    return fig

def pie_line(clickData=None):
    if clickData != None:
        line = clickData["points"][0]["y"]
    else:
        line = 'K40'
    data = annual_operating.loc[line]
    total = data['Net Quantity Produced'].sum()/1e6
    fig = px.pie(data, values='Net Quantity Produced', names=data.index,
                title='Production distribution 2019 ({:.1f}M kg)'.format(total))
    fig.update_layout({
                "plot_bgcolor": "#F9F9F9",
                "paper_bgcolor": "#F9F9F9",
    })
    return fig

def calculate_opportunity(quantile=0.9):
    data = opportunity.reorder_levels([0,2,1,3]).sort_index()\
                .loc['Additional Days', quantile].groupby('Line').sum()
    data['Total'] = data.sum(axis=1)
    return "{:.1f}".format(data.sum()[3]), \
            "{:.1f}".format(data.sum()[0]), \
            "{:.1f}".format(data.sum()[1]), \
            "{:.1f}".format(data.sum()[2])
# Describe the layout/ UI of the app

app.layout = html.Div([
html.H1('Gendorf OpEx Assessment'),
html.H2('Data Storyboard'),
html.Br(),
html.H3(["Product Margin Optimization"]),
html.Div([
html.Div([
dcc.Markdown('''
###### Key Finding: ######
There are a fair number of low to negative margin products
that should be reviewed. All groups &#150 business, manufacturing, supply
chain &#150 need to work together to improve these margins by using a combination
of potential levers: Price Increase, Production Rules, Minimum Order Sizes, Campaigning, etc.

**Implementation Phase:** Caravel partners work with group teams to strategize
products around margin levers.

### Est. Impact € 3.5-6 M/Yr ###
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "350px"},
    id='explain1a',
),
html.Div([
dcc.Markdown('''

###### Demonstrates margin disparity and product buckets. ######


The default view of the following interactive charts show that of all
possible combinations of thicknesses, widths, base types, treatments, colors,
polymers and product groups and families, **53 were statistically influential
on EBITDA.** Ex: Selecting all products that are described by the 10 most positively
influential of those descriptors accounts for 47% of EBITDA for 2019 and 16%
of the production volume i.e. a significant production effort is spent on
products that do not give a positive contribution to EBITDA. **All 53 descriptors
are made available here.**

------

* Descriptors can be selected from eight categories:
    * thickness, width, base type, treatment, color, polymer, product family & group
* Descriptors are sorted by either best (describe high EBITDA products) or
worst (describe low EBITDA products)
* The range bar updates what descriptors are shown in the violin plot and EBITDA
by Product Family Plot as well as what is calculated in EBITDA, unique products, and volume displays

------

A violin plot of EBITDA values is constructed of each descriptor
selected by the range bar. A violin plot is a method of plotting
distributions. It is similar to a box plot, with the addition of a rotated
kernel density (kde) plot on each side. **The benefit of the kde is to visualize
the density of the data without obstructing key outliers** *(ex: 200-400K EBITDA
outliers in 2D Coil Coating and Base Type 153/07)*

Clicking on a distribution in the violin
plot expands the sunburst chart to its right. A sunburst chart is a way of
representing hierarchical data structures. In this case it is showing the
product breakdown for a given descriptor. For instance, products with base
types of 202/14 fall within the Construction category, with PVC polymer, ZZZ
treatment, and OP color. The bandwidths that lie on each ring indicate the
production volume fraction for that given descriptor while color indicates
the average EBITDA for all products described by that section of the sunburst *(ex:
in the default view, highest EBITDA base type 202/14 products have a width of 955
while lowest EBITDA have a width of 400 and each of these count for 1 production
run out of 23 for this product group).* Thickness and width can be toggled on the sunburst chart for clarity.

Descriptors in the violin plot are overlayed onto the EBITDA by Product Family
chart. In this way, product descriptors can be evaluated within the broader portfolio
*(ex: toggling the best/worst rank selector above
will alternate highlighting the high margin and negative margin products within
each family, respectively).*
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "350px",
          "overflow": "scroll"},
   id='explain1b',
),
], className='row container-display',
),
    html.Div([
        html.Div([
            html.H6(id='margin-new-rev'), html.P('Adjusted EBITDA')
        ], className='mini_container',
           id='margin-rev',

        ),
        html.Div([
            html.H6(id='margin-new-rev-percent'), html.P('Unique Products')
        ], className='mini_container',
           id='margin-rev-percent',
        ),
        html.Div([
            html.H6(id='margin-new-products'), html.P('Volume')
        ], className='mini_container',
           id='margin-products',
        ),
    ], className='row container-display',
        # style={'border-color': '#ED2222',
        #        'background-color': '#aec7e8'},
    ),
    html.Div([
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
                        options=[{'label': i, 'value': j} for i, j in \
                                [['Low EBITDA', 'Worst'],
                                ['High EBITDA', 'Best']]],
                        value='Best',
                        labelStyle={'display': 'inline-block'},
                        style={"margin-bottom": "10px"},),
            html.P('Toggle Violin/Descriptor Data onto EBITDA by Product Family:'),
            daq.BooleanSwitch(
              id='daq-violin',
              on=False,
              style={"margin-bottom": "10px", "margin-left": "0px",
              'display': 'inline-block'}),
                ], className='mini_container',
                    id='descriptorBlock',
                ),
            html.Div([
                dcc.Graph(
                            id='ebit_plot',
                            figure=make_ebit_plot(production_df)),
                ], className='mini_container',
                   id='ebit-family-block'
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
               style={'margin-bottom': '50px'},
            ),
html.H5(["Margin Velocity"]),
html.Div([
html.Div([
dcc.Markdown('''
###### Key Finding: ######
There is clear segmentation in line and product families
in their margin velocity. High EBITDA per Hr product lines should be expanded
while low EBITDA per Hr product lines should be discontinued or augmented
with pricing and other levers.
'''),
], className='pretty_container',
style={"background-color": "#ffffff",
       "maxHeight": "300px"},
    id='explain2a',
),
html.Div([
dcc.Markdown('''
###### Looks at margin velocity by product family and line. ######


A product can have a very high margin. But if it takes 4x as long to make it
vs other products the margin velocity, and hence its value, may be much less than
previously thought.
Margin velocity gives you a sense of which products should be growing and
which ones should be removed *(ex: in the default view of the
following chart, we would like to prioritize all products appearing to the right,
(high EBITDA per Hr) pushing them further up the y-axis (Adjusted EBITDA) by
increasing their Size (production volume)).*
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "300px"},
   id='explain2b',
),
], className='row container-display',
),
    html.Div([
        html.Div([
            html.Div([
                html.P('X-axis'),
                dcc.Dropdown(id='x-select',
                             options=[{'label': i, 'value': i} for i in \
                                        ['Rate', 'Yield', 'EBITDA per Hr Rank',\
                                         'Adjusted EBITDA', 'Net Sales Quantity in KG']],
                            value='EBITDA per Hr Rank',),
                     ],  className='mini_container',
                         id='x-select-box',
                     ),
            html.Div([
                html.P('Y-axis'),
                dcc.Dropdown(id='y-select',
                             options=[{'label': i, 'value': i} for i in \
                                        ['EBITDA per Hr', 'Adjusted EBITDA',\
                                         'Net Sales Quantity in KG']],
                            value='Adjusted EBITDA',),
                    ],className='mini_container',
                      id='y-select-box',
                    ),
            html.Div([
                html.P('Color'),
                dcc.Dropdown(id='color-select',
                             options=[{'label': i, 'value': i} for i in \
                                        ['Line', 'Thickness Material A',\
                                         'Width Material Attri', 'Product Family']],
                            value='Line',),
                    ],className='mini_container',
                      id='color-select-box',
                    ),
            ], className='row container-display',
            ),
        ],
        ),
    html.Div([
        dcc.Graph(
            id='bubble_plot',
            figure=make_bubble_chart(),
        ),
        ], className='mini_container',
            style={'margin-bottom': '100px'},
        ),
html.H3(["Asset Performance Analysis"]),
html.Div([
html.Div([
dcc.Markdown('''
###### Key Finding: ######
If sales can not come through with additional volumes,
Lines such as E26, K06 should be considered for Consolidation. There is
evidence to suggest that consolidating these lines into higher performing
lines is possible.

**Implementation Phase:** Caravel partners will assist in unutilized capacity
being be monetized.

### Est. Impact € 2-4 M/Yr ###
'''),
], className='pretty_container',
style={"background-color": "#ffffff",
       "maxHeight": "350px"},
    id='explain3a',
),
html.Div([
dcc.Markdown('''
###### Explores key variables that affect rate, yield, and uptime ######

In this graphic, scores reflect whether or not a group (line or product family) is
improving uptime, rate, or yield. The statistical test is similar to that
performed for the product descriptors in the margin analysis.

While groups were determined to be statistically impactful
(null hypothesis < 0.01) it does not guarantee decoupling. For
instance, PSL has a very negative impact on rate and yield, however, the only
line that runs PSL is E28 and is rated similarly.
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "350px"},
   id='explain3b',
),
], className='row container-display',
),
    html.Div([
        dcc.Graph(
                    id='scores_plot',
                    figure=make_culprits()),
        html.Pre(id='slider-data'),
        html.Pre(id='click-data'),
            ], className='mini_container',
            style={'margin-bottom': '50px'},
            ),
html.H5(["Line Performance"]),
html.Div([
html.Div([
dcc.Markdown('''
###### Key Finding: ######
Newest and most state-of-the-art lines are E27, K06, & K17
with stable yield, uptime, and rate performance relative to the others.
 K40, E26, E28, K10 have the most upside opportunity.
'''),
], className='pretty_container',
style={"background-color": "#ffffff",
       "maxHeight": "300px"},
    id='explain4a',
),
html.Div([
dcc.Markdown('''
###### Quantifies the opportunity in each line in terms of equivalent days of production

Unutilized capacity should be monetized. Priority for capturing increased asset
capability should be on Lines E27, K40 -
This will take a sharper focus on true continuous improvement.
The organization tracks daily operating parameters, but there does not appear
to be a concerted effort with a project mentality on thinking in strategical
improvement terms to capture hidden plant opportunities (increases in yield, uptime and rate).

------

In the following charts, selecting a quantile on the range bar will update
the predicted upside. This effectively pushes each line into its upper quantiles
in relation to rate, yield, and uptime. Selecting a line in the Annualized opportunity
chart will pareto out product family areas.
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "300px"},
   id='explain4b',
),
], className='row container-display',
),
    html.Div([
        html.Div([
            html.H6(id='new-rev'), html.P('Total Days of Production Saved')
        ], className='mini_container',
           id='rev',
        ),
        html.Div([
            html.H6(id='new-rev-percent'), html.P('Rate (days)')
        ], className='mini_container',
           id='rev-percent',
        ),
        html.Div([
            html.H6(id='new-products'), html.P('Yield (days)')
        ], className='mini_container',
           id='products',
        ),
        html.Div([
            html.H6(id='new-products-percent'), html.P('Uptime (days)')
        ], className='mini_container',
           id='products-percent',
        ),
    ], className='row container-display'

    ),
    html.Div([
        html.Div([
            html.H6(id='slider-selection'),
            dcc.Slider(id='quantile_slider',
                        min=0.51,
                        max=0.99,
                        step=0.01,
                        value=.82,
                        included=False,
                        className="dcc_control"),
            dcc.Graph(
                        id='bar_plot',
                        figure=make_days_plot()),
                ], className='mini_container',
                    id='opportunity',
                ),
            ], className='row container-display',
            ),
    html.Div([
        html.Div([
            dcc.Graph(
                        id='pareto_plot',
                        figure=pareto_product_family())
                ], className='mini_container',
                   id='pareto',
                ),
        html.Div([
            dcc.Graph(
                        id='pie_plot',
                        figure=pie_line())
                ], className='mini_container',
                   id='pie',
                ),
            ], className='row container-display',
            style={'margin-bottom': '50px'},
            ),
html.Div([
dcc.Markdown('''
###### Identifies where broad distributions are taking place in performance ######

The afformentioned opportunity comes from tightening distributions around rate, yield,
and uptime. In the default view, K40 is shown to have
wide distributions around rate and yield. Switching the Line view to E27 will
show how this contrasts with a much better performing line.

The bottom chart shows the utilization for all lines in 2019.
'''),
], className='pretty_container',
   style={"background-color": "#ffffff"},
),
    html.Div([
        html.Div([
            html.Div([
                html.P('Line'),
                dcc.Dropdown(id='line-select',
                             options=[{'label': i, 'value': i} for i in \
                                        lines],
                            value='K40',),
                     ],  className='mini_container',
                         id='line-box',
                     ),
            html.Div([
                html.P('Pareto'),
                dcc.Dropdown(id='pareto-select',
                             options=[{'label': 'Thickness', 'value': 'Thickness Material A'},
                                     {'label': 'Product', 'value': 'Product'}],
                            value='Product',),
                    ],className='mini_container',
                      id='pareto-box',
                    ),
            html.Div([
                html.P('Marginal'),
                dcc.Dropdown(id='marginal-select',
                             options=[{'label': 'None', 'value': 'none'},
                                    {'label': 'Rug', 'value': 'rug'},
                                     {'label': 'Box', 'value': 'box'},
                                     {'label': 'Violin', 'value': 'violin'},
                                    {'label': 'Histogram', 'value': 'histogram'}],
                            value='histogram',
                             style={'width': '120px'}),
                    ],className='mini_container',
                      id='marginal-box',
                    ),
            ], className='row container-display',
            ),
        ],
        ),
    html.Div([
        dcc.Graph(
                    id='metric-plot',
                    figure=make_metric_plot()),
            ], className='mini_container',
                id='metric',
            ),
    html.Div([
        dcc.Graph(
                    id='utilization_plot',
                    figure=make_utilization_plot()),
            ], className='mini_container',
                id='util',
                style={'margin-bottom': '50px'},
            ),
html.H5("Potential Line Consolidations"),
html.Div([
html.Div([
dcc.Markdown('''
###### Key Finding: ######
The data indicates E26 may be consolidated into E27 and K06 into
K40.
'''),
], className='pretty_container',
style={"background-color": "#ffffff",
       "maxHeight": "300px"},
    id='explain6a',
),
html.Div([
dcc.Markdown('''
###### Uses product overlap and quantile performances to determine line consolidation feasibility.

With the given line performances there is an opportunity for
consolidation. 'Days Needed' are computed from rate, yield and
the total production for 'Line to Remove' in 2019.
'Days Available' is computed from rate, yield, and uptime
improvements in 'Line to Overload'. A manual overide is
available to remove uptime consideration. In this case, uptime
can be manually inputed, with a maximum value based on the
downtime days for that line in 2019.

The sunburst chart to the right shows the product overlap for the two
selected lines.
'''),
], className='pretty_container',
   style={"background-color": "#ffffff",
          "maxHeight": "300px"},
   id='explain6b',
),
], className='row container-display',
),
    html.Div([
        html.Div([
            html.Div([
                html.P("Line to Remove"),
                dcc.Dropdown(id='line-in-selection',
                            options=[{'label': i, 'value': i} for i in \
                                     lines],
                            value='E26',),
                    ], className='mini_container',
                       id='line-in',
                    ),
            html.Div([
                html.P("Line to Overload"),
                dcc.Dropdown(id='line-out-selection',
                            options=[{'label': i, 'value': i} for i in \
                                     lines],
                            value='E27',),
                    ], className='mini_container',
                       id='line-out',
                    ),
            html.Div([
                html.P('Uptime manual overide'),
                daq.BooleanSwitch(
                  id='daq-switch',
                  on=False,
                  style={"margin-bottom": "10px"}),
                dcc.Slider(id='uptime-slider',
                            min=0,
                            max=10,
                            step=1,
                            value=9,
                            included=True,
                            className="dcc_control"),
                    ], className='mini_container',
                        id='switch',
                    ),
                ], className='row container-display',
                ),
        ],
        ),
        html.Div([
            html.Div([
                dcc.Graph(
                            id='consolidate_plot',
                            figure=make_consolidate_plot()),
                    ], className='mini_container',
                        id='consolidate-box',
                    ),
            html.Div([
                dcc.Graph(
                            id='product-sunburst',
                            figure=make_product_sunburst()),
                    ], className='mini_container',
                        id='product-box',
                    ),
                ], className='row container-display',
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
    local_df = local_df.reset_index(drop=True)
    col = local_df['descriptor'][select[0]]
    val = local_df['group'][select[0]]
    return make_sunburst_plot(clickData, toAdd, col, val)

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
    [Output('margin-new-rev', 'children'),
     Output('margin-new-rev-percent', 'children'),
     Output('margin-new-products', 'children')],
    [Input('sort', 'value'),
    Input('select', 'value'),
    Input('descriptor_dropdown', 'value')]
)
def display_opportunity(sort, select, descriptors):
    return calculate_margin_opportunity(sort, select, descriptors)

@app.callback(
    Output('metric-plot', 'figure'),
    [Input('line-select', 'value'),
    Input('pareto-select', 'value'),
    Input('marginal-select', 'value')]
)
def display_opportunity(line, pareto, marginal):
    return make_metric_plot(line, pareto, marginal)

@app.callback(
    Output('bubble_plot', 'figure'),
    [Input('x-select', 'value'),
    Input('y-select', 'value'),
    Input('color-select', 'value')]
)
def display_opportunity(x, y, color):
    return make_bubble_chart(x, y, color)

@app.callback(
    [Output('new-rev', 'children'),
     Output('new-rev-percent', 'children'),
     Output('new-products', 'children'),
     Output('new-products-percent', 'children')],
    [Input('quantile_slider', 'value')]
)
def display_opportunity(quantile):
    return calculate_opportunity(quantile)

@app.callback(
    Output('uptime-slider', 'disabled'),
    [Input('daq-switch', 'on')])
def display_click_data(on):
    return on == False

@app.callback(
    Output('uptime-slider', 'max'),
    [Input('line-out-selection', 'value')])
def display_click_data(line):
    days = np.round(oee.loc[oee['Line'] == line]['Uptime'].sum()/24)
    return days

@app.callback(
    Output('bar_plot', 'figure'),
    [Input('quantile_slider', 'value')])
def display_click_data(quantile):
    return make_days_plot(quantile)

@app.callback(
    Output('consolidate_plot', 'figure'),
    [Input('line-in-selection', 'value'),
     Input('line-out-selection', 'value'),
     Input('daq-switch', 'on'),
     Input('uptime-slider', 'value')]
     )
def display_click_data(inline, outline, switch, uptime):
    if switch == True:
        return make_consolidate_plot(inline, outline, ['Rate', 'Yield'], uptime)
    else:
        return make_consolidate_plot(inline, outline)

@app.callback(
    Output('product-sunburst', 'figure'),
    [Input('line-in-selection', 'value'),
     Input('line-out-selection', 'value')]
     )
def display_click_data(inline, outline):
    lines = [inline, outline]
    return make_product_sunburst(lines)

# @app.callback(
#     Output('quantile-target', 'children'),
#     [Input('line-in-selection', 'value'),
#      Input('line-out-selection', 'value'),
#      Input('daq-switch', 'on'),
#      Input('uptime-slider', 'value')]
#      )
# def display_click_data(inline, outline, switch, uptime):
#     if switch == True:
#         quantile, final = find_quantile(inline, outline, ['Rate', 'Yield'],
#                     uptime)
#         return "Quantile-Performance Target: {} + {} Uptime Days"\
#             .format(quantile, uptime)
#     else:
#         quantile, final = find_quantile(inline, outline)
#         return "Quantile-Performance Target: {}".format(quantile)

@app.callback(
    Output('pareto_plot', 'figure'),
    [Input('quantile_slider', 'value'),
     Input('bar_plot', 'clickData')])
def display_click_data(quantile, clickData):
    return pareto_product_family(quantile, clickData)

@app.callback(
    Output('pie_plot', 'figure'),
    [Input('bar_plot', 'clickData')])
def display_click_data(clickData):
    return pie_line(clickData)

@app.callback(
    Output('slider-selection', 'children'),
    [Input('quantile_slider', 'value')])
def display_click_data(quantile):
    return "Quantile: {}".format(quantile)

if __name__ == "__main__":
    app.run_server(debug=True)
