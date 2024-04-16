import os
import io
import pandas as pd
import numpy as np
import math
import datetime as dt
import matplotlib.pyplot as plt
from random import gauss

import dash
from dash import Dash, dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import base64
import datetime
from datetime import date

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.preprocessing import MinMaxScaler

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.layers import LSTM, Bidirectional, GRU
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from arch import arch_model


import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], use_pages=True, pages_folder="", suppress_callback_exceptions=True)
server = app.server

model_options = ['ARCH', 'GARCH', 
                             'Support Vector Regression', 'Random Forest', 'Gradient Boosting', 'Lasso Regression',
                             'GRU', 'LSTM', 'BiLSTM']

ogDF = None
df = None
df_summary = pd.DataFrame()
metrics_df = pd.DataFrame()

def skeleton():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dcc.Markdown('# Dashboard', style={'textAlign': 'center', 'margin-top':'2rem'})
            ], width=12)
        ]),
    
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H2('Traditional Models'),
                    html.Li('ARCH'),
                    html.Li('GARCH')
                ] , style={'border':'0.2rem solid blue', 'padding': '1rem', 'font-size':'1.5rem', 'margin-top':'2rem'})
            ], width=4),
            
            dbc.Col([
                html.Div([
                    html.H2('Machine Learning Models'),
                    html.Li('Random Forest'),
                    html.Li('Gradient Boosting'),
                    html.Li('Support Vector Regression'),
                    html.Li('Lasso Regression')
                ] , style={'border':'0.2rem solid blue', 'padding': '1rem', 'font-size':'1.5rem', 'margin-top':'2rem'})
            ], width=4),
            
            dbc.Col([
                html.Div([
                    html.H2('Deep Learning Models'),
                    html.Li('LSTM'),
                    html.Li('GRU'),
                    html.Li('BiLSTM')
                ] , style={'border':'0.2rem solid blue', 'padding': '1rem', 'font-size':'1.5rem', 'margin-top':'2rem'})
            ], width=4), 
        ]),
    ])


def add_model(num):
    id_dropdown = 'model' + str(num) + '-dropdown'
    first_layer = 'slider' + str(num) + '-first'
    second_layer = 'slider' + str(num) + '-second'
    header = 'Model ' + str(num) + ' Selection'
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H5(header),
                    dcc.Dropdown(
                        id=id_dropdown,
                        options=[{'label': option, 'value': option} for option in model_options],
                        value=None
                    ),            
                ])                    
            ], width=4),
            
            dbc.Col([
                html.H5('First Layer Units'),
                html.Div([
                    dcc.Slider(
                        0, 64, 4,
                        value=14,
                        id=first_layer
                    )
                ])
            ], width=4),
            
            dbc.Col([
                html.H5('Second Layer Units'),
                html.Div([
                    dcc.Slider(
                        0, 20, 2,
                        value=0,
                        id=second_layer
                    )
                ])
            ], width=4)       
        ])
    ], style={'textAlign': 'center', 'margin-top':'2rem'})


def compare_models():
    return html.Div([
        dbc.Row([
            
            dbc.Col([
                html.Div([
                    html.H5('Upload the dataset : '),
    
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        # Allow multiple files to be uploaded
                        multiple=True
                    )
                    
                ])
                
            ], width = 4),
            
            dbc.Col([
                html.Div([
                    html.H5('Range of dataset'),
    
                    dcc.DatePickerRange(
                        id='my-date-picker-range',
                        min_date_allowed=None,
                        max_date_allowed=None,
                        initial_visible_month=date(2019, 8, 5),
                        start_date=date(2011, 8, 5),
                        end_date=date(2024, 3, 5)
                    )
                ], style = {'textAlign':'center'})
                
            ], width = 4),

            dbc.Col([
                html.Div([
                    html.H5('Target'),
                        dcc.Dropdown(
                            id='target-dropdown',
                            options=['Price', 'Volatility'],
                            value=None
                        ),
                ])
            ], width = 4),


        ],style={'textAlign': 'center', 'margin-top':'2rem'}),

        dbc.Row([
    
            dbc.Col([
                    html.H5('Number of training cycles'),
    
                html.Div([
                    dcc.Slider(20, 200, 20,
                               value=100,
                               id='epoch-slider'
                    )
                ])
            ], width = 4),
            
            
            dbc.Col([
                    html.H5('Historical data used'),
    
                html.Div([
                    dcc.Slider(min=1, max=15, step=1, 
                               value=5,
                               id='my-slider'
                    )
                ])
            ], width = 4),
            
            dbc.Col([
                    html.H5('Forecast Period (Days) '),
    
                html.Div([
                    dcc.Slider(1, 7, 1,
                               value=1,
                               id='forecastPeriod-slider'
                    )
                ])
            ], width = 4),
            
        ], style={'textAlign': 'center', 'margin-top':'2rem'}),

        dbc.Row([
            dbc.Col([
                html.Div(id = 'analysis-dropdown-output',  style = {'display':'none'})
            ], width = 4),


            dbc.Col([
                html.Div([
                    html.H5('Analysis type'),
                        dcc.Dropdown(
                            id='analysis-dropdown',
                            options=['Single', 'Compare'],
                            value=None
                        ),
                ])
            ], width = 4),


            
            dbc.Col([
                
            ], width = 4),
            
        ], style={'textAlign': 'center', 'margin-top':'2rem'}),

    ])

    
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

dash.register_page("home", path='/', layout=html.Div([
    skeleton(),
    compare_models(),
    html.Div(id = 'add-model-output'),
    html.Div(id = 'model1-dropdown-output', style = {'display':'none'}),
    html.Div(id = 'model2-dropdown-output', style = {'display':'none'}),
    html.Div(id = 'target-dropdown-output', style = {'display':'none'}),
    html.Div(id = 'slider1-first-output', style = {'display':'none'}),
    html.Div(id = 'slider1-second-output', style = {'display':'none'}),
    html.Div(id = 'slider2-first-output', style = {'display':'none'}),
    html.Div(id = 'slider2-second-output', style = {'display':'none'}),

    dbc.Row([
        dcc.Markdown('### Model Summary:'),
#         html.Div(id='dataset-output-container'),
        html.Div(id='modelSelection-output-container'),

        html.Div(id='output-container-date-picker-range'),
        html.Div(id='slider-output-container'),
        html.Div(id='forecast-slider-output'),
        html.Div(id='output-modelSubmit-button', children='model'),
        
        html.Div(id='output-targetSubmit-button', children='target')

        
        ],style={'textAlign': 'center', 'margin-top':'2rem'}
    ) ,

    
    dbc.Row([
        html.Div(id='graph-output'),
        html.Div(id='output-data-upload'),
        html.Div(id='output-summary-table')
    ], style = {'margin-top':'2rem'}),
    dbc.Row([
        dbc.Col([
            
        ], width = 3),
        dbc.Col([
            html.Div(id = 'compare-analysis-output'),

        ], width = 6),
        dbc.Col([
            
        ], width = 3)
    ], style = {'margin-top': '2rem'}),

## Single page added from here...
    
    dbc.Row([
        dbc.Col([
            html.Div(id='graph2-output'),
        ], width=6),
        dbc.Col([
            html.Div(id='output-metrics', style={'margin-top':'6.5rem'})
        ], width = 6)
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='graph3-output')
        ], width=4),
        
        dbc.Col([
            html.Div(id='graph4-output')
        ], width=4),
        
        dbc.Col([
            html.Div(id='graph5-output')
        ], width=4)
    ]),

    dbc.Row([
        html.Div(id='graph6-output')
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='graph_ml1-output'),

        ], width=6),
        dbc.Col([
            html.Div(id='graph_ml2-output'),
        ], width=6)
    ], style={'margin-top':'-6rem'}),

    dbc.Row([
        dbc.Col([
            
        ], width=3),

        dbc.Col([
            html.Div(id='outputML-metrics', style={'margin-top':'1rem', 'textAlign': 'center'})
        ], width=6), 
        
        dbc.Col([
            
        ], width=3)
        
    ]),
    
    dbc.Row([
        html.Div(id='graph_ml3-output')

    ]), 

    dbc.Row([

        html.Div(id='timeSeries-graph1'),

        html.Div(id='timeSeries-graph2'),


    ]), 

    dbc.Row([
        dbc.Col([], width=4),
        dbc.Col([
            html.Div(id='outputTSA-metrics', style={'textAlign':'center'})
        ], width=4),
        dbc.Col([], width=4)
    ])


    

], style={'margin':'0 auto', 'max-width':'1600px'}))


app.layout = html.Div([
    # html.Div([
    #     html.Div(
    #         dcc.Link(f"{page['name']} - {page['path']}", href=page["relative_path"])
    #     ) for page in dash.page_registry.values()
    # ]),
    dash.page_container,
])




############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

@callback(

    Output('output-container-date-picker-range', 'children'),
    
    Output('slider-output-container', 'children'),
    Output('forecast-slider-output','children'),
    Output('output-modelSubmit-button', 'children'),
    Output('output-targetSubmit-button', 'children'),
    Input('my-date-picker-range', 'start_date'),
    Input('my-date-picker-range', 'end_date'),
    
    Input('my-slider', 'value'),
    Input('forecastPeriod-slider', 'value'),
    Input('model1-dropdown-output', 'children'),
    Input('model2-dropdown-output', 'children'),
    Input('target-dropdown-output', 'children')
    
    
)
def update_output(start_date, end_date, window_size, forecast_period, model1, model2, target):
    string_prefix = ''
    if start_date is not None:
        start_date_object = date.fromisoformat(start_date)
        start_date_string = start_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'Start Date: ' + start_date_string + ' | '
    if end_date is not None:
        end_date_object = date.fromisoformat(end_date)
        end_date_string = end_date_object.strftime('%B %d, %Y')
        string_prefix = string_prefix + 'End Date: ' + end_date_string
    if len(string_prefix) == len(''):
        return string_prefix, 'Window size: {} days'.format(window_size), 'Forecast period: {} days'.format(forecast_period), ['Model 1: {}'.format(model1), ' | Model 2: {}'.format(model2)], 'Target: {}'.format(target)
    else:
        return string_prefix, 'Window size: {} days'.format(window_size), 'Forecast period: {} days'.format(forecast_period), ['Model 1: {}'.format(model1), ' | Model 2: {}'.format(model2)], 'Target: {}'.format(target)


####

@callback(
    [Output('my-date-picker-range', 'min_date_allowed'),
     Output('my-date-picker-range', 'max_date_allowed')],
    [Input('my-date-picker-range', 'min_date_allowed'),
     Input('my-date-picker-range', 'max_date_allowed')]
)
def update_date_range(min_date_allowed, max_date_allowed):
    global ogDF
    if ogDF is not None:
        min_date_allowed = ogDF['Date'].iloc[0]
        max_date_allowed = ogDF['Date'].iloc[-1]
    else:
        min_date_allowed = date(2001, 1, 1)
        max_date_allowed = date(2024, 1, 1)

    return min_date_allowed, max_date_allowed

####  

def parse_contents(start_date, end_date, contents, filename, date):
    global df
    global ogDF
    global df_summary
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
#             df['Date'] =  pd.to_datetime(df['Date'], format='%Y-%m-%d')
            df.iloc[:, 1:] = df.iloc[:, 1:].round(2)
            ogDF = df.copy()
            df = df.loc[(df['Date'] >= start_date) & (df['Date'] < end_date)]
            df_summary = df.describe()
#             df_summary.drop(columns=['Date'], inplace=True)
            df_summary = df_summary.reset_index()
            df_summary = df_summary.round(2)

            df_summary = df_summary.rename(columns={'index': 'Statistic'})
            closedf = df[['Date', 'Close']]
            fig = px.line(df, x='Date', y=df['Close'],labels={'Date':'Date','Close':'Asset Price'})
            fig.update_traces(marker_line_width=2, opacity=0.8, marker_line_color='orange')
            fig.update_layout(title_text='Asset price for the selected timeframe',
                              plot_bgcolor='white', font_size=15, font_color='black', title_x=0.5,  
                            title_y=0.9)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False)

        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        dcc.Graph(figure=fig),

        html.H5(filename),

        dash_table.DataTable(
            df.to_dict('records'),
            [{'name': i, 'id': i} for i in df.columns], 
            page_size = 10
        ),

        html.Hr(),  # horizontal line
        
        html.H5('Summary table'),
        dash_table.DataTable(
            df_summary.to_dict('records'),
            [{'name': i, 'id': i} for i in df_summary.columns], 
            page_size = 10
        ),

        html.Hr(),  # horizontal line
        
        

    ])

####

@callback(Output('output-data-upload', 'children'),
              Input('my-date-picker-range', 'start_date'),
              Input('my-date-picker-range', 'end_date'),
              Input('upload-data', 'contents'),
              Input('analysis-dropdown-output', 'children'),

              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(start_date, end_date, list_of_contents, analysis, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(start_date, end_date, c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        if analysis == 'Single':
            return children
        else:
            return None
            
@callback(
    Output('output-summary-table', 'children'),
    Input('analysis-dropdown-output', 'children')
)

def output_summary(analysis):
    global df_summary
    global df
    if analysis == 'Compare' and df is not None:
        return html.Div([
            html.H5('Summary table'),
            dash_table.DataTable(
                df_summary.to_dict('records'),
                [{'name': i, 'id': i} for i in df_summary.columns], 
                page_size = 10
            ),
        ])
    else:
        return None

####

# Define callback to plot data

####

@callback(
    Output('graph-output', 'children'),
    
    Input('analysis-dropdown-output', 'children')
    
)
def plot_data(analysis):
    global ogDF
    if ogDF is not None and analysis == 'Single':
        names = cycle(['Close Price','High Price','Low Price'])
        colors = ['red', 'green', 'blue', 'orange']

        fig = px.line(ogDF, x='Date', y=[ogDF['Close'], 
                                                  ogDF['High'], ogDF['Low']],
                     labels={'Date': 'Date','value':'Asset Price'})
        for i, trace in enumerate(fig.data):
            trace.line.color = colors[i]

        fig.update_layout(title_text='Price chart', font_size=15, font_color='black',legend_title_text='Parameters', title_x=0.5,  
                        title_y=0.9, plot_bgcolor='white')
        fig.for_each_trace(lambda t:  t.update(name = next(names)))
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        
        return dcc.Graph(figure=fig)
    else:
        return None

####


####
@callback(
    Output('analysis-dropdown-output', 'children'),
    Input('analysis-dropdown', 'value')
)
def update_analysis(analysis):
    return analysis

####

####

@callback(
    Output('target-dropdown-output', 'children'),
    Input('target-dropdown', 'value')
)
def update_target(target):
    return target

##

# Callback to update target value
# @callback(
#     Output('target-dropdown-output', 'children'),
#     Input('train-button', 'n_clicks'),
#     State('target-dropdown', 'value')
# )

# def update_target(n_clicks, target):
#     if n_clicks > 0:
#         return target


####
@callback(
    Output('add-model-output', 'children'),
    Input('analysis-dropdown-output', 'children'),
)
def is_add_model(analysis):
    if analysis == 'Compare':
        return [add_model(1), add_model(2)]
    elif analysis == 'Single':
        return [add_model(1)]
    return None

####


####

@callback(
    Output('model1-dropdown-output', 'children'),
    Input('model1-dropdown', 'value'),
    # Input('train-button', 'n_clicks')
)
def update_model1(model1):
    return model1


####

####
@callback(
    Output('model2-dropdown-output', 'children'),
    Input('model2-dropdown', 'value')
)
def update_model1(model2):
    return model2

####

@callback(
    Output('slider1-first-output', 'children'),
    Input('slider1-first', 'value')
)

def update_slider1_first(slider):
    return slider

#

@callback(
    Output('slider1-second-output', 'children'),
    Input('slider1-second', 'value')
)

def update_slider1_second(slider):
    return slider

#

@callback(
    Output('slider2-first-output', 'children'),
    Input('slider2-first', 'value')
)

def update_slider2_first(slider):
    return slider

#

@callback(
    Output('slider2-second-output', 'children'),
    Input('slider2-second', 'value')
)

def update_slider2_second(slider):
    return slider

####

# Callback to update model1 variable
# @callback(
#     Output('model1-dropdown-output', 'children'),
#     Input('train-button', 'n_clicks'),
#     Input('model1-dropdown', 'value')
# ) 

# def update_selected_model(n_clicks, selected_model):
#     if n_clicks > 0:
#         return selected_model

# ####


# # Callback to update model1 variable
# @callback(
#     Output('model2-dropdown-output', 'children'),
#     Input('train-button', 'n_clicks'),
#     Input('model2-dropdown', 'value')
# ) 

# def update_selected_model(n_clicks, selected_model):
#     if n_clicks > 0:
#         return selected_model

####

@callback(
    [
        Output('timeSeries-graph1', 'children'),
        Output('timeSeries-graph2', 'children'),
        Output('outputTSA-metrics', 'children')

    ],
    [
        Input('model1-dropdown-output', 'children'),
        Input('target-dropdown-output', 'children'),
        Input('my-slider', 'value'), 
        Input('forecastPeriod-slider', 'value'),
        Input('analysis-dropdown-output', 'children')
    ]
)

def time_series_models(model_selected, target, window_size, forecast_period, analysis):
    global df
    global metrics_df
    metrics_df = pd.DataFrame()


    if df is not None and analysis == 'Single' and (model_selected == 'ARCH' or model_selected == 'GARCH'):
        # series = np.array(df['Close'])
        # log_returns = np.log(series[1:] / series[:-1]) * 100
        # vols = np.sqrt(log_returns**2)
        # series = log_returns
        # test_size = int(len(series)*0.2)
        # lookback = 7
        
        # X = []
        # for i in range(lookback, len(series)):
        #     window = series[i-lookback:i]
        #     X.append(window)
            
        # weekly_variances = np.sqrt(np.var(X, axis=1))
        if target == 'Volatility':
            vols = price_to_volatility()
        else:
            vols = np.array(df['Close'])
        print('Target', target)
        test_size = int(len(vols)*0.2)

        series = [x*gauss(0,1) for x in vols]
    
        train, test = series[:-test_size], series[-test_size:]        
    
        rolling_pred = []
        title = ''
        for i in range(test_size):
            train = series[:-(test_size-i)]
            if model_selected == 'GARCH':
                model = arch_model(train, p=window_size, q=window_size)
                title = 'Volatility Prediction of the test set using GARCH({}, {})'.format(window_size, window_size) 
            else:
                model = arch_model(train, p=window_size, q=0)  
                title = 'Volatility Prediction of the test set using ARCH({})'.format(window_size) 

            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=forecast_period)
            rolling_pred.append(np.sqrt(pred.variance.values[-1,:][0]))
        
        true = vols[-test_size:]

        metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']

        metrics_df['Test'] = [
            math.sqrt(mean_squared_error(true,rolling_pred)),
            mean_squared_error(true,rolling_pred),
            mean_absolute_error(true,rolling_pred),
            mean_absolute_percentage_error(true,rolling_pred),
            r2_score(true,rolling_pred)
        ]
        metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

        
        fig2 = go.Figure()
    
        fig2.add_trace(go.Scatter(x=np.arange(len(true)), y=true, mode='lines', name='True Volatility'))
    
        fig2.add_trace(go.Scatter(x=np.arange(len(rolling_pred)), y=rolling_pred, mode='lines', name='Predicted Volatility', line=dict(color='orange')))

        
        # Update layout
        fig2.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Returns',
            plot_bgcolor='white',
            title_x=0.5,  
            title_y=0.9,

        )

    
        return [graph_time_series(vols, series, rolling_pred), dcc.Graph(figure=fig2), 
               html.Div([html.H5('Performance metrics'), 
                    dash_table.DataTable(metrics_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in metrics_df.columns])])]
    else:
        return [None, None, None]


###

@callback(
    [
        Output('graph_ml1-output', 'children'),
        Output('graph_ml2-output', 'children'), 
        Output('graph_ml3-output', 'children'),
        Output('outputML-metrics', 'children')

    ],
    [
        Input('model1-dropdown-output', 'children'),
        Input('target-dropdown-output', 'children'),
        Input('my-slider', 'value'), 
        Input('forecastPeriod-slider', 'value'),
        Input('analysis-dropdown-output', 'children')

    ]
)


def machine_learning_models(model_selected, target, window_size, forecast_period, analysis):
    global df
    global metrics_df
    metrics_df = pd.DataFrame()

    if df is not None and analysis == 'Single':
        # print('Target1: ', target)
        # Determine if we want to predict volatility or price.
        if target == "Volatility":
            sequential_data = price_to_volatility()
        else:
            sequential_data = np.array(df['Close'])
        X = []
        y = []
        for i in range(len(sequential_data) - window_size - forecast_period):
            window = sequential_data[i:i+window_size]
            tgt = sequential_data[i+window_size+forecast_period-1]
            X.append(window)
            y.append(tgt)
        # Split data into training and testing sets
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        

        # Train Random Forest model
        if model_selected == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            r_squared = r2_score(y_test, y_pred)
            
            metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
            metrics_df['Train'] = [
                math.sqrt(mean_squared_error(y_train,y_pred_train)),
                mean_squared_error(y_train,y_pred_train),
                mean_absolute_error(y_train,y_pred_train),
                mean_absolute_percentage_error(y_train,y_pred_train),
                r2_score(y_train,y_pred_train)
            ]
    
            metrics_df['Test'] = [
                math.sqrt(mean_squared_error(y_test,y_pred)),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                mean_absolute_percentage_error(y_test,y_pred)  ,
                r2_score(y_test,y_pred)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

            return [pred_actual_plot(y_train, y_pred_train, 'Train set'),
                    pred_actual_plot(y_test, y_pred, 'Test set'), 
                    train_test_overlay_plot(model_selected, target, window_size, forecast_period, y_pred_train, y_pred), 
                    html.Div([html.H5('Performance metrics'), 
                    dash_table.DataTable(metrics_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in metrics_df.columns])])]


    
        elif model_selected == "Gradient Boosting":
            model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            r_squared = r2_score(y_test, y_pred)
            
            metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
            metrics_df['Train'] = [
                math.sqrt(mean_squared_error(y_train,y_pred_train)),
                mean_squared_error(y_train,y_pred_train),
                mean_absolute_error(y_train,y_pred_train),
                mean_absolute_percentage_error(y_train,y_pred_train),
                r2_score(y_train,y_pred_train)
            ]
    
            metrics_df['Test'] = [
                math.sqrt(mean_squared_error(y_test,y_pred)),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                mean_absolute_percentage_error(y_test,y_pred)  ,
                r2_score(y_test,y_pred)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

            return [pred_actual_plot(y_train, y_pred_train, 'Train set'),
                    pred_actual_plot(y_test, y_pred, 'Test set'), 
                    train_test_overlay_plot(model_selected, target, window_size, forecast_period, y_pred_train, y_pred), 
                    html.Div([html.H5('Performance metrics'), 
                    dash_table.DataTable(metrics_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in metrics_df.columns])])]


            
        elif model_selected == 'Support Vector Regression':
            model = SVR(kernel='linear')
            X_train = [x * 0.001 for x in X_train]
            y_train = [x * 0.001 for x in y_train]
            X_test = [x * 0.001 for x in X_test]
            y_test = [x * 0.001 for x in y_test]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            
            y_train = [x * 1000 for x in y_train]
            y_test = [x * 1000 for x in y_test]
            y_pred_train = [x * 1000 for x in y_pred_train]
            y_pred = [x * 1000 for x in y_pred]
            
            r_squared = r2_score(y_test, y_pred)
            
            metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
            metrics_df['Train'] = [
                math.sqrt(mean_squared_error(y_train,y_pred_train)),
                mean_squared_error(y_train,y_pred_train),
                mean_absolute_error(y_train,y_pred_train),
                mean_absolute_percentage_error(y_train,y_pred_train),
                r2_score(y_train,y_pred_train)
            ]
    
            metrics_df['Test'] = [
                math.sqrt(mean_squared_error(y_test,y_pred)),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                mean_absolute_percentage_error(y_test,y_pred)  ,
                r2_score(y_test,y_pred)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

            return [pred_actual_plot(y_train, y_pred_train, 'Train set'),
                    pred_actual_plot(y_test, y_pred, 'Test set'), 
                    train_test_overlay_plot(model_selected, target, window_size, forecast_period, y_pred_train, y_pred), 
                    html.Div([html.H5('Performance metrics'), 
                    dash_table.DataTable(metrics_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in metrics_df.columns])])]


        elif model_selected == 'Lasso Regression':
            model = Lasso(alpha=0.1)  # You can adjust the regularization parameter alpha
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            r_squared = r2_score(y_test, y_pred)
            
            metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
            metrics_df['Train'] = [
                math.sqrt(mean_squared_error(y_train,y_pred_train)),
                mean_squared_error(y_train,y_pred_train),
                mean_absolute_error(y_train,y_pred_train),
                mean_absolute_percentage_error(y_train,y_pred_train),
                r2_score(y_train,y_pred_train)
            ]
    
            metrics_df['Test'] = [
                math.sqrt(mean_squared_error(y_test,y_pred)),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                mean_absolute_percentage_error(y_test,y_pred)  ,
                r2_score(y_test,y_pred)
            ]

            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

            return [pred_actual_plot(y_train, y_pred_train, 'Train set'),
                    pred_actual_plot(y_test, y_pred, 'Test set'), 
                    train_test_overlay_plot(model_selected, target, window_size, forecast_period, y_pred_train, y_pred), 
                    html.Div([html.H5('Performance metrics'), 
                    dash_table.DataTable(metrics_df.to_dict('records'),
                    [{'name': i, 'id': i} for i in metrics_df.columns])])]


            
        else:
            return [None, None, None, None]
        
    else:
        return [None, None, None, None]
 
###

@callback(
    Output('graph2-output', 'children'),
    Output('graph3-output', 'children'),
    Output('graph4-output', 'children'),
    Output('graph5-output', 'children'),
    Output('graph6-output', 'children'),
    Output('output-metrics', 'children'),

    [Input('my-date-picker-range', 'start_date'),
    Input('model1-dropdown-output', 'children'),
    Input('target-dropdown-output', 'children'),
    Input('epoch-slider', 'value'), 
    Input('my-slider', 'value'), 
    Input('slider1-first-output', 'children'),
    Input('slider1-second-output', 'children'), 
    Input('forecastPeriod-slider', 'value'),
    Input('analysis-dropdown-output', 'children')
    ]
) 
        
    
def deep_learning_models(start_date, model_selected, target, epochs, window_size, first_layer_units, second_layer_units, forecast_period, analysis):
    global df, metrics_df
    metrics_df = pd.DataFrame()

    if df is not None and analysis == 'Single' and (model_selected == 'LSTM' or model_selected == 'BiLSTM' or model_selected == 'GRU'):
        closedf = df[['Date', 'Close']]
        closedf = closedf[closedf['Date']>= start_date]
        close_stock = closedf.copy()
        
        closedf.index = closedf.pop('Date')
        if target == 'Volatility':
            closedf = price_to_volatility()
            
        closedf_og = closedf.copy()
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))
        
        training_size = int(len(closedf) * .6)
        test_size = int(len(closedf) * .8)
        train_data,val_data, test_data=closedf[0:training_size,:],closedf[training_size:test_size,:1], closedf[test_size:len(closedf),:1]
        
        time_step = window_size
        X_train, y_train = create_dataset(train_data, time_step, forecast_period)
        X_val, y_val = create_dataset(val_data, time_step, forecast_period)
        X_test, y_test = create_dataset(test_data, time_step, forecast_period)


        X_train, X_val, X_test = transform_data(X_train, X_val, X_test)

        model = Sequential()

        if model_selected == 'LSTM':
            if second_layer_units > 0:
                model.add(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True))
                model.add(LSTM(units=second_layer_units, input_shape=()))
            else:
                model.add(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu"))
        elif model_selected == 'GRU':
            if second_layer_units > 0:
                model.add(GRU(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True))
                model.add(GRU(units=second_layer_units, input_shape=()))
            else:
                model.add(GRU(units=first_layer_units, input_shape=(time_step, 1), activation="relu"))
        elif model_selected == 'BiLSTM':
            if second_layer_units > 0:
                model.add(Bidirectional(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True)))
                model.add(Bidirectional(LSTM(units=second_layer_units, input_shape=())))
            else:
                model.add(Bidirectional(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu")))



                

        model.add(Dense(units = 1))        

        model.compile(loss="mean_squared_error",optimizer="adam", metrics=['mean_absolute_error', r_squared])
        
        history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=32,verbose=1)


        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(loss))
        
        
        train_predict=model.predict(X_train)
        val_predict = model.predict(X_val)
        test_predict=model.predict(X_test)

        
        # write a function to inverse transform
        train_predict = scaler.inverse_transform(train_predict)
        val_predict = scaler.inverse_transform(val_predict)
        test_predict = scaler.inverse_transform(test_predict)
        closedf = scaler.inverse_transform(closedf)

        original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
        original_yval = scaler.inverse_transform(y_val.reshape(-1,1))
        original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
        
        metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
        metrics_df['Train'] = [
            math.sqrt(mean_squared_error(original_ytrain,train_predict)),
            mean_squared_error(original_ytrain,train_predict),
            mean_absolute_error(original_ytrain,train_predict),
            mean_absolute_percentage_error(original_ytrain,train_predict),
            r2_score(original_ytrain, train_predict)

        ]

        metrics_df['Test'] = [
            math.sqrt(mean_squared_error(original_ytest,test_predict)),
            mean_squared_error(original_ytest,test_predict),
            mean_absolute_error(original_ytest,test_predict),
            mean_absolute_percentage_error(original_ytest,test_predict)  ,
            r2_score(original_ytest, test_predict)

        ]
        metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

        return graph_results(model_selected, target, epochs, loss, val_loss, time_step, forecast_period, closedf, train_predict, 
                             val_predict, test_predict, close_stock, original_ytrain, original_yval, original_ytest)
        
    else:
        return [None, None, None, None, None, None]





###

@callback(
    Output('compare-analysis-output', 'children'),
    [Input('analysis-dropdown-output', 'children'),
     Input('model1-dropdown-output', 'children'),
     Input('model2-dropdown-output', 'children'), 
     Input('target-dropdown-output', 'children'),
     Input('my-slider', 'value'), 
     Input('forecastPeriod-slider', 'value'),
     Input('epoch-slider', 'value'), 
     Input('slider1-first-output', 'children'),
     Input('slider1-second-output', 'children'), 
     Input('slider2-first-output', 'children'),
     Input('slider2-second-output', 'children'), 


    ]
)

def compare_analysis(analysis, model1, model2, target, window_size, forecast_period, epochs, first_layer_units1, second_layer_units1, first_layer_units2, second_layer_units2):
    global df
    global metrics_df
    metrics_df = pd.DataFrame()

    if analysis == 'Compare' and (model1 is not None or model2 is not None):
        model_analysis(model1, target, window_size, forecast_period, epochs, first_layer_units1, second_layer_units1)
        model_analysis(model2, target, window_size, forecast_period, epochs, first_layer_units2, second_layer_units2)
        return html.Div([
            html.H5('Performance metrics'), 
            dash_table.DataTable(
                data=metrics_df.to_dict('records'),
                columns=[{'name': i, 'id': i} for i in metrics_df.columns]
            )
        ])
    return None


def model_analysis(m, target, window_size, forecast_period, epochs, first_layer_units, second_layer_units):
    global df
    global metrics_df
    metrics_df['Names'] = ['RMSE', 'MSE', 'MAE', 'MAPE', 'r_squared']
    if df is not None:
        if m == 'ARCH' or m == 'GARCH':
            vols = []
            if target == 'Volatility':
                vols = price_to_volatility()
            else:
                vols = np.array(df['Close']) * 0.01
            
            test_size = int(len(vols)*0.2)
    
            series = [x*gauss(0,1) for x in vols]
        
            train, test = series[:-test_size], series[-test_size:]        
        
            rolling_pred = []
            title = 'Volatility Prediction of the test set using {}({})'.format(m, window_size)         
            for i in range(test_size):
                train = series[:-(test_size-i)]
                if m == 'ARCH':
                    model = arch_model(train, p=window_size, q=0)  
                    model_fit = model.fit(disp='off')
                else:
                    model = arch_model(train, p=window_size, q=window_size)  
                    model_fit = model.fit(disp='off')
    
                pred = model_fit.forecast(horizon=forecast_period)
                rolling_pred.append(np.sqrt(pred.variance.values[-1,:][0]))
            
            true = vols[-test_size:]
    
            metrics_df[m] = [
                math.sqrt(mean_squared_error(true,rolling_pred)),
                mean_squared_error(true,rolling_pred),
                mean_absolute_error(true,rolling_pred),
                mean_absolute_percentage_error(true,rolling_pred),
                r2_score(true,rolling_pred)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

    
        elif m == 'Random Forest' or m == 'Gradient Boosting' or m == 'Lasso Regression':
            if target == "Volatility":
                sequential_data = price_to_volatility()
            else:
                sequential_data = np.array(df['Close'])
    
            X = []
            y = []
            for i in range(len(sequential_data) - window_size - forecast_period):
                window = sequential_data[i:i+window_size]
                tgt = sequential_data[i+window_size+forecast_period-1]
                X.append(window)
                y.append(tgt)
            # Split data into training and testing sets
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            if m == 'Random Forest':
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif m == 'Gradient Boosting':
                model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            elif m == 'Lasso Regression':
                model = Lasso(alpha=0.1)  
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            elif model_selected == 'Support Vector Regression':
                model = SVR(kernel='linear')
                X_train = [x * 0.001 for x in X_train]
                y_train = [x * 0.001 for x in y_train]
                X_test = [x * 0.001 for x in X_test]
                y_test = [x * 0.001 for x in y_test]
    
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                y_train = [x * 1000 for x in y_train]
                y_test = [x * 1000 for x in y_test]
                y_pred = [x * 1000 for x in y_pred]
                

            metrics_df[m] = [
                math.sqrt(mean_squared_error(y_test,y_pred)),
                mean_squared_error(y_test,y_pred),
                mean_absolute_error(y_test,y_pred),
                mean_absolute_percentage_error(y_test,y_pred),
                r2_score(y_test,y_pred)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     


        elif m == 'LSTM' or m == 'GRU' or m == 'BiLSTM':
            closedf = df['Close']            
            if target == 'Volatility':
                closedf = price_to_volatility()
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))
            
            training_size = int(len(closedf) * .6)
            test_size = int(len(closedf) * .8)
            train_data,val_data, test_data=closedf[0:training_size,:],closedf[training_size:test_size,:1], closedf[test_size:len(closedf),:1]
            
            time_step = window_size
            X_train, y_train = create_dataset(train_data, time_step, forecast_period)
            X_val, y_val = create_dataset(val_data, time_step, forecast_period)
            X_test, y_test = create_dataset(test_data, time_step, forecast_period)
    
    
            X_train, X_val, X_test = transform_data(X_train, X_val, X_test)
    
            model = Sequential()
            
            if m == 'LSTM':
                if second_layer_units > 0:
                    model.add(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True))
                    model.add(LSTM(units=second_layer_units, input_shape=()))
                else:
                    model.add(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu"))
            elif m == 'GRU':
                if second_layer_units > 0:
                    model.add(GRU(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True))
                    model.add(GRU(units=second_layer_units, input_shape=()))
                else:
                    model.add(GRU(units=first_layer_units, input_shape=(time_step, 1), activation="relu"))
            elif m == 'BiLSTM':
                if second_layer_units > 0:
                    model.add(Bidirectional(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu", return_sequences=True)))
                    model.add(Bidirectional(LSTM(units=second_layer_units, input_shape=())))
                else:
                    model.add(Bidirectional(LSTM(units=first_layer_units, input_shape=(time_step, 1), activation="relu")))
    
            model.add(Dense(units = 1))        
    
            model.compile(loss="mean_squared_error",optimizer="adam", metrics=['mean_absolute_error', r_squared])
            
            history = model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=epochs,batch_size=32,verbose=1)

            train_predict=model.predict(X_train)
            val_predict = model.predict(X_val)
            test_predict=model.predict(X_test)
    
            
            # write a function to inverse transform
            test_predict = scaler.inverse_transform(test_predict)
            original_ytest = scaler.inverse_transform(y_test.reshape(-1,1))

            metrics_df[m] = [
                math.sqrt(mean_squared_error(original_ytest,test_predict)),
                mean_squared_error(original_ytest,test_predict),
                mean_absolute_error(original_ytest,test_predict),
                mean_absolute_percentage_error(original_ytest,test_predict)  ,
                r2_score(original_ytest, test_predict)
            ]
            metrics_df.iloc[:, 1:] = metrics_df.iloc[:, 1:].round(4)     

            





def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def r_squared(y_true, y_pred):
    SS_res =  tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res/(SS_tot + tf.keras.backend.epsilon())

def transform_data(*args):
    return [arg.reshape(arg.shape[0], arg.shape[1], 1) for arg in args]


def create_dataset(dataset, time_step=1, forecast_period=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-forecast_period):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100
        dataX.append(a)
        dataY.append(dataset[i + time_step + forecast_period-1, 0])
    return np.array(dataX), np.array(dataY)


def price_to_volatility():
    global df
    weekly_variances = []
    if df is not None:
        series = np.array(df['Close'])
        log_returns = np.log(series[1:] / series[:-1]) * 100
        vols = np.sqrt(log_returns**2)
        series = log_returns
        lookback = 6
        
        X = []
        for i in range(lookback, len(series)):
            window = series[i-lookback:i]
            X.append(window)
            
        weekly_variances = np.sqrt(np.var(X, axis=1))
    # print('Size of weekly_variance: {}, and close: {}'.format(len(weekly_variances), len(df['Close'])))
    return weekly_variances

def graph_results(model_selected, target, epochs, loss, val_loss, time_step, forecast_period, closedf, train_predict, val_predict, test_predict, close_stock, original_ytrain, original_yval, original_ytest):
    global df
    offset = 0
    if target == 'Volatility':
        offset = 7
        
    plt.clf()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.legend(loc=0)
    
    fig1 = go.Figure()
    for line in plt.gca().get_lines():
        fig1.add_trace(go.Scatter(x=line.get_xdata(), y=line.get_ydata(), mode='lines', name=line.get_label()))
    fig1.update_layout(
        title='Training and validation loss',
        xaxis_title='Epochs',
        yaxis_title='Loss',
        plot_bgcolor='white',
        title_x=0.5,  
        title_y=0.9
    )
    plt.clf()

    look_back=time_step
    trainPredictPlot = np.empty_like(closedf)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back+forecast_period:len(train_predict)+look_back+forecast_period, :] = train_predict
    # print("Train predicted data: ", trainPredictPlot.shape)

    # shift test predictions for plotting
    valPredictPlot = np.empty_like(closedf)
    valPredictPlot[:, :] = np.nan

    valPredictPlot[len(train_predict)+(look_back+forecast_period)*2:len(train_predict)+len(val_predict)+(look_back+forecast_period)*2, :] = val_predict
    # print("Validation predicted data: ", valPredictPlot.shape)

    # # unseen
    testPredictPlot = np.empty_like(closedf)
    testPredictPlot[:, :] = np.nan

    testPredictPlot[len(train_predict)+len(val_predict)+(look_back+forecast_period)*3:len(closedf), :] = test_predict
    # print("Test predicted data: ", testPredictPlot.shape)


    names = cycle(['Original','Train','Validation', 'Test'])


    # plotdf = pd.DataFrame({'date': close_stock['Date'],
    #                        'original_close': close_stock['Close'],
    #                        'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
    #                        'val_predicted_close': valPredictPlot.reshape(1,-1)[0].tolist(),
    #                        'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    plotdf = pd.DataFrame({'date': close_stock['Date'][offset:],
                       'original_close': closedf.reshape(1,-1)[0].tolist(),
                       'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                       'val_predicted_close': valPredictPlot.reshape(1,-1)[0].tolist(),
                       'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

    fig2 = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['val_predicted_close'], plotdf['test_predicted_close']],
                  labels={'value':'Asset Price','date': 'Date'})

    colors = ['blue', 'green', 'red', 'orange']

    for i, trace in enumerate(fig2.data):
        trace.line.color = colors[i]

    fig2.update_layout(title_text='Original vs Predicted {} for {}'.format(target, model_selected),
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Legend', title_x=0.5,  
                        title_y=0.9)
    fig2.for_each_trace(lambda t:  t.update(name = next(names)))

    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(showgrid=False)

    
    return [dcc.Graph(figure=fig1), 
            pred_actual_plot(original_ytrain.flatten(), train_predict.flatten(), 'Train set', width=500, height=500), 
            pred_actual_plot(original_yval.flatten(), val_predict.flatten(), 'Validation set', width=500, height=500), 
            pred_actual_plot(original_ytest.flatten(), test_predict.flatten(), 'Test set', width=500, height=500), 
            dcc.Graph(figure=fig2), 
            html.Div([html.H5('Performance metrics'), 
            dash_table.DataTable(metrics_df.to_dict('records'),
            [{'name': i, 'id': i} for i in metrics_df.columns])])]

def train_test_overlay_plot(model_selected, target, window_size, forecast_period, train_predict, test_predict):
    global df
    offset = 0
    
    if target == 'Volatility':
        series = price_to_volatility()
        offset = 7
    else:
        series = df['Close']
        
    trainPredictPlot = np.zeros(len(series))
    trainPredictPlot[:] = np.nan
    trainPredictPlot[window_size+forecast_period:len(train_predict)+window_size+forecast_period] = train_predict

    testPredictPlot = np.zeros(len(series))
    testPredictPlot[:] = np.nan

    testPredictPlot[len(train_predict)+(window_size+forecast_period):len(series)] = test_predict
    
    names = cycle(['Original','Train','Test'])
    colors = ['blue', 'green', 'red', 'orange']

    plotdf = pd.DataFrame({'date': df['Date'][offset:],
                       'original_close': series,
                      'train_predicted_close': trainPredictPlot,
                      'test_predicted_close': testPredictPlot})

    fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                              plotdf['test_predicted_close']],labels={'value':'Asset price','date': 'Date'})
    for i, trace in enumerate(fig.data):
        trace.line.color = colors[i]

    fig.update_layout(title_text='Original vs Predicted {} using {}'.format(target, model_selected),
                      plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Legend', title_x=0.5,  
                        title_y=0.9)
    fig.for_each_trace(lambda t:  t.update(name = next(names)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return dcc.Graph(figure = fig)

def pred_actual_plot(x, y, title, width=600, height=600):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', showlegend=False))
    fig.add_trace(go.Scatter(x=[min(x), max(x)], y=[min(x), max(x)],mode='lines',showlegend=False))
    fig.update_layout(
        title=title,
        xaxis_title='True Value',
        yaxis_title='Predicted Value',
        width=width,  
        height=height,
        title_x=0.5,  
        title_y=0.9,
        plot_bgcolor='white'

    )
    return dcc.Graph(figure = fig)

def graph_time_series(vols, series, rolling_pred):
    test_size = int(len(series)*0.2)

    fig1 = go.Figure()
    
    fig1.add_trace(go.Scatter(x=np.arange(len(series)), y=series, mode='lines', name='Series returns'))

    # Add the volatility trace
    fig1.add_trace(go.Scatter(x=np.arange(len(vols)), y=vols, mode='lines', name='Volatility', line=dict(color='red')))
    # Update layout
    fig1.update_layout(
        title='Bitcoin Returns and Volatility',
        xaxis_title='Time',
        yaxis_title='Returns',
        plot_bgcolor='white',
        title_x=0.5,  
        title_y=0.9,

    )
    return dcc.Graph(figure = fig1)

    



if __name__ == '__main__':
    app.run(debug=True)

