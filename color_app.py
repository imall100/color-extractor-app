import pandas as pd
import numpy as np                        
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
import re
from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
import os
from PIL import Image
import urllib
from colory.color import Color
import base64
from io import BytesIO




def color_classification(hex):
    return Color(hex,'wiki').name

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def image_dataframe(image, input_cluster):

    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = input_cluster)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    
    hex_cols = []
    for i in range(len(ordered_colors)):
        hex = RGB2HEX(ordered_colors[i])
        hex_cols = np.append(hex_cols, hex)
    
    
    count_props = []
    
    for i in range(len(list(counts.values()))):
        total = sum(list(counts.values()))
        prop = round(((list(counts.values()))[i]/total), 4)
        count_props = np.append(count_props, prop)
    
    data_dic = {'hex_code': hex_cols, 'rgb_code':ordered_colors, 'count':list(counts.values()), 'proportion':count_props}
    
    dataframe = pd.DataFrame(data_dic)
    
    colors = []

    for i in range(input_cluster):
        hex_ = dataframe['hex_code'][i]
        color = color_classification(hex_)
        colors = np.append(colors, color)

    dataframe['color_approximation'] = colors

    
    return dataframe



def get_base64(input_string):
    split = input_string.split(',')
    base64 = split[1]
    return base64

def generate_table(dataframe, input_cluster):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), input_cluster))
        ])
    ])
    
#Master


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H1(children='Color Extractor', style={'textAlign': 'center'}),
    html.Div(children='A tool to analyze the color composition of any image', style={'marginBottom': 50,
                                                                                     'textAlign': 'center'
                                                                                     }),
    html.Div([
        html.Center(children=[
            html.Div([
                dcc.Upload(
                    id='upload-image',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select Files')]),


                style={
                'width': '60%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'})])
        ]),
            
        html.Div([    
            html.Div([
                    dcc.Input(id='input_cluster', 
                              type='number', 
                              placeholder='number of clusters',
                              min=2, max=50)],
                style = {'textAlign': 'right'}),
            
            html.Div([
                html.Div(children='The number of clusters determines the amount of colors in the breakdown. More clusters will give a more granular breakdown'
                         )],
                style = {'marginRight': 430,
                         'marginBottom': 0,
                         'color': 'grey',
                         'textAlign': 'left'
                         })
            ], style={'columnCount': 2}
              
        
        
        )
        # Allow multiple files to be uploaded
        
    ]),
    html.Center(children=[
        html.Div(id='output-image-upload', style={'textAlign':'center',
                                                  'marginTop': 40}),
        dcc.Loading(id='loading', children=[html.Div(id='graph'),
                                            html.Div(id='table'),                     
                        ], style={'marginTop': 55,
                                  'textAlign': 'center',
                                  'display': 'flex',
                                  'align-items': 'center',
                                  'justify-content': 'center'})])
])


def get_image(contents):
    return contents


@app.callback(Output('output-image-upload', 'children'),
              [Input('upload-image', 'contents')])
def update_image(list_of_contents):
    html_img = html.Div(html.Img(src=list_of_contents, style={'height':'10%', 
                                                                  'width':'10%',
                                                                  'Align': 'center'
                                                                 }))
    return html_img
               
               
@app.callback(Output('loading', 'children'),
              [Input('upload-image', 'contents'),
               Input('input_cluster', 'value')],
              )
def update_output(list_of_contents, input_cluster):
    if list_of_contents and input_cluster is not None:
        try:
            b64_string = get_base64(list_of_contents)
            decoded = BytesIO(base64.b64decode(b64_string))
            image = Image.open(decoded).convert('RGB')
            open_cv_image = np.array(image)

            dataframe = image_dataframe(open_cv_image, input_cluster)

            fig = go.Figure(data=[go.Pie(values=list(dataframe['count']), labels=list(dataframe['hex_code']))])
            colors = list(dataframe['hex_code'])
            fig.update_traces(hoverinfo='label+percent', marker=dict(colors=colors))
            fig.update(layout_showlegend=False)
            fig.update_traces(textinfo='none')


            pie = html.Div([
                dcc.Graph(
                    id='graph',
                    figure = fig)
            ])

            table = generate_table(dataframe, input_cluster)

            csv_string = dataframe.to_csv(index=False, encoding='utf-8')
            csv_string = "data:text/csv;charset=utf-8," + urllib.parse.quote(csv_string)
            download_link = html.Div(
                                html.A(
                                        'Download Data',
                                        id='download-link',
                                        download="rawdata.csv",
                                        href=csv_string,
                                        target="_blank"                           
                                        ),
                                style={'Align':'center',
                                       'marginTop': 30,
                                       'marginBottom':60})

            return pie, table, download_link
        
        except:
            return 'Error: please use only PNG or JPG files'
    



   
app.run_server(debug=True, use_reloader=False)




