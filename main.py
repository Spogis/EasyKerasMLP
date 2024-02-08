# Importações necessárias
import base64
import datetime
import io
import os

import dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash import dash_table
import pandas as pd

from KerasMLP_OPT import *
from KerasMLP import *
from KerasPredict import *

Input_Columns = None
Output_Columns = None
Dataset = None

# Inicializa o app Dash
app = dash.Dash(__name__,
                suppress_callback_exceptions=True,
                external_stylesheets=[dbc.themes.BOOTSTRAP],
                meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1.0"}],)

app.title = "Easy Keras MLP Regression"
server = app.server

app.layout = html.Div([
    html.Div([
        html.Img(src='assets/logo.png', style={'height': '100px', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ], style={'text-align': 'center', 'margin-bottom': '10px'}),

    html.Div([
        dcc.Tabs(id='tabs', value='tab1', children=[
            dcc.Tab(label='Dataset Information', value='tab1'),
            dcc.Tab(label='Simple Keras MLP', value='tab2'),
            dcc.Tab(label='Optimized Keras MLP', value='tab3'),
            dcc.Tab(label='Predict Values', value='tab4'),
        ], style={'align': 'center', 'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto'}),
    ]),
    dcc.Store(id='store', storage_type='memory'),
    html.Div(id='tabs-content'),
])

dataset_layout = html.Div([
    html.Br(),
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Arraste e Solte ou ',
            html.A('Selecione um Arquivo Excel ou CSV (Seu Dataset)')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
        },
        # Permite múltiplos arquivos a serem carregados
        multiple=False
    ),
    html.Br(),
    html.Label('Selecione quais serão os Inputs da sua MLP:'),
    dcc.Dropdown(
        id='column-input-selector',
        multi=True,
        placeholder='Selecione as colunas após carregar um arquivo'
    ),
    html.Br(),
    dash_table.DataTable(
        id='input-table',
        page_size=3,  # Número de linhas a mostrar por página
    ),
    html.Br(),
    html.Label('Selecione quais serão os Otputs da sua MLP:'),
    dcc.Dropdown(
        id='column-output-selector',
        multi=True,
        placeholder='Selecione as colunas após carregar um arquivo'
    ),
    html.Br(),
    dash_table.DataTable(
        id='output-table',
        page_size=3,  # Número de linhas a mostrar por página
    ),
], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

simple_layout = html.Div([
    html.Button('RUN PREDEFINED KERAS MLP!',
                        id='run-MLP-button',
                        disabled=False,
                        style={'display': 'flex', 'width': '500px', 'justifyContent': 'center',
                               'color': 'white', 'fontWeight': 'bold', 'background-color': 'green',
                               'margin-left': 'auto', 'margin-right': 'auto',
                               'margin-top': '10px', 'margin-bottom': '10px'}),
    html.Br(),
    dbc.Spinner(html.Div(id="loading-output1"), spinner_style={"width": "3rem", "height": "3rem"}),
    html.H2("r² score:"),
    dcc.Textarea(
        id='r2-simple-mlp-textarea',
        style={'width': '100%', 'height': 200, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        readOnly=True
    ),
    html.Br(),
    html.Div(id='button-output'),
], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

advanced_layout = html.Div([
    html.Button('OPTIMIZE KERAS MLP!',
                        id='run-OPTMLP-button',
                        disabled=False,
                        style={'display': 'flex', 'width': '500px', 'justifyContent': 'center',
                               'color': 'white', 'fontWeight': 'bold', 'background-color': 'red',
                               'margin-left': 'auto', 'margin-right': 'auto',
                               'margin-top': '10px', 'margin-bottom': '10px'}),
    html.Br(),
    dbc.Spinner(html.Div(id="loading-output2"), spinner_style={"width": "3rem", "height": "3rem"}),
    html.H2("r² score:"),
    dcc.Textarea(
        id='r2-opt-mlp-textarea',
        style={'width': '100%', 'height': 200, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        readOnly=True
    ),
    html.Br(),
    html.H2("Best Hyperparameters:"),
    dcc.Textarea(
        id='best-hps-textarea',
        style={'width': '100%', 'height': 200, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        readOnly=True
    ),
    html.H2("Best Model Architecture:"),
    dcc.Textarea(
        id='model-summary-textarea',
        style={'width': '100%', 'height': 200, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        readOnly=True
    ),
    html.Br(),
    html.Div(id='button-output-advanced'),
], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})

def create_predict_layout():
    predict_layout = html.Div([
        html.Div([
            html.H5("Input Values (coma separeted):"),
            dcc.Textarea(id='input-variables-textarea',
                         value=Input_Columns,
                         readOnly=True,
                         style={'width': '50%', 'height': 10, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'}),
        ]),

        dcc.Textarea(
            id='input-text',
            value='',
            style={'width': '50%', 'height': 10, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        ),
        html.Br(),
        html.Br(),

        html.Button('Predict Values!', id='predict-button', n_clicks=0,
                    style={'color': 'white', 'fontWeight': 'bold', 'background-color': 'green'}),
        html.Br(),
        html.Br(),

        dcc.Textarea(
            id='output-text',
            readOnly=True,
            style={'width': '50%', 'height': 300, 'resize': 'none', 'color': 'white', 'fontWeight': 'bold'},
        ),
    ], style={'width': '80%', 'justifyContent': 'center', 'margin-left': 'auto', 'margin-right': 'auto', 'padding': '20px'})
    return predict_layout

predict_layout = create_predict_layout()

def parse_contents(contents, filename):
    global Dataset
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'xlsx' in filename:
            # Assume que é um arquivo Excel
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'csv' in filename:
            # Assume que é um arquivo CSV
            df = pd.read_csv(io.BytesIO(decoded))
        else:
            return html.Div([
                'Tipo de arquivo não suportado.'
            ])
    except Exception as e:
        print(e)
        return html.Div([
            'Houve um erro ao processar o arquivo.'
        ])

    Dataset = df

    return df


# Callback para atualizar o conteúdo da aba com base na seleção
@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])

def update_tab_content(selected_tab):
    if selected_tab == 'tab1':
        return dataset_layout
    elif selected_tab == 'tab2':
        return simple_layout
    elif selected_tab == 'tab3':
        return advanced_layout
    elif selected_tab == 'tab4':
        predict_layout = create_predict_layout()
        return predict_layout


# Define o callback para atualizar a caixa de texto de saída
@app.callback(
    Output('output-text', 'value'),
    [Input('predict-button', 'n_clicks')],
    [dash.dependencies.State('input-text', 'value')]
)
def update_output(n_clicks, input_value):
    try:
        input_data = np.array([list(map(float, input_value.split(',')))])
        ypred = PredictValues(input_data)

        predicted_str = ""
        for i in range(len(ypred)):
            valor_formatado = f"{ypred[i]:.2f}"
            predicted_str += f"{Output_Columns[i]}:  {valor_formatado}\n"

    except Exception as e:
        predicted_str = ""
    return predicted_str

@app.callback(
    [Output("loading-output1", "children", allow_duplicate=True),
     Output("button-output", "children", allow_duplicate=True),
     Output('r2-simple-mlp-textarea', 'value')],
    Input("run-MLP-button", "n_clicks"),
    prevent_initial_call=True
)

def MLP(n_clicks):
    r2_str = RunMLP(Dataset, Input_Columns, Output_Columns)

    # Caminho do diretório contendo as imagens
    directory_path = 'assets/images'

    # Lista para armazenar os componentes de imagem
    image_components = []

    # Lista de extensões de arquivo para considerar como imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(directory_path):
        # Verifica se o arquivo é uma imagem
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Cria o caminho completo do arquivo
            file_path = os.path.join(directory_path, filename)
            # Cria um componente de imagem e adiciona à lista
            image_components.append(html.Img(src=file_path, style={'width': '50%', 'height': 'auto'}))

    loading_status = ""
    return loading_status, image_components, r2_str

@app.callback(
    [Output("loading-output2", "children", allow_duplicate=True),
     Output("button-output-advanced", "children", allow_duplicate=True),
     Output('best-hps-textarea', 'value'),
     Output('model-summary-textarea', 'value'),
     Output('r2-opt-mlp-textarea', 'value')],
    Input("run-OPTMLP-button", "n_clicks"),
    prevent_initial_call=True
)

def OPTMLP(n_clicks):
    best_hps_str, model_summary_str, r2_str  = RunOptimizedMLP(Dataset, Input_Columns, Output_Columns)

    # Caminho do diretório contendo as imagens
    directory_path = 'assets/images'

    # Lista para armazenar os componentes de imagem
    image_components = []

    # Lista de extensões de arquivo para considerar como imagens
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

    # Itera sobre todos os arquivos no diretório
    for filename in os.listdir(directory_path):
        # Verifica se o arquivo é uma imagem
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            # Cria o caminho completo do arquivo
            file_path = os.path.join(directory_path, filename)
            # Cria um componente de imagem e adiciona à lista
            image_components.append(html.Img(src=file_path, style={'width': '50%', 'height': 'auto'}))

    loading_status = ""
    return loading_status, image_components, best_hps_str, model_summary_str, r2_str

@app.callback(
    Output('column-input-selector', 'options'),
    Output('column-input-selector', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(list_of_contents, list_of_names):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names)
        return [{'label': col, 'value': col} for col in df.columns], df.columns.tolist()
    return [], []

@app.callback(
    Output('column-output-selector', 'options'),
    Output('column-output-selector', 'value'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_dropdown(list_of_contents, list_of_names):
    if list_of_contents is not None:
        df = parse_contents(list_of_contents, list_of_names)
        return [{'label': col, 'value': col} for col in df.columns], df.columns.tolist()
    return [], []

@app.callback(
    Output('input-table', 'columns'),
    Output('input-table', 'data'),
    Input('column-input-selector', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(selected_columns, list_of_contents, list_of_names):
    global Input_Columns
    if list_of_contents is not None and selected_columns is not None:
        df = parse_contents(list_of_contents, list_of_names)
        filtered_df = df[selected_columns]
        columns = [{"name": col, "id": col} for col in filtered_df.columns]
        data = filtered_df.to_dict('records')
        Input_Columns = selected_columns
        return columns, data
    return [], []

@app.callback(
    Output('output-table', 'columns'),
    Output('output-table', 'data'),
    Input('column-output-selector', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_table(selected_columns, list_of_contents, list_of_names):
    global Output_Columns
    if list_of_contents is not None and selected_columns is not None:
        df = parse_contents(list_of_contents, list_of_names)
        filtered_df = df[selected_columns]
        columns = [{"name": col, "id": col} for col in filtered_df.columns]
        data = filtered_df.to_dict('records')
        Output_Columns = selected_columns
        return columns, data
    return [], []

# Roda o app
if __name__ == '__main__':
    app.run_server(debug=False)