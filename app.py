import base64
import datetime
import io
import os
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_table
import requests
import numpy as np
from ase.io import read
from schnetpack import AtomsData
from gnn_model import gnn_pred

import pandas as pd


app = dash.Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

app.layout = html.Div([
    html.H2('Bandgap prediction for molecular crystals using SchNet model', style={'color': 'yellow'}),
    dbc.Row(
        [
        html.P('Upload a CIF file to get the bandgap value prediction', style={'color': 'yellow'}),
        dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CIF File')
        ]),
    ),
    html.Div(id='output-data-upload'),
    dbc.Modal(
        [
            dbc.ModalHeader("Your Header"),
            dbc.ModalBody("This is the content of the modal...file not XLS"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ml-auto")
            ),
        ],
        id="modal",
        is_open=False,
    )

    ],style={
            'margin': '0',
            'position': 'absolute',
            'color': 'yellow',
            'top': '50%',
            'left': '50%',
            '-ms-transform': 'translate(-50%, -50%)',
            'transform': 'translate(-50%, -50%)'
        }),
    
],
style={
"background-image": 'url("/assets/molecule.jpg")', 
'verticalAlign':'middle',
  'textAlign': 'center','position':'fixed',
  'width':'100%',
  'height':'100%',
  'top':'0px',
  'left':'0px',
  'z-index':'1000'}
)

def save_file(filename, content):
    """Create a Plotly Dash 'A' element that downloads a file from the app."""
    with open(os.path.join("./", filename), "wb") as fp:
        fp.write(content)


def convert_to_db(file_name):
    mol_list = []
    property_list = []
    atoms = read(os.path.join("./", file_name), index=':')
    property_list.append({'band_gap': np.array([-97208.40600498248], dtype=np.float32)})
    mol_list.extend(atoms)
    if os.path.exists("./cod_predict.db"):
        os.remove("./cod_predict.db")
    new_dataset = AtomsData('./cod_predict.db', available_properties=['band_gap'])
    new_dataset.add_systems(mol_list, property_list)


@app.callback([Output('output-data-upload', 'children'),
                Output('modal', 'is_open')],
              [Input('upload-data', 'contents'),
               Input('close', 'n_clicks')],
              [State('upload-data', 'filename'),
               State('upload-data', 'last_modified')])
def update_output(list_of_contents, modal_close, list_of_names, list_of_dates):
    
    print(list_of_contents)
    ctx = dash.callback_context
    user_clicked = ctx.triggered[0]['prop_id'].split('.')[0]
    if not user_clicked or user_clicked == 'close':
        return dash.no_update, False


    if list_of_contents is not None:

        content_type, content_string = list_of_contents.split(',')
        cif_data = base64.b64decode(content_string)
        try:
            if "cif" in list_of_names:
                print(list_of_names)
                save_file(list_of_names, cif_data)
                convert_to_db(list_of_names)
                bandgap = gnn_pred()
                return [html.Br()," The predicted cif value is "+str(bandgap)], False
            else:
                print('dddd')
                return [],True

        except Exception as e:
            print(e)
            return [],True
        # print(list_of_contents)
        # print(list_of_names)
        # # children = [
        # #     parse_contents(c, n, d) for c, n, d in
        # #     zip(list_of_contents, list_of_names, list_of_dates)]
        # return list_of_contents
    else:
        return [], False



if __name__ == '__main__':
    app.run_server(debug=True)