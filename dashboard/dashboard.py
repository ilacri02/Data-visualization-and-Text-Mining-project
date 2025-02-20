import dash
from dash import dcc, html
import webbrowser

external_stylesheets = ['/assets/styles.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "My Dashboard"

# Layout della dashboard
app.layout = html.Div(
    style={'backgroundColor': '#1E1E2F', 'color': 'white', 'font-family': 'Arial, sans-serif', 'padding': '30px'},
    children=[
        html.H1("Analysis of Defence Documents", style={'text-align': 'center'}),
        
        # Dropdown per la selezione (senza lo stile inline)
        dcc.Dropdown(
            id='section-dropdown',
            options=[
                {'label': 'EDA', 'value': 'EDA'},
                {'label': 'LSTM', 'value': 'model1'},
                {'label': 'BERT', 'value': 'model2'}
            ],
            value='EDA',  # Valore di default
            style={  # Rimuoviamo lo stile inline specifico, gestito dal CSS esterno
                'width': '30%',
                'padding': '10px',
                'font-size': '18px',
                # Nota: Gli altri stili saranno gestiti dal CSS esterno
            }
        ),

        html.Div(id='eda-section', children=[
            # Aggiungi l'immagine in questa sezione EDA
            html.Img(
                src='/assets/dep_distribution.png',  # Assicurati che il file si chiami 'eda_plot.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/entity_distribution.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/pos_distribution.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/iob_distribution.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/iob_distribution_no_O.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/tfidf_top_terms.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/tfidf_top_terms_file2.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
        ], style={'display': 'block'}),
        html.Div(id='model1-section', children=[
            html.Img(
                src='/assets/Copia di training and val loss accuracy NER.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/Copia di confusion matrix predictions on test set.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
        ], style={'display': 'block'}),
        html.Div(id='model2-section', children=[
            html.Img(
                src='/assets/train_validation_loss.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/validation_accuracy.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix_percentage.png',  # Assicurati che il file si chiami 'eda_plot3.png' nella cartella assets
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            )


        ], style={'display': 'block'})
    ]
)

# Callback per aggiornare la sezione selezionata
@app.callback(
    [
        dash.dependencies.Output('eda-section', 'style'),
        dash.dependencies.Output('model1-section', 'style'),
        dash.dependencies.Output('model2-section', 'style')
    ],
    [dash.dependencies.Input('section-dropdown', 'value')]
)
def update_section(selected_section):
    eda_style = {'display': 'none'}
    model1_style = {'display': 'none'}
    model2_style = {'display': 'none'}

    if selected_section == 'EDA':
        eda_style = {'display': 'block'}
    elif selected_section == 'model1':
        model1_style = {'display': 'block'}
    elif selected_section == 'model2':
        model2_style = {'display': 'block'}

    return eda_style, model1_style, model2_style

if __name__ == '__main__':
    # Aprire automaticamente il browser
    webbrowser.open('http://127.0.0.1:8050', new=2)
    app.run_server(debug=True)
