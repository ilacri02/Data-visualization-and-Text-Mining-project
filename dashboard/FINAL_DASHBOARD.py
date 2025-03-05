

from transformers import AutoModelForTokenClassification, AutoTokenizer
from safetensors.torch import load_file
import torch
import dash
from dash import dcc, html
import webbrowser
from dash.dependencies import Input, Output, State
from tensorflow.keras.models import load_model
import numpy as np

from tensorflow.keras.preprocessing.text import tokenizer_from_json

# 1 tokenizer loading
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# Load the tokenizer
with open("tokenizer1.pkl", "rb") as f:
    tokenizer1 = pickle.load(f)

# 2 parameters
embed_dim = 100
lstm_out = 100
max_sequence_len = 73
vocabulary_size_sentence = 3902
num_classes = 21

# 3 embedding glove
import numpy as np
import os
import numpy as np
import urllib.request
import zipfile
import numpy as np

def load_glove_embedding_matrix(word_index, embed_dim, glove_file_path):
    """Load GloVe embeddings."""
    
    # Dictionnary for storing word embeddings
    embeddings_index = {}
    
    # Carica i vettori di embedding da GloVe
    with open(glove_file_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    print(f'Found {len(embeddings_index)} word vectors.')

    # Crea la matrice di embedding (con dimensione vocab_size x embedding_dim)
    embedding_matrix = np.zeros((vocabulary_size_sentence + 1, embed_dim))

    # Popola la matrice di embedding
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Esegui il codice
USE_GLOVE = True
glove_matrix = None

if USE_GLOVE:
    embedding_dim = 100  # o un altro valore se hai un file con una dimensione diversa
    glove_file_path = 'C:/Users/ilaria/Desktop/dashboard/glove.6B.100d.txt'  # Imposta il percorso del tuo file GloVe
    glove_matrix = load_glove_embedding_matrix(tokenizer1.word_index, embedding_dim, glove_file_path)

#4 weights and plot paths

weights_path = "C:/Users/ilaria/Desktop/dashboard/lstm_conll_complete_model.h5"
from tensorflow.keras.models import load_model
model1 = load_model(weights_path)

lda_html_path = "C:/Users/ilaria/Desktop/dashboard/assets/lda_visualization.html"
lda_html_path2 = "C:/Users/ilaria/Desktop/dashboard/assets/lda_visualization2.html"

#6 iob map

from sklearn.preprocessing import LabelEncoder

IOB_tags = ['O', 'B-Organisation', 'I-Organisation', 'B-Temporal', 'I-Temporal',
 'B-Nationality', 'B-Location', 'I-Location', 'B-Person', 'I-Person',
 'B-DocumentReference', 'I-DocumentReference', 'B-Money', 'I-Money',
 'B-Quantity', 'B-MilitaryPlatform', 'I-MilitaryPlatform', 'B-Weapon',
 'I-Weapon', 'I-Quantity', 'I-Nationality']
tag_encoder = LabelEncoder()
tag_encoder.fit(IOB_tags)
tag_to_int = {tag: idx for idx, tag in enumerate(tag_encoder.classes_)}
int_to_tag = {v: k for k, v in tag_to_int.items()}

# 7 prediction function

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Percorso alla cartella del modello
model_path = "C:/Users/ilaria/Desktop/dashboard/final_model"


model = AutoModelForTokenClassification.from_pretrained(model_path).to("cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tag2unique = {
        'B-DocumentReference': 0,
        'B-Location': 1,
        'B-MilitaryPlatform': 2,
        'B-Money': 3,
        'B-Nationality': 4,
        'B-Organisation': 5,
        'B-Person': 6,
        'B-Quantity': 7,
        'B-Temporal': 8,
        'B-Weapon': 9,
        'I-DocumentReference': 10,
        'I-Location': 11,
        'I-MilitaryPlatform': 12,
        'I-Money': 13,
        'I-Nationality': 14,
        'I-Organisation': 15,
        'I-Person': 16,
        'I-Quantity': 17,
        'I-Temporal': 18,
        'I-Weapon': 19,
        'O': 20
    }
print("✅ Modello caricato con successo!")

# app Dash
external_stylesheets = ['/assets/styles.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "My Dashboard"

# Layout
app.layout = html.Div(
    style={'backgroundColor': '#1E1E2F', 'color': 'white', 'font-family': 'Arial, sans-serif', 'padding': '30px'},
    children=[ 
        html.H1("Analysis of Defence Documents", style={'text-align': 'center'}),
        
        # Dropdown menu
        dcc.Dropdown(
            id='section-dropdown',
            options=[
                {'label': 'EDA', 'value': 'EDA'},
                {'label': 'LSTM', 'value': 'model1'},
                {'label': 'BERT', 'value': 'model2'}
            ],
            value='EDA',
            style={  
                'width': '30%',
                'padding': '10px',
                'font-size': '18px',
            }
        ),
        
        # sections
        html.Div(id='eda-section', children=[
            html.Img(
                src='/assets/dep_distribution.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/entity_distribution.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/pos_distribution.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/iob_distribution.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/iob_distribution_no_O.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/tfidf_top_terms.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/tfidf_top_terms_file2.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.H3("LDA Topic Visualization", style={'text-align': 'center'}),
            html.Iframe(
                srcDoc=open(lda_html_path, 'r').read(),
                style={"width": "100%", "height": "800px", "border": "none"}
            ),
            html.H3("LDA Topic Visualization 2", style={'text-align': 'center'}),
            html.Iframe(
                srcDoc=open(lda_html_path2, 'r').read(),
                style={"width": "100%", "height": "800px", "border": "none"}
            )
        ], style={'display': 'block'}),
        html.Div(id='model1-section', children=[
            html.Img(
                src='/assets/training_validation_loss_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/training_validation_accuracy_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/report_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix1_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix2_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix_percentage_lstm.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),

            html.Div([
                html.H4("Insert a sentence:"),
                dcc.Input(id='input-sentence-lstm', type='text', placeholder="Insert here a sentence...", style={'width': '80%', 'padding': '10px'}),
                html.Button('Predict', id='predict-btn-lstm', style={'padding': '10px', 'font-size': '16px', 'margin-top': '10px'})
            ], style={'margin-top': '30px'}),


            html.Div(id='prediction-output-lstm', style={'margin-top': '20px'})  # For LSTM

        ], style={'display': 'block'}),
        html.Div(id='model2-section', children=[  
            html.Img(
                src='/assets/train_validation_loss_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/train_and_validation_accuracy_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/report_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix1_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix2_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Img(
                src='/assets/confusion_matrix_percentage_bert.png',
                style={'width': '80%', 'display': 'block', 'margin': '20px auto'}
            ),
            html.Div([
                html.H4("Insert a sentence:"),
                dcc.Input(id='input-sentence-bert', type='text', placeholder="Insert here a sentence...", style={'width': '80%', 'padding': '10px'}),
                html.Button('Predict', id='predict-btn-bert', style={'padding': '10px', 'font-size': '16px', 'margin-top': '10px'})
            ], style={'margin-top': '30px'}),

            # Output
            html.Div(id='prediction-output-bert', style={'margin-top': '20px'})  # For BERT

        ], style={'display': 'none'})
    ]
)

# Callback
@app.callback(
    [
        Output('eda-section', 'style'),
        Output('model1-section', 'style'),
        Output('model2-section', 'style')
    ],
    [Input('section-dropdown', 'value')]
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

@app.callback(
    [Output('prediction-output-lstm', 'children'), Output('prediction-output-bert', 'children')],
    [Input('predict-btn-lstm', 'n_clicks'), Input('predict-btn-bert', 'n_clicks')],
    [State('input-sentence-lstm', 'value'), State('input-sentence-bert', 'value'), State('section-dropdown', 'value')]
)
def predict_tags(n_clicks_lstm, n_clicks_bert, input_sentence_lstm, input_sentence_bert, selected_section):

    n_clicks = n_clicks_lstm if selected_section == 'model1' else n_clicks_bert

    if n_clicks is None:
        return "", ""

    input_sentence = input_sentence_lstm if selected_section == 'model1' else input_sentence_bert

    if not input_sentence:
        return "", ""
    try:
        predictions = []

        if selected_section == 'model1':  # LSTM Model
            def predict_tags_for_lstm(sentence):
                seq = tokenizer1.texts_to_sequences([sentence])
                X_input = pad_sequences(seq, maxlen=max_sequence_len, padding='post')
                predictions = model1.predict(X_input)[0]
                print(f"Predictions: {predictions}")

                predicted_tags = []
                for pred in predictions:
                    tag_index = np.argmax(pred)
                    if tag_index in int_to_tag:
                        predicted_tags.append(int_to_tag[tag_index])
                    else:
                        predicted_tags.append("O")
                print(f"Predicted tags: {predicted_tags}")
                return list(zip(sentence.split(), predicted_tags))

            predictions = predict_tags_for_lstm(input_sentence)  # Predizione con LSTM

        elif selected_section == 'model2':  # BERT Model
            def predict_tags_for_sentence(sentence, model, tokenizer, tag2unique):
                model.eval()


                tokens = sentence.split()
                inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True,
                                   padding=True).to(model.device)


                with torch.no_grad():
                    outputs = model(**inputs)


                predicted_labels = torch.argmax(outputs.logits, dim=-1).squeeze().cpu().numpy()


                unique2tag = {v: k for k, v in tag2unique.items()}


                predicted_tags = []
                word_ids = inputs.word_ids()

                for label, word_id in zip(predicted_labels, word_ids):
                    if word_id is None:
                        continue

                    else:
                        predicted_tags.append(unique2tag[label])


                return list(zip(tokens, predicted_tags))

            predictions = predict_tags_for_sentence(input_sentence, model, tokenizer, tag2unique)

        if not predictions:
            raise ValueError("Nessuna previsione è stata generata")


        result_str = ""
        for token, tag in predictions:
            result_str += f"{token}: {tag}\n"

        return (
            html.Div([  
                html.H4(f"Frase: {input_sentence}"),
                html.Pre(result_str)
            ]),
            html.Div()
        ) if selected_section == 'model1' else (
            html.Div(),
            html.Div([  
                html.H4(f"Sentence: {input_sentence}"),
                html.Pre(result_str)
            ])
        )


    except Exception as e:
        print(f"Error: {e}")
        return html.Div([html.P(f"Error: {e}")])

if __name__ == '__main__':

    webbrowser.open('http://127.0.0.1:8050', new=2)
    app.run_server(debug=True)
