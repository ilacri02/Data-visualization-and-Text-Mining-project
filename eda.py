# 1.1) DATA CLEANING

# SETTING DIRECTORY
import os

directory_path = 'C:\\Users\\Loren\\OneDrive\\Desktop\\Progetto'
os.chdir(directory_path)

print(os.listdir())  # Elenca i file nella directory corrente

# IMPORTING THE FILES

def read_iob2_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content


file1 = read_iob2_file('file1.ann')
file2 = read_iob2_file('file2.ann')

print("Contenuto di file1:")
print(file1)

print("\nContenuto di file2:")
print(file2)

# NON LATIN CHARACTERS AND NA ELIMINATION

import re

def pulisci_file(input_filename):
    # Legge tutte le righe dal file di input
    with open(input_filename, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filtra le righe che non contengono caratteri arabi e non sono vuote o "NA"
    filtered_lines = [
        line for line in lines
        if line.strip() and not re.search(r'[\u0600-\u06FF]', line)  # Rimuove righe vuote e con caratteri arabi
    ]

    # Concatena le righe filtrate in una singola stringa mantenendo i ritorni a capo originali
    return ''.join(filtered_lines)


# Pulisci i file
file1_clean = pulisci_file('file1.ann')  # Pulisce file1
file2_clean = pulisci_file('file2.ann')  # Pulisce file2


# 1.2) Exploratory Data Analysis


# setting up the LDA
import pandas as pd

# Carica il file specificando i delimitatori e i nomi delle colonne
df = pd.read_csv('file1_clean', sep='\s+', header=None, names=['Token', 'Tag'])


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_df=0.95, min_df=0.001, stop_words='english')
dtm = cv.fit_transform(df['Token'])
dtm

# LDA

from sklearn.decomposition import LatentDirichletAllocation

LDA = LatentDirichletAllocation(n_components=7,random_state=42)

# This can take awhile, we're dealing with a large amount of documents!
LDA.fit(dtm)

# showing stored words

len(cv.get_feature_names_out())

# Showing Top Words Per Topic
len(LDA.components_)

single_topic = LDA.components_[0]

# Returns the indices that would sort this array.
single_topic.argsort()

# Top 10 words for this topic:
single_topic.argsort()[-10:]

top_word_indices = single_topic.argsort()[-10:]

for index in top_word_indices:
    print(cv.get_feature_names_out()[index])


for index,topic in enumerate(LDA.components_):
    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')
    print([cv.get_feature_names_out()[i] for i in topic.argsort()[-15:]])
    print('\n')
    
    
