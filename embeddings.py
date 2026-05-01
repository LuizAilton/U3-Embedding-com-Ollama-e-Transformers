'''

OBJETIVOS:

    Desenvolver um script em Python ou um notebook Jupyter que execute as seguintes ações:
        - Conectar ao Ollama localmente.
        - Fornecer um texto simples (por exemplo, uma frase curta em português ou inglês).
        - Gerar os vetores de embeddings correspondentes ao texto utilizando o Ollama.
        - Exibir o vetor de embeddings, seja de forma integral ou por meio de um sumário, como o tamanho do vetor e alguns valores iniciais.

'''
# Pré-requisitos:
    #pip install ollama
    #pip install langchain_community
    #pip install transformers

#Importar as bibliotecas necessárias
import ollama
import pandas as pd
from transformers import AutoTokenizer

#Escolhe o modelo pré-treinado (você pode escolher outros modelos disponíveis no Hugging Face)
modelo1 = "neuralmind/bert-base-portuguese-cased"
modelo2 = "neuralmind/bert-large-portuguese-cased"
modelo3 = "distilbert/distilbert-base-multilingual-cased"
modelo4 = "pablocosta/bertabaporu-base-uncased"

#Carregar um Tokenizador pré-treinado
tokenizer = AutoTokenizer.from_pretrained(modelo1)

#Exemplo de texto para tokenização
texto = 'O impossível só existe até você decidir tentar'

#Tokenização do texto
tokens = tokenizer.tokenize(texto)

#Obter os embeddings
#Neste caso, usando o modelo "nomic-embed-text-v2-moe" da Ollama para gerar os embeddings dos tokens
response = ollama.embed(
    model='nomic-embed-text-v2-moe',
    input=tokens)

# Cria o DataFrame
df = pd.DataFrame({"Token": tokens})

# Mostra uma tabela com os tokens e os embeddings
df["Embeddings"] = response.embeddings
print(df[["Token", "Embeddings"]])

#Mostra a quantidade de embeddings
print(f"\nQuantidade de embeddings: {len(response.embeddings)}")

#Mostra a dimensão dos embeddings
print(f"Dimensão dos embeddings: {len(response.embeddings[0])}")
