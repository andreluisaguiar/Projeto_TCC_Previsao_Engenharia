import pandas as pd
import spacy

# Carregar o modelo spaCy para processamento de texto
nlp = spacy.load("pt_core_news_md")  # Modelo para português

# Função para calcular a similaridade entre o título e a Engenharia
def calcular_similaridade(Titulo, Engenharia):
    # Transformar o título e a Engenharia em objetos spaCy
    doc_Titulo = nlp(Titulo)
    doc_Engenharia = nlp(Engenharia)
    
    # Calcular a similaridade
    return doc_Titulo.similarity(doc_Engenharia)

# Função para detectar outliers baseado na similaridade semântica
def detectar_outliers_titulos(df, Titulo_col, Engenharia_col):
    """
    Função para detectar outliers com base na similaridade entre Título e Engenharia.
    Títulos com similaridade muito baixa com a Engenharia atribuída são considerados outliers.
    """
    df_outliers = pd.DataFrame()
    df_sem_outliers = df.copy()
    
    limite_similaridade = 0.3  # Limite de similaridade para considerar outlier (valor entre 0 e 1)

    for index, row in df.iterrows():
        Titulo = str(row[Titulo_col]).lower()  # Converte o título para minúsculas
        Engenharia = str(row[Engenharia_col]).lower()  # Converte a Engenharia para minúsculas

        # Verifica se a palavra "Engenharia" não está na coluna "Engenharia"
        if "engenharia" not in Engenharia:
            # Se não tiver a palavra "Engenharia", marca como outlier
            df_outliers = pd.concat([df_outliers, row.to_frame().T])
            df_sem_outliers = df_sem_outliers.drop(index)
        else:
            # Caso contrário, calcular a similaridade
            similaridade = calcular_similaridade(Titulo, Engenharia)
            
            # Se a similaridade for abaixo do limite, marcar como outlier
            if similaridade < limite_similaridade:
                df_outliers = pd.concat([df_outliers, row.to_frame().T])
                df_sem_outliers = df_sem_outliers.drop(index)
    
    return df_outliers, df_sem_outliers
