import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

def previsao():
    st.title("Previsão da Engenharia Cursada")

    # Caminho fixo onde os modelos estão salvos
    caminho_pasta = r"C://Users//User//Documents//Projeto_TCC_Previsao_Engenharia//models"

    # Carregar o modelo salvo
    modelo_nome = st.selectbox("Escolha o modelo salvo", ["Random Forest", "XGBoost", "MultinomialNB", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier"])
    
    try:
        # Carregar o modelo
        caminho_modelo = f"{caminho_pasta}\\{modelo_nome}_modelo.pkl"
        modelo = joblib.load(caminho_modelo)
    except FileNotFoundError:
        st.error(f"Modelo '{modelo_nome}' não encontrado! Certifique-se de que o modelo foi salvo corretamente.")
        return
    
    # Carregar o TfidfVectorizer usado para transformar o título
    try:
        caminho_tfidf = f"{caminho_pasta}\\{modelo_nome}_tfidf.pkl"
        tfidf = joblib.load(caminho_tfidf)
    except FileNotFoundError:
        st.error(f"TfidfVectorizer do modelo '{modelo_nome}' não encontrado! Certifique-se de que foi salvo corretamente.")
        return

    # Carregar o LabelEncoder utilizado durante o treinamento
    try:
        caminho_le = f"{caminho_pasta}\\{modelo_nome}_le.pkl"
        le = joblib.load(caminho_le)
    except FileNotFoundError:
        st.error(f"LabelEncoder do modelo '{modelo_nome}' não encontrado! Certifique-se de que foi salvo corretamente.")
        return

    # Exibir informações do modelo
    st.subheader(f"Informações do Modelo: {modelo_nome}")
    st.write(f"Modelo: {modelo}")
    
    # Permitir que o usuário insira um título para previsão
    titulo = st.text_input("Digite um título para prever a engenharia:", "")
    
    if titulo:
        # Transformar o título utilizando o mesmo TF-IDF
        titulo_transformado = tfidf.transform([titulo])
        
        # Fazer a previsão
        predicao = modelo.predict(titulo_transformado)
        
        # Verificar se o modelo tem o método predict_proba
        if hasattr(modelo, 'predict_proba'):
            probabilidades = modelo.predict_proba(titulo_transformado)
            st.subheader("Probabilidade para cada Engenharia:")

            # Criar um DataFrame com probabilidades e ordenar do maior para o menor
            probabilidade_df = pd.DataFrame(probabilidades, columns=le.classes_).T
            probabilidade_df.columns = ['Probabilidade']
            probabilidade_df = probabilidade_df.sort_values(by="Probabilidade", ascending=True)

            # Gráfico de Pirâmide (Barras Horizontais)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(probabilidade_df.index, probabilidade_df['Probabilidade'], color='skyblue')
            ax.set_xlabel("Probabilidade")
            ax.set_ylabel("Engenharia")
            ax.set_title(f"Distribuição das Probabilidades - {modelo_nome}")

            # Adicionar valores nas barras
            for i, v in enumerate(probabilidade_df['Probabilidade']):
                ax.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=10)

            st.pyplot(fig)

        else:
            st.warning(f"O modelo {modelo_nome} não possui previsão de probabilidades.")
        
        # Mapear a previsão de volta para a engenharia original
        engenharia_prevista = le.inverse_transform(predicao)
        
        st.subheader(f"Previsão: {engenharia_prevista[0]}")

if __name__ == "_main_":
    previsao()  # Chama a função corretamente