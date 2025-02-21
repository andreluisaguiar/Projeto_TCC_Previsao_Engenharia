import pandas as pd
import streamlit as st
import io

# Função para carregar múltiplos arquivos e concatená-los
def carregar_arquivos(files):
    dados = []
    for file in files:
        # Ler o arquivo Excel em um DataFrame
        df = pd.read_excel(file)
        dados.append(df)
    # Concatenar todos os DataFrames
    df_combinado = pd.concat(dados, ignore_index=True)
    return df_combinado

# Função para remover duplicatas
def remover_duplicatas(df):
    # Remover duplicatas
    df_sem_duplicatas = df.drop_duplicates()
    return df_sem_duplicatas

# Função principal
def exibir_remocao_duplicatas():
    # Título da aplicação
    st.subheader("Remoção de Duplicatas")
    
    # Carregar múltiplos arquivos
    uploaded_files = st.file_uploader("Carregar arquivos Excel", type=["xlsx"], accept_multiple_files=True)

    if uploaded_files:
        # Carregar e juntar os arquivos
        st.write("Carregando os arquivos...")
        df_combinado = carregar_arquivos(uploaded_files)

        # Exibir o DataFrame combinado
        st.write("Dados combinados de todos os arquivos:")
        st.dataframe(df_combinado)

        # Botão para remover duplicatas
        if st.button("Remover Duplicatas"):
            df_sem_duplicatas = remover_duplicatas(df_combinado)

            # Exibir os dados sem duplicatas
            st.write("Dados sem duplicatas:")
            st.dataframe(df_sem_duplicatas)

            # Salvar como arquivo Excel
            st.write("Baixar o arquivo sem duplicatas:")

            # Salvar o DataFrame sem duplicatas em um arquivo Excel em memória
            output = io.BytesIO()
            df_sem_duplicatas.to_excel(output, index=False, engine='openpyxl')
            output.seek(0)  # Volta ao início para o Streamlit ler o arquivo

            # Oferecer o arquivo para download
            st.download_button(
                label="Baixar Arquivo Excel",
                data=output,
                file_name="dataset_sem_duplicatas.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
