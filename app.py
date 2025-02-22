import streamlit as st
import pandas as pd
import src.data_collection as data_collection
import src.data_duplicate as data_duplicate
import src.data_outliers as data_outliers  
import src.data_analysis as data_analysis  
import src.data_prediction as data_prediction 
import webbrowser  # Para abrir links no navegador
import io

# Função para o menu lateral
def menu():
    option = st.sidebar.selectbox(
        "ESCOLHA A FUNCIONALIDADE: ",
        ("Coleta de Dados", "Remoção de Duplicatas", "Detecção de Outliers", "Analises e Treinamento","Predição de engenharia")
    )
    
    # # Adicionando o botão para abrir o outro projeto de Análise Descritiva
    # if st.sidebar.button("IR PARA ANÁLISE DESCRITIVA"):
    #     webbrowser.open("http://localhost:8502", new=2)  # Isso abrirá o link em uma nova aba do navegador

    return option

# Função principal para exibir a funcionalidade selecionada
def main():
    st.header("Utilizando Processamento de Linguagem Natural para Prever a Escolha de Engenharia a Partir de Títulos de TCC no BICT")

    escolha = menu()

    if escolha == "Coleta de Dados":
        st.write("Você escolheu Coleta de Dados.")
        # Chama a funcionalidade de coleta de dados
        url_input = st.text_input("Digite a URL do site", " ")
        if st.button("Buscar Monografias"):
            if url_input:
                st.write("Iniciando a extração...")
                excel_data = data_collection.scrape_monografias(url_input)
                if excel_data:
                    st.download_button(
                        label="Baixar Arquivo Excel",
                        data=excel_data,
                        file_name="dataset_monografias.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            else:
                st.error("Por favor, insira uma URL válida.")

    elif escolha == "Remoção de Duplicatas":
        st.write("Você escolheu Remoção de Duplicatas.")
        # Chama a funcionalidade de remoção de duplicatas
        data_duplicate.exibir_remocao_duplicatas()

    elif escolha == "Detecção de Outliers":
        st.write("Você escolheu Detecção de Outliers.")
        # Chama a funcionalidade de detecção de outliers
        exibir_outliers()

    elif escolha == "Analises e Treinamento":
        st.write("Você escolheu Treinamento E Predição.")
        # Chama a funcionalidade de Treinamento E Predição
        exibir_analise_predicao()

    elif escolha == "Predição de engenharia":
        st.write("Você escolheu Predição de engenharia.")
        data_prediction.previsao()  # Alterado para chamar a função correta do módulo previsao


# Função para exibir a detecção de outliers no Streamlit
def exibir_outliers():
    st.subheader("Detecção de Outliers - titulos e engenharia")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Envie o arquivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        # Carregar o arquivo em um DataFrame
        df = pd.read_excel(uploaded_file)

        st.write("Dados carregados:")
        st.dataframe(df)

        # Seleção das colunas para titulo e engenharia
        titulo_col = st.selectbox("Selecione a coluna de titulo", df.columns)
        engenharia_col = st.selectbox("Selecione a coluna de engenharia", df.columns)

        # Verificar se as colunas são iguais
        if titulo_col == engenharia_col:
            st.error("As colunas de titulo e engenharia não podem ser as mesmas. Por favor, selecione colunas diferentes.")
        # Verificar se as colunas não correspondem a "titulo" e "engenharia"
        elif titulo_col.lower() != "titulo" or engenharia_col.lower() != "engenharia":
            st.error("As colunas devem ser exclusivamente 'titulo' e 'engenharia'. Por favor, selecione as colunas corretas.")
        else:
            # Detectar outliers
            df_outliers, df_sem_outliers = data_outliers.detectar_outliers_titulos(df, titulo_col, engenharia_col)

            # Exibir os outliers
            st.write("Outliers detectados:")
            st.dataframe(df_outliers)

            st.write("Dados sem outliers:")
            st.dataframe(df_sem_outliers)

            # Salvar arquivos para download
            df_outliers_buffer = io.BytesIO()
            df_sem_outliers_buffer = io.BytesIO()

            df_outliers.to_excel(df_outliers_buffer, index=False, engine='openpyxl')
            df_sem_outliers.to_excel(df_sem_outliers_buffer, index=False, engine='openpyxl')

            df_outliers_buffer.seek(0)
            df_sem_outliers_buffer.seek(0)

            # Botões de download
            st.download_button(
                label="Baixar Arquivo com Outliers",
                data=df_outliers_buffer,
                file_name="outliers_detectados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            st.download_button(
                label="Baixar Arquivo sem Outliers",
                data=df_sem_outliers_buffer,
                file_name="dados_sem_outliers.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# Função para exibir a Treinamento E Predição no Streamlit
def exibir_analise_predicao():
    st.subheader("Treinamento E Predição - Escolha de engenharia")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Envie o arquivo Excel", type=["xlsx"])

    if uploaded_file is not None:
        # Carregar o arquivo em um DataFrame
        df = pd.read_excel(uploaded_file)

        # Chama a função do módulo de Treinamento E Predição
        data_analysis.analise_predicao(df)

if __name__ == "__main__":
    main()
