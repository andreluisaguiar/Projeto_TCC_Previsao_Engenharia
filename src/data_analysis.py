import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib
import streamlit as st
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import stopwords
import os

# Baixar stopwords
nltk.download('stopwords')

def analise_predicao(dados):
    st.title("Análise preditiva das engenharias cursadas após BICT")

    # Carregar stopwords em português
    stop_words = set(stopwords.words('portuguese'))
    
    if {'titulo', 'engenharia'}.issubset(dados.columns):
        st.subheader("Distribuição das engenharias")
        distrib_eng = dados['engenharia'].value_counts()
        st.bar_chart(distrib_eng)

        X = dados['titulo']
        y = dados['engenharia']
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Usando stop_words='english' para o TfidfVectorizer e depois removendo as stopwords manualmente
        tfidf = TfidfVectorizer(max_features=500, min_df=2, max_df=0.8, stop_words='english')  # Usamos 'english'
        X_tfidf = tfidf.fit_transform(X)

        # Filtrando as stopwords manualmente após a transformação
        X_tfidf_filtered = []
        for doc in X:
            filtered_doc = ' '.join([word for word in doc.split() if word.lower() not in stop_words])
            X_tfidf_filtered.append(filtered_doc)
        
        # Aplicando novamente o TfidfVectorizer no texto filtrado
        X_tfidf_filtered = tfidf.fit_transform(X_tfidf_filtered)

        # Gráfico de barras ANTES do SMOTE
        st.subheader("Distribuição das classes ANTES do SMOTE")
        contagem_antes = pd.Series(y_encoded).value_counts().sort_index()
        contagem_antes.index = le.classes_  # Usando os nomes das classes
        fig, ax = plt.subplots()
        sns.barplot(x=contagem_antes.values, y=contagem_antes.index, ax=ax, palette="viridis", orient="h")
        ax.set_title("Distribuição das classes ANTES do SMOTE")
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("Engenharia")
        st.pyplot(fig)

        smote = SMOTE(random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X_tfidf_filtered, y_encoded)

        # Gráfico de barras APÓS o SMOTE
        st.subheader("Distribuição das classes APÓS o SMOTE")
        contagem_apos = pd.Series(y_balanced).value_counts().sort_index()
        contagem_apos.index = le.classes_  # Usando os nomes das classes
        fig, ax = plt.subplots()
        sns.barplot(x=contagem_apos.values, y=contagem_apos.index, ax=ax, palette="viridis", orient="h")
        ax.set_title("Distribuição das classes APÓS o SMOTE")
        ax.set_xlabel("Quantidade")
        ax.set_ylabel("Engenharia")
        st.pyplot(fig)

        # Seleção do tamanho de treino/teste
        test_size_options = [0.1, 0.2, 0.3]  # 90-10, 80-20, 70-30
        test_size = st.selectbox(
            'Escolha o tamanho do split de treino/teste',
            test_size_options,
            format_func=lambda x: f'{int((1 - x) * 100)}-{int(x * 100)}'
        )

        # Ajustando o tamanho do treino/teste com base na escolha
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=test_size, random_state=42)

        # Adicionando os modelos ao dicionário
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', enable_categorical=False, random_state=42),
            "MultinomialNB": MultinomialNB(),
            "SVC": SVC(class_weight='balanced', random_state=42, probability=True),
            "KNeighborsClassifier": KNeighborsClassifier(),
            "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42)
        }

        escolha_modelo = st.selectbox("Escolha o modelo para análise:", ["Selecione", "Random Forest", "XGBoost", "MultinomialNB", "SVC", "KNeighborsClassifier", "DecisionTreeClassifier", "Melhor Modelo"])
        
        if escolha_modelo != "Selecione" and escolha_modelo != "Melhor Modelo":
            modelo = models[escolha_modelo]
            
            try:
                # Validação cruzada para o modelo selecionado
                scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy', error_score='raise')
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)
                report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)

                st.subheader(f"Resultados do Modelo {escolha_modelo}")
                st.write(f"Acurácia Média (Validação Cruzada): {np.mean(scores):.2f}")
                st.write(f"Acurácia no Teste: {accuracy_score(y_test, y_pred):.2f}")
                
                # Exibindo o relatório de classificação
                relatorio_df = pd.DataFrame(report).T.reset_index().rename(columns={
                    "index": "engenharia",
                    "precision": "Precisão",
                    "recall": "Revocação",
                    "f1-score": "F1-Score",
                    "support": "Suporte"
                })

                st.subheader("Relatório de Classificação")
                st.dataframe(relatorio_df)

                # Exibindo a matriz de confusão
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
                ax.set_title(f'Matriz de Confusão - {escolha_modelo}')
                ax.set_xlabel('Predição')
                ax.set_ylabel('Real')
                st.pyplot(fig)
            
            except Exception as e:
                st.error(f"Erro ao treinar o modelo {escolha_modelo}: {e}")

        elif escolha_modelo == "Melhor Modelo":
            melhor_modelo = None
            melhor_acuracia = 0
            melhor_report = None
            melhor_matrix = None
            melhor_scores = None

            # Testando todos os modelos e armazenando o melhor
            for nome, modelo in models.items():
                try:
                    modelo.fit(X_train, y_train)
                    y_pred = modelo.predict(X_test)
                    acuracia = accuracy_score(y_test, y_pred)

                    if acuracia > melhor_acuracia:
                        melhor_acuracia = acuracia
                        melhor_modelo = nome
                        melhor_report = classification_report(y_test, y_pred, output_dict=True, target_names=le.classes_)
                        melhor_matrix = confusion_matrix(y_test, y_pred)
                        # Validação cruzada para o melhor modelo
                        melhor_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy', error_score='raise')

                except Exception as e:
                    st.error(f"Erro ao treinar o modelo {nome}: {e}")

            # Exibindo o melhor modelo
            st.subheader(f"Melhor Modelo: {melhor_modelo}")
            st.write(f"Acurácia: {melhor_acuracia:.2f}")
            st.write(f"Acurácia Média (Validação Cruzada): {np.mean(melhor_scores):.2f}")

            # Exibindo o relatório de classificação
            relatorio_df = pd.DataFrame(melhor_report).T.reset_index().rename(columns={
                "index": "engenharia",
                "precision": "Precisão",
                "recall": "Revocação",
                "f1-score": "F1-Score",
                "support": "Suporte"
            })
            st.subheader("Relatório de Classificação")
            st.dataframe(relatorio_df)

            # Exibindo a matriz de confusão
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(melhor_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
            ax.set_title(f'Matriz de Confusão - {melhor_modelo}')
            ax.set_xlabel('Predição')
            ax.set_ylabel('Real')
            st.pyplot(fig)

            # Adicionando o botão para salvar o modelo
            if st.button('Salvar Modelo'):
                # Caminho fixo para salvar os arquivos
                caminho_pasta = r"C://Users//User//Documents//Projeto_TCC_Previsao_Engenharia//models"
                
                # Verificar se o caminho da pasta existe
                if not os.path.exists(caminho_pasta):
                    st.error(f"A pasta '{caminho_pasta}' não existe.")
                else:
                    # Salvar o modelo
                    caminho_modelo = os.path.join(caminho_pasta, f"{melhor_modelo}_modelo.pkl")
                    joblib.dump(models[melhor_modelo], caminho_modelo)

                    # Salvar o TfidfVectorizer
                    caminho_tfidf = os.path.join(caminho_pasta, f"{melhor_modelo}_tfidf.pkl")
                    joblib.dump(tfidf, caminho_tfidf)

                    # Salvar o LabelEncoder
                    caminho_le = os.path.join(caminho_pasta, f"{melhor_modelo}_le.pkl")
                    joblib.dump(le, caminho_le)

                    st.success(f"Arquivos salvos com sucesso em {caminho_pasta}!")
    
    else:
        st.error("Certifique-se de que o arquivo contém as colunas 'titulo' e 'engenharia'.")