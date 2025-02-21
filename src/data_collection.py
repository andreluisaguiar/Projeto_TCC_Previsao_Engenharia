from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from selenium.webdriver.chrome.service import Service
import io

# Função de scraping
def scrape_monografias(url):
    # Configuração do WebDriver
    driver_path = "E:/UFMA-2024(1)/tcc/trabalho-organizado/chromedriver.exe"
    options = webdriver.ChromeOptions()
    driver = webdriver.Chrome(service=Service(driver_path), options=options)

    try:
        # Acessa o site da URL
        driver.get(url)

        # Clicando no botão de busca no site
        wait = WebDriverWait(driver, 20)
        search_button = wait.until(EC.element_to_be_clickable((By.ID, "form:buscar")))
        search_button.click()

        # Aguardando a tela carregar
        time.sleep(10)

        # Localizar a tabela de monografias
        table = driver.find_element(By.CLASS_NAME, "table_lt")
        rows = table.find_elements(By.TAG_NAME, "tr")

        # Extrair dados da tabela
        data = []
        for i in range(1, len(rows) - 1):
            row = rows[i]
            cols = row.find_elements(By.TAG_NAME, "td")

            if len(cols) >= 5:
                ano = cols[0].text.strip()
                date = cols[1].text.strip()
                aluno = cols[2].text.strip()
                orientador = cols[3].text.strip()
                curso = cols[4].text.strip()

                # Verifica se a próxima linha contém o título
                next_row = rows[i + 1]
                next_cols = next_row.find_elements(By.TAG_NAME, "td")

                # Verificar se o <td> possui "colspan" e contém o texto "Título:"
                titulo = ""
                if len(next_cols) == 1 and "Título:" in next_cols[0].text:
                    titulo = next_cols[0].text.split("Título:")[1].strip()

                # Adicionando dados à lista
                data.append([ano, date, aluno, orientador, curso, titulo])

        # Converter para DataFrame
        df = pd.DataFrame(data, columns=["Ano", "Data", "Aluno", "Orientador", "Curso", "Título"])

        # Salvar como arquivo Excel em memória
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)  # Retorna ao início do arquivo para download

        return output
    
    except Exception as e:
        print(f"Erro durante o scraping: {e}")
        return None
    
    finally:
        driver.quit()
