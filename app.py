import streamlit as st
import xgboost as xgb
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Configurações da Página
st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",

)
# CSS para imagem
st.markdown(
    """
    <style>

    img {
        opacity: 0.8; 
        position: static;
        bottom: 20px; 
        right: 10px;  
    }

    </style>
    """,
    unsafe_allow_html=True
)

# Inicializa o estado da página
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Menu Lateral
st.sidebar.title("Navegação")
selection = st.sidebar.radio(
    "",
    ["Home", "Previsão de Obesidade", "Análise de Dados", "Sobre o Projeto"]
)

# Controle do menu lateral
if selection != st.session_state.get('last_selection', "Home"): # Compara com a última seleção conhecida
    st.session_state.page = selection
    st.session_state.last_selection = selection # Guarda a seleção para a próxima rodada

# Definição das Páginas
def home_page():
    st.title("Sistema de Apoio à Decisão: Previsão de Obesidade ⚕️")
    # Separando espaço para imagem
    col_vazia, col_imagem = st.columns([3, 1])
    
    with col_vazia:
        
        st.write("""
        Bem-vindo ao sistema de apoio à decisão para a previsão de obesidade.
        Esta ferramenta utiliza um modelo de Machine Learning para auxiliar
        profissionais de saúde na identificação precoce do risco de obesidade em pacientes.
        Nosso objetivo é fornecer um diagnóstico preditivo para apoiar a equipe médica
        na tomada de decisões, contribuindo para a prevenção e tratamento desta
        condição de saúde cada vez mais prevalente.
        """)
    with col_imagem:

        st.image('img-plano.png', width = 250)

    st.markdown("---")
    st.header("Como funciona?")
    st.write(
        """
        1. **Preencha os Dados:** Vá para a seção "Previsão de Obesidade" no menu lateral.
           Insira as informações solicitadas do paciente.
        2. **Obtenha a Previsão:** Nosso modelo irá analisar os dados e fornecer um
           nível preditivo de obesidade.
        3. **Explore Insights:** Na seção "Análise de Dados", você pode explorar
           visualizações e insights sobre a base de dados utilizada no treinamento
           do modelo.
        """
    )
 
    if st.button("Iniciar Previsão"):
        st.session_state.page = "Previsão de Obesidade"
        st.rerun()

def prediction_page():
    st.title("Diagnóstico Preditivo de Obesidade")
    st.write(
        """
        Por favor, insira as informações do paciente abaixo para obter uma previsão
        do nível de obesidade.
        """
    )

    # Seção de Inputs
    st.header("Dados do Paciente")

    with st.form("obesity_prediction_form"):
        # Usando colunas para organizar os inputs lado a lado
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Informações Pessoais")
            gender = st.selectbox("Gênero", ["Masculino", "Feminino"], help="Gênero do paciente.")
            age = st.number_input("Idade (anos)", min_value=0, max_value=120, value=25, step=1, help="Idade do paciente em anos.")
            height = st.number_input("Altura (metros)", min_value=0.5, max_value=2.5, value=1.70, step=0.01, format="%.2f", help="Altura do paciente em metros.")
            weight = st.number_input("Peso (kgs)", min_value=10.0, max_value=300.0, value=70.0, step=0.1, format="%.1f", help="Peso do paciente em quilogramas.")
            family_history = st.selectbox("Histórico Familiar de Obesidade?", ["Não", "Sim"], help="Algum membro da família sofreu ou sofre de excesso de peso?")
            smoke = st.selectbox("Fuma?", ["Não", "Sim"], help="O paciente fuma?")


        with col2:
            st.subheader("Hábitos e Estilo de Vida")
            favc = st.selectbox("Come alimentos altamente calóricos com frequência?", ["Não", "Sim"], help="Consumo frequente de alimentos de alto valor calórico?")
            fcvc = st.selectbox("Costuma comer vegetais nas refeições?", ["Nunca", "Algumas vezes", "Sempre"], help="Frequência de consumo de vegetais.")
            ncp = st.number_input("Quantas refeições principais por dia?", min_value=1, max_value=3, value=2, step=1, help="Número de refeições principais diárias.")
            caec = st.selectbox("Come algo entre as refeições?", ["Nunca", "Às vezes", "Frequente", "Sempre"], help="Frequência de lanches entre refeições.")
            ch2o = st.selectbox("Quanta água bebe diariamente?", ["Pouco", "Médio", "Muito"], help="Quantidade de água consumida diariamente.")
            scc = st.selectbox("Monitora as calorias que ingere diariamente?", ["Não", "Sim"], help="O paciente monitora a ingestão de calorias?")
            faf = st.selectbox("Com que frequência pratica atividade física?", ["Nenhuma", "Pouca", "Moderada", "Muita"], help="Nível de atividade física.")
            tue = st.selectbox("Quanto tempo usa dispositivos tecnológicos?", ["Pouco", "Médio", "Muito"], help="Tempo de uso de celular, videogame, TV, computador, etc.")
            calc = st.selectbox("Com que frequência bebe álcool?", ["Nunca", "Às vezes", "Frequente", "Sempre"], help="Frequência de consumo de álcool.")
            mtrans = st.selectbox("Qual meio de transporte usa?", ["Automóvel", "Caminhada", "Bicicleta", "Transporte Público", "Moto"], help="Meio de transporte mais comum.")


        # Botão de submissão do formulário
        submitted = st.form_submit_button("Obter Previsão")

    if submitted:

        # Carregar o modelo treinado
        try:
            modelo = joblib.load('modelo.joblib')
            # Obter a lista exata de features que o modelo espera, na ordem correta
            colunas_treinamento = modelo.feature_names_in_.tolist()
        except FileNotFoundError:
            st.error("Erro: O arquivo 'modelo.joblib' não foi encontrado. Certifique-se de que o modelo foi treinado e salvo corretamente.")
            st.stop()
        except AttributeError:
            st.error("Erro: O modelo carregado não possui o atributo 'feature_names_in_'. Verifique se o modelo foi salvo corretamente após o treinamento.")
            st.stop()

        # Carregar o LabelEncoder treinado
        try:
            le = joblib.load('label_encoder.joblib')
            # Criar o dicionário de mapeamento de inteiro para rótulo original (inglês)
            int_to_label_english = {i: label for i, label in enumerate(le.classes_)}
        except FileNotFoundError:
            st.error("Erro: O arquivo 'label_encoder.joblib' não foi encontrado. Certifique-se de que o LabelEncoder foi treinado e salvo corretamente.")
            st.stop()
        except Exception as e:
            st.error(f"Erro ao carregar o LabelEncoder: {e}")
            st.stop()

        # Dicionário de tradução das classes de inglês para português
        translation_map = {
            'Insufficient_Weight': 'Peso Insuficiente',
            'Normal_Weight': 'Peso Normal',
            'Obesity_Type_I': 'Obesidade Tipo I',
            'Obesity_Type_II': 'Obesidade Tipo II',
            'Obesity_Type_III': 'Obesidade Tipo III',
            'Overweight_Level_I': 'Sobrepeso Nível I',
            'Overweight_Level_II': 'Sobrepeso Nível II'
        }

        # Definir as colunas categóricas originais que foram usadas para get.dummie
        original_categorical_cols = [
            'Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
        ]

        # Mapeamento strings para número
        mapeamento_FCVC = {
            "Nunca": 1,
            "Algumas vezes": 2,
            "Sempre": 3
        }

        mapeamento_CH2O = {
            "Pouco": 1,
            "Médio": 2,
            "Muito": 3
        }
        mapeamento_FAF = {
            "Nenhuma": 0,
            "Pouca": 1,
            "Moderada": 2,
            "Muita": 3
        }

        mapeamento_TUE = {
            "Pouco": 0,
            "Médio": 1,
            "Muito": 2
        }
      

        # Mapeando as entradas do Streamlit para o formato esperado pelo modelo
        Gender_mapped = gender.replace("Masculino", 'Male').replace("Feminino", 'Female')
        Age = age
        Height = height
        Weight = weight
        family_history = family_history.replace("Sim", "yes").replace("Não", "no")
        SMOKE = smoke.replace("Sim", "yes").replace("Não", "no")
        FAVC = favc.replace("Sim", "yes").replace("Não", "no")
        FCVC = mapeamento_FCVC.get(fcvc)
        NCP = ncp
        CAEC = caec.replace("Às vezes", 'Sometimes').replace("Frequente", 'Frequently').replace("Sempre", 'Always').replace("Nunca", 'no')
        CH2O = mapeamento_CH2O.get(ch2o)
        SCC = scc.replace("Sim", "yes").replace("Não", "no")
        FAF = mapeamento_FAF.get(faf)
        TUE = mapeamento_TUE.get(tue)
        CALC = calc.replace("Às vezes", 'Sometimes').replace("Frequente", 'Frequently').replace("Sempre", 'Always').replace("Nunca", 'no')
        MTRANS = mtrans.replace("Transporte Público", 'Public_Transportation').replace("Caminhada", 'Walking').replace("Automóvel", 'Automobile').replace("Moto", 'Motorbike').replace("Bicicleta", 'Bike')
        
        # Criar um DataFrame com os dados brutos do usuário
        user_data_raw = {
            'Gender': Gender_mapped,
            'Age': Age,
            'Height': Height,
            'Weight': Weight,
            'family_history': family_history,
            'FAVC': FAVC,
            'FCVC': FCVC,
            'NCP': NCP,
            'CAEC': CAEC,
            'SMOKE': smoke,
            'CH2O': CH2O,
            'SCC': SCC,
            'FAF': FAF,
            'TUE': TUE,
            'CALC': CALC,
            'MTRANS': MTRANS
        }
        df_user_input = pd.DataFrame([user_data_raw])

        # Calcular IMC
        if df_user_input['Height'].iloc[0] == 0:
            st.error("Altura não pode ser zero para calcular o IMC.")
            st.stop()
        df_user_input['IMC'] = df_user_input['Weight'] / (df_user_input['Height']**2)


        # Garanta que IMC seja float64
        df_user_input['IMC'] = pd.to_numeric(df_user_input['IMC'], errors='coerce').astype(np.float64)
        df_user_input['IMC'].fillna(0.0, inplace=True) # Preencher NaNs se houver (ex: altura zero)


        #Aplicar Dummie Encoding e Alinhar Features
        df_encoded = pd.get_dummies(df_user_input, columns=original_categorical_cols, dtype=bool)

        df_final_para_previsao = pd.DataFrame(columns=colunas_treinamento)
        df_final_para_previsao.loc[0] = False # Preenche a primeira linha com False

        for col in df_encoded.columns:
            if col in colunas_treinamento:
                df_final_para_previsao[col] = df_encoded[col].iloc[0]


        # Garantir que as colunas numéricas sejam float64 e as booleanas sejam bool
        for col in colunas_treinamento:
            if col in ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'IMC']:
                df_final_para_previsao[col] = df_final_para_previsao[col].astype(np.float64)
            else:
                df_final_para_previsao[col] = df_final_para_previsao[col].astype(bool)



        #Previsão
        try:
            # Faça a previsão
            predicao_numerica = modelo.predict(df_final_para_previsao)[0] # Pega o primeiro (e único) valor da previsão
            predicao_proba = modelo.predict_proba(df_final_para_previsao)[0] # Pega as probabilidades da primeira (e única) linha

            # Decodificar a previsão numérica para o nome da classe em inglês e depois traduzir
            classe_prevista_english = int_to_label_english[predicao_numerica]
            classe_prevista_portugues = translation_map.get(classe_prevista_english, classe_prevista_english) # Usa .get para segurança

        except Exception as e:
            st.error(f"Ocorreu um erro durante a previsão: {e}")
            st.warning("Verifique se os dados de entrada estão no formato correto para o seu modelo.")       



        # Previsão
        with st.spinner('Processando previsão...'):


            if classe_prevista_portugues == 'Peso Insuficiente': 
                prediction_result = 'Peso Insuficiente'
                recommendation = "É crucial buscar avaliação profissional para investigar a causa do baixo peso e desenvolver um plano alimentar adequado."
                result_color = "warning"

            if classe_prevista_portugues == 'Peso Normal': 
                prediction_result = 'Peso Normal'
                recommendation = "Parabéns! Mantenha seus hábitos saudáveis."
                result_color = "success"

            if classe_prevista_portugues == 'Sobrepeso Nível I': 
                prediction_result = 'Sobrepeso Nível I'
                recommendation = "Atenção ao peso! Considere ajustar hábitos alimentares e aumentar a atividade física."
                result_color = "info"

            if classe_prevista_portugues == 'Sobrepeso Nível II': 
                prediction_result = 'Sobrepeso Nível II'
                recommendation = "Atenção ao peso! Considere ajustar hábitos alimentares e aumentar a atividade física. Você já é quase Obeso"
                result_color = "info"

            if classe_prevista_portugues == 'Obesidade Tipo I': 
                prediction_result = 'Obesidade Tipo I'
                recommendation = "Recomenda-se buscar orientação de um especialista em saúde para acompanhamento e planejamento nutricional/físico."
                result_color = "warning"

            if classe_prevista_portugues == 'Obesidade Tipo II': 
                prediction_result = 'Obesidade Tipo II'
                recommendation = "Recomenda-se buscar orientação de um especialista em saúde para acompanhamento e planejamento nutricional/físico."
                result_color = "warning"

            if classe_prevista_portugues == 'Obesidade Tipo III': 
                prediction_result = 'Obesidade Tipo III'
                recommendation = "Recomenda-se buscar orientação de um especialista em saúde para acompanhamento e planejamento nutricional/físico."
                result_color = "warning"


        st.markdown("---")
        st.subheader("Resultado da Previsão:")
        if result_color == "success":
            st.success(f"O paciente foi classificado com: **{prediction_result}**")
        elif result_color == "warning":
            st.warning(f"O paciente foi classificado com: **{prediction_result}**")
        else:
            st.info(f"O paciente foi classificado com: **{prediction_result}**")

        st.write(f"**Recomendação:** {recommendation}")



def analytics_page():
    #%%
    #Espaço para analises e dados
    import matplotlib.pyplot as plt
    import pandas as pd
    df_pt = pd.read_csv("df_pt.csv")

    #%%


    #%%






    st.title("Insights sobre Obesidade")
    st.write(
        """
        Nesta seção, você encontrará visualizações e insights extraídos da base de dados
        utilizada para treinar o modelo. Estes gráficos podem ajudar a equipe médica
        a compreender melhor os fatores relacionados à obesidade.
        """
    )
    st.markdown("---")
    st.header("Dados da pesquisa")
    st.write(
        """
        Para embasar este projeto, foram cuidadosamente analisados **2111** registros de pacientes, constituindo uma base de dados robusta para a compreensão da obesidade.  
        * **Gênero do indivíduo**: Sexo do participante.
        * **Idade**: Idade do participante em anos.
        * **Altura**: Altura do participante em metros.
        * **Peso**: Peso do participante em quilogramas.
        * **Histórico familiar de excesso de peso**: Presença de histórico familiar de obesidade.
        * **Frequência de consumo de alimentos altamente calóricos (FAVC)**: Regularidade da ingestão de itens de alta densidade energética.
        * **Frequência de consumo de vegetais nas refeições (FCVC)**: Regularidade da ingestão de vegetais nas refeições.
        * **Número de refeições principais diárias (NCP)**: Quantidade de refeições principais consumidas por dia.
        * **Consumo de alimentos entre as refeições (CAEC)**: Hábito de consumir alimentos fora das refeições principais.
        * **Status de fumante (SMOKE)**: Indicação se o participante é fumante.
        * **Consumo diário de água (CH2O)**: Volume de água ingerido diariamente.
        * **Monitoramento diário de calorias (SCC)**: Hábito de monitorar a ingestão calórica diária.
        * **Frequência de atividade física (FAF)**: Regularidade da prática de atividade física.
        * **Tempo de uso de dispositivos tecnológicos (TER)**: Horas dedicadas ao uso de dispositivos tecnológicos.
        * **Frequência de consumo de álcool (CALC)**: Regularidade da ingestão de bebidas alcoólicas.
        * **Meio de transporte utilizado (MTRANS)**: Principal meio de locomoção do participante.
        """
    )
    st.markdown("---")
    st.subheader("Distribuição dos Níveis de Obesidade")

    contagem_diagnostico = df_pt['Diagnóstico'].value_counts()
    diag_na_ordem = ['Peso Insuficiente', 'Peso Normal', 'Sobrepeso Nível I',  'Sobrepeso Nível II', 'Obesidade Tipo I', 'Obesidade Tipo II', 'Obesidade Tipo III']
    contagem_diagnostico_ordenada = contagem_diagnostico.reindex(diag_na_ordem, fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
    contagem_diagnostico_ordenada.index,
    contagem_diagnostico_ordenada.values,
    color='#1f77b4', # Uma cor padrão do Matplotlib (azul)
    edgecolor='black'
    )
    ax.set_xlabel('Diagnóstico', fontsize=12, labelpad=10)
    ax.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax.set_title('Distribuição de Pacientes por Diagnóstico', fontsize=14, pad=15)
    ax.set_xticks(contagem_diagnostico_ordenada.index)
    ax.set_xticklabels(contagem_diagnostico_ordenada.index, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    col1, col2 = st.columns([0.8, 0.2])
    with col1:
        st.pyplot(fig)
    with col2:
        ""
    st.write("*Obesidade Tipo I* apresenta a maior prevalência entre as categorias de diagnóstico")
    st.markdown("---")
    st.warning("Para o restante da análise, utilizaremos apenas dados de pacientes diagnosticados com Obesidade (Tipo I, Tipo II ou Tipo III).")

    #Distribuição de Genero com obesidade
    st.markdown("---")
    st.subheader("Análise da Distribuição de Gênero em Pacientes com Diagnóstico de Obesidade")

    niveis_obesidade = ['Obesidade Tipo I', 'Obesidade Tipo II', 'Obesidade Tipo III']
    df_obesidade = df_pt[df_pt['Diagnóstico'].isin(niveis_obesidade)].copy()

    df_fm = df_obesidade['Gênero'].value_counts()
    gender_counts = pd.Series(df_fm)

    labels = gender_counts.index
    sizes = gender_counts.values

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.pie(sizes, autopct='%1.1f%%', startangle=90, labels=labels)
    ax.axis('equal')
    ax.set_title('Distribuição de Gênero')
    col1, col2 = st.columns([0.5, 0.5]) # Divide o espaço em duas colunas, 50% para cada

    with col1: # Coloque o gráfico na primeira coluna
        st.pyplot(fig) # Exibe a figura do Matplotlib

    with col2: # Esta coluna pode ser usada para texto ou outros elementos
        st.write("Embora haja uma leve predominância masculina nesta amostra, é crucial ressaltar que essa diferença percentual mínima não se configura como um **fator determinante**")
        st.write(f"Total de indivíduos: {sum(sizes)}")
        st.write(f"Homens: {sizes[0]} ({labels[0]})")
        st.write(f"Mulheres: {sizes[1]} ({labels[1]})")
        
    #Distribuiçao de obesidade por faixa etaria
    st.markdown("---") 
    bins = [0, 18, 25, 35, 45, 60, float('inf')]
    age_labels = ['Até 18 anos', '18 a 25 anos', '25 a 35 anos', '35 a 45 anos', '45 a 60 anos', '60+ anos']
    df_obesidade['Age_Group'] = pd.cut(df_obesidade['Idade'], bins=bins, labels=age_labels, right=True)
    age_group_counts = df_obesidade['Age_Group'].value_counts().reindex(age_labels, fill_value=0)

    st.subheader("Distribuição de Indivíduos diagnosticados com Obesidade por Grupo de Idade")

    fig_age_bar, ax_age_bar = plt.subplots(figsize=(10, 6))

    ax_age_bar.bar(
        age_group_counts.index,
        age_group_counts.values,
        color='#1f77b4', 
        edgecolor='black'
    )

    ax_age_bar.set_xlabel('Grupo de Idade', fontsize=12, labelpad=10)
    ax_age_bar.set_ylabel('Número de Indivíduos', fontsize=12, labelpad=10)
    ax_age_bar.set_title('Distribuição de Pacientes por Faixa Etária', fontsize=14, pad=15)

    ax_age_bar.set_xticks(age_group_counts.index)
    ax_age_bar.set_xticklabels(age_group_counts.index, rotation=45, ha='right', fontsize=10)

    ax_age_bar.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    col1, col2 = st.columns([0.7, 0.3]) 
    with col1:

        st.pyplot(fig_age_bar)
    with col2:

        most_common_age_group = age_group_counts.idxmax()
        most_common_age_percentage = (age_group_counts.max() / age_group_counts.sum()) * 100

        st.write(
            f"A faixa etária **{most_common_age_group}** apresenta a maior prevalência entre as categorias de diagnóstico, representando aproximadamente **{most_common_age_percentage:.1f}%** da amostra."
        )

    #Histórico familiar 

    st.markdown("---")
    st.subheader("Histórico Familiar de Obesidade")

    # Contar a ocorrência de 'Sim' e 'Não'
    # Garantir a ordem 'Sim', 'Não' para consistência no gráfico
    family_history_counts = df_obesidade['Histórico familiar'].value_counts().reindex(['sim', 'não'], fill_value=0)

    # Criar a figura e os eixos para o gráfico de barras de histórico familiar
    fig_family_history, ax_family_history = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_family_history.bar(
        family_history_counts.index,
        family_history_counts.values,
        color=['#1f77b4', '#d62728'], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_family_history.set_xlabel('Histórico Familiar', fontsize=12, labelpad=10)
    ax_family_history.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_family_history.set_title('Distribuição por Histórico Familiar de Obesidade', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_family_history.set_xticks(family_history_counts.index)
    ax_family_history.set_xticklabels(family_history_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_family_history.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_family_history)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_family_history = family_history_counts.idxmax()
        most_common_family_history_percentage = (family_history_counts.max() / family_history_counts.sum()) * 100

        st.write(
        f"A maioria dos pacientes (**{most_common_family_history_percentage:.1f}%**) **{most_common_family_history.lower()}** possui histórico familiar de obesidade, o que pode indicar uma predisposição genética ou ambiental significativa."
         )
    st.markdown("---")    


    # Consulmo de alimentos altamente caloricos

    st.subheader("Consulmo de alimentos altamente calóricos")

    cal_food_counts = df_obesidade['Alimentos calóricos'].value_counts().reindex(['sim', 'não'], fill_value=0)
    
    fig_cal_food, ax_cal_food = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_cal_food.bar(
        cal_food_counts.index,
        cal_food_counts.values,
        color=['#1f77b4', '#d62728'], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_cal_food.set_xlabel('Consome alimentos altamente calóricos', fontsize=12, labelpad=10)
    ax_cal_food.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_cal_food.set_title('Distribuição por consulmo de alimentos altamente calóricos', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_cal_food.set_xticks(cal_food_counts.index)
    ax_cal_food.set_xticklabels(cal_food_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_cal_food.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_cal_food)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_cal_food = cal_food_counts.idxmax()
        most_common_cal_food_percentage = (cal_food_counts.max() / cal_food_counts.sum()) * 100

        st.write(
        f"Assim como a maioria dos pacientes com obesidade possui histórico familiar, grande parte desses indivíduos também consome alimentos altamente calóricos com frequência (**{most_common_cal_food_percentage:.1f}%**)"
         )







    # Consulmo de alimentos entre refeições

    st.subheader("Consulmo de alimentos entre as refeições")

    entre_ref_counts = df_obesidade['Entre Refeições'].value_counts().reindex(['Sempre', 'Frequentemente', 'Às vezes', 'não'], fill_value=0)
    
    fig_ente_ref, ax_entre_ref = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_entre_ref.bar(
        entre_ref_counts.index,
        entre_ref_counts.values,
        color='#1f77b4', # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_entre_ref.set_xlabel('Consulmo de alimentos entre as refeições', fontsize=12, labelpad=10)
    ax_entre_ref.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_entre_ref.set_title('Distribuição por Consulmo de alimentos entre as refeições', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_entre_ref.set_xticks(entre_ref_counts.index)
    ax_entre_ref.set_xticklabels(entre_ref_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_entre_ref.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_ente_ref)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_entre_ref = entre_ref_counts.idxmax()
        most_common_entre_ref_percentage = (entre_ref_counts.max() / entre_ref_counts.sum()) * 100

        st.write(
        f"A grande maioria dos entrevistados diagnosticados com obesidade ({most_common_entre_ref_percentage:.1f}%) relata consumir alimentos entre as refeições **Às Vezes**. Essa frequência ocasional, por si só, não se mostra um fator determinante para o desenvolvimento da obesidade."
         )
    st.markdown('---')






    # Fumantes

    st.subheader("Status de Fumante")

    smoke_counts = df_obesidade['Fumante'].value_counts().reindex(['sim', 'não'], fill_value=0)
    
    fig_smoke, ax_smoke = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_smoke.bar(
        smoke_counts.index,
        smoke_counts.values,
        color=['#1f77b4', "#572323"], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_smoke.set_xlabel('Status de Fumante', fontsize=12, labelpad=10)
    ax_smoke.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_smoke.set_title('Distribuição por Status de Fumante', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_smoke.set_xticks(smoke_counts.index)
    ax_smoke.set_xticklabels(smoke_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_smoke.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_smoke)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_smoke = smoke_counts.idxmax()
        most_common_smoke_percentage = (smoke_counts.min() / smoke_counts.sum()) * 100

        st.write(
        f"Apenas ({most_common_smoke_percentage:.1f}%) dos indivíduos diagnosticados com obesidade são fumantes. Essa baixa proporção sugere que, nesta amostra, o tabagismo não se mostra um fator preponderante para a obesidade."
         )

    st.markdown('---')


    #Consulmo de agua

    df_obesidade['Água'] = df_obesidade['Água'].round()
    mapeamento_substituicao_agua = {
        1.0: 'Pouco',
        2.0: 'Medio',
        3.0: 'Muito'
    }
    df_obesidade['Água'] = df_obesidade['Água'].replace(mapeamento_substituicao_agua)
    
    st.subheader("Consulmo de água")

    agua_counts = df_obesidade['Água'].value_counts().reindex(['Pouco', 'Medio', 'Muito'], fill_value=0)
    
    fig_agua, ax_agua = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_agua.bar(
        agua_counts.index,
        agua_counts.values,
        color=['#1f77b4', "#073b5f"], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_agua.set_xlabel('Consulmo de água', fontsize=12, labelpad=10)
    ax_agua.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_agua.set_title('Distribuição por Consulmo de água', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_agua.set_xticks(agua_counts.index)
    ax_agua.set_xticklabels(agua_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_agua.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_agua)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_agua = agua_counts.idxmax()
        less_commom_agua = agua_counts.idxmin()
        middle_category_count = agua_counts.get('Muito', 0)
        middle_common_agua_percentage = (agua_counts.max() / agua_counts.sum()) * 100
        most_common_agua_percentage = (middle_category_count / agua_counts.sum()) * 100
        min_common_agua_percentage = (agua_counts.min() / agua_counts.sum()) * 100

        st.write(
        f"Em relação ao consumo diário de água, a análise dos dados revela que **{min_common_agua_percentage:.1f}**% dos entrevistados relatam um consumo baixo, enquanto **{most_common_agua_percentage:.1f}**% indicam um consumo médio. Por sua vez, **{middle_common_agua_percentage:.1f}**% afirmam consumir uma quantidade elevada de água."
         )


    st.markdown('---')

#Dispositivos tech

    df_obesidade['Dispositivos tech'] = df_obesidade['Dispositivos tech'].round()
    mapeamento_substituicao_tech = {
        0.0: 'Pouco',
        1.0: 'Medio',
        2.0: 'Muito'
    }
    df_obesidade['Dispositivos tech'] = df_obesidade['Dispositivos tech'].replace(mapeamento_substituicao_tech)
    
    st.subheader('Tempo de uso de dispositivos tecnológicos')

    tech_counts = df_obesidade['Dispositivos tech'].value_counts().reindex(['Pouco', 'Medio', 'Muito'], fill_value=0)


    fig_tech, ax_tech = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_tech.bar(
        tech_counts.index,
        tech_counts.values,
        color=['#1f77b4', "#073b5f"], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_tech.set_xlabel('Tempo de uso de dispositivos tecnológicos', fontsize=12, labelpad=10)
    ax_tech.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_tech.set_title('Distribuição por Tempo de uso de dispositivos tecnológicos', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_tech.set_xticks(tech_counts.index)
    ax_tech.set_xticklabels(tech_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_tech.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_tech)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_tech = tech_counts.idxmax()
        most_common_tech_percentage = (tech_counts.max() / tech_counts.sum()) * 100

        st.write(
        f"Com {most_common_tech_percentage:.1f}% dos pacientes diagnosticados com obesidade reportando pouco uso de dispositivos tecnológicos, e considerando que apenas uma pequena porcentagem (se disponível, adicione o valor aqui) relata uso excessivo, pode-se inferir que o tempo prolongado em dispositivos tecnológicos não se mostra um fator primário ou altamente correlacionado com a obesidade nesta amostra."
         )

    st.markdown('---')
    #Alcool
    st.subheader('Frequência de consumo de álcool')

    alcool_counts = df_obesidade['Álcool'].value_counts().reindex(['Frequentemente', 'Às vezes', 'não'], fill_value=0)

    fig_alcool, ax_alcool = plt.subplots(figsize=(8, 5)) # Tamanho ajustado para 2 barras

    # Criar o gráfico de barras
    ax_alcool.bar(
        alcool_counts.index,
        alcool_counts.values,
        color=['#1f77b4', "#073b5f"], # Cores: Verde para 'Sim', Vermelho para 'Não'
        edgecolor='black'
    )

    # Definir rótulos e título
    ax_alcool.set_xlabel('Frequência de consumo de álcool', fontsize=12, labelpad=10)
    ax_alcool.set_ylabel('Número de Pacientes', fontsize=12, labelpad=10)
    ax_alcool.set_title('Distribuição por consumo de álcool', fontsize=14, pad=15)

    # Remover ticks do eixo X se houver apenas 'Sim' e 'Não' para um visual mais limpo
    ax_alcool.set_xticks(alcool_counts.index)
    ax_alcool.set_xticklabels(alcool_counts.index, rotation=0, ha='center', fontsize=10) # Rotação 0 para rótulos curtos

    # Adicionar grid no eixo Y
    ax_alcool.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajustar layout
    plt.tight_layout()

    col1, col2 = st.columns([0.5, 0.5])
    with col1:

    # Exibir o gráfico no Streamlit
        st.pyplot(fig_alcool)
    with col2:
    # Identificar a categoria mais prevalente para a frase de destaque
        most_common_alcool = alcool_counts.idxmax()
        min_common_alcool = alcool_counts.idxmin()
        most_common_alcool_percentage = (alcool_counts.max() / alcool_counts.sum()) * 100
        lest_common_alcool_percentage = (alcool_counts.min() / alcool_counts.sum()) * 100
        middle_common_alcool_percentage = (alcool_counts.get('não', 0) / alcool_counts.sum()) * 100

        st.write(
        f"Em relação à frequência de consumo de álcool, observa-se que apenas **{lest_common_alcool_percentage:.1f}**% dos entrevistados relatam consumir álcool frequentemente. A maior parte **{most_common_alcool_percentage:.1f}**% indica um consumo ocasional (Ás vezes), enquanto **{middle_common_alcool_percentage:.1f}**% afirmam não consumir álcool."
         )


    st.markdown('---')

    st.subheader('Conclusão')
    st.write(
        """
        Com base na análise de **972 diagnósticos de pacientes**, a **Obesidade Tipo I** é a categoria mais prevalente.

        Em termos demográficos, a faixa etária de **18 a 25 anos** é a mais representativa **(44.5%)**, e a leve predominância masculina (50.5% vs. 49.5%) não se mostra um fator determinante isolado.

        Os dados indicam uma forte associação com:

        * **Histórico familiar de obesidade (99.2%)**.

        * **Consumo frequente de alimentos altamente calóricos (98.0%)**.

        * **Consumo ocasional de alimentos entre refeições (98.1%)**, embora essa frequência, por si só, não seja um fator determinante.

        Fatores como **tabagismo (2.3%), uso excessivo de tecnologia (baixa prevalência) e consumo frequente de álcool (1.6%)** não se mostram preponderantes nesta amostra.
        """
    )

    if st.button("Sobre o Projeto"):
        st.session_state.page = "Sobre o Projeto"
        st.rerun()




def about_page():
    st.title("Sobre o Projeto e o Modelo")
    st.write(
        """
        Este projeto foi desenvolvido como parte do **Tech Challenge - Fase 04**
        de **Data Analytics POS TECH**.
        """
    )
    st.markdown("---")
    st.subheader("Objetivo do Modelo")
    st.write(
        """
        O principal objetivo é criar um modelo de Machine Learning capaz de prever
        o nível de obesidade de um indivíduo com base em diversos fatores, auxiliando
        a equipe médica no diagnóstico precoce e na tomada de decisões estratégicas.
        """
    )

    st.subheader("Tecnologias Utilizadas")
    st.markdown(
        """
        * **Linguagem:** Python
        * **Framework de Aplicação:** Streamlit
        * **Bibliotecas de ML:** Scikit-learn, XGBoost
        * **Manipulação de Dados:** Pandas
        * **Visualização de Dados:** Matplotlib, Seaborn, Plotly
        """
    )

    st.subheader("Métricas do Modelo")
    st.info(f"Acurácia do Modelo: **98,58%**.")
    st.write("O modelo foi treinado e validado para garantir uma alta assertividade nas predições.")

    st.subheader("Limitações")
    st.warning(
        """
        É importante ressaltar que este sistema é uma **ferramenta de apoio** ao diagnóstico
        e **não substitui** a avaliação e o julgamento clínico de um profissional de saúde.
        As previsões são baseadas nos dados de treinamento e podem variar com novos dados
        ou com particularidades individuais não capturadas pelo modelo.
        """
    )

    st.subheader("Links Úteis")
    st.markdown("https://www.youtube.com/watch?v=XX3bv4_tWnY")
    st.markdown("https://github.com/antonioaugusto79/Sistema_Preditivo_de_Obesidade")


# Exibir a página selecionada
if st.session_state.page == "Home":
    home_page()
elif st.session_state.page == "Previsão de Obesidade":
    prediction_page()
elif st.session_state.page == "Análise de Dados":
    analytics_page()
elif st.session_state.page == "Sobre o Projeto":
    about_page()
# %%
