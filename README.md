# App Streamlit - Previsão e Clustering

Este é um aplicativo Streamlit focado em análise de crimes com duas funcionalidades principais: **Previsão de Crimes Específicos** e **Clustering de Crimes**.

## 📋 Funcionalidades

### 📈 Dashboard Geral
- Visão geral dos dados de crimes
- Métricas principais (total de ocorrências, bairros afetados, tipos de crime)
- Visualizações interativas dos dados
- Mapa interativo de localização dos crimes
- Análise temporal dos crimes

### 🔮 Previsão de Crimes Específicos
- Modelos de Machine Learning para prever crimes específicos
- Comparação entre Random Forest e Regressão Linear
- Métricas de avaliação (MAE, MSE, R²)
- Análise temporal e por bairro dos crimes específicos
- Gráficos de dispersão para comparar previsões vs valores reais

### 🎯 Clustering de Crimes
- Análise de agrupamentos usando K-Means
- Visualização PCA dos clusters
- Análise detalhada das características de cada cluster
- Heatmap de características
- Distribuição dos crimes por cluster

## 🛠️ Tecnologias Utilizadas

- **Streamlit**: Framework para aplicações web
- **Pandas**: Manipulação e análise de dados
- **Scikit-learn**: Algoritmos de machine learning
- **Plotly**: Visualizações interativas
- **Folium**: Mapas interativos
- **Matplotlib/Seaborn**: Visualizações estáticas

## 📁 Estrutura do Projeto

```
.
├── app.py                          # Aplicação principal Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                      # Documentação do projeto
```

## 🚀 Como Executar

### 0.1 Criar um ambiente virtual python
```bash
python -m venv venv
```

### 0.2 Ativar ambiente virtual
```bash
.\venv\Scripts\activate
```

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Executar o aplicativo
```bash
python -m streamlit run app.py
```

### 3. Acessar o aplicativo
O aplicativo será aberto automaticamente no navegador no endereço:
```
http://localhost:8501
```

## 📊 Dados

O aplicativo utiliza um dataset da policia civil de pernambuco que contém informações sobre ocorrências criminais incluindo:

- **id_ocorrencia**: Identificador único da ocorrência
- **data_ocorrencia**: Data da ocorrência
- **bairro**: Bairro onde ocorreu o crime
- **tipo_crime**: Tipo do crime cometido
- **descricao_modus_operandi**: Descrição do modus operandi
- **arma_utilizada**: Arma utilizada no crime
- **quantidade_vitimas**: Número de vítimas
- **quantidade_suspeitos**: Número de suspeitos
- **sexo_suspeito**: Sexo do suspeito
- **idade_suspeito**: Idade do suspeito
- **orgao_responsavel**: Órgão responsável pela investigação
- **status_investigacao**: Status da investigação
- **latitude/longitude**: Coordenadas geográficas

## 🎯 Modelos de Machine Learning

### Previsão de Crimes Específicos
- **Random Forest Regressor**: Modelo ensemble baseado em árvores de decisão
- **Regressão Linear**: Modelo linear para previsão de valores contínuos
- **Métricas**: MAE (Mean Absolute Error), MSE (Mean Squared Error), R²

### Clustering
- **K-Means**: Algoritmo de clustering não supervisionado
- **PCA**: Redução de dimensionalidade para visualização
- **Padronização**: StandardScaler para normalizar os dados

## 💡 Insights Principais

### Previsão de Crimes
- Identificação de bairros com maior tendência a crimes especificos
- Análise temporal para entender padrões sazonais
- Comparação de diferentes modelos de previsão

### Clustering de Crimes
- Identificação de padrões criminais similares
- Agrupamento de crimes por características comuns
- Insights para estratégias de segurança direcionadas
