import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuração da página streamlit
st.set_page_config(
    page_title="Análise de Crimes - ML App",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Import do dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./dataset_ocorrencias_delegacia.csv')
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
        return df
    except FileNotFoundError:
        st.error("dataset não encontrado")
        return None
# preparação da previsão
@st.cache_data
def prepare_prediction_data(df):
    df_pred = df.copy()
    df_pred['month'] = df_pred['data_ocorrencia'].dt.month
    df_pred['year'] = df_pred['data_ocorrencia'].dt.year
    
    # Filtrar apenas latrocínios
    df_latrocinio = df_pred[df_pred['tipo_crime'] == 'Latrocínio'].copy()
    
    # Agregar por bairro, ano e mês
    df_agg = df_latrocinio.groupby(['bairro', 'year', 'month']).size().reset_index(name='crime_count')
    df_agg['date'] = pd.to_datetime(df_agg[['year', 'month']].assign(day=1))
    df_agg = df_agg.sort_values(by=['bairro', 'date'])
    df_agg['previous_month_crime_count'] = df_agg.groupby('bairro')['crime_count'].shift(1)
    df_agg.fillna(0, inplace=True)
    
    return df_agg

# treinamento do modelo e metricas
@st.cache_data
def train_prediction_model(df_agg):
    """Treina os modelos de previsão"""
    # Preparar dados
    df_model = df_agg.copy()
    df_model = pd.get_dummies(df_model, columns=['bairro'])
    
    # Encoding da data
    encoder = LabelEncoder()
    df_model['date'] = encoder.fit_transform(df_model['date'])
    
    # Separar features e target
    X = df_model.drop(columns=['crime_count'])
    y = df_model['crime_count']
    
    # Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar Random Forest
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Treinar Regressão Linear
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # Fazer previsões
    y_pred_rf = rf_model.predict(X_test)
    y_pred_lr = lr_model.predict(X_test_scaled)
    
    # Calcular métricas
    metrics = {
        'Random Forest': {
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'MSE': mean_squared_error(y_test, y_pred_rf),
            'R²': r2_score(y_test, y_pred_rf)
        },
        'Linear Regression': {
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'MSE': mean_squared_error(y_test, y_pred_lr),
            'R²': r2_score(y_test, y_pred_lr)
        }
    }
    
    return {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'encoder': encoder,
        'metrics': metrics,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'y_pred_rf': y_pred_rf,
        'y_pred_lr': y_pred_lr,
        'feature_names': X.columns.tolist()
    }

# preparação para clustering
@st.cache_data
def prepare_clustering_data(df):
    X = df.drop(columns=['id_ocorrencia']).copy()
    
    X['year'] = X['data_ocorrencia'].dt.year
    X['month'] = X['data_ocorrencia'].dt.month
    X['day'] = X['data_ocorrencia'].dt.day
    X = X.drop(columns=['data_ocorrencia'])
    
    # One-hot encoding
    categorical_cols = ['bairro', 'tipo_crime', 'descricao_modus_operandi', 
                       'arma_utilizada', 'sexo_suspeito', 'orgao_responsavel', 
                       'status_investigacao']
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    return X

# clustering
@st.cache_data
def perform_clustering(X, n_clusters=3):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    return {
        'clusters': clusters,
        'X_pca': X_pca,
        'kmeans': kmeans,
        'scaler': scaler,
        'pca': pca
    }

def main():
    # titulo principal
    st.title("Previsão e Análise de Crimes")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navegação")
    page = st.sidebar.selectbox(
        "Escolha uma análise:",
        ["📈 Dashboard Geral", "🔮 Previsão de Latrocínios", "🧫 Clustering de Crimes"]
    )
    
    # Carregar dados
    df = load_data()
    if df is None:
        return
    
    # Dashboard Geral
    if page == "📈 Dashboard Geral":
        st.header("📈 Dashboard Geral dos Crimes")
        
        # Métricas gerais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total de Ocorrências", len(df))
        with col2:
            st.metric("Bairros Afetados", df['bairro'].nunique())
        with col3:
            st.metric("Tipos de Crime", df['tipo_crime'].nunique())
        with col4:
            latrocinios = len(df[df['tipo_crime'] == 'Latrocínio'])
            st.metric("Latrocínios", latrocinios)
        
        st.markdown("---")
        
        # Visualizações
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏘️ Crimes por Bairro")
            crime_counts = df['bairro'].value_counts().head(10)
            fig_bar = px.bar(
                x=crime_counts.values,
                y=crime_counts.index,
                orientation='h',
                title="Top 10 Bairros com Mais Crimes",
                labels={'x': 'Número de Crimes', 'y': 'Bairro'}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("---")
        
        with col2:
            st.subheader("📊 Tipos de Crime")
            crime_types = df['tipo_crime'].value_counts()
            fig_pie = px.pie(
                values=crime_types.values,
                names=crime_types.index,
                title="Distribuição dos Tipos de Crime"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("---")
        
        # Análise temporal
        st.subheader("📅 Análise Temporal")
        
        # Filtro para tipo de crime
        crime_types_temporal = ["Todos"] + sorted(df['tipo_crime'].unique().tolist())
        selected_crime_type = st.selectbox(
            "Filtrar por tipo de crime:",
            crime_types_temporal,
            key="temporal_crime_filter"
        )
        
        # Aplicar filtro
        df_temporal = df.copy()
        if selected_crime_type != "Todos":
            df_temporal = df_temporal[df_temporal['tipo_crime'] == selected_crime_type]
        
        if len(df_temporal) > 0:
            df_temporal['mes'] = df_temporal['data_ocorrencia'].dt.month
            df_temporal['ano'] = df_temporal['data_ocorrencia'].dt.year
            
            temporal_data = df_temporal.groupby(['ano', 'mes']).size().reset_index(name='crimes')
            temporal_data['mes_nome'] = temporal_data['mes'].map({
                1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
            })
            temporal_data['ano_str'] = temporal_data['ano'].astype(str)
            
            title_suffix = f" - {selected_crime_type}" if selected_crime_type != "Todos" else ""
            
            fig_bar_temporal = px.bar(
                temporal_data,
                x='mes_nome',
                y='crimes',
                color='ano_str',
                title=f"Distribuição de Crimes por Mês e Ano{title_suffix}",
                labels={'mes_nome': 'Mês', 'crimes': 'Número de Crimes', 'ano_str': 'Ano'},
                barmode='group'
            )
            fig_bar_temporal.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']}
            )
            st.plotly_chart(fig_bar_temporal, use_container_width=True)
            
            # Mostrar estatísticas do filtro
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Ocorrências", len(df_temporal))
            with col2:
                if len(temporal_data) > 0:
                    pico_mes = temporal_data.loc[temporal_data['crimes'].idxmax(), 'mes_nome']
                    pico_ano = temporal_data.loc[temporal_data['crimes'].idxmax(), 'ano_str']
                    st.metric("Pico de Crimes", f"{pico_mes}/{pico_ano}")
                else:
                    st.metric("Pico de Crimes", "N/A")
            with col3:
                if len(temporal_data) > 0:
                    media_mensal = temporal_data['crimes'].mean()
                    st.metric("Média Mensal", f"{media_mensal:.1f}")
                else:
                    st.metric("Média Mensal", "0")
        else:
            st.warning(f"Nenhum dado encontrado para o tipo de crime: {selected_crime_type}")
        st.markdown("---")
        
        # Mapa de crimes
        st.subheader("🗺️ Mapa de Localização dos Crimes")
        
        # Filtros para o mapa
        col1, col2 = st.columns(2)
        with col1:
            crime_type_filter = st.selectbox(
                "Tipo de Crime:",
                ["Todos"] + list(df['tipo_crime'].unique())
            )
        with col2:
            modus_filter = st.selectbox(
                "Modus Operandi:",
                ["Todos"] + list(df['descricao_modus_operandi'].unique())
            )
        
        # Filtrar dados
        df_map = df.copy()
        if crime_type_filter != "Todos":
            df_map = df_map[df_map['tipo_crime'] == crime_type_filter]
        if modus_filter != "Todos":
            df_map = df_map[df_map['descricao_modus_operandi'] == modus_filter]

        if len(df_map) > 0:
            # Criar mapa
            center_lat = df_map['latitude'].mean()
            center_lon = df_map['longitude'].mean()
            
            m = folium.Map(location=[center_lat, center_lon], zoom_start=11)
            
            # Limitar pontos para performance
            df_sample = df_map.sample(min(500, len(df_map)))
            
            marker_cluster = MarkerCluster().add_to(m)
            
            for _, row in df_sample.iterrows():
                folium.Marker(
                    location=[row['latitude'], row['longitude']],
                    popup=f"<b>Bairro:</b> {row['bairro']}<br><b>Crime:</b> {row['tipo_crime']}<br><b>Data:</b> {row['data_ocorrencia'].strftime('%d/%m/%Y')}",
                    tooltip=f"{row['bairro']} - {row['tipo_crime']}"
                ).add_to(marker_cluster)
            
            folium_static(m)
        else:
            st.warning("Nenhum dado encontrado para os filtros selecionados.")
    
    # Previsão de Latrocínios
    elif page == "🔮 Previsão de Latrocínios":
        st.header("🔮 Previsão de Latrocínios")
        st.markdown("""
        ### Contexto do Problema
        Esta análise visa prever o número de latrocínios que podem ocorrer nos bairros do Recife,
        auxiliando a Polícia Civil na alocação eficiente de recursos e medidas preventivas.
        """)
        
        # Preparar dados e treinar modelo
        with st.spinner("Preparando dados e treinando modelos..."):
            df_agg = prepare_prediction_data(df)
            model_results = train_prediction_model(df_agg)
        
        # Métricas dos modelos
        st.subheader("📊 Performance dos Modelos")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Random Forest**")
            st.write(f"MAE: {model_results['metrics']['Random Forest']['MAE']:.2f}")
            st.write(f"MSE: {model_results['metrics']['Random Forest']['MSE']:.2f}")
            st.write(f"R²: {model_results['metrics']['Random Forest']['R²']:.2f}")
        
        with col2:
            st.write("**Regressão Linear**")
            st.write(f"MAE: {model_results['metrics']['Linear Regression']['MAE']:.2f}")
            st.write(f"MSE: {model_results['metrics']['Linear Regression']['MSE']:.2f}")
            st.write(f"R²: {model_results['metrics']['Linear Regression']['R²']:.2f}")
        
        # Gráficos de dispersão
        st.subheader("📈 Comparação: Valores Reais vs Previsões")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Random Forest
        ax1.scatter(model_results['y_test'], model_results['y_pred_rf'], alpha=0.6)
        ax1.plot([model_results['y_test'].min(), model_results['y_test'].max()], 
                [model_results['y_test'].min(), model_results['y_test'].max()], 'r--')
        ax1.set_title('Random Forest')
        ax1.set_xlabel('Valores Reais')
        ax1.set_ylabel('Previsões')
        
        # Linear Regression
        ax2.scatter(model_results['y_test'], model_results['y_pred_lr'], alpha=0.6)
        ax2.plot([model_results['y_test'].min(), model_results['y_test'].max()], 
                [model_results['y_test'].min(), model_results['y_test'].max()], 'r--')
        ax2.set_title('Regressão Linear')
        ax2.set_xlabel('Valores Reais')
        ax2.set_ylabel('Previsões')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Análise por bairro
        st.subheader("🏘️ Análise de Latrocínios por Bairro")
        
        latrocinios_bairro = df[df['tipo_crime'] == 'Latrocínio']['bairro'].value_counts()
        
        fig_lat = px.bar(
            x=latrocinios_bairro.index,
            y=latrocinios_bairro.values,
            title="Número de Latrocínios por Bairro",
            labels={'x': 'Bairro', 'y': 'Número de Latrocínios'}
        )
        fig_lat.update_xaxes(tickangle=45)
        st.plotly_chart(fig_lat, use_container_width=True)
        
        # Tendência temporal
        st.subheader("📅 Tendência Temporal dos Latrocínios")
        
        df_lat_temporal = df[df['tipo_crime'] == 'Latrocínio'].copy()
        df_lat_temporal['ano_mes'] = df_lat_temporal['data_ocorrencia'].dt.to_period('M')
        lat_temporal = df_lat_temporal.groupby('ano_mes').size().reset_index(name='count')
        lat_temporal['ano_mes'] = lat_temporal['ano_mes'].astype(str)
        
        fig_temporal = px.line(
            lat_temporal,
            x='ano_mes',
            y='count',
            title="Evolução dos Latrocínios ao Longo do Tempo",
            labels={'ano_mes': 'Período (Ano-Mês)', 'count': 'Número de Latrocínios'}
        )
        fig_temporal.update_xaxes(tickangle=45)
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Clustering
    elif page == "🎯 Clustering de Crimes":
        st.header("🎯 Análise de Clustering dos Crimes")
        st.markdown("""
        ### Análise de Agrupamentos
        Esta análise identifica padrões e agrupa crimes similares baseado em suas características,
        ajudando a entender perfis criminais e otimizar estratégias de segurança.
        """)
        
        # Parâmetros do clustering
        st.sidebar.subheader("⚙️ Parâmetros do Clustering")
        n_clusters = st.sidebar.slider("Número de Clusters", 2, 8, 3)
        
        # Preparar dados e realizar clustering
        with st.spinner("Preparando dados e realizando clustering..."):
            X_clustering = prepare_clustering_data(df)
            clustering_results = perform_clustering(X_clustering, n_clusters)
        
        # Adicionar clusters ao dataframe original
        df_clustered = df.copy()
        df_clustered['cluster'] = clustering_results['clusters']
        
        # Visualização PCA
        st.subheader("📊 Visualização dos Clusters (PCA)")
        
        # Criar DataFrame para visualização
        pca_df = pd.DataFrame({
            'PC1': clustering_results['X_pca'][:, 0],
            'PC2': clustering_results['X_pca'][:, 1],
            'Cluster': clustering_results['clusters'].astype(str)
        })
        
        fig_pca = px.scatter(
            pca_df,
            x='PC1',
            y='PC2',
            color='Cluster',
            title="Visualização dos Clusters usando PCA",
            labels={'PC1': 'Primeira Componente Principal', 'PC2': 'Segunda Componente Principal'}
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
        # Análise dos clusters
        st.subheader("🔍 Análise dos Clusters")
        
        # Mostrar características de cada cluster
        for cluster_id in range(n_clusters):
            with st.expander(f"📋 Cluster {cluster_id}"):
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total de Crimes", len(cluster_data))
                with col2:
                    st.metric("Bairros Distintos", cluster_data['bairro'].nunique())
                with col3:
                    st.metric("Tipos de Crime", cluster_data['tipo_crime'].nunique())
                
                # Características principais do cluster
                st.write("**Principais Características:**")
                
                # Bairros mais frequentes
                top_bairros = cluster_data['bairro'].value_counts().head(3)
                st.write(f"- **Bairros principais:** {', '.join(top_bairros.index)}")
                
                # Tipos de crime mais frequentes
                top_crimes = cluster_data['tipo_crime'].value_counts().head(3)
                st.write(f"- **Crimes principais:** {', '.join(top_crimes.index)}")
                
                # Armas mais utilizadas
                top_armas = cluster_data['arma_utilizada'].value_counts().head(3)
                st.write(f"- **Armas principais:** {', '.join(top_armas.index)}")
        
        # Distribuição dos clusters
        st.subheader("📈 Distribuição dos Clusters")
        
        cluster_counts = df_clustered['cluster'].value_counts().sort_index()
        
        fig_cluster_dist = px.bar(
            x=[f"Cluster {i}" for i in cluster_counts.index],
            y=cluster_counts.values,
            title="Distribuição de Crimes por Cluster",
            labels={'x': 'Cluster', 'y': 'Número de Crimes'}
        )
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
        
        # Heatmap de características
        st.subheader("🌡️ Heatmap de Características dos Clusters")
        
        # Criar tabela de características por cluster
        cluster_features = []
        for cluster_id in range(n_clusters):
            cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
            features = {
                'Cluster': f'Cluster {cluster_id}',
                'Total_Crimes': len(cluster_data),
                'Avg_Vitimas': cluster_data['quantidade_vitimas'].mean(),
                'Avg_Suspeitos': cluster_data['quantidade_suspeitos'].mean(),
                'Avg_Idade_Suspeito': cluster_data['idade_suspeito'].mean()
            }
            cluster_features.append(features)
        
        cluster_features_df = pd.DataFrame(cluster_features)
        cluster_features_df = cluster_features_df.set_index('Cluster')
        
        fig_heatmap = px.imshow(
            cluster_features_df.T,
            title="Características dos Clusters (Normalizado)",
            labels={'x': 'Cluster', 'y': 'Características', 'color': 'Valor Normalizado'},
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center'>
        <p>📊 Desenvolvido com Streamlit | 🔍 Análise de Dados Criminais | 🤖 Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()