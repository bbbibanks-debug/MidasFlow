import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
from scipy import stats
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

# Estrutura de persistência interna unificada
DATA_STORE = {
    'df': None,
    'columns': [],
    'selected_var': None,
    'tables_data_c': None,
    'tables_data_r': None,
    'graphs': []
}

def formatar_numero_terminal(valor):
    """Formata com precisão padrão do mercado: duas casas decimais e separadores brasileiros."""
    if isinstance(valor, (int, float, np.number)):
        if np.isnan(valor):
            return "—"
        s_val = f"{valor:,.2f}"
        return s_val.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(valor)

def processar_metricas_lista(df, col):
    """Calcula todas as 16 métricas e extrai listas de dicionários limpas para os boxes."""
    col_data = df[col].dropna()
    if col_data.empty:
        return [], []
        
    valores = col_data.to_numpy()
    n = len(valores)
    
    # 01. Medidas Clássicas
    m_aritmetica = np.mean(valores)
    mediana = np.median(valores)
    
    moda_res = stats.mode(valores, keepdims=True)
    moda = moda_res.mode[0] if len(moda_res.mode) > 0 else "—"
    
    ponto_medio = (np.max(valores) + np.min(valores)) / 2
    
    try:
        m_geometrica = stats.gmean(valores) if np.all(valores > 0) else "REQUER VALORES > 0"
    except Exception:
        m_geometrica = "ERRO"
        
    try:
        m_harmonica = stats.hmean(valores) if np.all(valores > 0) else "REQUER VALORES > 0"
    except Exception:
        m_harmonica = "ERRO"
        
    m_quadratica = np.sqrt(np.mean(valores**2))
    m_cubica = np.cbrt(np.mean(valores**3))
    
    # 02. Medidas Robustas e Modificadas
    m_aparada = stats.trim_mean(valores, proportiontocut=0.10)
    
    valores_winsor = stats.mstats.winsorize(valores, limits=[0.05, 0.05]).compressed()
    m_winsorizada = np.mean(valores_winsor)
    
    valores_sub = valores
    if n > 400:
        np.random.seed(42)
        valores_sub = np.random.choice(valores, size=400, replace=False)
    
    par_means = (valores_sub[:, None] + valores_sub) / 2.0
    m_hodges_lehmann = np.median(par_means)
    pseudomediana = m_hodges_lehmann
    
    pesos = np.arange(1, n + 1)
    valores_ordenados = np.sort(valores)
    m_ponderada_pop = np.average(valores_ordenados, weights=pesos)
    
    try:
        m_geo_winsor = stats.gmean(valores_winsor) if np.all(valores_winsor > 0) else "REQUER VALORES > 0"
    except Exception:
        m_geo_winsor = "ERRO"
        
    valores_aparados = valores[(valores >= np.percentile(valores, 10)) & (valores <= np.percentile(valores, 90))]
    try:
        m_harm_aparada = stats.hmean(valores_aparados) if np.all(valores_aparados > 0) else "REQUER VALORES > 0"
    except Exception:
        m_harm_aparada = "ERRO"
        
    try:
        kde = stats.gaussian_kde(valores)
        amostras_kde = np.linspace(np.min(valores), np.max(valores), 1000)
        moda_geometrica = amostras_kde[np.argmax(kde(amostras_kde))]
    except Exception:
        moda_geometrica = m_aritmetica

    metricas_c = ["Média Aritmética", "Mediana", "Moda", "Ponto Médio", "Média Geométrica", "Média Harmônica", "Média Quadrática", "Média Cúbica"]
    valores_c = [m_aritmetica, mediana, moda, ponto_medio, m_geometrica, m_harmonica, m_quadratica, m_cubica]
    data_c = [{"name": m, "value": formatar_numero_terminal(v)} for m, v in zip(metricas_c, valores_c)]
    
    metricas_r = ["Média Aparada (10%)", "Média Winsorizada (5%)", "Média Hodges-Lehmann", "Média Ponderada", "Média Geom. Winsorizada", "Média Harm. Aparada", "Pseudomediana", "Moda Geométrica"]
    valores_r = [m_aparada, m_winsorizada, m_hodges_lehmann, m_ponderada_pop, m_geo_winsor, m_harm_aparada, pseudomediana, moda_geometrica]
    data_r = [{"name": m, "value": formatar_numero_terminal(v)} for m, v in zip(metricas_r, valores_r)]
    
    return data_c, data_r

@app.route('/', methods=['GET'])
def home():
    return render_template(
        'index.html', 
        tables_data_c=DATA_STORE['tables_data_c'], 
        tables_data_r=DATA_STORE['tables_data_r'], 
        graphs=DATA_STORE['graphs'], 
        columns=DATA_STORE['columns'], 
        selected_var=DATA_STORE['selected_var']
    )

@app.route('/', methods=['POST'])
def handle_post():
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        try:
            df = pd.read_excel(file)
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if num_cols:
                DATA_STORE['df'] = df
                DATA_STORE['columns'] = num_cols
                DATA_STORE['tables_data_c'] = None
                DATA_STORE['tables_data_r'] = None
                DATA_STORE['graphs'] = []
                DATA_STORE['selected_var'] = None
        except Exception:
            pass

    elif 'target_variable' in request.form and DATA_STORE['df'] is not None:
        df = DATA_STORE['df']
        selected_var = request.form.get('target_variable')
        
        if selected_var in df.columns:
            try:
                DATA_STORE['selected_var'] = selected_var
                dc, dr = processar_metricas_lista(df, selected_var)
                DATA_STORE['tables_data_c'] = dc
                DATA_STORE['tables_data_r'] = dr
                
                fig = px.line(df, y=selected_var)
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Plus Jakarta Sans", size=10, color="#a5a6b5"),
                    margin=dict(l=40, r=20, t=20, b=40),
                    xaxis=dict(gridcolor="#1c1c27", linecolor="#23232f", title=None),
                    yaxis=dict(gridcolor="#1c1c27", linecolor="#23232f", title=None)
                )
                
                DATA_STORE['graphs'] = [json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)]
            except Exception:
                pass

    return redirect(url_for('home'))

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
