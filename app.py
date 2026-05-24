import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
from scipy import stats
from flask import Flask, render_template, request

app = Flask(__name__)

DATA_STORE = {}

def formatar_numero_terminal(valor):
    """Garante formatação uniforme padrão mercado: duas casas decimais com vírgula."""
    if isinstance(valor, (int, float, np.number)):
        if np.isnan(valor):
            return "—"
        s_val = f"{valor:,.2f}"
        return s_val.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(valor)

def gerar_tabela_elegante(metricas_classicas, valores_classicos, metricas_robustas, valores_robustas):
    """Cria uma estrutura de tabela institucional de alta performance visual dividida em categorias."""
    html = '<div class="space-y-6">'
    
    # Bloco 1: Tendência Central Clássica
    html += '<div>'
    html += '<h4 class="text-[10px] uppercase font-bold text-zinc-500 tracking-wider mb-2">// 01. Medidas Clássicas</h4>'
    html += '<table class="w-full text-left border-collapse text-[11px] font-mono font-light tracking-tight">'
    html += '<tbody>'
    for k, v in zip(metricas_classicas, valores_classicos):
        html += f'<tr class="border-b border-[#171921] hover:bg-[#1a1e29]"><td class="py-2 text-zinc-400 font-medium">{k}</td><td class="py-2 text-right text-[#00c087] font-bold">{v}</td></tr>'
    html += '</tbody></table></div>'
    
    # Bloco 2: Tendência Central Robusta e Modificada
    html += '<div>'
    html += '<h4 class="text-[10px] uppercase font-bold text-[#ff9800] tracking-wider mb-2">// 02. Medidas Robustas e Modificadas</h4>'
    html += '<table class="w-full text-left border-collapse text-[11px] font-mono font-light tracking-tight">'
    html += '<tbody>'
    for k, v in zip(metricas_robustas, valores_robustas):
        html += f'<tr class="border-b border-[#171921] hover:bg-[#1a1e29]"><td class="py-2 text-zinc-400 font-medium">{k}</td><td class="py-2 text-right text-lime-neon font-bold">{v}</td></tr>'
    html += '</tbody></table></div>'
    
    html += '</div>'
    return html

def analisar_coluna_especifica(df, col):
    """Executa o ecossistema estatístico completo de 16 variáveis sob demanda."""
    col_data = df[col].dropna()
    if col_data.empty:
        return "<span class='text-amber-500 font-mono'>DADOS INSUFICIENTES</span>"
        
    valores = col_data.to_numpy()
    n = len(valores)
    
    # --- GRUPO 1: MEDIDAS CLÁSSICAS ---
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
    
    # --- GRUPO 2: MEDIDAS ROBUSTAS E MODIFICADAS ---
    # 1. Média Aparada (Truncada a 10% de cada cauda)
    m_aparada = stats.trim_mean(valores, proportiontocut=0.10)
    
    # 2. Média Winsorizada (Substitui extremos abaixo de 5% e acima de 95%)
    valores_winsor = stats.mstats.winsorize(valores, limits=[0.05, 0.05]).compressed()
    m_winsorizada = np.mean(valores_winsor)
    
    # Sub-amostragem inteligente de segurança para métricas combinatórias O(N^2)
    valores_sub = valores
    if n > 400:
        np.random.seed(42)
        valores_sub = np.random.choice(valores, size=400, replace=False)
    
    # 3. Média de Hodges-Lehmann (Mediana das médias dos pares)
    # 7. Pseudomediana (Equivalente matemático de posto de Wilcoxon para centro do vetor)
    par_means = (valores_sub[:, None] + valores_sub) / 2.0
    m_hodges_lehmann = np.median(par_means)
    pseudomediana = m_hodges_lehmann  # Estatisticamente idênticas em vetores contínuos univariados
    
    # 4. Média Ponderada pela População (Neste escopo univariado, ponderada por ranqueamento de magnitude linear)
    pesos = np.arange(1, n + 1)
    valores_ordenados = np.sort(valores)
    m_ponderada_pop = np.average(valores_ordenados, weights=pesos)
    
    # 5. Média Geométrica Winsorizada
    try:
        m_geo_winsor = stats.gmean(valores_winsor) if np.all(valores_winsor > 0) else "REQUER VALORES > 0"
    except Exception:
        m_geo_winsor = "ERRO"
        
    # 6. Média Harmônica Aparada
    valores_aparados = valores[(valores >= np.percentile(valores, 10)) & (valores <= np.percentile(valores, 90))]
    try:
        m_harm_aparada = stats.hmean(valores_aparados) if np.all(valores_aparados > 0) else "REQUER VALORES > 0"
    except Exception:
        m_harm_aparada = "ERRO"
        
    # 8. Moda Geométrica (Estimador de densidade Kernel contínuo gaussiano)
    try:
        kde = stats.gaussian_kde(valores)
        amostras_kde = np.linspace(np.min(valores), np.max(valores), 1000)
        moda_geometrica = amostras_kde[np.argmax(kde(amostras_kde))]
    except Exception:
        moda_geometrica = m_aritmetica

    # Listagem de Rótulos
    metricas_c = ["Média Aritmética", "Mediana", "Moda", "Ponto Médio", "Média Geométrica", "Média Harmônica", "Média Quadrática", "Média Cúbica"]
    valores_c = [formatar_numero_terminal(x) for x in [m_aritmetica, mediana, moda, ponto_medio, m_geometrica, m_harmonica, m_quadratica, m_cubica]]
    
    metricas_r = ["Média Aparada (10%)", "Média Winsorizada (5%)", "Média Hodges-Lehmann", "Média Ponderada (Rank)", "Média Geom. Winsorizada", "Média Harm. Aparada", "Pseudomediana", "Moda Geométrica (KDE)"]
    valores_r = [formatar_numero_terminal(x) for x in [m_aparada, m_winsorizada, m_hodges_lehmann, m_ponderada_pop, m_geo_winsor, m_harm_aparada, pseudomediana, moda_geometrica]]
    
    return gerar_tabela_elegante(metricas_c, valores_c, metricas_r, valores_r)

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []
    columns = DATA_STORE.get('columns', [])
    selected_var = None
    msg = None

    if request.method == 'POST':
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_excel(file)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if num_cols:
                    DATA_STORE['df'] = df
                    DATA_STORE['columns'] = num_cols
                    columns = num_cols
                    msg = "MATRIZ OPERACIONAL CARREGADA. SELECIONE A VARIÁVEL ACIMA."
                else:
                    msg = "EXCEÇÃO: PLANILHA NÃO CONTÉM COLUNAS NUMÉRICAS VÁLIDAS."
            except Exception as e:
                msg = f"FALHA CRÍTICA DE LEITURA: {str(e)}"

        elif 'target_variable' in request.form and 'df' in DATA_STORE:
            df = DATA_STORE['df']
            columns = DATA_STORE.get('columns', [])
            selected_var = request.form.get('target_variable')
            
            if selected_var in df.columns:
                try:
                    tables = analisar_coluna_especifica(df, selected_var)
                    
                    fig = px.line(df, y=selected_var)
                    fig.update_traces(line=dict(color="#ccff00", width=2))
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(family="Plus Jakarta Sans", size=10, color="#a5a6b5"),
                        margin=dict(l=40, r=20, t=20, b=40),
                        xaxis=dict(gridcolor="#1c1c27", linecolor="#23232f", title=None),
                        yaxis=dict(gridcolor="#1c1c27", linecolor="#23232f", title=None)
                    )
                    graphs.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    msg = f"ESTATÍSTICAS COMPILADAS PARA: {selected_var.upper()}"
                except Exception as e:
                    msg = f"ERRO OPERACIONAL DE COMPILAÇÃO: {str(e)}"

    return render_template('index.html', tables=tables, graphs=graphs, columns=columns, selected_var=selected_var, msg=msg)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
