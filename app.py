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

def formatar_numero(valor):
    """Trata todos os números para o padrão com duas casas decimais após a vírgula."""
    if isinstance(valor, (int, float, np.number)):
        if np.isnan(valor):
            return "-"
        return f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return str(valor)

def gerar_tabela_html(metricas, valor):
    """Monta uma tabela HTML limpa e minimalista no estilo Bloomberg."""
    html = '<table class="w-full text-left border-collapse text-xs font-mono">'
    html += '<thead class="text-zinc-500 border-b border-[#2a2e39]"><tr><th class="pb-2">STATISTIC_METRIC</th><th class="pb-2 text-right">VALUE</th></tr></thead>'
    html += '<tbody>'
    for k, v in zip(metricas, valor):
        html += f'<tr class="border-b border-[#1f222e]"><td class="py-2 text-zinc-400">{k}</td><td class="py-2 text-right text-[#00ff66] font-bold">{v}</td></tr>'
    html += '</tbody></table>'
    return html

def calcular_metricas_variavel(df, col):
    """Calcula estritamente as 8 medidas tratadas para uma coluna específica."""
    col_data = df[col].dropna()
    if col_data.empty or not pd.api.types.is_numeric_dtype(df[col]):
        return "<span class='text-red-400'>A VARIÁVEL SELECIONADA NÃO É NUMÉRICA.</span>"
        
    valores = col_data.to_numpy()
    
    # Execução das 8 métricas solicitadas
    m_aritmetica = np.mean(valores)
    mediana = np.median(valores)
    
    moda_res = stats.mode(valores, keepdims=True)
    moda = moda_res.mode[0] if len(moda_res.mode) > 0 else "-"
    
    ponto_medio = (np.max(valores) + np.min(valores)) / 2
    
    try:
        m_geometrica = stats.gmean(valores) if np.all(valores > 0) else "N/A (Requer > 0)"
    except Exception:
        m_geometrica = "Erro"
        
    try:
        m_harmonica = stats.hmean(valores) if np.all(valores > 0) else "N/A (Requer > 0)"
    except Exception:
        m_harmonica = "Erro"
        
    m_quadratica = np.sqrt(np.mean(valores**2))
    m_cubica = np.cbrt(np.mean(valores**3))
    
    metricas = [
        "Média Aritmética", "Mediana", "Moda", "Ponto Médio",
        "Média Geométrica", "Média Harmônica", "Média Quadrática", "Média Cúbica"
    ]
    
    valores_formatados = [
        formatar_numero(m_aritmetica),
        formatar_numero(mediana),
        formatar_numero(moda),
        formatar_numero(ponto_medio),
        formatar_numero(m_geometrica),
        formatar_numero(m_harmonica),
        formatar_numero(m_quadratica),
        formatar_numero(m_cubica)
    ]
    
    return gerar_tabela_html(metricas, valores_formatados)

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []
    columns = []
    selected_var = None
    msg = None

    if request.method == 'POST':
        # CASO 1: Nova Planilha Carregada
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_excel(file)
                # Filtra apenas colunas com dados numéricos para evitar falhas operacionais
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if num_cols:
                    DATA_STORE['df'] = df
                    DATA_STORE['columns'] = num_cols
                    columns = num_cols
                    msg = "REPOSITÓRIO DE DADOS ATUALIZADO. AGUARDANDO SELEÇÃO DE VARIÁVEL."
                else:
                    msg = "FALHA: NENHUMA COLUNA DE DADOS NUMÉRICOS ENCONTRADA."
            except Exception as e:
                msg = f"ERRO OPERACIONAL DE LEITURA: {str(e)}"

        # CASO 2: Seleção de Variável no Dropdown Único
        elif 'target_variable' in request.form and 'df' in DATA_STORE:
            df = DATA_STORE['df']
            columns = DATA_STORE.get('columns', [])
            selected_var = request.form.get('target_variable')
            
            if selected_var in df.columns:
                try:
                    # 1. Processa e exibe a tabela vertical estilizada
                    tables = calcular_metricas_variavel(df, selected_var)
                    
                    # 2. Renderiza gráfico sequencial linear refinado padrão Bloomberg
                    fig = px.line(df, y=selected_var, title=f"REAL-TIME TIMELINE/SEQUENCE: {selected_var.upper()}")
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#d1d4dc",
                        xaxis=dict(gridcolor="#1f222e", title="INDEX"),
                        yaxis=dict(gridcolor="#1f222e", title="VALUE"),
                        line=dict(color="#2962ff", width=2)
                    )
                    graphs.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    msg = f"ANÁLISE EXECUTADA PARA A VARIÁVEL: {selected_var.upper()}"
                except Exception as e:
                    msg = f"ERRO NA GERAÇÃO DOS COMPONENTES: {str(e)}"

    return render_template('index.html', tables=tables, graphs=graphs, columns=columns, selected_var=selected_var, msg=msg)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
