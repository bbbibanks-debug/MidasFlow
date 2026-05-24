import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.utils
from scipy import stats
from flask import Flask, render_template, request

app = Flask(__name__)

# Cache de dados em memória volátil para o escopo do projeto
DATA_STORE = {}

def formatar_numero_terminal(valor):
    """Garante formatação uniforme e elegante padrão Bloomberg: duas casas decimais com vírgula."""
    if isinstance(valor, (int, float, np.number)):
        if np.isnan(valor):
            return "—"
        # Converte para string com formatação americana e depois inverte para o padrão de mercado solicitado
        s_val = f"{valor:,.2f}"
        return s_val.replace(",", "X").replace(".", ",").replace("X", ".")
    return str(valor)

def gerar_tabela_elegante(metricas, valores):
    """Cria uma estrutura de tabela institucional de alta performance visual."""
    html = '<table class="w-full text-left border-collapse text-[11px] font-mono font-light tracking-tight">'
    html += '<thead class="text-zinc-600 border-b border-[#1e222d]"><th class="pb-2.5 font-medium">STAT_ANALYSIS</th><th class="pb-2.5 text-right font-medium">VALUE</th></thead>'
    html += '<tbody>'
    for k, v in zip(metricas, valores):
        html += f'<tr class="border-b border-[#171921] hover:bg-[#1a1e29]"><td class="py-2.5 text-zinc-400 font-medium">{k}</td><td class="py-2.5 text-right text-[#00c087] font-bold">{v}</td></tr>'
    html += '</tbody></table>'
    return html

def analisar_coluna_especifica(df, col):
    """Aplica o motor estatístico estritamente para a variável parametrizada."""
    col_data = df[col].dropna()
    if col_data.empty:
        return "<span class='text-amber-500 font-mono'>DADOS INSUFICIENTES</span>"
        
    valores = col_data.to_numpy()
    
    # 8 Medidas de Tendência Central Rígidas
    m_aritmetica = np.mean(valores)
    mediana = np.median(valores)
    
    moda_res = stats.mode(valores, keepdims=True)
    moda = moda_res.mode[0] if len(moda_res.mode) > 0 else "—"
    
    ponto_medio = (np.max(valores) + np.min(valores)) / 2
    
    try:
        m_geometrica = stats.gmean(valores) if np.all(valores > 0) else "REQUER VALORES > 0"
    except Exception:
        m_geometrica = "FALHA_CÁLCULO"
        
    try:
        m_harmonica = stats.hmean(valores) if np.all(valores > 0) else "REQUER VALORES > 0"
    except Exception:
        m_harmonica = "FALHA_CÁLCULO"
        
    m_quadratica = np.sqrt(np.mean(valores**2))
    m_cubica = np.cbrt(np.mean(valores**3))
    
    metricas = [
        "Média Aritmética", "Mediana", "Moda", "Ponto Médio",
        "Média Geométrica", "Média Harmônica", "Média Quadrática", "Média Cúbica"
    ]
    
    valores_formatados = [
        formatar_numero_terminal(m_aritmetica),
        formatar_numero_terminal(mediana),
        formatar_numero_terminal(moda),
        formatar_numero_terminal(ponto_medio),
        formatar_numero_terminal(m_geometrica),
        formatar_numero_terminal(m_harmonica),
        formatar_numero_terminal(m_quadratica),
        formatar_numero_terminal(m_cubica)
    ]
    
    return gerar_tabela_elegante(metricas, valores_formatados)

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []
    columns = DATA_STORE.get('columns', [])
    selected_var = None
    msg = None

    if request.method == 'POST':
        # EVENTO 1: Ingestão de Planilha
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_excel(file)
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if num_cols:
                    DATA_STORE['df'] = df
                    DATA_STORE['columns'] = num_cols
                    columns = num_cols
                    msg = "MATRIZ DE DADOS SUCESSO. SELECIONE A VARIÁVEL ALVO PARA INICIAR COMPILAÇÃO."
                else:
                    msg = "EXCEÇÃO: PLANILHA NÃO CONTÉM VETORES NUMÉRICOS VÁLIDOS."
            except Exception as e:
                msg = f"FALHA CRÍTICA DE ESTRUTURAÇÃO: {str(e)}"

        # EVENTO 2: Isolamento e Análise de Variável Única
        elif 'target_variable' in request.form and 'df' in DATA_STORE:
            df = DATA_STORE['df']
            columns = DATA_STORE.get('columns', [])
            selected_var = request.form.get('target_variable')
            
            if selected_var in df.columns:
                try:
                    # Rendeira a tabela analítica minimalista
                    tables = analisar_coluna_especifica(df, selected_var)
                    
                    # Desenha o gráfico de dispersão/linha de alta fidelidade
                    fig = px.line(df, y=selected_var)
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_family="JetBrains Mono",
                        font_size=10,
                        font_color="#b2b5be",
                        margin=dict(l=40, r=20, t=30, b=40),
                        xaxis=dict(
                            gridcolor="#171921", 
                            linecolor="#1e222d", 
                            title=None, 
                            tickfont=dict(color="#63666e")
                        ),
                        yaxis=dict(
                            gridcolor="#171921", 
                            linecolor="#1e222d", 
                            title=None, 
                            tickfont=dict(color="#63666e")
                        ),
                        line=dict(color="#2962ff", width=1.5)
                    )
                    graphs.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    msg = f"ESTATÍSTICAS ATUALIZADAS PARA O ATIVO: {selected_var.upper()}"
                except Exception as e:
                    msg = f"ERRO OPERACIONAL DE COMPILAÇÃO: {str(e)}"

    return render_template('index.html', tables=tables, graphs=graphs, columns=columns, selected_var=selected_var, msg=msg)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
