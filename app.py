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

def calcular_estatisticas_completas(df):
    """Calcula as 8 medidas de tendência central para colunas numéricas."""
    num_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(num_cols) == 0:
        return None
        
    resultados = {}
    
    # Índices das métricas solicitadas
    metricas = [
        "Média Aritmética", "Mediana", "Moda", "Ponto Médio",
        "Média Geométrica", "Média Harmônica", "Média Quadrática", "Média Cúbica"
    ]
    
    for col in num_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        valores = col_data.to_numpy()
        
        # 1. Média Aritmética
        media_aritmetica = np.mean(valores)
        
        # 2. Mediana
        mediana = np.median(valores)
        
        # 3. Moda (Retorna o primeiro valor se houver mais de uma)
        moda_res = stats.mode(valores, keepdims=True)
        moda = moda_res.mode[0] if len(moda_res.mode) > 0 else "-"
        
        # 4. Ponto Médio
        ponto_medio = (np.max(valores) + np.min(valores)) / 2
        
        # 5. Média Geométrica (Apenas para valores positivos)
        try:
            media_geometrica = stats.gmean(valores) if np.all(valores > 0) else "N/A (Requer > 0)"
        except Exception:
            media_geometrica = "Erro no cálculo"
            
        # 6. Média Harmônica (Apenas para valores positivos)
        try:
            media_harmonica = stats.hmean(valores) if np.all(valores > 0) else "N/A (Requer > 0)"
        except Exception:
            media_harmonica = "Erro no cálculo"
            
        # 7. Média Quadrática (RMS)
        media_quadratica = np.sqrt(np.mean(valores**2))
        
        # 8. Média Cúbica
        media_cubica = np.cbrt(np.mean(valores**3))
        
        # Formata os valores numéricos para visualização clara no terminal
        def formatar(v):
            return f"{v:,.4f}" if isinstance(v, (int, float, np.number)) else str(v)

        resultados[col] = [
            formatar(media_aritmetica),
            formatar(mediana),
            formatar(moda),
            formatar(ponto_medio),
            formatar(media_geometrica),
            formatar(media_harmonica),
            formatar(media_quadratica),
            formatar(media_cubica)
        ]
        
    # Converte o dicionário em um DataFrame organizado para exibição estilo Bloomberg
    res_df = pd.DataFrame(resultados, index=metricas)
    
    # Transforma em HTML estruturado
    html_table = res_df.to_html(classes="min-w-full text-left border-collapse border border-zinc-800 text-xs text-emerald-400 table-auto")
    return html_table.replace('border="1"', '').replace('class="dataframe', 'class="')

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []
    columns = []
    selected_x = None
    selected_y = None
    msg = None

    if request.method == 'POST':
        # ETAPA 1: Upload do Arquivo
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_excel(file)
                DATA_STORE['df'] = df
                columns = df.columns.tolist()
                
                # Executa o cálculo das 8 métricas customizadas
                tables = calcular_estatisticas_completas(df)
                if tables:
                    DATA_STORE['tables'] = tables
                    msg = "ARQUIVO PROCESSADO. TODAS AS 8 MEDIDAS CALCULADAS COM SUCESSO."
                else:
                    msg = "AVISO: NENHUMA COLUNA NUMÉRICA DETECTADA NA PLANILHA."
            except Exception as e:
                msg = f"ERRO CRÍTICO DE PROCESSAMENTO: {str(e)}"

        # ETAPA 2: Geração do Gráfico Dinâmico
        elif 'axis_x' in request.form and 'df' in DATA_STORE:
            df = DATA_STORE['df']
            tables = DATA_STORE.get('tables')
            columns = df.columns.tolist()
            
            selected_x = request.form.get('axis_x')
            selected_y = request.form.get('axis_y')

            try:
                if selected_x and selected_y:
                    fig = px.scatter(df, x=selected_x, y=selected_y, title=f"ANALYSIS: {selected_y.upper()} vs {selected_x.upper()}")
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#00ff66"
                    )
                    graphs.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    msg = "GRÁFICO GERADO VIA SISTEMA."
            except Exception as e:
                msg = f"ERRO AO GERAR GRÁFICO: {str(e)}"

    return render_template('index.html', tables=tables, graphs=graphs, columns=columns, selected_x=selected_x, selected_y=selected_y, msg=msg)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
