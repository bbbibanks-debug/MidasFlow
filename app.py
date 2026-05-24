import os
import json
import pandas as pd
import plotly.express as px
import plotly.utils
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []

    if request.method == 'POST':
        file = request.files.get('file')
        if file and file.filename != '':
            # Ler planilha Excel usando Pandas
            df = pd.read_excel(file)
            
            # 1. Gerar Estatísticas Descritivas (Tabela HTML estilizada estilo Bloomberg)
            stats_df = df.describe(include='all').fillna('-')
            tables = stats_df.to_html(classes="min-w-full text-left border-collapse border border-zinc-800 text-xs text-emerald-400 table-auto")
            tables = tables.replace('border="1"', '').replace('class="dataframe', 'class="')

            # 2. Identificar variáveis numéricas para criar gráficos automáticos
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            if len(numeric_cols) >= 1:
                # Gráfico 1: Linha do tempo ou evolução sequencial da primeira variável numérica
                fig1 = px.line(df, y=numeric_cols[0], title=f"ANALYSIS: {numeric_cols[0].upper()} SEQUENCE")
                fig1.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#00ff66"
                )
                graphs.append(json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder))

            if len(numeric_cols) >= 2:
                # Gráfico 2: Dispersão (Correlação) se houver duas ou mais variáveis numéricas
                fig2 = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=f"CORRELATION: {numeric_cols[0].upper()} vs {numeric_cols[1].upper()}")
                fig2.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#45f3ff"
                )
                graphs.append(json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder))

    return render_template('index.html', tables=tables, graphs=graphs)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
