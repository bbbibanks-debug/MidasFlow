import os
import json
import pandas as pd
import plotly.express as px
import plotly.utils
from flask import Flask, render_template, request

app = Flask(__name__)

# Memória temporária global para simplificar o projeto sem banco de dados
# Nota: Em produção real com muitos usuários usaríamos sessões, mas para o MVP atende perfeitamente.
DATA_STORE = {}

@app.route('/', methods=['GET', 'POST'])
def home():
    tables = None
    graphs = []
    columns = []
    selected_x = None
    selected_y = None
    msg = None

    if request.method == 'POST':
        # ETAPA 1: O usuário enviou o arquivo original
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            try:
                df = pd.read_excel(file)
                # Salva o DataFrame na memória para a próxima etapa
                DATA_STORE['df'] = df
                columns = df.columns.tolist()
                
                # Gera estatísticas iniciais
                stats_df = df.describe(include='all').fillna('-')
                tables = stats_df.to_html(classes="min-w-full text-left border-collapse border border-zinc-800 text-xs text-emerald-400 table-auto")
                tables = tables.replace('border="1"', '').replace('class="dataframe', 'class="')
                DATA_STORE['tables'] = tables
                msg = "ARQUIVO CARREGADO COM SUCESSO. SELECIONE AS VARIÁVEIS ABAIXO."
            except Exception as e:
                msg = f"ERRO AO PROCESSAR ARQUIVO: {str(e)}"

        # ETAPA 2: O usuário escolheu as colunas nos dropdowns e enviou
        elif 'axis_x' in request.form and 'df' in DATA_STORE:
            df = DATA_STORE['df']
            tables = DATA_STORE.get('tables')
            columns = df.columns.tolist()
            
            selected_x = request.form.get('axis_x')
            selected_y = request.form.get('axis_y')

            try:
                # Criando gráfico dinâmico baseado na escolha do usuário
                if selected_x and selected_y:
                    fig = px.scatter(df, x=selected_x, y=selected_y, title=f"ANALYSIS: {selected_y.upper()} vs {selected_x.upper()}")
                    fig.update_layout(
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#00ff66"
                    )
                    graphs.append(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
                    msg = "GRÁFICO GERADO COM SUCESSO."
            except Exception as e:
                msg = f"ERRO AO GERAR GRÁFICO: {str(e)}"

    return render_template('index.html', tables=tables, graphs=graphs, columns=columns, selected_x=selected_x, selected_y=selected_y, msg=msg)

if __name__ == '__main__':
    porta = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=porta)
