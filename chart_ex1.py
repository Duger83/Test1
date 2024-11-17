import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objs as go

def mathplot_chart(mass):
    plt.plot(mass)
    plt.grid()
    plt.xlabel('ось Х')
    plt.ylabel('ось У') 
    plt.title('y=cos(20x)/(x+0.1) by mathplotlib')
    plt.show()

def sea_chart(mass): 
    sns.set_style("ticks",{'axes.grid' : True})
    df = pd.DataFrame({'ось X': np.arange(0, 4.01, 0.01), 'ось Y': mass})
    sns.lineplot(df, color='red', x='ось X', y='ось Y')
    plt.title('y=cos(20x)/(x+0.1) by seaborn')
    plt.show()

def plotly_chart(mass):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(0, 4.01, 0.01), y=mass, line=dict(color='Green')))
    fig.update_layout(title='y=cos(20x)/(x+0.1) by plotly', xaxis_title='ось X', yaxis_title='ось У')
    fig.show()

a = np.array([np.cos(20*x)/(x+0.1) for x in np.arange(0, 4.01, 0.01)])

mathplot_chart(a)
sea_chart(a)
plotly_chart(a)