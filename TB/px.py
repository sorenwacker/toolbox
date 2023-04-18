import plotly.express as px
import plotly.graph_objects as go

def px_heatmap(df, colorscale='jet_r', layout_kws=None):
    fig = go.Figure(data=go.Heatmap(
            z=df.values,
            y=df.index,
            x=df.columns,
            colorscale=colorscale
            )
    )
    fig.update_layout(**layout_kws)
    return fig