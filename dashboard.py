import pandas as pd
import numpy as np
import warnings
from scipy import stats
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
warnings.filterwarnings("ignore")
gps_data = pd.read_csv('gps_data.csv')
imu_data = pd.read_csv('imu_data.csv')
missing_GPSvalues = gps_data.isnull().sum()
print(missing_GPSvalues)
missing_imuvalues = imu_data.isnull().sum()
print(missing_imuvalues)
# Merge based on both Serial Number and NZDT (Mutiple selection by '[]')
merged_data = pd.merge(gps_data, imu_data, on=['Serial Number','NZDT'])
merged_data['NZDT'] = pd.to_datetime(merged_data['NZDT'], format='%d/%m/%y %H:%M:%S')
def angle_standard(angle1,angle2):
    res = np.abs(angle1-angle2)%360
    return np.where(res > 180, 360 - res, res)
merged_data["difference"] = angle_standard(merged_data['Heading'], merged_data['GPS Heading'])
merged_data["difference"].describe()
sd = 12.03
mean = 16.99
standard_error = sd / np.sqrt(3309)
print(f"Standard Error is:{standard_error}")
#μ0 (difference between two group) = 0
t_stastistic = (mean - 0) / standard_error
print(f"T_stastistics is :{t_stastistic}" )
df = 3308
print(f"Degree of freedom is :{df}" )
p_value = 2 * (1 - stats.t.cdf(abs(81.246), df=3308))
print(f"P-value is :{p_value}" )

# Now do something fancy
merged_data['Heading'] = merged_data['Heading'] % 360
merged_data['GPS Heading'] = merged_data['GPS Heading'] % 360
map_center = {"lat": merged_data['Latitude'].mean(), "lon": merged_data['Longitude'].mean()}
zoom_level = 20

merged_data['Adjusted GPS Heading'] = merged_data['Heading'] + merged_data['difference']
merged_data['Adjusted GPS Heading'] = merged_data['Adjusted GPS Heading'] % 360

def smooth_heading_transition(heading_series):
    adjusted_heading = heading_series.copy()
    for i in range(1, len(heading_series)):
        if abs(adjusted_heading[i] - adjusted_heading[i - 1]) > 180:
            if adjusted_heading[i] > adjusted_heading[i - 1]:
                adjusted_heading[i] -= 360
            else:
                adjusted_heading[i] += 360
    return adjusted_heading

merged_data['Adjusted Heading'] = smooth_heading_transition(merged_data['Heading'])
merged_data['Adjusted GPS Heading'] = smooth_heading_transition(merged_data['Adjusted GPS Heading'])

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Cow Movement Analysis Dashboard"),
        dcc.Graph(
        id='map-graph',
        figure=px.scatter_mapbox(
            merged_data,
            lat='Latitude',
            lon='Longitude',
            color='Serial Number',
            mapbox_style="carto-positron",
            title='Cow Movements'
        ).update_layout(
            mapbox=dict(
                center=map_center,
                zoom=zoom_level
            )
        )
    ),

    dcc.Graph(
        id='difference-histogram',
        figure=px.histogram(
            merged_data,
            x='difference',
            nbins=30,
            title='Histogram of Heading Differences'
        )
    ),

    dcc.Graph(
        id='anomaly-graph',
        figure=go.Figure(
            data=[
                go.Scatter(
                    x=merged_data['NZDT'],
                    y=merged_data['difference'],
                    mode='markers',
                    marker=dict(
                        color=['red' if diff > 60 else 'blue' for diff in merged_data['difference']]
                    ),
                    name='Heading Difference'
                )
            ],
            layout=go.Layout(
                title='Heading Differences with Anomalies Highlighted'
            )
        )
    ),
    dcc.Dropdown(
        id='cow-selector',
        options=[{'label': serial, 'value': serial} for serial in merged_data['Serial Number'].unique()],
        value=merged_data['Serial Number'].unique()[0]
    ),
    dcc.Graph(id='heading-plot'),
    dcc.Graph(id='histogram-plot'),
    html.Div(id='anomaly-frequency')
])

@app.callback(
    Output('heading-plot', 'figure'),
    Input('cow-selector', 'value')
)
def update_heading_plot(selected_cow):
    cow_data = merged_data[merged_data['Serial Number'] == selected_cow]
    figure = {
        'data': [
            go.Scatter(
                x=cow_data['NZDT'], y=cow_data['Adjusted Heading'], mode='lines', name='Heading (IMU)'
            ),
            go.Scatter(
                x=cow_data['NZDT'], y=cow_data['Adjusted GPS Heading'], mode='lines', name='GPS Heading'
            )
        ],
        'layout': go.Layout(
            title=f'Time Series of Adjusted Headings for Cow {selected_cow}',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Adjusted Heading'},
            hovermode='closest'
        )
    }
    return figure

@app.callback(
    Output('histogram-plot', 'figure'),
    Input('cow-selector', 'value')
)
def update_histogram(selected_cow):
    cow_data = merged_data[merged_data['Serial Number'] == selected_cow]
    if cow_data.empty:
        return {
            'data': [],
            'layout': go.Layout(
                title='No data available',
                xaxis={'title': 'Heading Difference'},
                yaxis={'title': 'Frequency'},
                hovermode='closest'
            )
        }
    figure = {
        'data': [
            go.Histogram(
                x=cow_data['difference'], nbinsx=30, name='Heading Difference'
            )
        ],
        'layout': go.Layout(
            title=f'Distribution of Heading Differences for Cow {selected_cow}',
            xaxis={'title': 'Heading Difference'},
            yaxis={'title': 'Frequency'},
            hovermode='closest'
        )
    }
    return figure
@app.callback(
    Output('anomaly-frequency', 'children'),
    Input('cow-selector', 'value')
)
def update_anomaly_frequency(selected_cow):
    cow_data = merged_data[merged_data['Serial Number'] == selected_cow]
    if cow_data.empty:
        return f'Number of Anomalies for Cow {selected_cow}: 0'
    mean_diff = cow_data['difference'].mean()
    std_diff = cow_data['difference'].std()
    threshold_upper = mean_diff + 3 * std_diff
    threshold_lower = mean_diff - 3 * std_diff
    anomalies = cow_data[(cow_data['difference'] > threshold_upper) | (cow_data['difference'] < threshold_lower)]
    anomaly_count = anomalies.shape[0]
    return f'Number of Anomalies for Cow {selected_cow}: {anomaly_count}'

if __name__ == '__main__':
    app.run_server(debug=True)