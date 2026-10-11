import dash
from dash import html, dcc, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import os

from config import algorithm_strategy_pairs, base_path

# Initialise Dash App
app = dash.Dash(__name__)
app.title = "Simulation of variance-aware algorithms for Stochastic Bandit Problems"

# Create a mapping of algorithm names with their corresponding IDs
algorithm_id_mapping = {f"{algo.name}_{index}": (algo, config) for index, (algo, config) in enumerate(algorithm_strategy_pairs)}

# Scenarios shown in the dashboard: (title, arm 1, arm 2)
scenarios = [
    ("Scenario A (Baseline): p1 = 0.8, p2 = 0.9", "800", "900"),
    ("Scenario B (Low-Variance Micro-Gap): p1 = 0.895, p2 = 0.9", "895", "900"),
    ("Scenario C (High-Variance Micro-Gap): p1 = 0.89, p2 = 0.895", "890", "895"),
]

# Horizon at which the metrics are evaluated
horizon = 1000000

def get_label_with_params(algo_name, params):
    param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
    return f"{algo_name} ({param_str})" if params else algo_name

def load_data(algorithm_with_index, arm_1, arm_2, arm_3):
    """
    Loads results and average results data for a specified algorithm and arm distribution.
    
    Parameters:
    algorithm_with_index (str): The name of the algorithm with index (e.g., "UCB_0").
    arm_1 (str): The selected value for the first arm distribution.
    arm_2 (str): The selected value for the second arm distribution.
    arm_3 (str): The selected value for the third arm distribution (can be "-").
    
    Returns:
    tuple: A tuple containing two pandas DataFrames:
        - df_results: DataFrame containing the results data.
        - df_average: DataFrame containing the average results data.
    """

    algorithm_id = algorithm_with_index
    
    # Extract the actual algorithm name from the combined string
    algorithm = algorithm_with_index.split('_')[0]
    
    # Combine arm distributions into a string for the file name
    if arm_3 == '-' or arm_3 is None:
        arm_distribution = f"{arm_1}_{arm_2}"
    else:
        arm_distribution = f"{arm_1}_{arm_2}_{arm_3}"

    # Find the corresponding algorithm configuration
    algo, algo_config = algorithm_id_mapping.get(algorithm_id, (None, None))
    #algo_config = next((config for algo, config in algorithm_strategy_pairs if algo.name == algorithm), None)
    if not algo_config:
        raise ValueError(f"No configuration found for algorithm: {algorithm}")

    # Generate the parameter directory string
    params = algo_config["params"]
    if params:
        param_dir = '_'.join([f'{key}_{str(value).replace(".", "_")}' for key, value in params.items()])
    else:
        param_dir = 'default'

    # Construct the directory path based on the algorithm and parameters
    results_dir = os.path.join(base_path, algorithm, param_dir)
    
    # Load results and average results
    results_path = os.path.join(results_dir, f"results_{arm_distribution}.csv")
    average_results_path = os.path.join(results_dir, f"average_results_{arm_distribution}.csv")

    # Check if the files exist in the parameter directory, otherwise check the default directory
    if not os.path.exists(results_path) or not os.path.exists(average_results_path):
        results_dir = os.path.join(base_path, algorithm, 'default')
        results_path = os.path.join(results_dir, f"results_{arm_distribution}.csv")
        average_results_path = os.path.join(results_dir, f"average_results_{arm_distribution}.csv")

    df_results = pd.read_csv(results_path)
    df_average = pd.read_csv(average_results_path)

    return df_results, df_average

# Line style per algorithm group, colors are assigned per configuration
line_styles = {"ETC": "solid", "Greedy": "solid", "UCB": "solid",
               "PAC-UCB": "dot", "UCB-Improved": "dot",
               "UCB-Tuned": "dash", "UCB-V": "dash", "EUCBV": "dash"}
colors = px.colors.qualitative.Dark24

metrics = ["Average Regret", "Reward Variance", "Subopt. Ratio"]
columns = ["Algorithm"] + metrics

def scenario_metrics(arm_1, arm_2):
    """
    Computes Average Regret, Reward Variance and Suboptimal Ratio per timestep
    for every algorithm configuration. Returns {label: (algorithm name, DataFrame)}.
    """
    out = {}
    for algorithm_id, (algo, config) in algorithm_id_mapping.items():
        df_results, df_average = load_data(algorithm_id, arm_1, arm_2, '-')
        per_step = df_results.groupby('Timestep').agg(**{
            "Average Regret": ('Total Regret', 'mean'),
            "Reward Variance": ('Total Reward', 'var'),
        })
        per_step["Subopt. Ratio"] = (df_average.set_index('Timestep')['Average Suboptimal Arms'] / df_average['Timestep'].values)
        out[get_label_with_params(algo.name, config["params"])] = (algo.name, per_step.reset_index())
    return out

def scenario_table(data):
    """
    One row per algorithm configuration with the metrics at the horizon, sorted by Average Regret.
    """
    rows = []
    for label, (_, df) in data.items():
        final = df[df['Timestep'] == horizon].iloc[0]
        rows.append({
            "Algorithm": label,
            "Average Regret": round(final["Average Regret"], 2),
            "Reward Variance": round(final["Reward Variance"], 2),
            "Subopt. Ratio": round(final["Subopt. Ratio"], 6),
        })
    return sorted(rows, key=lambda r: r["Average Regret"])

def metric_figure(data, metric):
    fig = go.Figure()
    for i, (label, (algo_name, df)) in enumerate(data.items()):
        fig.add_trace(go.Scatter(
            x=df['Timestep'],
            y=df[metric],
            mode='lines',
            name=label,
            line=dict(color=colors[i % len(colors)], dash=line_styles.get(algo_name, "solid"))
        ))
    fig.update_layout(
        title=f'{metric} over Time',
        xaxis_title="Timesteps",
        yaxis_title=metric,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font={'color': 'black', "size": 9},
        legend={'font': {'size': 8}},
        height=450
    )
    return fig

def scenario_section(title, arm_1, arm_2):
    data = scenario_metrics(arm_1, arm_2)
    return html.Div(
        style={'margin-bottom': '40px'},
        children=[
            html.H2(title),
            html.Div(
                style={'display': 'flex', 'flexWrap': 'wrap'},
                children=[
                    html.Div(style={'flex': '1 1 30%', 'minWidth': '400px', 'padding': '10px'},
                             children=[dcc.Graph(figure=metric_figure(data, m))])
                    for m in metrics
                ]
            ),
            dash_table.DataTable(
                data=scenario_table(data),
                columns=[{"name": c, "id": c} for c in columns],
                sort_action='native',
                style_cell={'textAlign': 'left', 'padding': '6px'},
                style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
            ),
        ]
    )

# Layout of the Dash App
app.layout = html.Div(
    style={'backgroundColor': 'white', 'color': 'black', 'font-family': 'Arial, sans-serif', 'padding': '20px'},
    children=[
        html.H1("Simulation of Variance-aware Algorithms for Stochastic Bandit Problems", style={'textAlign': 'center'}),
        html.P(f"Averaged over 100 independent trials; tables show metrics at T = {horizon:,}.", style={'textAlign': 'center'}),
        *[scenario_section(title, arm_1, arm_2) for title, arm_1, arm_2 in scenarios],
    ]
)

# Start Dash App
if __name__ == '__main__':

    app.run_server(debug=True)
