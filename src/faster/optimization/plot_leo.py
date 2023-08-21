# Finding out how to plot Maria Alice's results

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Generic file_graph_nodes locations
csv_file_graph_information = 'D:\\OP_pycharm\\Operwas_pump\\inputs\\wwtp_locations_many.csv'

# Files no pumping
# 'D:\OP_pycharm\Operwas_pump\optimization_results\no_pumping\run_2023-08-18-17-18-52-125\solutions_2023-08-18-17-18-52-125.csv'
# 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\no_pumping\\run_2023-08-18-17-18-52-125\\solutions_2023-08-18-17-18-52-125'
# experiment_folder = 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\no_pumping\\'
# experiment_id = 'run_2023-08-18-17-18-52-125'
# experiment_full_path = experiment_folder + experiment_id + '\\'

# Files genetic pumping
# 'D:\OP_pycharm\Operwas_pump\optimization_results\genetic_pumping\run_2023-08-20-16-38-44-738\solutions_2023-08-20-16-38-44-738.csv'

experiment_folder = 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\genetic_pumping\\'
experiment_id = 'run_2023-08-20-16-38-44-738'
experiment_full_path = experiment_folder + experiment_id + '\\'

# CSV Files
csv_file_wwtp_labels = experiment_full_path + 'solutions' + experiment_id.lstrip("run") + '.csv'  # solutions
csv_file_wwtp_results_total = experiment_full_path + 'results_total' + experiment_id.lstrip(
    "run") + '.csv'  # results_total

# Read CSV file with graph information
graph_data = pd.read_csv(csv_file_graph_information)

# Read CSV file with 'WWTP' labels
wwtp_labels = pd.read_csv(csv_file_wwtp_labels)

# Read CSV file with information about the WWTP configuration
overall_info = pd.read_csv(csv_file_wwtp_results_total)


# Create directed graph the results of genetic_pumping

def generate_and_plot_graph(row_index, graph_nodes_data, graph_connections_data, df_overall_info, save_path=None):
    # Create a directed graph (DiGraph)
    G = nx.DiGraph()

    # Add nodes with attributes
    for _, node_row in graph_nodes_data.iterrows():
        node_id = node_row['id']
        x_coord = node_row['X']
        y_coord = node_row['Y']
        G.add_node(node_id, pos=(x_coord, y_coord))
        # You can also add other attributes as needed

    # Add edges based on connections information
    connection_row = graph_connections_data.iloc[row_index]
    for i in range(32):  # Fix the range to match the indices
        to_node = connection_row['connect_to_{}'.format(i)]
        if to_node >= 0:
            G.add_edge(to_node, i)
        # You can also add other attributes as needed

    # Get node positions
    node_positions = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Draw nodes with positions, use gray for nodes without positions
    nx.draw(G, pos=node_positions, with_labels=False, font_weight='bold', node_size=500, node_color='lightgray')

    # Highlight nodes with red color
    red_nodes = [node for node, data in G.nodes(data=True) if G.out_degree(node) == 0]
    nx.draw_networkx_nodes(G, pos=node_positions, nodelist=red_nodes, node_color='red', node_size=500)

    # Create a dictionary to map nodes to their labels
    node_labels = {}
    for node in G.nodes:
        if int(node) == node:  # Check if the node value is an integer
            node_labels[node] = str(int(node))  # Display integer nodes as integers
        else:
            node_labels[node] = ''  # Display non-integer nodes as empty strings

    # Draw node labels
    nx.draw_networkx_labels(G, pos=node_positions, labels=node_labels, font_size=10, font_color='black')

    # Extract the desired information from the DataFrame
    total_population = df_overall_info.loc[row_index, 'Total population supplied (inhab)']
    total_benefits = df_overall_info.loc[row_index, 'Total benefits  (ILS)']
    total_costs = df_overall_info.loc[row_index, 'Total costs (ILS)']
    coverage_region = df_overall_info.loc[row_index, 'Coverage (region)']
    benefits_costs_ratio = df_overall_info.loc[row_index, 'Benefits/costs']
    centralization_degree_huang = df_overall_info.loc[row_index, 'Centralization degree Huang']

    # Create the text box
    text_box = '''
    Total Population supplied: {:,.0f}
    Total Benefits (ILS): {:,.0f}
    Total Costs (ILS): {:,.0f}
    Coverage (% Region): {:.2f}
    Benefits/Costs: {:.2f}
    Centralization Degree (Huang): {:.2f}
    '''.format(total_population, total_benefits, total_costs, coverage_region, benefits_costs_ratio,
               centralization_degree_huang)

    plt.gca().text(1.05, 0.5, text_box, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')


    plt.title('Graph solution ' + str(row_index) + ' in file results ' + experiment_id)
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # Save the figure if save_path is provided

    plt.show()


# Call the function with the desired row index
run_id = 4900
generate_and_plot_graph(run_id, graph_data, wwtp_labels, overall_info, save_path='experiment_full_path' + 'graph.png')  # Replace with the desired row index

# Optimisation results to analyse

# ************************** Hossein***********************

# Read edges CSV file
edges_data = pd.read_csv(
    r'D:\IHE-Delft\2023\MSc 3-month theses\Hossein Tavakoligargari 2023\model\pipes.csv')  # Replace with your file path
# Read nodes CSV file
nodes_data = pd.read_csv(
    r'D:\IHE-Delft\2023\MSc 3-month theses\Hossein Tavakoligargari 2023\model\manholes.csv')  # Replace with your file path

# Create a graph
G = nx.DiGraph()

# Add nodes with attributes
for _, node_row in nodes_data.iterrows():
    node_name = node_row['Name']
    x_coord = node_row['X-Coord']
    y_coord = node_row['Y-Coord']
    G.add_node(node_name, x=x_coord, y=y_coord)
    # You can also add other attributes as needed

# Add edges with attributes
for _, edge_row in edges_data.iterrows():
    from_node = edge_row['From Node']
    to_node = edge_row['To Node']
    edge_name = edge_row['Name']
    G.add_edge(from_node, to_node, name=edge_name)
    # You can also add other attributes as needed

# # Draw the graph
# node_positions = {node_name: (x_coord, y_coord) for node_name, x_coord, y_coord in nodes_data[['Name', 'X-Coord', 'Y-Coord']].values}
# nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=100)
# #edge_labels = nx.get_edge_attributes(G, 'name')
# #nx.draw_networkx_edge_labels(G, pos=node_positions, edge_labels=edge_labels
#
# plt.title('Graph with Nodes and Edges')
# plt.xlabel('X-coordinate')
# plt.ylabel('Y-coordinate')
# plt.grid(False)
# plt.show()

given_node_id = 331


# Function to get upstream nodes recursively
def get_upstream_nodes_recursive(graph, node, visited=None):
    if visited is None:
        visited = set()
    visited.add(node)
    for predecessor in graph.predecessors(node):
        if predecessor not in visited:
            get_upstream_nodes_recursive(graph, predecessor, visited)
    return visited


# Get upstream nodes (all predecessors)
upstream_nodes = get_upstream_nodes_recursive(G, given_node_id)

# Convert to DataFrame
upstream_nodes_df = pd.DataFrame({'NodeID': list(upstream_nodes)})

print(upstream_nodes_df)


# ****************************************** HOSSEIN ******************


# Produce graph with wttp locations, connections and type of ndoes


# # Plotting last generation:
# coverage_last_gen = overall_info.loc[overall_info['idx_evaluation'] >= 0, 'Coverage (region)']
# ben_cos_last_gen = overall_info.loc[overall_info['idx_evaluation'] >= 0, 'Benefits/costs']
# central_last_gen = overall_info.loc[overall_info['idx_evaluation'] >= 0, 'Centralization degree Huang']
# num_wwtp_last_gen = overall_info.loc[overall_info['idx_evaluation'] >= 0, 'Number WWTP']
#
# # Identify Pareto front points
# pareto_front = []
# for i in range(len(coverage_last_gen)):
#     is_pareto = True
#     for j in range(len(coverage_last_gen)):
#         if i != j and coverage_last_gen[j] >= coverage_last_gen[i] and ben_cos_last_gen[j] >= ben_cos_last_gen[i]:
#             is_pareto = False
#             break
#     if is_pareto:
#         pareto_front.append((coverage_last_gen[i], ben_cos_last_gen[i]))
#
# # Plotting all generations:
# scatter_plot = overall_info.plot.scatter(x='Coverage (region)', y='Benefits/costs', c='Centralization degree Huang', s='Number WWTP')
#
# # Plot the Pareto front points on top
# pf_x = [point[0] for point in pareto_front]
# pf_y = [point[1] for point in pareto_front]
# scatter_plot.scatter(pf_x, pf_y, label='Pareto Front', color='black', marker='o', alpha=.1, s=40)
#
# # Annotate Pareto front points with their indices
# for i, point in enumerate(pareto_front):
#     scatter_plot.annotate(f'{i}', (point[0], point[1]), textcoords="offset points", xytext=(0, 10), ha='center')
#
#
# scatter_plot.legend()
# #scatter_plot.title('Scatter Plot with Pareto Front')
# scatter_plot.grid(True)
# plt.show(block=True)


# Function to generate a graph (when user clicks scatter with Pareto)
def generate_graph(selected_row, df_graph_data, df_wwtp_labels, df_overall_info):
    #    plt.figure()  # Create a new figure for the graph
    # ... Generate the graph using selected_row data

    # Get WWTP labels for nodes
    wwtp_labels_dict = {row['idx_run']: row for _, row in df_wwtp_labels.iterrows()}
    labels_for_this_idx_run = wwtp_labels_dict.get(selected_row, None)

    # Create a graph
    G = nx.DiGraph()

    # Add nodes with coordinates
    for index, row in df_graph_data.iterrows():
        G.add_node(row['id'], pos=(row['X'], row['Y']))

    # Add edges based on 'connects_to'
    for index, row in df_graph_data.iterrows():
        if pd.notna(row['connects_to']):
            G.add_edge(row['id'], row['connects_to'])

    # Get node positions
    node_positions = {node: (df_graph_data.loc[graph_data['id'] == node, 'X'].values[0],
                             df_graph_data.loc[graph_data['id'] == node, 'Y'].values[0])
                      for node in G.nodes}

    # Plot the graph with equal aspect ratio
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Create labels and colors for nodes

    node_labels = {}
    node_colors = []
    for node in G.nodes:
        if node in wwtp_labels_dict:
            label = wwtp_labels_dict[selected_row]['connect_to_{}'.format(
                int(node))]  # label = wwtp_labels_dict[node][selected_row]['connect_to_{}'.format(int(node))]
            if label == 'NOTHING':
                node_labels[node] = str(int(node))
                node_colors.append('lightblue')  # Non-'WWTP' nodes color
            else:
                node_labels[node] = '{}'.format(int(node))  # Two lines
                node_colors.append('red')  # 'WWTP' nodes color
        else:
            node_labels[node] = str(int(node))
            node_colors.append('lightblue')  # Non-'WWTP' nodes color

    # Draw the graph with node labels and colors
    nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=500, node_color=node_colors,
            labels=node_labels)

    # Extract the desired information from the DataFrame
    total_population = df_overall_info.loc[selected_row, 'Total population supplied (inhab)']
    total_benefits = df_overall_info.loc[selected_row, 'Total benefits  (ILS)']
    total_costs = df_overall_info.loc[selected_row, 'Total costs (ILS)']
    coverage_region = df_overall_info.loc[selected_row, 'Coverage (region)']
    benefits_costs_ratio = df_overall_info.loc[selected_row, 'Benefits/costs']
    centralization_degree_huang = df_overall_info.loc[selected_row, 'Centralization degree Huang']

    # Create the text box
    text_box = '''
    Total Population supplied: {:,.0f}
    Total Benefits (ILS): {:,.0f}
    Total Costs (ILS): {:,.0f}
    Coverage (% Region): {:.2f}
    Benefits/Costs: {:.2f}
    Centralization Degree (Huang): {:.2f}
    '''.format(total_population, total_benefits, total_costs, coverage_region, benefits_costs_ratio,
               centralization_degree_huang)

    plt.gca().text(1.05, 0.5, text_box, transform=plt.gca().transAxes, fontsize=10, verticalalignment='center')

    plt.title('Configuration of WWTPs - solution ' + str(selected_row))
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


# Create a function to handle click event
def on_click(event):
    if event.button == 1:  # Left mouse button click
        print('Clicked!')
        ind = event.ind[0]  # Get the index of the clicked point
        print('Index is: ' + str(ind))
        selected_row = overall_info.iloc[ind]  # Get the corresponding row from the DataFrame
        generate_graph(selected_row, graph_data, overall_info)  # Call the generate_graph function with selected_row


# # Connect the pick event to the generate_graph function
# scatter_plot.figure.canvas.mpl_connect('pick_event', generate_graph)

# **************************************
# plot WWTP locations

#
# # Create a graph
# G = nx.Graph()
#
# # Add nodes with coordinates
# for index, row in data.iterrows():
#     G.add_node(row['id'], pos=(row['X'], row['Y']))
#
# # Add edges based on 'connects_to'
# for index, row in data.iterrows():
#     if pd.notna(row['connects_to']):
#         G.add_edge(row['id'], row['connects_to'])
#
# # Get node positions
# node_positions = {node: (data.loc[data['id'] == node, 'X'].values[0],
#                          data.loc[data['id'] == node, 'Y'].values[0])
#                   for node in G.nodes}
#
# # Plot the graph
# plt.figure(figsize=(6, 10))
# plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
# nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=500, node_color='lightblue')
# plt.title('Graph from CSV')
# plt.show()

generate_graph(63043, graph_data, wwtp_labels, overall_info)
