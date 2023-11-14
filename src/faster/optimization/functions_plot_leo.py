# Finding out how to plot Maria Alice's results

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# ***************************** Visualising results no_pumping ****************************
# ***********in 'D:\OP_pycharm\Operwas_pump\optimization_results\no_pumping\ **************
# *****************************************************************************************


def generate_graph_nodes_no_pump(selected_row, df_graph_data, df_node_labels, df_overall_info, graph_title):
    '''# ... Generate the graph using selected_row data
    selected_row; one of the solutions in file "results_total...csv"

    df_graph_data: node locations and connections in inputs\\wwtp_locations_many.csv'

    df_node_labels: as found in solutions_xxx.csv, which has the form:
       idx_run,connect_to_0,connect_to_1...
       0,      WWTP        ,NOTHING     ,...

    df_overal_info: as found in "results_total...csv", which has the form:
       Total population supplied (inhab),Benefit wastewater (Total with PV summed),Total benefit reclaimed wastewater (total PV),...
       73114.7217,                       39207908.58071944,                        67641180.31590986,...
    pump: if results are coming from pump analysis, set to True.
    example
    generate_graph_nodes(selected_row, graph_data, all_node_labels, overall_info)
    '''
    # Get WWTP labels for nodes
    wwtp_labels_dict = {row['idx_run']: row for _, row in df_node_labels.iterrows()}
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
    node_positions = {node: (df_graph_data.loc[df_graph_data['id'] == node, 'X'].values[0],
                             df_graph_data.loc[df_graph_data['id'] == node, 'Y'].values[0])
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
            if label == 'NOTHING':  # node is neither a WWTP nor a WWPS
                node_labels[node] = str(int(node))
                node_colors.append('lightblue')  # Non-'WWTP' nodes color
            elif label == 'WWPS':  # node is a pumping station
                node_labels[node] = str(int(node))
                node_colors.append('blue')  # 'WWPS' nodes color
            else:  # node is a treatment plant
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

    plt.title('Configuration from solution ' + str(selected_row) + ' in ' + graph_title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def generate_graph_nodes_pump(selected_row, df_graph_data, df_node_labels, df_overall_info, graph_title):
    pass
    # todo:  df_node_labels is different than for no_pumps, found in solutions_xxx.csv. It has the form:
    #        idx_run,connect_to_0,connect_to_1...
    #        0,      x        ,-2     ,-1...
    #        where -2, if node mentioned in the heading is a NOTHING point
    #              -1, if node mentioned in the heading is a WWTP point
    #               x, if node mentioned in the heading is connected to node x by means of a WWPS

    # Get WWTP labels for nodes
    wwtp_labels_dict = {row['idx_run']: row for _, row in df_node_labels.iterrows()}
    labels_for_this_idx_run = wwtp_labels_dict.get(selected_row, None)

    # Create a graph
    G = nx.DiGraph()

    # Add nodes with coordinates
    for index, row in df_graph_data.iterrows():
        G.add_node(row['id'], pos=(row['X'], row['Y']))

    # Get node positions
    node_positions = {node: (df_graph_data.loc[df_graph_data['id'] == node, 'X'].values[0],
                             df_graph_data.loc[df_graph_data['id'] == node, 'Y'].values[0])
                      for node in G.nodes}

    # Determine node colors and edges
    node_colors = []
    edges = []

    for node in G.nodes:
        label_row = df_node_labels.loc[df_node_labels['idx_run'] == selected_row]
        label = label_row['connect_to_' + str(int(node))].values[0]

        if label == -2:
            node_colors.append('lightblue')
        elif label == -1:
            node_colors.append('red')
        elif label >= 0:
            node_colors.append('blue')
            edges.append((node, label))  # Create edges based on your rules

    # Plot the graph with equal aspect ratio
    plt.figure(figsize=(10, 10))
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio

    # Create labels for nodes
    node_labels = {node: str(int(node)) for node in G.nodes}

    # Draw the graph with node labels and colors
    nx.draw(G, pos=node_positions, with_labels=True, font_weight='bold', node_size=500, node_color=node_colors,
            labels=node_labels)

    # Draw edges based on the edges list
    nx.draw_networkx_edges(G, pos=node_positions, edgelist=edges, edge_color='blue')

    # Create a legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10,
                   label='Non-selected candidate'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='TP'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='PS'),
    ]

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

    # Add the legend to the graph
    plt.legend(handles=legend_elements)

    plt.title('Configuration from solution ' + str(selected_row) + ' in ' + graph_title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def generate_graph_nodes_n_wwtps(selected_row, df_graph_data, df_node_labels, df_overall_info, graph_title):
    ''' ... Generate the graph using selected_row data
    selected_row; one of the solutions in file "results_total...csv"

    df_graph_data: node locations and connections in inputs\\wwtp_locations_many.csv'

    df_node_labels: as found in solutions_xxx.csv, which for n=4 wwtps has the form:
    idx_run,connect_to_0,connect_to_1,connect_to_2,connect_to_3,connect_to_4,...
    0      ,5           ,24          ,18          ,18
    ...
    so, this is equivalent to say that all nodes are NOTHING except 5, 24, 18 and 18, which are all WWTPs
    An alternative is to convert this to the form:
    0  , NOTHING, NOTHING, NOTHING, NOTHING, WWTP, NOTHING...
    ... and use the function generate_graph_nodes_no_pump

    df_overal_info: as found in "results_total...csv", which has the form:
       Total population supplied (inhab),Benefit wastewater (Total with PV summed),Total benefit reclaimed wastewater (total PV),...
       73114.7217,                       39207908.58071944,                        67641180.31590986,...
    pump: if results are coming from pump analysis, set to True.
    example
    generate_graph_nodes(selected_row, graph_data, all_node_labels, overall_info)
    '''

    # Create a new DataFrame with the same column names
    # Converting this to the form:
    # 0  , NOTHING, NOTHING, NOTHING, NOTHING, WWTP, NOTHING...

    # Extract the values from the DataFrame
    # Initialize an empty list to store the new DataFrames
    new_dfs = []

    # Iterate through each row in the original DataFrame
    for index, row in df_node_labels.iterrows():
        values = row[1:].values  # Extract values from the row
        new_data = {'idx_run': [row['idx_run']]}  # Create new data dictionary for the row

        # Create the new row based on values
        for i in range(33):  # Assuming you have 33 columns
            if i in values:
                new_data[f'connect_to_{i}'] = 'WWTP'
            else:
                new_data[f'connect_to_{i}'] = 'NOTHING'

        new_df = pd.DataFrame(new_data)  # Create a new DataFrame for the row
        new_dfs.append(new_df)  # Append the new DataFrame to the list

    # Concatenate all the new DataFrames into one
    new_df_node_labels = pd.concat(new_dfs, ignore_index=True)

    # Get WWTP labels for nodes
    wwtp_labels_dict = {row['idx_run']: row for _, row in new_df_node_labels.iterrows()}
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
    node_positions = {node: (df_graph_data.loc[df_graph_data['id'] == node, 'X'].values[0],
                             df_graph_data.loc[df_graph_data['id'] == node, 'Y'].values[0])
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
            if label == 'NOTHING':  # node is neither a WWTP nor a WWPS
                node_labels[node] = str(int(node))
                node_colors.append('lightblue')  # Non-'WWTP' nodes color
            elif label == 'WWPS':  # node is a pumping station
                node_labels[node] = str(int(node))
                node_colors.append('blue')  # 'WWPS' nodes color
            else:  # node is a treatment plant
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

    plt.title('Configuration from solution ' + str(selected_row) + ' in ' + graph_title)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


def plot_all_pop_with_pareto(df_results_total, title_str):
    # # *****************************************************************************
    # # Plotting given generations with identified Pareto front:
    # # *****************************************************************************
    coverage_last_gen = df_results_total.loc[df_results_total['idx_evaluation'] >= 0, 'Coverage (region)']
    ben_cos_last_gen = df_results_total.loc[df_results_total['idx_evaluation'] >= 0, 'Benefits/costs']
    central_last_gen = df_results_total.loc[df_results_total['idx_evaluation'] >= 0, 'Centralization degree Huang']
    num_wwtp_last_gen = df_results_total.loc[df_results_total['idx_evaluation'] >= 0, 'Number WWTP']

    # Identify Pareto front points
    # pareto_front = []
    # for i in range(len(coverage_last_gen)):
    #     is_pareto = True
    #     for j in range(len(coverage_last_gen)):
    #         if i != j and coverage_last_gen[j] >= coverage_last_gen[i] and ben_cos_last_gen[j] >= ben_cos_last_gen[i]:
    #             is_pareto = False
    #             break
    #     if is_pareto:
    #         pareto_front.append((coverage_last_gen[i], ben_cos_last_gen[i]))

    # Plotting all generations:
    scatter_plot = df_results_total.plot.scatter(x='Coverage (region)', y='Benefits/costs', c='Centralization degree Huang',
                                             s='Number WWTP')

    # Plot the Pareto front points on top
    # pf_x = [point[0] for point in pareto_front]
    # pf_y = [point[1] for point in pareto_front]
    # scatter_plot.scatter(pf_x, pf_y, label='Pareto Front', color='black', marker='o', alpha=.1, s=40)

    # Annotate Pareto front points with their indices
    # for i, point in enumerate(pareto_front):
    #    scatter_plot.annotate(f'{i}', (point[0], point[1]), textcoords="offset points", xytext=(0, 10), ha='center')

    scatter_plot.legend()
    scatter_plot.grid(True)
    plt.title(title_str)
    plt.show(block=True)





def generate_graph_only(df_graph_data, graph_title):
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
    node_positions = {node: (df_graph_data.loc[df_graph_data['id'] == node, 'X'].values[0],
                             df_graph_data.loc[df_graph_data['id'] == node, 'Y'].values[0])
                      for node in G.nodes}

    # Plot the graph with equal aspect ratio
    plt.figure(figsize=(8, 6))
    plt.gca().set_aspect('equal', adjustable='box')  # Set equal aspect ratio
    nx.draw(G, pos=node_positions, with_labels=False, font_weight='bold', node_size=200, node_color='lightblue',arrows=True, connectionstyle="arc3, rad=0.3", edgecolors='blue')

    # Format node labels as integers
    node_labels = {node: str(int(node)) for node in G.nodes}

    # Draw node labels
    nx.draw_networkx_labels(G, pos=node_positions, labels=node_labels, font_size=8, font_color='black')

    plt.title(graph_title)
    plt.show()

# ***************************************************************************************************************
# ************************** Hossein - uses graphs to facilitate analysis of SWIMM models ***********************
# ***************************************************************************************************************
#
# # Read edges CSV file
# edges_data = pd.read_csv(
#     r'D:\IHE-Delft\2023\MSc 3-month theses\Hossein Tavakoligargari 2023\model\pipes.csv')  # Replace with your file path
# # Read nodes CSV file
# nodes_data = pd.read_csv(
#     r'D:\IHE-Delft\2023\MSc 3-month theses\Hossein Tavakoligargari 2023\model\manholes.csv')  # Replace with your file path
#
# # Create a graph
# G = nx.DiGraph()
#
# # Add nodes with attributes
# for _, node_row in nodes_data.iterrows():
#     node_name = node_row['Name']
#     x_coord = node_row['X-Coord']
#     y_coord = node_row['Y-Coord']
#     G.add_node(node_name, x=x_coord, y=y_coord)
#     # You can also add other attributes as needed
#
# # Add edges with attributes
# for _, edge_row in edges_data.iterrows():
#     from_node = edge_row['From Node']
#     to_node = edge_row['To Node']
#     edge_name = edge_row['Name']
#     G.add_edge(from_node, to_node, name=edge_name)
#     # You can also add other attributes as needed
#
# # Draw the graph (this takes time)
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
#
# given_node_id = 331
#
#
# # Function to get upstream nodes recursively
# def get_upstream_nodes_recursive(graph, node, visited=None):
#     if visited is None:
#         visited = set()
#     visited.add(node)
#     for predecessor in graph.predecessors(node):
#         if predecessor not in visited:
#             get_upstream_nodes_recursive(graph, predecessor, visited)
#     return visited
#
#
# # Get upstream nodes (all predecessors)
# upstream_nodes = get_upstream_nodes_recursive(G, given_node_id)
#
# # Convert to DataFrame
# upstream_nodes_df = pd.DataFrame({'NodeID': list(upstream_nodes)})
#
# print(upstream_nodes_df)

# ****************************************** HOSSEIN ******************


