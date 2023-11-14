# Finding out how to plot Maria Alice's results

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import functions_plot_leo

# Generic file_graph_nodes locations
csv_file_graph_information = 'D:\\OP_pycharm\\Operwas_pump\\inputs\\wwtp_locations_many.csv'

# ************************ Visualising results genetic pumping ****************************
# ******in 'D:\OP_pycharm\Operwas_pump\optimization_results\genetic_pumping\ **************
# *****************************************************************************************

experiment_folder_pump = 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\genetic_pumping\\'
experiment_id_pump = 'run_2023-10-29-20-34-29-653'
experiment_full_path_pump = experiment_folder_pump + experiment_id_pump + '\\'

# CSV pump files
csv_file_gen_pump_solutions = experiment_full_path_pump + 'solutions' + experiment_id_pump.lstrip("run") + '.csv'  # solutions
csv_file_gen_pump_results_total = experiment_full_path_pump + 'results_total' + experiment_id_pump.lstrip(
    "run") + '.csv'  # results_total

# Read CSV file with graph information
graph_data = pd.read_csv(csv_file_graph_information)
#functions_plot_leo.generate_graph_only(graph_data, "Connection among candidates following natural flows by gravity")


# Read CSV file with 'WWTP' labels genetic_pumping
wwtp_labels_pump = pd.read_csv(csv_file_gen_pump_solutions)

# Read CSV file with information about the WWTP configuration with no pumping
overall_info_pump = pd.read_csv(csv_file_gen_pump_results_total)

solution_id = 38060
graph_title_pump = 'Solution with pumps ' + str(solution_id) + ' experiment ' + experiment_id_pump
#functions_plot_leo.generate_graph_nodes_pump(solution_id, graph_data, wwtp_labels_pump, overall_info_pump, graph_title_pump)
#functions_plot_leo.plot_all_pop_with_pareto(overall_info_pump, "Solutions with pumps")

# ***************************** Visualising results no_pumping ****************************
# ***********in 'D:\OP_pycharm\Operwas_pump\optimization_results\no_pumping\ **************
# *****************************************************************************************

# Files no pumping
# 'D:\OP_pycharm\Operwas_pump\optimization_results\no_pumping\run_2023-08-18-17-18-52-125\solutions_2023-08-18-17-18-52-125.csv'
# 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\no_pumping\\run_2023-08-18-17-18-52-125\\solutions_2023-08-18-17-18-52-125'
experiment_folder_no_pump = 'D:\\OP_pycharm\\Operwas_pump\\optimization_results\\no_pumping\\'
experiment_id_no_pump = 'run_2023-08-18-17-18-52-125'
experiment_full_path_no_pump = experiment_folder_no_pump + experiment_id_no_pump + '\\'

# CSV Files
csv_file_no_pump_solutions = experiment_full_path_no_pump + 'solutions' + experiment_id_no_pump.lstrip("run") + '.csv'  # solutions_xxx.csv
csv_file_no_pump_results_total = experiment_full_path_no_pump + 'results_total' + experiment_id_no_pump.lstrip(
    "run") + '.csv'  # results_total

# Read CSV file with graph information
graph_data_no_pump = pd.read_csv(csv_file_graph_information) #(already read in the beginning of this file)

# Read CSV file with node labels ('WWTP' or "WWPS')
all_node_labels = pd.read_csv(csv_file_no_pump_solutions)

# Read CSV file with information about the WWTP configuration with no pumping
overall_info_no_pump = pd.read_csv(csv_file_no_pump_results_total)

solution_id = 25004
graph_title_no_pump = 'Solution with no pumps ' + str(solution_id) + ' experiment ' + experiment_id_no_pump
#functions_plot_leo.generate_graph_nodes_no_pump(solution_id, graph_data_no_pump, all_node_labels, overall_info_no_pump, graph_title_no_pump)
#functions_plot_leo.plot_all_pop_with_pareto(overall_info_no_pump, "Solutions with no pumps")




# Analyse solutions with no-pumping, for 4 WWTPs, using Hossein's costs

experiment_folder_no_pump_4_wwtp_Hos = r'D:\OP_pycharm\Operwas_pump\optimization_results\no_pumping_n_wwtps\\'
experiment_id_no_pump_4_wwtp_Hos = 'run_2023-10-30-18-19-20-825'
experiment_full_path_no_pump_4_wwtp = experiment_folder_no_pump_4_wwtp_Hos + experiment_id_no_pump_4_wwtp_Hos + '\\'

# CSV Files
csv_file_no_pump_solutions = experiment_full_path_no_pump_4_wwtp + 'solutions' + experiment_id_no_pump_4_wwtp_Hos.lstrip("run") + '.csv'  # solutions_xxx.csv
csv_file_no_pump_results_total = experiment_full_path_no_pump_4_wwtp + 'results_total' + experiment_id_no_pump_4_wwtp_Hos.lstrip(
    "run") + '.csv'  # results_total

# Read CSV file with graph information
graph_data_no_pump_4 = pd.read_csv(csv_file_graph_information) #(already read in the beginning of this file)

# Read CSV file with node labels (n values indicating nodes)
all_node_labels = pd.read_csv(csv_file_no_pump_solutions)

# Read CSV file with information about the WWTP configuration with no pumping
overall_info_no_pump_4 = pd.read_csv(csv_file_no_pump_results_total)

solution_id = 1489
graph_title_no_pump = 'Solution with no pumps ' + str(solution_id) + ' experiment ' + experiment_id_no_pump
functions_plot_leo.generate_graph_nodes_n_wwtps(solution_id, graph_data_no_pump_4, all_node_labels, overall_info_no_pump_4, graph_title_no_pump)
functions_plot_leo.plot_all_pop_with_pareto(overall_info_no_pump_4, "Solutions with 4 WWTP, network costs with hydraulic model")


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