# This code is only to produce initial solutions for combinations of n wwtps < 5 out of 33, to be evaluated by the operwas code
# This needed to be done because the current way operwas works does not allow for small number of combinations, and
# Hossein's work only allows for a max combinaiton of n=4

from itertools import combinations

# Number of columns and the number of 1s
num_columns = 33
num_ones = 4

# Generate all possible combinations of positions for three 1s
combinations_of_ones = list(combinations(range(num_columns), num_ones))

# Create a list to represent each row
all_rows = []

# Create rows with 0s and 1s based on the combinations
for combination in combinations_of_ones:
    row = [1 if i in combination else 0 for i in range(num_columns)]
    all_rows.append(row)

# Define the file name
file_name = r"D:\OP_pycharm\Operwas_pump\src\faster\optimization\combinations.txt"

# Write the rows to a CSV file with '[' and ']' around each row
with open(file_name, 'w') as file:
    for row in all_rows:
        formatted_row = '[' + ','.join(map(str, row)) + ']'
        file.write(formatted_row + "\n")

print(f"{len(all_rows)} combinations written to {file_name}")


# ------------------

def network_cost(files_directory, WWTP_IDs):
    # Convert WWTP IDs to numbers and sort them

    WWTP_IDs = [int(WWTP_ID) for WWTP_ID in WWTP_IDs]
    WWTP_IDs.sort()

    # Create the comb variable as a string
    comb = ""

    for WWTP_ID in WWTP_IDs:
        comb += f"{WWTP_ID},"

    file_path = files_directory + '\\' + str(len(WWTP_IDs)) + '.txt'
    cost_file = open(file_path, "r")

    for line in cost_file:
        if line.startswith(comb):
            cost = line.split(',')[len(WWTP_IDs)]
            cost_file.close()
            return cost

    cost_file.close()
    return 0
