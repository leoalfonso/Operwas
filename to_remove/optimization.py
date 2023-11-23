from timeit import default_timer as timer

import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from platypus import NSGAII, Problem, Subset

from inputs import *
from .operwa_library import AOKP, LimitedBinary


def Operwa(nr_wwtp, nr_runs, checks):

    print('Optimization 1')

    # if not all([check.get() for check in checks]):
    #     print('Not set!!!')
    #     return
    #
    try:
        nr_wwtp = int(nr_wwtp)
    except:
        print('Could not convert number of wwtp', nr_wwtp, ' to an int (try again)')
        return
    try:
        nr_runs = int(nr_runs)
    except:
        print('Could not convert number of runs', nr_runs, ' to an int (try again)')
        return

    t_start = timer()
    print('running start time: ', t_start)
    print('Running set for: ', nr_runs, 'runs')

    # get the decision variable lists
    junc = loadtxt(file_with_possible_locations, skiprows=1, delimiter=';')
    # junc = loadtxt(file_with_possible_locations, delimiter=';')
    if len(junc.shape) == 1:
        junc = np.expand_dims(junc, axis=0)
    print('junc is', junc)
    print('type of junc is ', type(junc))
    tuple_junctions = tuple(tuple(row) for row in junc)

    print(tuple_junctions)
    # set the problem via Class Problem

    class Operwa(Problem):

        def __init__(self):
            super(Operwa, self).__init__(1, 2)
            self.types[:] = Subset(tuple_junctions, nr_wwtp)
            print('tuple_junctions', tuple_junctions)
            self.directions[0] = Problem.MAXIMIZE
            self.directions[1] = Problem.MAXIMIZE
            self._maria_sols = []
            self._maria_coords = []
            self._iteration = 0

        def evaluate(self, solution):
            print('solution variables', solution.variables[:])
            coordinates = solution.variables[:]
            print(coordinates)
            coordinates = np.array(coordinates[0])
            print(coordinates)
            # try:
            _sol = AOKP(coordinates)
            print(("Finished iteration {}".format(self._iteration)))
            # except:
            # sol = (0, 0)
            # print(('iteration failed: ', sys.exc_info()[1]))
            self._iteration += 1
            _sol_tuple = tuple(_sol)
            solution.objectives[:] = _sol_tuple
            solution.constraints[:] = ">(0,0)"

            to_save = "{},{},\"{}\"\n".format(_sol_tuple[0], _sol_tuple[1], coordinates)
            with open(file_all_generations, 'a') as f:
                f.write(to_save)

    # optimize the problem using function evaluations
    prob = Operwa()
    nr_runs = int(nr_runs)

    if nr_runs < 100:
        nr_pop = nr_runs
    else:
        nr_pop = 100

    print("nr_pop is ", nr_pop)
    print("type  of nr_pop is ", type(nr_pop))
    algorithm = NSGAII(prob, nr_pop)  # second parameter is for size of initial population
    # default initial population (without second parameter) is 100

    print('type of nr_runs is', type(nr_runs))
    print('nr_runs is ', nr_runs)
    algorithm.run(nr_runs)

    with open(file_with_results, 'w+') as file_handler:
        for item in algorithm.result:
            file_handler.write("{}\n".format(item))

    # Plot data in graph
    x_coverage = [s.objectives[1] for s in algorithm.result]
    y_bencost = [s.objectives[0] for s in algorithm.result]
    print("Nr. of calculations = ", nr_runs)
    print("Nr. of results =", len(x_coverage))
    plt.scatter(x_coverage, y_bencost)
    if len(x_coverage) == 0 and len(y_bencost) == 0:
        max_x_coverage = 1
        max_y_bencost = 1
    else:
        max_x_coverage = max(x_coverage)
        max_y_bencost = max(y_bencost)
    plt.xlim(xmin=0, xmax=1.1 * max_x_coverage)
    plt.ylim(ymin=0, ymax=1.1 * max_y_bencost)
    plt.xlabel("$Coverage$")
    plt.ylabel("$Benefit/Costs$")

    plt.grid()
    plt.show()

    print('End of optimization run')

    t_end = timer()
    print("Calculation took ", t_end - t_start, " seconds = ",
          (t_end - t_start) / 60, "minutes = ",
          ((t_end - t_start) / 60) / 60, "hours")

    for solution in algorithm.result:
        print((solution.objectives))


def Operwas2(nr_runs):

    # if not all([check.get() for check in checks]):
    #     print('Not set!!!')
    #     return
    #
    # try:
    #     nr_runs = int(nr_runs)
    # except:
    #     print('Could not convert number of runs', nr_runs, ' to an int (try again)')
    #     return

    t_start = timer()
    print('running start time: ', t_start)
    print('Running set for: ', nr_runs, 'runs')

    # set up folders

    list_of_paths = [path_results, path_temp, path_pour_points, path_outputs]
    for i in list_of_paths:
        if not os.path.exists(i):
            os.makedirs(i)

    # reset data file
    with open(file_all_generations, 'w') as f:
        pass

    # get the decision variable lists
    junc = loadtxt(file_with_possible_locations, skiprows=1, delimiter=';')
    if len(junc.shape) == 1:
        junc = np.expand_dims(junc, axis=0)
    print('junc is', junc)
    print('type of junc is ', type(junc))
    tuple_junctions = tuple(tuple(row) for row in junc)
    print(tuple_junctions)

    # set the problem via Class Problem
    class Operwa2(Problem):

        def __init__(self):
            super(Operwa2, self).__init__(1, 2)
            self.types[:] = LimitedBinary(len(junc))
            self.directions[0] = Problem.MAXIMIZE
            self.directions[1] = Problem.MAXIMIZE
            self._maria_sols = []
            self._maria_coords = []
            self._iteration = 0

        def evaluate(self, solution):
            include_junctions = solution.variables[0]
            coordinates = [junc[i] for i in range(len(junc)) if include_junctions[i]]
            coordinates = np.array(coordinates)
            # try:
            _sol = AOKP(coordinates)
            print(("Finished iteration {}".format(self._iteration)))
            # except:
            #     _sol = (0,0)
            #     print(('iteration failed: ', sys.exc_info()[1]))
            self._iteration += 1
            _sol_tuple = tuple(_sol)
            solution.objectives[:] = _sol_tuple
            solution.constraints[:] = ">(0,0)"

            nr_wwtp = sum(include_junctions)

            to_save = "{},{},{},\"{}\"\n".format(
                nr_wwtp, _sol_tuple[0], _sol_tuple[1], include_junctions)
            with open(file_all_generations, 'a') as f:
                f.write(to_save)

    # optimize the problem using 10,000 function evaluations
    prob = Operwa2()
    if nr_runs < 100:
        nr_pop = nr_runs
    else:
        nr_pop = 100
    print("number of population is ", nr_pop)
    algorithm = NSGAII(prob, nr_pop)  # second parameter is for size of population
    algorithm.run(nr_runs)

    with open(file_with_results, 'w+') as file_handler:
        for item in algorithm.result:
            file_handler.write("{}\n".format(item))

    x_coverage = [s.objectives[1] for s in algorithm.result]
    y_bencost = [s.objectives[0] for s in algorithm.result]
    z_wwtp = [sum(s.variables) for s in algorithm.result]

    w_coord = [s.variables for s in algorithm.result]

    if len(x_coverage) == 0 and len(y_bencost) == 0:
        max_x_coverage = 1
        max_y_bencost = 1
    else:
        max_x_coverage = max(x_coverage)
        max_y_bencost = max(y_bencost)

    df = pd.DataFrame(dict(x=x_coverage, y=y_bencost, s=z_wwtp, w=w_coord))
    df.to_csv(file_with_last_population, index=None, header=True)

    for key, row in df.iterrows():
        plt.scatter(row['x'], row['y'], s=row['s'] * 10, alpha=.5)
        plt.annotate(int(row['s']), xy=(row['x'], row['y']), ha='center', va='center')

    # plt.scatter(x, y, s=z*1000, alpha=0.5)
    plt.xlim(xmin=0, xmax=1.2*max_x_coverage)
    plt.ylim(ymin=0, ymax=1.2*max_y_bencost)
    plt.xlabel("$Coverage$")
    plt.ylabel("$Benefits/Costs$")

    plt.grid()
    plt.show()

    # # # For each point, we add a text inside the bubble
    # for s in algorithm.result:
    #     plt.text(plt.x.iloc[s.objectives[1]], plt.y.iloc[s.objectives[0]], plt.z.iloc[sum(s.variables)], horizontalalignment='center', size='medium', color='black',
    #             weight='semibold')

    # bubbles = plt.subplots()
    # produce a legend with a cross section of sizes from the scatter
    # handles, labels = graph.legend(prop="sizes", alpha=0.5)
    # legend2 = bubbles.legend(handles, labels, loc="upper right", title="Sizes")

    print('End of optimization run')

    t_end = timer()
    print("Calculation took ", t_end - t_start, " seconds = ",
          (t_end - t_start) / 60, "minutes = ",
          ((t_end - t_start) / 60) / 60, "hours")

    # for solution in algorithm.result:
    #     print(solution.objectives)

    for solution in algorithm.result:
        print(df)
