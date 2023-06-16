# Operwas changes

## Story time

- Question: what happens when you consider the possibility of pumping stations in the Operwas optimization system?
- Solution: add pumping stations to the system and find out.
- Original problem:
  - 33 decision variables, choose from:
  - Node x is nothing
  - Node x is a WWTP
- New problem:
  - 33 decision variables, choose from:
  - Node x is nothing
  - Node x is a WWTP
  - Node x pumps to node y
  - Node x pumps to node z
  - etc.
- In other words: the solution space goes from 2^33 to (2+33)^33.
- The naive way of letting the genetic algorithm pick the connections leads to many configurations being evaluated that are "infeasible"
- A configuration is infeasible when:
  - It contains a cycle
  - At least one WWPS is pumping to a node that is a "nothing"
- Trying to "teach" the genetic algorithm not to create infeasible solutions by giving them very bad scores takes a lot of iterations. Plus there is a high chance that you are simply teaching the algorithm to be too reluctant to pump at all.
- Also, the current Operwas algorithm is very slow. This has the following reasons:
  - It keeps recalculating the same things over and over.
  - It takes very long to do the subcatchment delineation.
- Three solutions:
  - Many configurations are infeasible and these should not be considered at all. Take time to create custom Platypus classes that are unable to create infeasible configurations during variation and mutation.
  - Gather data beforehand and use interpolation during the runs to obtain approximately correct data
  - Gather data beforehand and use the fact that the subcatchments are a river network and can be efficiently handled using a tree/graph data structure. Subcatchment delineation then only needs to be performed once.

## Changes

2 main changes:
- **New custom Platypus pumping graph type (solution feasibility)**
  - Created a custom solution class that contains the entire configuration graph
  - Created a custom way of generating randomized graphs that are feasible (required for initialization of the population)
  - Created a custom mutator that is able to mutate graphs in a way that keeps them feasible (required for mutation)
  - Created a custom crossover operator that combines two graphs in a way that keeps them feasible (required for variation)
  - To reduce the solution space further, only feasible pumping connections are considered. A set of feasible pumping connections is created for each outlet, based on:
    - Cannot pump to yourself
    - If there is a "downstream" path from node A to B then it is considered infeasible to pump from A to B.
    - Minimum and maximum pumping height
    - Pipe length (pump, grav, and total)
- **Join calcul pre-calculations (speed)**
  - Created a bunch of script that run the "experiments", they collect the necessary data in corresponding folders. These only need to be run once, afterwards the data can be reused.
  - Use a tree (graph) to store subcatchment data
    - Use the tree for fast accumulations
      - Network length
      - Population served
    - Store stuff like land price at the outlet location in the nodes of the tree
  - Use data interpolators for everything to do with the circular reuse buffers
    - Get reuse type and buffer radius based on flow at the WWTP
    - Use the buffer radius to get other stuff
      - Population per town
      - Reuse network length
  - Pre-calculate all the pumping stuff between each outlet
    - Pumping height
    - Pipe length (pump)
    - Pipe length (grav)

Some other miscellaneous stuff:
- Now using a custom `NodeType` class to contain all the possible node-types (`NOTHING`, `WWTP`, `WWPS`)
- Now using a custom `ReuseType` to contain all the possible reuse-types (`AGRICULTURAL`, `URBAN`)
- No more "mbr with no reuse" possible, applied reuse flow is always equal to the flow available for reuse.
- Added type-hints to make the code more readable and ready it for static type-checking
- Added some plotting functionality to plot the graph
- Imports of user-inputs and paths and such are now behind a namespace. This makes it easier to see where variables are coming from. I.e. `import x as y` instead of `from x import *`.
- Cleaned up the code and improved performance by utilizing `numpy` vectorized computation where possible.
- Added a config script that contains all the parameters that control the newly added code
- Bug fixed where Operwas results did not match the coordinates because of reordering of subcatchments.