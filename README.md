# Vehicle Routing Problem with Drones

This repository contains code and data related to the "Vehicle Routing Problem with Drones" project.

## Repository Contents

1. `Dataset.ipynb`: A Jupyter notebook that contains array lists of cities used for testing and evaluating the project.
2. `helper.py`: A Python script that consists of all the function definitions necessary for the operations in the project.
3. `project_prototype`: This file consists of the main codebase for the "Vehicle Routing Problem with Drones".

## Prerequisites

To run the code in this repository, you'll need to have the following software/packages installed:

- Python version 3.11
- Jupyter Notebook / Visual Studio Code
- Gurobi optimization solver version 10.0.2.0

Additionally, you'll need the following Python libraries:
- numpy
- gurobipy (For interfacing with the Gurobi solver)
- plotly
- helper (local module)
- tabulate
- matplotlib
- seaborn

## Installation

1. Install the required Python packages using `pip`:

    ```bash
    pip install numpy gurobipy plotly tabulate matplotlib seaborn
    ```

    **Note**: `gurobipy` requires the Gurobi optimizer installed with a valid license.

## License for Gurobi Optimization Solver

This project uses Gurobi optimization solver version 10.0.2.0. Please note that the use of Gurobi is subject to its own terms and licensing. This project is meant for academic purposes, and you are required to have an appropriate academic license for Gurobi to use it with this project.

If you don't have a Gurobi license already, you can apply for an academic license on the [official Gurobi website](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Getting Started

1. Download the zip file.
2. Navigate to the directory where you have stored these files.
3. Launch Jupyter Notebook (`jupyter notebook` in terminal or command prompt) and open `Dataset.ipynb` to view the city datasets.
4. To run the main prototype, ensure that the Gurobi solver is correctly set up
5. Copy the desired dataset from `dataset.ipynb` and paste it in the 2nd cell replacing the current `city_list` arraylist and then execute the `project_prototype`.

## References

1. Saturn Cloud (2023) 2-opt algorithm: Solving the travelling salesman problem in Python, Saturn Cloud Blog. Available at: https://saturncloud.io/blog/2opt-algorithm-solving-the-travelling-salesman-problem-in-python/ (Accessed: 24 July 2023). 
