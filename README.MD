# 8 Queens - Genetic Algorithm Solution

Solving the 8 queens puzzle using [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm). The goal of the program is to place 8 queens on a board in such a way that no queens are attacking each other.  

Change global variables N and mutation_pct to experiment with different population number and mutation percentage.

## Requirements

* Python 3.5 or above
* [Numpy 1.16](https://www.numpy.org/) - For easy data array manipulation
* [Matplotlib](https://matplotlib.org/) - For plots

## Installing

cd into cloned directory. Then:

* `virtualenv env`
* `source env/bin/activate`
* `pip install -r requirements.txt`
* `python main.py`


## Discussion

Mutation rate seemed to play a significant part in finding the solution. Whenever the algorithm
converged to a single value, the mutation enabled the algorithm to still find a solution (But a solution is still not guaranteed in Genetic Algorithm).

Setting mutation percentage to a low value (1%) caused the algorithm to run to take anywhere between 1000-5000+ iterations to find a solution. Setting mutation percentage to 5% significantly attenuated this problem. The algorithm was able to obtain a solution under 1000 iterations most of the time. 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details