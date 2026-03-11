# rSQP : A Noise Tolerant Sequential Quadratic Programming Algorithm with Relaxations

rSQP is a line search sequential quadratic programming (SQP) method for nonlinear optimization with noisy inequality constraints. To make the algorithm robust to the presence of bounded noise, the line search is relaxed, and the quadratic programming (QP) subproblem is modified by allowing relaxations to its feasible set. 
    
## Reference

Please refer to the paper *A Noise Tolerant SQP Algorithm for Inequality Constrained Optimization* by Figen Oztoprak and Richard Byrd.  The link to the paper coming up soon.

You can contact us via figenoeztoprak@gmail.com for any comments or questions.

## Usage

Install the package locally by cloning the repository and running ```pip install .```

Call the noise aware SQP solver via:

``` rsqp_solve(evalFC, evalGA, evalH, nVars, nCons, bl, bu, x_initial, options)```

Here,
    - evalFC is a routine that returns objective and constraint evaluations
    - evalGA is a routine that returns objective gradient and Jacobian evaluations
    - evalH is a routine that returns Hessian of the Lagrangian using given multipliers (only needed if options.hessType=1)
    - nVars is the number of variables of the problem
    - nCons is the number of constraints of the problem
    - bl and bu are lower and upper bounds of the variables respectively (set to -numpy.inf/numpy.inf if no bounds)
    - x_initial is the initial solution point to start the algorithm
    - options is the set of options of type rsqp_options with default values defined as follows.
    rsqp_options:
        maxiter: int = 1000
        opttol: float = 1e-8
        feastol: float = 1e-8
        noiseLevelObj: float = 0.0
        noiseLevelCons: float = 0.0
        noiseLevelGrad: float = 0.0
        noiseLevelJac: float = 0.0
        hessType: int = 2
        verbose: bool = True
        qp_solver: str = "scipy"

The current version of rSQP handles inequality constraints and bound constraints only (equality constraints are not accepted).  Also, noise level estimations are not computed internally in the current version, these estimations must be provided by the user via rsqp_options. 

## Examples

Examples using rSQP are provided under the *examples* folder 

## Testing on Cutest problems

Cutest problems can be solved via pycutest, by calling:

   ``` python rsqp_cutest PROBLEM_NAME ``` 

Please find the script under the folder *cutest*. For experimental purposes, the final objective can be compared to the true solution to the problem (obtained without injecting noise) by passing the true optimal objective value to the script as the second argument:

   ``` python rsqp_cutest PROBLEM_NAME f_optimal_true ``` 
    


