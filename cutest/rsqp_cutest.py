'''
rsqp_cutest.py
author: Figen Oztoprak, figenoeztoprak@gmail.com
last updated: 9.Mar.2026
'''

import pycutest
import rsqp_noiseaware as rsqp
import numpy as np
from time import process_time
import sys
import warnings

INFTY = 1e+20

def evalFC_exact (x):
    #constraints in form c(x)<=0
    y = np.reshape(x,prob.n)
    obj , c = prob.objcons(y)
    for i in range(prob.m) : 
        if(prob.cl[i]>-INFTY): #>= constraint
            c[i] = prob.cl[i]-c[i]
        elif (prob.cu[i]<INFTY): #<= constraint
            c[i] = c[i]-prob.cu[i]
        else :
            sys.exit("InputError: A constraint with no bounds")
    c = np.reshape(c,(prob.m,1))
    return obj , c

def evalFC (x):
    obj, c = evalFC_exact(x)
    if(noise_type=="U"):
        obj = obj + eps_func*np.random.uniform(-1,1)
        c = c + eps_func*np.random.uniform(-1,1,prob.m)
    if(noise_type=="N"):
        obj = obj + np.random.normal(0.0,0.5*eps_func)
        c = c + np.random.normal(0.0,0.5*eps_func,prob.m)        
    #c = np.reshape(c,(prob.m,1))
    return obj , c

def evalGA (x):
    y = np.reshape(x,prob.n)
    gradf, jac = prob.lagjac(y)
    #jac = np.reshape(jac,(prob.n*prob.m))
    for i in range(prob.m) :
        if(prob.cl[i]>-INFTY): 
            jac[i,:] = -1*jac[i,:]
    if(noise_type=="U"):
        gradf = gradf + eps_grad*np.random.uniform(-1,1,prob.n)    
        if(prob.m>0):
            jac = jac + eps_grad*np.random.uniform(-1,1,np.shape(jac)) 
    if(noise_type=="N"):
        gradf = gradf + np.random.normal(0.0,0.5*eps_grad,prob.n)   
        if(prob.m>0): 
            jac = jac + np.random.normal(0.0,0.5*eps_grad,np.shape(jac)) 
    gradf = np.reshape(gradf,(prob.n,1))   
    return gradf, jac

def evalH (x,l):
    y = np.reshape(x,prob.n)
    if(prob.m>0):
        v = np.reshape(l,prob.m)
    for i in range(prob.m) : 
        if(prob.cl[i]>-INFTY): 
            v[i]=-1*v[i]
    hess = prob.hess(y, v=v) 
    return hess

def call_rsqp(prob):        
    #Variables
    n = prob.n

    #Constraints; do not accept equality constraints
    m = prob.m
    for i in range(m):
        if(prob.is_eq_cons[i]):
           sys.exit("EXIT: The current version of rSQP does not accept equality constraints")
    for i in range(m):
        if(prob.cl[i]>-INFTY and prob.cu[i]<INFTY):
           sys.exit("EXIT: The current version of rSQP does not accept range constraints")
    if(prob.n_fixed>0) :
        sys.exit("EXIT: The current version of rSQP does not accept fixed variables")
    
    #Set initial solution
    x0 = prob.x0
        
    #Parameters
    options = rsqp.rsqp_options()
    options.maxiter = 100
    options.noiseLevelObj = eps_func
    options.noiseLevelCons = eps_func
    options.noiseLevelGrad = eps_grad
    options.noiseLevelJac = eps_grad
    options.qp_solver = "quadprog"

    if(len(sys.argv)>2):
        fopt_true = float(sys.argv[2]) #the true optimal solution of the problem
  
    # Solve!
    t_start = process_time()
    solution = rsqp.rsqp_solve(evalFC, evalGA, evalH, n, m, prob.bl, prob.bu, x0, options) 
    t_stop = process_time()    
    fexact , cexact = evalFC_exact (solution.x_final)

    if(len(sys.argv)>2):
        fopt_gap = np.fabs(fopt_true-fexact)
    else:
        fopt_true = INFTY
        fopt_gap = INFTY

    # Print information
    print("Termination : %d" % solution.termination) 
    print("Final Objective Value : %e" % solution.f_final)
    print("CPU Time Elapsed (s)  : %f " % (t_stop-t_start))    
    print("Final Objective Value (exact) : %e" % fexact)
    print("Final Constraint Violation (exact, infnorm) : %e" 
          % np.linalg.norm(np.maximum(cexact,np.zeros(m)), np.inf))
    if(len(sys.argv)>2):
        print("Optimality gap (in objective value): %e" % fopt_gap)
    print("\n")

#MAIN
global prob
global eps_func
global eps_grad
global noise_type
warnings.filterwarnings("ignore")

#set artificial noise type and level
np.random.seed(101)
#noise_type = "N" #normal
noise_type = "U" #uniform
eps_func = 1e-8 
eps_grad = 1e-4

#get problem
p = sys.argv[1]
pycutest.clear_cache(p)
prob = pycutest.import_problem(p)
#prob=pycutest.import_problem(p, sifParams={'N':50})

#solve via rSQP
call_rsqp(prob)
pycutest.clear_cache(p)

#end rsqp_cutest
