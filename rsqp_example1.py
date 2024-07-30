import rsqp
import numpy as np
import scipy.linalg as linalg
from time import process_time
import sys
import warnings
import matplotlib.pyplot as plt

class rsqp_options:
    maxiter = 1000
    tol = 1e-8
    xOpt = None
    noiseLevelObj = 0.0
    noiseLevelCons = 0.0
    noiseLevelGrad = 0.0
    noiseLevelJac = 0.0
    hessType = 0
    
class rsqp_solution:
    x_final = None
    f_final = None
    evaluations_FC = 0
    evaluations_GA = 0
    iterations = 0
    termination = 0
    ls_failures = 0

def evalFC_exact (x):
    #constraints in form c(x)<=0
    m=4
    a=1e-4
    c = np.zeros((m,1))
    obj = x[0]+x[1]
    c[0] = a-x[0]*x[0]
    c[1] = a+x[0]*x[0] + x[1]
    c[2] = -x[0]
    c[3] = -x[1]
    #print(c)
    return obj , c

def evalFC (x):
    #constraints in form c(x)<=0
    m=4
    obj , c = evalFC_exact(x)
    if(noise_type=="U"):
        obj = obj + eps_func*np.random.uniform(-1,1)
        #in call_CONDFO we make sure prob.cl=prob.cu
        c = c + eps_func*np.random.uniform(-1,1,(m,1))
    if(noise_type=="N"):
        obj = obj + np.random.normal(0.0,0.5*eps_func)
        #in call_CONDFO we make sure prob.cl=prob.cu
        c = c + np.random.normal(0.0,0.5*eps_func,m)        
    c = np.reshape(c,(m,1))
    #print(c)
    return obj , c

def evalGA (x):
    n=2
    m=4
    gradf=np.zeros((n,1))
    gradf[0]=1
    gradf[1]=1
    jac=np.zeros((m,n))
    jac[0,0]=-2*x[0]
    jac[1,0]= 2*x[0]
    jac[1,1]= 1
    jac[2,0]=-1
    jac[3,1]=-1    
    #jac = np.reshape(jac,(prob.n*prob.m))
    if(noise_type=="U"):
        gradf = gradf + eps_grad*np.random.uniform(-1,1,(n,1))    
        jac = jac + eps_grad*np.random.uniform(-1,1,np.shape(jac)) 
    if(noise_type=="N"):
        gradf = gradf + np.random.normal(0.0,0.5*eps_grad,n)    
        jac = jac + np.random.normal(0.0,0.5*eps_grad,np.shape(jac)) 
    gradf = np.reshape(gradf,(n,1))   
    return gradf, jac

def call_rsqp():        
    #Variables; do not accept bounds on variables
    n = 2
    #for i in range(n): 
    #    if(prob.bl[i]>-infinity or prob.bu[i]<infinity):
    #        return -1
     
    #Constraints; do not accept equality constraints
    m = 4

    #Set initial position
    x0 = np.array([1.0, 1.0])
    
    #Termination parameters
    options = rsqp_options()
    options.maxiter = 1000
    options.noiseLevelObj = eps_func
    options.noiseLevelCons = eps_func
    options.noiseLevelGrad = eps_grad
    options.noiseLevelJac = eps_grad
    options.hessType = 2
 
    #if(len(sys.argv)>2):
    #	fopt = float(sys.argv[2])

    #Set target solution
    xast = None
    xast = np.array([0.0, 0.0]) 
    options.x_target = xast

    bl = np.array([-1e+20, -1e+20])
    bu = np.array([1e+20, 1e+20])
    
    #f = open("rsqp_output_"+prob.name+"_"+noise_type+".txt","a")
    #f.write("prob \t noise_type \t eps_fun_one \t eps_grad_one \t exit \t iter \t finalf \t bestxgap \t LSfail \n")

    #for k in range(1000):
    #    np.random.seed(101+k)        
    t_start = process_time()
    # Solve!
    solution = rsqp.rsqp_infnorm(evalFC, evalGA, n, m, bl, bu, x0, options, None) 
    t_stop = process_time()    

    #f.write("%s \t %s \t %e \t %e \t %d \t %d \t %e \t %e \t %d \n" % (prob.name,noise_type,eps_func,eps_grad,solution.termination, solution.iteration,solution.f_final,solution.best_dist_x_target,solution.ls_failures))
    #f.close()

    # Print information
    
    print("Termination :", solution.termination) 
    print("Final Objective Value :", solution.f_final)
    print("CPU Time Elapsed (s)  :", (t_stop-t_start))
    print(solution.x_final)
    
    fexact , cexact = evalFC_exact (solution.x_final)
    print("Final Objective Value (exact) :", fexact)
    print("Final Constraint Violation (exact, infnorm) :",np.linalg.norm(np.maximum(cexact,np.zeros(m)), np.inf))

    print("\n")

#MAIN
global prob
global eps_func
global eps_grad
global infinity
global noise_type
warnings.filterwarnings("ignore")
#noise_type = "N" #normal
noise_type = "U" #uniform
#eps_func = 1e-20
#eps_grad = 1e-20
eps_func = 1e-2
eps_grad = 1e-2
infinity = 1e+20
np.random.seed(101)
call_rsqp()


