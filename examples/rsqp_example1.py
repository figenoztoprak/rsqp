import rsqp_noiseaware as rsqp
import numpy as np
from time import process_time
import warnings

INFTY = 1e+20

def evalFC_exact (x):
    #constraints in form c(x)<=0
    c = np.zeros((4,1))
    obj = 0.1*x[0]+x[1]
    c[0] = (x[0]-0.5)*(x[0]-0.5)+(x[1]-0.5)*(x[1]-0.5)-1
    #c[1] = x[0]*x[1]-1.0
    c[1] = -x[0]-x[1]+1.0
    c[2] = -x[0]
    c[3] = -x[1]
    return obj[0], c

def evalFC (x):
    #constraints in form c(x)<=0
    m=4
    obj, c = evalFC_exact(x)
    if(noise_type=="U"):
        obj = obj + eps_func*np.random.uniform(-1,1)
        c = c + eps_func*np.random.uniform(-1,1,(m,1))
    if(noise_type=="N"):
        obj = obj + np.random.normal(0.0,0.5*eps_func)
        c = c + np.random.normal(0.0,0.5*eps_func,m)        
    c = np.reshape(c,(m,1))
    return obj, c

def evalGA (x):
    n=2
    m=4
    gradf=np.zeros((n,1))
    gradf[0]=0.1
    gradf[1]=1
    jac=np.zeros((m,n))
    jac[0,1]=2*(x[1]-0.5)
    jac[0,0]=2*(x[0]-0.5)
    jac[1,0]=-1 #x[1]
    jac[1,1]=-1 #x[0]
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
    #Variables
    n = 2
    #Constraints
    m = 4

    #Set initial position
    x0 = np.array([0.1, 0.9])
       
    #Termination parameters
    options = rsqp.rsqp_options()
    options.maxiter = 20
    options.noiseLevelObj = eps_func
    options.noiseLevelCons = eps_func
    options.noiseLevelGrad = eps_grad
    options.noiseLevelJac = eps_grad
    options.hessType = 2
    options.qp_solver = "quadprog"
     
    bl = np.array([-INFTY, -INFTY])
    bu = np.array([INFTY, INFTY])
        
    t_start = process_time()
    # Solve!
    solution = rsqp.rsqp_solve(evalFC, evalGA, None, n, m, bl, bu, x0, options)
    t_stop = process_time()    

   
    # Print information
    fexact , cexact = evalFC_exact (solution.x_final)
    print("Termination : %d" % solution.termination) 
    print("Final Objective Value : %e" % solution.f_final)
    print("CPU Time Elapsed (s)  : %f " % (t_stop-t_start))    
    print("Final Objective Value (exact) : %e" % fexact)
    print("Final Constraint Violation (exact, infnorm) : %e" 
          % np.linalg.norm(np.maximum(cexact,np.zeros(m)), np.inf))
    print("\n")

#MAIN
global prob
global eps_func
global eps_grad
global noise_type
warnings.filterwarnings("ignore")

#set artificial noise level
np.random.seed(1010)
#noise_type = "N" #normal
noise_type = "U" #uniform
eps_func = 1e-16
eps_grad = 1e-8

#solve via rsqp
call_rsqp()



