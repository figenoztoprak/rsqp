import pycutest
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

def evalFC (x):
    #constraints in form c(x)<=0
    y = np.reshape(x,prob.n)
    obj , c = prob.objcons(y)
    for i in range(prob.m) : 
        if(prob.cl[i]>-infinity): #>= constraint
            #print("ci=%e, cl=%e" % (c[i],prob.cl[i]))
            c[i] = prob.cl[i]-c[i]
        elif (prob.cu[i]<infinity): #<= constraint
            #print("ci=%e, cu=%e" % (c[i],prob.cu[i]))
            c[i] = c[i]-prob.cu[i]
        else :
            print('POTENTIAL BUG : constraint with no bounds!!')
    if(noise_type=="U"):
        obj = obj + eps_func*np.random.uniform(-1,1)
        #in call_CONDFO we make sure prob.cl=prob.cu
        c = c + eps_func*np.random.uniform(-1,1,prob.m)
    if(noise_type=="N"):
        obj = obj + np.random.normal(0.0,0.5*eps_func)
        #in call_CONDFO we make sure prob.cl=prob.cu
        c = c + np.random.normal(0.0,0.5*eps_func,prob.m)        
    c = np.reshape(c,(prob.m,1))
    #print(c)
    return obj , c

def evalFC_exact (x):
    #constraints in form c(x)<=0
    y = np.reshape(x,prob.n)
    obj , c = prob.objcons(y)
    for i in range(prob.m) : 
        if(prob.cl[i]>-infinity): #>= constraint
            #print("ci=%e, cl=%e" % (c[i],prob.cl[i]))
            c[i] = prob.cl[i]-c[i]
        elif (prob.cu[i]<infinity): #<= constraint
            #print("ci=%e, cu=%e" % (c[i],prob.cu[i]))
            c[i] = c[i]-prob.cu[i]
        else :
            print('POTENTIAL BUG : constraint with no bounds!!')
    c = np.reshape(c,(prob.m,1))
    #print(c)
    return obj , c

def evalGA (x):
    y = np.reshape(x,prob.n)
    gradf, jac = prob.lagjac(y)
    #jac = np.reshape(jac,(prob.n*prob.m))
    for i in range(prob.m) :
        if(prob.cl[i]>-infinity): #>= constraint
            #print(jac[i,:])
            jac[i,:] = -1*jac[i,:]
            #print(jac)
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
        if(prob.cl[i]>-infinity): #>= constraint
            v[i]=-1*v[i]
    hess = prob.hess(y, v=v) 
    return hess

def call_rsqp(prob):        
    #Variables; do not accept bounds on variables
    n = prob.n
    #for i in range(n): 
    #    if(prob.bl[i]>-infinity or prob.bu[i]<infinity):
    #        return -1
     
    #Constraints; do not accept equality constraints
    m = prob.m
    for i in range(m):
        if(prob.is_eq_cons[i]):
           print("EXIT : Equality constraints exist")
           return -1
    for i in range(m):
        if(prob.cl[i]>-infinity and prob.cu[i]<infinity):
           print("EXIT : Range constraints exist")
           return -1

    #Set initial position
    x0 = prob.x0
    
    #Termination parameters
    options = rsqp_options()
    options.maxiter = 1000
    options.noiseLevelObj = eps_func
    options.noiseLevelCons = eps_func
    options.noiseLevelGrad = eps_grad
    options.noiseLevelJac = eps_grad
    options.hessType = 2

    if(len(sys.argv)>2):
        fopt_true = float(sys.argv[2])
        #fstopval = fopt+eps_func*np.maximum(1.0,np.abs(fopt))
        #fstopval = fopt+1e-1*np.maximum(1.0,np.abs(fopt))	
        #options.ftarget = fstopval

    #Set target solution
    xast = None
    if prob.name =="POWELLBS":
        xast = np.array([1.09816327627e-05, 9.10610687252e+00]) #POWELLBS
    if prob.name =="HS7":
        xast = np.array([-1.19107236145e-11, 1.73205080757e+00]) #HS7
    if prob.name =="HS40":
        xast = np.array([7.93700525984e-01, 7.07106781187e-01, 5.29731547180e-01, 8.40896415254e-01])
    if prob.name =="BT11":
        xast = np.array([1.26757595974e+00, 9.65300462289e-01, 3.51043816586e-01, -1.36415761259e-02, -7.32424040259e-01])
    if prob.name =="HS79":
        xast = np.array([1.19112745621e+00, 1.36260316523e+00, 1.47281793186e+00, 1.63501662016e+00, 1.67908143631e+00])
    options.x_target = xast
    
    #f = open("rsqp_output_"+prob.name+"_"+noise_type+".txt","a")
    f = open("rsqp_output_"+noise_type+".txt","a")
    #f.write("prob \t noise_type \t eps_fun_one \t eps_grad_one \t exit \t iter \t finalf \t feaserr \t |d_qp| \t pi \t truef \t truefeaserr \t nQNskips \t true_fopt \t fopt_gap \n")

    #for k in range(1000):
    #    np.random.seed(101+k)        
    # Solve!
    t_start = process_time()
    solution = rsqp.rsqp_infnorm(evalFC, evalGA, n, m, prob.bl, prob.bu, x0, options, evalH) 
    t_stop = process_time()    
    fexact , cexact = evalFC_exact (solution.x_final)

    if(len(sys.argv)>2):
        fopt_gap = np.fabs(fopt_true-fexact)
    else:
        fopt_true = 1e+20
        fopt_gap = 1e+20

    f.write("%s \t %s \t %e \t %e \t %d \t %d \t %e \t %e \t %e \t %e \t %e \t %e \t %d \t %e \t %e\n" % (prob.name,noise_type,eps_func,eps_grad,solution.termination, solution.iteration,solution.f_final,solution.feaserr,solution.opterr,solution.pi, 
fexact,np.linalg.norm(np.maximum(cexact,np.zeros(m)), np.inf), solution.nQNskips, 
fopt_true, fopt_gap))
    f.close()

    # Print information
    
    print("Termination : %d" % solution.termination) 
    print("Final Objective Value : %e" % solution.f_final)
    print("CPU Time Elapsed (s)  : %f " % (t_stop-t_start))
    print(solution.x_final)
    
    print("Final Objective Value (exact) : %e" % fexact)
    print("Final Constraint Violation (exact, infnorm) : %e" % np.linalg.norm(np.maximum(cexact,np.zeros(m)), np.inf))

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
eps_func = 1e-2 #1e-1
eps_grad = 1e-1
#eps_func = 0.0 #1e-1
#eps_grad = 0.0
infinity = 1e+20
np.random.seed(101)
p = sys.argv[1]
pycutest.clear_cache(p)
prob = pycutest.import_problem(p)
#p='READING4'
#prob=pycutest.import_problem(p, sifParams={'N':50})
print(prob)
if(prob.n_fixed>0) :
    print('EXIT : n_fixed>0')
else :
    call_rsqp(prob)
pycutest.clear_cache(p)
