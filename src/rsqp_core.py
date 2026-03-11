'''
rsqp_core.py
author: Figen Oztoprak, figenoeztoprak@gmail.com
last updated: 9.Mar.2026
'''

from dataclasses import dataclass
import numpy as np
import scipy.optimize as optimize
from qpsolvers import solve_qp

#Definitions
RSQP_INFINITY = 1e+20
RSQP_ZERO = 2e-16

@dataclass
class rsqp_options:
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

@dataclass
class rsqp_solution:
    x_final : np.ndarray = None
    f_final : float = np.inf
    evaluations_FC: int = 0
    evaluations_GA: int = 0
    iterations: int = 0
    termination: int = 0
    feaserr : float = np.inf
    opterr : float = np.inf
    pi : float = 0.0
    nQNskips : int = 0

'''
Add bound constraints on variables to the original set of inequalities
'''
def revise_c_for_bounds (n,m,num_bounds,c,xk,bl,bu) :
    inf = RSQP_INFINITY
    caug = np.zeros((m,1))
    for i in range(m-num_bounds) :
        caug[i] = c[i]
    j=m-num_bounds
    for i in range(n) :
        if(bl[i]>-inf):
            caug[j]=bl[i]-xk[i]
            j = j+1
        if(bu[i]<inf):
            caug[j]=xk[i]-bu[i]
            j = j+1
    return caug
   
'''
Add the gradient of bound constraints to the Jacobian 
'''      
def revise_J_for_bounds (n,m,num_bounds,J,xk,bl,bu) :
    inf = RSQP_INFINITY
    Jaug = np.zeros((m,n))
    Jaug[0:m-num_bounds,:] = J
    j=m-num_bounds
    for i in range(n) :
        if(bl[i]>-inf):
            Jaug[j,i]=-1.0
            j = j+1
        if(bu[i]<inf):
            Jaug[j,i]=1.0 
            j = j+1
    return Jaug        

'''
Compute feasibility error based on the infinity norm
'''   
def get_feaserr_standard_ineq(c):
    feaserr = np.linalg.norm(np.maximum(c,np.zeros(np.shape(c))), np.inf)
    return feaserr

def model_obj(H,q,x):
    return 0.5*x.T@ H @ x+x.T@ q

def model_con(A,b,x):
    return b-A@ x

def get_linear_feasibility_improvement(n,m,Delta,c,J,xk):
    #min r : c + Jd <= r, -Delta<d<Delta, r>=0;
    clp = np.zeros(n+1)
    clp[n] = 1.0
    ublp = Delta*np.ones(n+1)
    ublp[n] = np.inf
    lblp = -Delta*np.ones(n+1)
    lblp[n] = 0.0
    boundslp=[]
    for i in range(n+1):
        boundslp.append((lblp[i], ublp[i]))
    blp = -1*c[:,0]
    Alp = -1*np.ones((m,n+1))
    Alp[:,0:n]=J
    x0lp = np.zeros(n+1)
    x0lp[0:n] = xk[:,0]
    optionslp = {'tol' : 1e-12}
    reslp = optimize.linprog(clp, A_ub=Alp, b_ub=blp, bounds=boundslp, 
                             options=optionslp, x0=x0lp)
    statuslp = reslp.status
    return statuslp, reslp

def quasiNewton_update(n, m_orig, hess, J, J_prev, g, g_prev, xk, x_prev, lk, 
                       lk_orig, epsG, epsJ, QN_skips, verbose):
    J_orig = J[0:m_orig,0:n]
    Jprev_orig = J_prev[0:m_orig,0:n]
    gL = g + J_orig.T@lk_orig
    gL_prev = g_prev + Jprev_orig.T@lk_orig
    yL = gL - gL_prev
    sL = xk - x_prev
    ys = yL.T@sL
    hess_s = hess @ sL
    #if(ys > 1e-7*np.linalg.norm(sL) and 
    #ys > (epsG+np.linalg.norm(lk,np.inf)*epsJ)*np.linalg.norm(sL)):
    skipThresh = 1e-3*sL.T@hess_s
    if(skipThresh < 1e-7*np.linalg.norm(sL)):
        skipThresh = 1e-7*np.linalg.norm(sL)
    if(ys > skipThresh and 
       ys > (epsG+np.linalg.norm(lk,np.inf)*epsJ)*np.linalg.norm(sL)):
        hess = hess + (1/(yL.T@sL))*(yL@yL.T) 
        - (1/(sL.T@hess_s))*(hess_s@hess_s.T)
    else:
        if(verbose):
            if(ys <= skipThresh):
                print("skipping BFGS update due to curvature, yTs=%e" % ys[0,:]);
            else:
                print("skipping BFGS update due to noise");
            QN_skips=QN_skips+1
    return hess, QN_skips

def update_penalty_param(pi, pimax, theta, linredfeas, g, hess_qp, dqp):
    linred = -g.T@ dqp + pi*linredfeas
    pidone=0
    while(pidone==0): 
        quadred = linred - 0.5*(dqp.T@hess_qp)@dqp
        if(pi>pimax):
          pidone = -1
        if(quadred >= theta*linredfeas):
          pidone = 1
        else :
          pi = pi*10
          linred = -g.T@ dqp + pi*linredfeas 
    return pidone, pi

def line_search(n, m, num_bounds, objcons, revise_c_for_bounds, relaxation, xk, 
                dqp, c, f, g, J, pi, feaserr, bl, bu, theta, alphamin):
        alpha=1.0
        lsdone=0
        linredalpha = -alpha*(g.T@ dqp) + pi*(feaserr-get_feaserr_standard_ineq(c+alpha*(J@ dqp)))
        trials = 0
        while(lsdone==0) :
          if(alpha<alphamin):
              lsdone=-1
          xtrial=xk+alpha*dqp
          ftrial, ctrial = objcons(xtrial)
          if(num_bounds>0):
              ctrial = revise_c_for_bounds(n,m,num_bounds,ctrial,xtrial,bl,bu)
          trials = trials + 1
          feastrial = np.linalg.norm(np.maximum(ctrial,np.zeros((m,1))), np.inf)
          if( (ftrial+pi*feastrial) - relaxation < ((f + pi*feaserr) - theta*linredalpha)):
             lsdone=1
          else :
             alpha = alpha*0.5   
             linredalpha = -alpha*(g.T@ dqp) + pi*(feaserr-get_feaserr_standard_ineq(c+alpha*(J@ dqp)))
        return trials, lsdone, alpha, xtrial, ftrial, ctrial
 
    
def print_iterate_info(k, feaserr, f, rlp, pi, alpha, opterr, dqp, relaxation):
    header_format = "{0:<5} {1:<9} {2:<9} {3:<9} {4:<9} {5:<9} {6:<9} {7:<9} {8:<9}"
    row_format = "{0:<5} {1:<.3e} {2:<.3e} {3:<.3e} {4:<.3e} {5:<.3e} {6:<.3e} {7:<.3e} {8:<.3e}" 
    row_format_0 = "{0:<5} {1:<.3e} {2:<.3e} {3:<9} {4:<.3e} {5:<9} {6:<.3e} {7:<9} {8:<9}" 
    if(k==0 or k%10==0):
        print(header_format.format("itr", "feaserr", "obj", "rlp", "pi", "alpha", "opterr", "norm(dqp)", "relax"))
    
    if(k==0):
        print(row_format_0.format(k, feaserr, f, "---------", pi, "---------", opterr, "---------", "---------"))
    else:
        ndqp = np.linalg.norm(dqp)
        print(row_format.format(k, feaserr, f, rlp, pi, alpha, opterr, ndqp, relaxation))

'''
This routine implements the noise tolerant SQP algorithm with relaxations (rSQP)
described in the paper "A Noise Tolerant SQP Algorithm for Inequality Constrained 
Optimization"
'''   
def rsqp_solve(objcons, gradjac, hessL, n, m, bl, bu, x_0, 
               options: rsqp_options()) -> rsqp_solution():
    
    #get options
    maxiter = options.maxiter
    opttol = options.opttol
    feastol = options.feastol
    hessType = options.hessType
    
    #noise level declared by the user
    epsC = options.noiseLevelCons
    epsF = options.noiseLevelObj
    epsG = options.noiseLevelGrad
    epsJ = options.noiseLevelJac
    
    #parameters  
    Delta = 1000.0
    beta = 1.0 #H=beta*I
    theta1 = 0.1 #in the update of pi
    theta2 = 1e-2 #in Armijo line search  
    alphamin = 1e-12 #minimum steplength
    dmin = RSQP_ZERO #zero qp step
    pimax = RSQP_INFINITY
    inf = RSQP_INFINITY
    
    #initialize
    numevalFC = 0
    numevalGA = 0
    xk = np.reshape(x_0,(n,1))
    f, c = objcons(xk)
    numevalFC = numevalFC + 1
    pi = 1.0
    k=0
    hess = beta*np.identity(n)
    QN_skips=0
    g = None
    J = None
    dqp = None
    rlp = None
  
    epsC = options.noiseLevelCons
    epsF = options.noiseLevelObj
    epsG = options.noiseLevelGrad
    epsJ = options.noiseLevelJac
    
    x_prev = None
    g_prev = None
       
    #bound constraints
    num_bounds=0
    m_orig = m
    for i in range(n):
        if(bl[i]>-inf or bu[i]<inf):
            num_bounds=num_bounds+1
    if(num_bounds>0):
        m = m + num_bounds
        c = revise_c_for_bounds(n,m,num_bounds,c,xk,bl,bu)
    lk = np.zeros((m,1))
    
    #main loop
    while(1):
        #check termination
        if(k>=maxiter):
            termination = 1
            break

        #feasibility error
        feaserr = get_feaserr_standard_ineq(c) 
        
        #get (approximate) derivatives
        if(k>0):
            g_prev = g
            J_prev = J
        g, J = gradjac(xk)
        if(num_bounds>0):
            J = revise_J_for_bounds(n,m,num_bounds,J,xk,bl,bu)
        numevalGA = numevalGA + 1
        
        #identify active constraints and update multiplier estimates
        if(k==0):
            inactivity = c
        else:
            inactivity = c + J@dqp - rlp*np.ones(m)
        actives = []
        for i in range(m):
            if(np.linalg.norm(inactivity[i])<1e-8):
                actives.append(i)
        J_actives = J[actives,:]
        regterm = np.identity(np.size(actives))*1e-10
        lk = np.zeros((m,1))
        lk[actives,:] = np.linalg.solve((J_actives@J_actives.T+regterm),(-J_actives@(g)))

        #optimality error
        opterr = np.linalg.norm(g + J.T@lk, np.inf)
        for i in range(m):
            comperr = np.fabs(lk[i]*c[i])
            opterr = max(opterr,comperr[0])

        if(options.verbose and k==0):
            print_iterate_info(k, feaserr, f, opterr, pi, None, opterr, None, None)
        if(np.min(lk)>(-opttol) and opterr<opttol and feaserr<feastol):
          termination=0
          if(options.verbose):
              print('local stationary point, opterr=%e' % opterr)
          break

        #linear feasibility improvement
        statuslp, reslp = get_linear_feasibility_improvement(n,m,Delta,c,J,xk)
        if (statuslp != 0):
            if(options.verbose):
                print('LP solve failed')
            termination=2
            break
        else :
            rlp = reslp.fun
        
        #compute Hessian
        lk_orig = lk[0:m_orig]
        if (hessType==0):
            hess_qp = hess
        if (hessType==1):
            hess = hessL(xk, lk_orig)
            eigHess, eigV = np.linalg.eig(hess)
            eigmin = np.min(eigHess)         
            if(eigmin>0):
                hess_qp = hess
            else:
                hess_qp = hess + (((1e-4)-eigmin)*np.identity(n))
        if (hessType==2):
            if(k>0):
                hess, QN_skips = quasiNewton_update(n, m_orig, hess, J, J_prev, g, g_prev, 
                                                xk, x_prev, lk, lk_orig, epsG, 
                                                epsJ, QN_skips, options.verbose)
            hess_qp = hess
        
        #direction computation
        qqp = np.zeros(n)
        qqp = g[:,0]
        ubqp = np.inf*np.ones(n)
        lbqp = -np.inf*np.ones(n)
        Aqp = J
        bqp = -1*c[:,0] + rlp*np.ones(m)
        Hqp = beta*np.eye(n)
        Hqp[0:n,0:n] = hess_qp
        qp_failed = False
        if(options.qp_solver=="scipy"):
            import functools
            obj_qp = functools.partial(model_obj,Hqp,qqp)
            con_qp = functools.partial(model_con,Aqp,bqp)
            res_qp = optimize.fmin_slsqp(x0=xk, func=obj_qp, f_ieqcons=con_qp,
                                         bounds=list(zip(lbqp,ubqp)), iter=500,
                                         iprint=0, full_output=True, acc=1e-8)
            if(res_qp[3]!=0):
                qp_failed=True
            else:
                dqp = res_qp[0]
        else: 
            resqp = solve_qp(Hqp,qqp,G=Aqp,h=bqp,A=None,b=None,lb=lbqp,ub=ubqp,
                             solver="quadprog")
            if (resqp is None):
                qp_failed = True
            else:
                dqp = resqp[0:n]
        if(qp_failed):
            termination=2
            if(options.verbose):
                print('QP solve failed')
            break
        
        dqp = np.reshape(dqp, (n,1))
        if(np.linalg.norm(dqp)<dmin):
          termination=0
          if(options.verbose):
              print('local stationary point, |dqp|=%e' % np.linalg.norm(dqp))
          break
      
        #check feasibility improvement
        linredfeas = feaserr-get_feaserr_standard_ineq((c+J@ dqp))
        if(linredfeas<0 and linredfeas>=-(rlp+feastol)):
           linredfeas = 0.0
        elif (linredfeas<-(rlp+feastol)):
           termination=4
           if(options.verbose):
               print("linredfeas=%e < 0" % linredfeas)
           break
       
        #penalty parameter (for the merit function)
        pidone, pi = update_penalty_param(pi, pimax, theta1, linredfeas, 
                                          g, hess_qp, dqp)
        if(pidone<0):
           termination=3
           if(options.verbose):
               print("penalty parameter > %e" % pimax)
           break
       
        #line search
        relaxation = 2*epsF + pi*2*epsC
        lstrials, lsdone, alpha, xtrial, ftrial, ctrial = line_search(n, m, 
                                            num_bounds, objcons, 
                                            revise_c_for_bounds, relaxation, xk, 
                                            dqp, c, f, g, J, pi, feaserr, 
                                            bl, bu, theta2, alphamin)
        numevalFC = numevalFC + lstrials
        if(lsdone<0):
            termination=3
            if(options.verbose):
                print('line search failed')
            break
        else: #accept the trial iterate
            x_prev = xk
            xk = xtrial
            f = ftrial 
            c = ctrial
        
        #increase iteration counter
        k = k+1

        #print information
        if(options.verbose):
            print_iterate_info(k, feaserr, f, rlp, pi, alpha, opterr, dqp, relaxation)
    
    #end while

    solution = rsqp_solution()
    solution.x_final = np.reshape(xk, (n,1))
    solution.f_final = f
    solution.evaluations_FC = numevalFC
    solution.evaluations_GA = numevalGA
    solution.iterations = k
    solution.termination = termination
    solution.feaserr = feaserr
    solution.opterr = opterr
    solution.pi = pi
    solution.nQNskips = QN_skips
    return solution

    #end rsqp_core       
