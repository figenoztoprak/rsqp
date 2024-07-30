import highspy
import numpy as np
import scipy.linalg as linalg
from time import process_time
import sys
import warnings
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from qpsolvers import solve_qp
#from qpsolvers import problem, solve_problem
import qpsolvers

class rsqp_solution:
    x_final = None
    f_final = None
    evaluations_FC = 0
    evaluations_GA = 0
    iterations = 0
    termination = 0
    best_dist_x_target = None
    feaserr = None
    opterr = None
    pi = None
    nQNskips = 0


def revise_c_for_bounds (n,m,num_bounds,c,xk,bl,bu) :
    inf = 1e+20
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
         
def revise_J_for_bounds (n,m,num_bounds,J,xk,bl,bu) :
    inf = 1e+20
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

def get_feaserr_standard_ineq (c):
    feaserr = np.linalg.norm(np.maximum(c,np.zeros(np.shape(c))), np.inf)
    return feaserr

def rsqp_infnorm(objcons, gradjac, n, m, bl, bu, x_0, options, hessL):

    warnings.filterwarnings("ignore")
    
    #get options
    maxiter = options.maxiter
    opttol = options.tol
    feastol = options.tol
    x_true = options.xOpt 
    if not(options.xOpt is None):
        x_true = np.reshape(options.xOpt,(n,1))
    hessType = options.hessType
      
    #parameters  
    Delta_max = 1e+4
    Delta_min = 1e-4
    Delta = 1000.0
    beta = 1.0 #H=beta*I
    theta1 = 0.1 #in the update of pi
    theta2 = 1e-2 #in Armijo line search  
    alphamin = 1e-12 #minimum steplength
    #dmin = opttol #zero qp step
    dmin = 0.0
    pimax = 1e+20
    inf = 1e+20

    #initialize
    numevalFC = 0
    numevalGA = 0
    xk = np.reshape(x_0,(n,1))
    #lk = np.zeros((m,1))
    f, c = objcons(xk)
    numevalFC = numevalFC + 1
    pi = 1.0
    k=0
    hess = beta*np.identity(n)
    QN_skips=0
  
    epsC = options.noiseLevelCons
    epsF = options.noiseLevelObj
    epsG = options.noiseLevelGrad
    epsJ = options.noiseLevelJac

    #bound constraints
    num_bounds=0
    m_orig = m
    for i in range(n):
        if(bl[i]>-inf):
            num_bounds=num_bounds+1
        if(bu[i]<inf):
            num_bounds=num_bounds+1
    if(num_bounds>0):
        m = m + num_bounds
        c = revise_c_for_bounds(n,m,num_bounds,c,xk,bl,bu)
    print("num_bounds=%d" % num_bounds)    
    lk = np.zeros((m,1))
    
    feaserr_list = []
    obj_list = []

    #main loop
    while(1):
        #check termination
        if(k>=maxiter):
            termination = 1
            break

        #get (approximate) derivatives
        if(k>0):
            g_prev = g
            J_prev = J
        g, J = gradjac(xk)
        if(num_bounds>0):
            J = revise_J_for_bounds(n,m,num_bounds,J,xk,bl,bu)
        numevalGA = numevalGA + 1

        feaserr = get_feaserr_standard_ineq(c) 

        #linear feasibility improvement
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
        #print(boundslp)
        blp = -1*c[:,0]
        #print(blp)
        #Alp = [J, -ones(m,1)];
        Alp = -1*np.ones((m,n+1))
        Alp[:,0:n]=J
        #print(Alp)
        x0lp = np.zeros(n+1)
        x0lp[0:n] = xk[:,0]
        #optionslp=None
        optionslp = {'tol' : 1e-12}
        #optionslp = {'method' : 'highs-ds'}
        #highs_options.primal_feasibility_tolerance=1e-10
        reslp = optimize.linprog(clp, A_ub=Alp, b_ub=blp, bounds=boundslp, options=optionslp, x0=x0lp)
        statuslp = reslp.status
        if (statuslp != 0):
            print('LP solve failed')
            termination=2
            break
        else :
            rlp = reslp.fun
            #print('Lp solve successful, r_lp= %e' % rlp)
   
        #print(reslp.x)
        #print(reslp.slack)
   
        #direction computation
        qqp = g[:,0]
        ubqp = np.inf*np.ones(n)
        #ubqp[n] = rlp
        lbqp = -np.inf*np.ones(n)
        #lbqp[n] = 0.0 #-rlp
        Aqp = J
        bqp = -1*c[:,0]+rlp*np.ones(m)
        #print(qqp)
        #print(lbqp)
        #print(ubqp)
        #Hqp = np.zeros((n+1,n+1))
        #Hqp[0:n,0:n] = beta*np.eye(n)
        Hqp = beta*np.eye(n)
        lk_orig = lk[0:m_orig]
        if (hessType==0):
            hess_qp = hess
            Hqp[0:n,0:n] = hess_qp
        if (hessType==1):
            hess = hessL(xk, lk_orig)
            #print("hess")
            #print(hess)
            eigHess, eigV = np.linalg.eig(hess)
            #print(eigHess)
            eigmin = np.min(eigHess)
            #print("mineigHess=%e"%eigmin)            
            if(eigmin>0):
                hess_qp = hess
            else:
                hess_qp = hess + (((1e-4)-eigmin)*np.identity(n))
            Hqp[0:n,0:n] = hess_qp
            #print(hess_qp)
        if (hessType==2):
            if(k>0):
                J_orig = J[0:m_orig,0:n]
                Jprev_orig = J_prev[0:m_orig,0:n]
                gL = g + J_orig.T@lk_orig
                gL_prev = g_prev + Jprev_orig.T@lk_orig
                yL = gL - gL_prev
                sL = xk - x_prev
                ys = yL.T@sL
                hess_s = hess @ sL
                #if(ys > 1e-7*np.linalg.norm(sL) and ys > (epsG+np.linalg.norm(lk,np.inf)*epsJ)*np.linalg.norm(sL)):
                skipThresh = 1e-3*sL.T@hess_s
                if(skipThresh < 1e-7*np.linalg.norm(sL)):
                    skipThresh = 1e-7*np.linalg.norm(sL)
                if(ys > skipThresh and ys > (epsG+np.linalg.norm(lk,np.inf)*epsJ)*np.linalg.norm(sL)):
                    hess = hess + (1/(yL.T@sL))*(yL@yL.T) - (1/(sL.T@hess_s))*(hess_s@hess_s.T)
                else:
                    if(ys <= skipThresh):
                        print("skipping BFGS update due to curvature, yTs=%e" % ys[0,:]);
                    else:
                        print("skipping BFGS update due to noise");
                    QN_skips=QN_skips+1
            hess_qp = hess
            Hqp[0:n,0:n] = hess_qp    
            #eigHess, eigV = np.linalg.eig(hess)
            #eigmin = np.min(eigHess)
            #print("mineigHess=%e"%eigmin) 
            #print(hess)           
        #print(Hqp)
        lk = np.zeros((m,1))
        #resqp = solve_qp(Hqp,qqp,G=Aqp,h=bqp,A=None,b=None,lb=lbqp,ub=ubqp,solver="highs")
        resqp = solve_qp(Hqp,qqp,G=Aqp,h=bqp,A=None,b=None,lb=lbqp,ub=ubqp,solver="quadprog")
        #resqp = solve_qp(Hqp,qqp,G=Aqp,h=bqp,A=None,b=None,lb=lbqp,ub=ubqp,solver="cvxopt",initvals=None)
        #resqp = qpsolvers.solvers.cvxopt_.cvxopt_solve_qp(Hqp,qqp,G=Aqp,h=bqp,A=None,b=None,lb=lbqp,ub=ubqp,initvals=None, verbose=True, maxiters=100)
        #prob = qpsolvers.Problem(Hqp,qqp,G=Alp,h=blp,A=None,b=None,lb=lbqp,ub=ubqp)
        #solution = qpsolvers.solve_problem(prob,solver="quadprog")
        #resqp = solution.x
        
        if (resqp is None):
            termination=2
            print('QP solve failed')
            break
        #else :
            #print('QP solve succesful')
        dqp = resqp[0:n]
        #print("n=%d"%n)
        #print(dqp)
        #print("rlp=%e, rqp=%e"%(rlp, resqp[n]))
        dqp = np.reshape(dqp, (n,1))
        #print(Alp[0:m,0:n]@dqp)
        #print(J@dqp+c)
        opterr = np.linalg.norm(dqp)
        if(np.linalg.norm(dqp)<dmin) :
          termination=0
          print('local stationary point, |dqp|=%e' % np.linalg.norm(dqp))
          break
        #lk = resqp[(n+1):(n+m_orig)]
        #lk = np.reshape(lk, (m_orig,1))

        #identify active constraints of qp and update multiplier estimates
        #inactivity_qp = c[0:m_orig] + J[0:m_orig,:]@dqp
        inactivity_qp = c + J@dqp
        actives = []
        for i in range(m):
            if(np.linalg.norm(inactivity_qp[i])<1e-8):
                actives.append(i)
        J_actives = J[actives,:]
        regterm = np.identity(np.size(actives))*1e-8
        lk[actives,:] = np.linalg.solve((J_actives@J_actives.T+regterm),(-J_actives@g))
        opterr2 = np.linalg.norm(g + J.T@lk, np.inf)
        #if(np.min(lk)>=0 and opterr2<opttol and feaserr<feastol):
        #  termination=0
        #  print('local stationary point, |opterr2|=%e' % opterr2)
        #  break
            
        #print("inactivity=%e, multiplier=%e" % (inactivity_qp[0],lk[0]))
        #print(np.linalg.lstsq((J_actives.T),(g))) 
        #print("norm(J)=%e, norm(c)=%e, norm(lambda)=%e, norm(H_qp)=%e, eigmin(H_orig)=%e" % (np.linalg.norm(Alp),np.linalg.norm(c),np.linalg.norm(lk),np.linalg.norm(Hqp),eigmin))
            
        #penalty parameter (for the merit function)
        #print(c+J@ dqp)
        linredfeas = feaserr-get_feaserr_standard_ineq((c+J@ dqp))
        if(linredfeas<0 and linredfeas>=-(rlp+feastol)):
           linredfeas = 0.0
        elif (linredfeas<-(rlp+feastol)):
           termination=4
           print("linredfeas=%e < 0" % linredfeas)
           break

        linred = -g.T@ dqp + pi*linredfeas
        #print("g.T@ dqp = %e\n" % (g.T@ dqp));
        pidone=0
        while(pidone==0): 
          quadred = linred - 0.5*(dqp.T@hess_qp)@dqp
          if(pi>pimax):
            pidone = -1
          if(quadred >= theta1*linredfeas):
            pidone = 1
          else :
            pi = pi*10
            linred = -g.T@ dqp + pi*linredfeas 
        #print("pi=%e\t feaserr=%e \t |dqp|=%e \t linredfeas=%e \t linred=%e\t quadred=%e" % (pi, feaserr, np.linalg.norm(dqp), linredfeas, linred, quadred));
        if(pidone<0):
           termination=3
           print("penalty parameter > %e" % pimax)
           break
    
        #line search  
        #print(c)  
        alpha=1.0
        lsdone=0
        relaxation = 2*epsF + pi*2*epsC
        linredalpha = -alpha*(g.T@ dqp) + pi*(feaserr-get_feaserr_standard_ineq(c+alpha*(J@ dqp)))
        while(lsdone==0) :
          if(alpha<alphamin):
              termination=3
              print('line search failed')
              lsdone=-1
          xtrial=xk+alpha*dqp
          ftrial, ctrial = objcons(xtrial)
          if(num_bounds>0):
              ctrial = revise_c_for_bounds(n,m,num_bounds,ctrial,xtrial,bl,bu)
          #print(ctrial)
          #print("alpha=%e, ftrial=%e, |x-xtrial|=%e, feastrial=%e" % (alpha, ftrial, np.linalg.norm(xtrial-xk), np.linalg.norm(np.maximum(ctrial,np.zeros((m,1))), np.inf)))
          numevalFC = numevalFC + 1
          feastrial = np.linalg.norm(np.maximum(ctrial,np.zeros((m,1))), np.inf)
          if( (ftrial+pi*feastrial) - relaxation < ((f + pi*feaserr) - theta2*linredalpha)):
             lsdone=1
             #print("alpha=%e" % alpha)
          else :
             alpha = alpha*0.5   
             linredalpha = -alpha*(g.T@ dqp) + pi*(feaserr-get_feaserr_standard_ineq(c+alpha*(J@ dqp)))
             #print("alpha=%e, linreadalpha=%e , feastrial=%e, ftrial= %e, f=%e, feas=%e" % (alpha,linredalpha,feastrial,ftrial,f,feaserr))    
        if(lsdone<0):
            break

        #PRINT ITERATION INFORMATION
        if(k==0 or k%10==0):
            print("itr     feaserr        obj \t     rlp          pi \t       alpha        Delta        norm(dqp)    relax    \t opterr2")
        print("%3d     %3.5e    %3.5e   %3.3e    %3.3e    %3.3e    %3.3e    %3.3e    %3.3e    %3.3e" %
                                   (k, feaserr, f, rlp, pi, alpha, Delta, np.linalg.norm(dqp), relaxation, opterr2))
        feaserr_list.append(feaserr)
        obj_list.append(f)

        #update Delta ; TODO : change to quadred
        ared = (ftrial+pi*feastrial)-(f + pi*feaserr)
        pred = linredalpha - alpha*alpha*0.5*(dqp.T@hess_qp)@dqp
        #if(ared<0.25*pred) :
        #    Delta = 0.5*alpha*np.linalg.norm(dqp, np.inf)
        #elif(ared>0.75*pred) :
        #    Delta = 2*alpha*np.linalg.norm(dqp, np.inf)
        #else :
        #    Delta = Delta = np.linalg.norm(alpha*dqp, np.inf)
        #Delta = np.minimum(Delta,Delta_max)
        #Delta = np.maximum(Delta,Delta_min)
        
        x_prev = xk
        xk = xtrial
        f = ftrial 
        c = ctrial
        #print(c)
        
        #record the solution with the best feasibility value in the last a hundred iterations
        if(maxiter>=100 and k==maxiter-100):
            bestfeaserr = feaserr
            bestf = f
            bestx = xk
            bestopterr = opterr
        if(maxiter>=100 and k>maxiter-100 and feaserr<bestfeaserr):
            bestfeaserr = feaserr
            bestf = f
            bestx = xk
            bestopterr = opterr
        if(maxiter>=100 and k==maxiter-1):
            feaserr = bestfeaserr
            f = bestf
            xk = bestx
            opterr = bestopterr    
        k = k+1

    markediter = maxiter
    for i in range(len(feaserr_list)):
        if ((feaserr_list[i] < feaserr+2*epsF) and (obj_list[i] < f+2*epsF)):
                markediter = i
                print("markediter=%d,feaserr=%e,feaserr_list[i]=%e,f=%e,obj_list[i]=%e\n" % (markediter,feaserr,feaserr_list[i],f,obj_list[i]))
                break
    maxfeaserr = -1e+20
    minfeaserr = 1e+20
    maxobj = -1e+20
    minobj = 1e+20
 
    for i in range(markediter,len(feaserr_list)):
        if(feaserr_list[i]>maxfeaserr):
            maxfeaserr = feaserr_list[i]
        if(feaserr_list[i]<minfeaserr):
            minfeaserr = feaserr_list[i]
        if(obj_list[i]>maxobj):
            maxobj = obj_list[i]
        if(obj_list[i]<minobj):
            minobj = obj_list[i]
    print("range_feaserr = %e, range_obj=%e\n" % (maxfeaserr-minfeaserr, maxobj-minobj))
    dosya = open("rsqp_output_extra_2epsf_"+str(epsF)+".txt","a")
    dosya.write("%d \t %e \t %e \n" % (markediter,(maxfeaserr-minfeaserr),(maxobj-minobj)))
    dosya.close()

    solution = rsqp_solution()
    solution.x_final = np.reshape(xk, (n,1))
    solution.f_final = f
    solution.evaluations_FC = numevalFC
    solution.evaluations_GA = numevalGA
    solution.iteration = k
    #solution.best_dist_x_target = best_dist_x_target
    solution.termination = termination
    solution.feaserr = feaserr
    solution.opterr = opterr
    solution.pi = pi
    solution.nQNskips = QN_skips
    return solution
#end CONDFO

