import math
import numpy as np
import random
from scipy.stats import norm
from scipy.optimize import minimize
from variationalQuantumEigensolver import *
import time

def kernel(theta,thetaprime,len_scale,sigma):
    delta = 1
    for j in range(len(theta)):
        delta = delta*math.exp((-2/len_scale**2)*math.sin(math.pi*(abs(theta[j]-thetaprime[j]))/2)**2)
    return (sigma**2)*delta

def Kernel(theta,len_scale,sigma):
    p = len(theta)
    K = np.zeros((p,p))
    for k in range(p):
        for j in range(p):
            K[k,j] = kernel(theta[k],theta[j],len_scale,sigma)
    return K

def mux(x,xdata,ydata,samples,noise,len_scale,sigma):
    bracket = Kernel(xdata,len_scale,sigma) + noise**2*np.eye(len(xdata))
    kn = np.zeros(len(xdata))
    for i in range(len(xdata)):
        kn[i] = kernel(x,xdata[i],len_scale,sigma)
    kn = np.transpose(kn)
    bracket = np.linalg.inv(bracket)
    mux = np.dot(kn,np.dot(bracket,ydata))
    return mux

def sigmax(x,xdata,ydata,samples,noise,len_scale,sigma):
    bracket = Kernel(xdata,len_scale,sigma) + noise**2*np.eye(len(xdata))
    kn = np.zeros(len(xdata))
    kx = kernel(x,x,len_scale,sigma)
    for i in range(len(xdata)):
        kn[i] = kernel(x,xdata[i],len_scale,sigma)
    knt = np.transpose(kn)
    bracket = np.linalg.inv(bracket)
    sigx = np.dot(knt,np.dot(bracket,kn))
    sigx = math.sqrt(sigx**2)
    return math.sqrt(sigx)

def EI(x,bestguess,xdata,ydata,samples,noise,len_scale,sigma):
    E_min = mux(bestguess,xdata,ydata,samples,noise,len_scale,sigma)
    muy = mux(x,xdata,ydata,samples,noise,len_scale,sigma)
    sigy = sigmax(x,xdata,ydata,samples,noise,len_scale,sigma)
    if sigy != 0:
        t = -1*(muy-E_min)
        E_I = t*norm.cdf(t/sigy) + sigy*norm.pdf(t/sigy)
    else:
        E_I = 0
    return -E_I

def POI(x,bestguess,xdata,ydata,samples,noise,len_scale,sigma):
    E_min = mux(bestguess,xdata,ydata,samples,noise,len_scale,sigma)
    muy = mux(x,xdata,ydata,samples,noise,len_scale,sigma)
    sigy = sigmax(x,xdata,ydata,samples,noise,len_scale,sigma)

    v = (muy-1*noise-E_min)/sigy
    P_I = norm.cdf(v)
    return -P_I

def BayesianOptimization(parameters):
    #print("Bayes called")
    samples = parameters['samples']
    len_scale = parameters['len_grid']
    sigma = parameters['bayesSigma']
    npara = parameters['nParaVQE']
    BITS = parameters['bayesIters']
    xdata = []
    ydata = []
    xbests = []

    # Build parametric circuit and theta_params for cvarVQE
    nqubits = parameters.get('nVQE', npara)
    nlayers = parameters.get('nLayersVQE', 1)
    parametric_circuit, theta_params = create_parametric_circuitVQE(nqubits, nlayers)
    session = None
    backend = None

    for k in range(samples):
        xdata.append([2 * math.pi * random.uniform(0, 1)] * npara)
        ydata.append(cvarVQE(xdata[-1], parameters, session, backend, parametric_circuit, theta_params))
    noisedata = []
    #for j in range(10):
    #    noisedata.append(cvarVQE(xdata[0], parameters, session, backend, parametric_circuit, theta_params))
    #noise = np.std(noisedata)
    noise = 0.1
    timePerIter = []
    tokTotal = time.perf_counter()
    for kb in range(BITS):
        tok = time.perf_counter()
        percent = ((kb + 1) / BITS) * 100
        #print('Percentage complete = ' + str(percent) + '%')
        xi = np.argmin(ydata)
        bestguess = xdata[xi]
        xbests.append(bestguess)
        starter = []
        guesses = []
        for k in range(25 * npara):
            xguess = [2 * math.pi * random.uniform(0, 1) for c in range(npara)]
            guesses.append(xguess)
            mu = mux(xguess, xdata, ydata, samples, noise, len_scale, sigma)
            starter.append(mu)
        sti = np.argmin(starter)
        guess = guesses[sti]
        AF_min = minimize(EI, guess, args=(bestguess, xdata, ydata, samples, noise, len_scale, sigma),
                          method='powell', options={'ftol': 1e-3, 'disp': False, 'maxfev': 10 ** 4, 'return_all': True})
        x_opt = AF_min.allvecs[-1]
        xdata.append(x_opt)
        ydata.append(cvarVQE(x_opt, parameters, session, backend, parametric_circuit, theta_params))
        tik = time.perf_counter()
        timePerIter.append(tik - tok)
        print("Time for iter", kb, ":", tik - tok)
    xi = np.argmin(ydata)
    tikTotal = time.perf_counter()
    timeTaken = tikTotal - tokTotal
    solution = thetaToSolutionVQE(xdata[xi], parameters)
    return solution, xbests, timeTaken, timePerIter
