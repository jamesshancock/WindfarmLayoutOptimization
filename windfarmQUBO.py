import numpy as np
import math
import matplotlib.pyplot as plt

def labelling(len_grid):
    offset = (len_grid - 1) / 2  # Calculate the offset to center the grid around 0
    pos_labels = {c: [k0 - offset, k1 - offset] for c, (k0, k1) in enumerate(((k0, k1) for k0 in range(len_grid) for k1 in range(len_grid)), 1)}
    return pos_labels

def cone_maker(turbine_i,dist,r,d):
    theta = d[0]
    pos = labelling(len_grid)
    for p in pos:
        x = pos[p][0]
        y = pos[p][1]
        x_new = x*math.cos(math.radians(theta)) - y*math.sin(math.radians(theta))
        y_new = x*math.sin(math.radians(theta)) + y*math.cos(math.radians(theta))
        pos[p] = [x_new,y_new]
    ith = pos[turbine_i]
    cone = {}
    for k in range(1,(len_grid**2)+1):
        kth = pos[k]
        if ((kth[1]-ith[1]) <= (dist+0.001)) and (kth[1] - ith[1] != 0):
            if abs(kth[0] - ith[0]) < abs(r*(kth[1]-ith[1])):
                cone[k] = 1
            else:
                cone[k] = 0
        else:
            cone[k] = 0
    cone[turbine_i] = 1
    for k in pos:
        if pos[k][1] - ith[1] < 0:
            cone[k] = 0
    return cone

def jansen(i,j,D,wf_para):
    turbine_radius = 0.33 #turbine radius is 1/3 of the site size
    pos = labelling(len_grid)

    I = pos[i]
    J = pos[j]
    dist = math.sqrt((I[0] - J[0])**2 + (I[1] - J[1])**2)

    free_speed = D[1]
    alpha = (wf_para['r'] - turbine_radius)/wf_para['x']
    C_T = 4*alpha*(1-alpha)

    a = 0.1

    reduced_speed = free_speed*(1-2*a/(1+alpha*(dist/wf_para['r'])**2)**2)
    return reduced_speed

def reduced_jansen_factor(i,j,d,wf_para):
    x = wf_para['x']
    r = wf_para['r']
    cone = cone_maker(i,x,r,d)
    pos = labelling(len_grid)
    if cone[j] == 1:
        RF =  jansen(i,j,d,wf_para)
    else:
        RF = d[1]
    return RF

def cone_matrix(cone):
    M = np.zeros((len_grid,len_grid))
    c = 1
    for k1 in range(len_grid):
        for k0 in range(len_grid):
            if cone[c] == 1:
                M[k0,k1] = 1
            else:
                M[k0,k1] = 0
            c += 1
    return M

def number_constraint(m,len_grid):
    C = np.zeros((len_grid**2,len_grid**2))
    for i in range(len_grid**2):
        for j in range(len_grid**2):
            if i == j:
                C[i,i] = 1 - 4*m
            else:
                C[i,j] = 2
    return C

def proximity_constraint(E, lenGrid):
    '''
    Creates a proximity constraint matrix for the QUBO problem.
    
    Parameters:
    E (float): The proximity threshold.
    lenGrid (int): The number of grid points in one dimension.
    
    Returns:
    C (numpy.ndarray): The proximity constraint matrix.
    '''
    C = np.zeros((lenGrid ** 2, lenGrid ** 2))
    labels = labelling(lenGrid)
    for i in range(lenGrid ** 2):
        for j in range(lenGrid ** 2):
            if i != j:
                x1, x2 = labels[i + 1], labels[j + 1]
                distance = math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2)
                if distance <= E:
                    C[i, j] = 1
    return C

def LocationConstraint(noGos, lenGrid):
    '''
    Creates a location constraint matrix for the QUBO problem.
    
    Parameters:
    noGos (list): A list of forbidden locations.
    lenGrid (int): The number of grid points in one dimension.
    
    Returns:
    C (numpy.ndarray): The location constraint matrix.
    '''
    C = np.zeros((lenGrid ** 2, lenGrid ** 2))
    for i in range(lenGrid ** 2):
        if i + 1 in noGos:
            C[i, i] = 1
    return C

def WindfarmQ(parameters):
    global x
    global r
    D = parameters['D']
    x = parameters['x']
    r = parameters['r']
    m = parameters['m']
    E = parameters['E']
    lam1 = parameters['lam1']
    lam2 = parameters['lam2']
    lam3 = parameters['lam3']
    global len_grid
    len_grid = parameters['len_grid']
    Q = np.zeros((len_grid**2,len_grid**2))
    for d in D:
        for k in range(1,(len_grid**2)+1):
            for j in range(1,(len_grid**2)+1):
                if k !=j:
                    reduced_speed = reduced_jansen_factor(k,j,d,parameters)
                    term = d[1]**3 - reduced_speed**3
                    Q[k-1,j-1] += -1/3*d[2]*term
                else:
                    Q[k-1,j-1] += 1/3*d[2]*(d[1]**3)
    Q = 1/2*(Q+Q.T)
    Q = Q - lam1*number_constraint(m,len_grid) - lam2*proximity_constraint(E,len_grid) - lam3*LocationConstraint(parameters['noGos'], len_grid)
    return -Q
#as we are looking for min not max

def Energy(parameters,solution):
    global x
    global r
    D = parameters['D']
    x = parameters['x']
    r = parameters['r']
    m = parameters['m']
    E = parameters['E']
    m = np.count_nonzero(solution)
    energy = 0
    global len_grid
    len_grid = parameters['len_grid']
    for d in D:
        angle = d[0]
        free_speed = d[1]
        prob = d[2]
        for k in range(1,len(solution)+1):
            pos = labelling(len_grid)
            if solution[k-1] == 1:
                cone = cone_maker(k,x,r,d)
                term = 0
                for c in cone:
                    if cone[c] == 1 and solution[c-1]:
                        if c != k:
                            reduced_speed = reduced_jansen_factor(k,c,d,parameters)
                            term += free_speed**3 - reduced_speed**3
                energy += 1/3*prob*(free_speed**3 - term)
    return energy

def wake_speeds(parameters,turbine_i,d):
    dist = parameters['x']
    r = parameters['r']
    global len_grid
    len_grid = parameters['len_grid']
    Cone = cone_maker(turbine_i,dist,r,d)
    mat = cone_matrix(Cone)
    c = 0
    mat2 = np.zeros((len_grid,len_grid))
    for u in range(len_grid):
        for v in range(len_grid):
            c += 1
            if mat[v,u] == 1:
                if c == turbine_i:
                    mat2[v,u] = d[1]
                else:
                    mat2[v,u] = reduced_jansen_factor(turbine_i,c,d,parameters)
            else:
                mat2[v,u] = d[1]
    return mat2

def solutionToGrid(solution, parameters):
    '''
    Converts a solution to a grid
    
    Parameters:
    solution (list): A list representing the wind farm layout solution
    parameters (dict): A dictionary containing wind farm parameters
    
    Returns:
    solutionGrid (numpy.ndarray): A grid representing the wind farm layout solution
    '''
    counter = 0
    solutionGrid = np.zeros((parameters['len_grid'], parameters['len_grid']))
    for i in range(parameters['len_grid']):
        for j in range(parameters['len_grid']):
            solutionGrid[j, i] = solution[counter]
            counter += 1
    return solutionGrid

def plot_wake_heatmap(solution, parameters):
    '''
    Generates a heatmap of reduced wind speeds due to wakes for a given solution and parameters.
    
    Parameters:
    solution (list): A list representing the wind farm layout solution.
    parameters (dict): A dictionary containing wind farm parameters.
    '''
    lenGrid = parameters['len_grid']
    D = parameters['D']

    # Initialize the heatmap matrix with free wind speeds
    heatmap = np.full((lenGrid, lenGrid), D[0][1])

    # Calculate the reduced wind speeds for each turbine in the solution
    for turbineI in range(1, len(solution) + 1):
        if solution[turbineI - 1] == 1:
            for d in D:
                wake_speed = wake_speeds(parameters, turbineI, d)
                heatmap = np.minimum(heatmap, wake_speed)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, cmap='viridis', origin='lower', extent=[0, lenGrid, 0, lenGrid])
    plt.colorbar(label='Wind Speed (m/s)')
    plt.title('Heatmap of Reduced Wind Speeds Due to Wakes')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')
    plt.show()
