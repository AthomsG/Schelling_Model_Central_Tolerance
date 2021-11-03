# IMPORT LIBRARIES
import numpy as np
import matplotlib as mpl
import random as rd
import matplotlib.pyplot as plt

# DEFINE AUXILIARY FUNCTIONS

# define color map
color_map = {-1: np.array([255, 0, 0]), # red
             0: np.array([255, 255, 255]), # green
             1: np.array([0, 0, 255])} # blue

def make_matrix(N, density, f1):
    matrix = list()

    for i in range(N):
        linha = list()
        for j in range(N):
            linha.append(0)
        matrix.append(linha)

    for linha in range(N):
        for coluna in range(N):
            if (rd.random() <= density):
                if (rd.random() <= f1):
                    matrix[linha][coluna] = 1
                else:
                    matrix[linha][coluna] = -1
    return matrix

offset = 1/2

def get_r (i, j, N): #i - indice linha; j - indice coluna; N - dimensão matriz
    return np.sqrt((i-(N/2 - offset))**2  +   (j-(N/2 - offset))**2)

def get_tol_matrix(N, tmax, tmin): #Returns matrix with tolerance value for each position
    r_matrix = list()
    Rmax     = get_r(0, 0, N)
    for i in range(N):
        linha = list()
        for j in range(N):
            r = get_r(i=i, j=j, N=N)
            #################################
            tol = tmin + r*(tmax - tmin)/Rmax
            #################################

            linha.append(tol)
        r_matrix.append(linha)
    return r_matrix

def print_matrix(matrix, marker=False, save=False, filename=None):
    N = len(matrix)
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(1, 1, 1)
    data_3d = np.ndarray(shape=(N, N, 3), dtype=int)
    for i in range(0, N):
        for j in range(0, N):
            data_3d[i][j] = color_map[matrix[i][j]]

    if(marker):
        x = [coord[1] for coord in marker]
        y = [coord[0] for coord in marker]

        ax.scatter(x, y, color='lime', s=300/N)

    ax.imshow(data_3d)
    ax.set_xticks([])
    ax.set_yticks([])
    if save:
        fig.savefig(filename)
    plt.show()

def get_neighbours(coordinates, N):
    neighbours = list()
    n_range    = [-1, 0, 1]
    [x, y]     = coordinates

    for i in n_range:
        for j in n_range:

            neighbour_coords_x = coordinates[0]+i
            neighbour_coords_y = coordinates[1]+j

            if (neighbour_coords_x > N - 1):
                neighbour_coords_x = 0

            if (neighbour_coords_x < 0):
                neighbour_coords_x = N - 1

            if (neighbour_coords_y > N - 1):
                neighbour_coords_y = 0

            if (neighbour_coords_y < 0):
                neighbour_coords_y = N - 1

            if (not(i == 0 and j == 0)):
                neighbours.append([neighbour_coords_x, neighbour_coords_y])

    return neighbours

def get_agents(matrix, val):
    empty_coords = list()
    for linha in range(len(matrix)):
        for coluna in range(len(matrix)):
            if matrix[linha][coluna] == val:
                empty_coords.append([linha, coluna])
    return empty_coords

def get_empty_space(matrix):
    empty_coords = list()
    for linha in range(len(matrix)):
        for coluna in range(len(matrix)):
            if matrix[linha][coluna] == 0:
                empty_coords.append([linha, coluna])
    return empty_coords

def check_neighbours(coordinates, matrix):
    N = len(matrix)
    neighbours = get_neighbours(coordinates,N)
    content = list()

    content = [matrix[pos[0]][pos[1]] for pos in neighbours]

    return content

#algo return 1 if agent gathers conditions that satisfy dissatisfaction and 0 if it doesn't

# algo(coordinates, matrix)
def get_dissatisfied(matrix, algo, tol_matrix):
    dissatisfied = list()
    for linha in range(len(matrix)):
        for coluna in range(len(matrix)):
            if algo([linha, coluna], matrix, tol_matrix):
                dissatisfied.append([linha, coluna])
    return dissatisfied

def dis_algo(coord, matrix, tol_matrix):

    ############## t comes from the tolerance matrix
    t = tol_matrix[coord[0]][coord[1]]

    agent_val = matrix[coord[0]][coord[1]]
    if (agent_val):
        n_agents  = check_neighbours(coord, matrix)

        n_1 = 0
        n0  = 0
        n1  = 0

        for agent in n_agents:
            if agent == -1:
                n_1 += 1
            if agent == 0:
                n0  += 1
            if agent == 1:
                n1  += 1

        if (n_1 + n1) == 0:
            return 0

        val_sum = {-1:n_1, 0:n0, 1:n1}

        if val_sum[agent_val]/(n_1 + n1) < t:
            return 1
        return 0

def get_happiness(matrix, algo, tol_matrix, diss_list=None):
    N = len(matrix)
    if not (diss_list):
        diss_list = get_dissatisfied(matrix, algo, tol_matrix)
    return (1 - len(diss_list)/N**2)

def iteration(matrix, algo, tol_matrix):
    new_matrix = matrix.copy()

    empty_space  = get_empty_space(matrix)
    dissatisfied = get_dissatisfied(matrix, algo, tol_matrix)

    if dissatisfied == []:
        return "STOP"

    for agent in dissatisfied:
        new_pos = rd.choice(empty_space) #Choosing a random empty space to be occupied by agent

        new_matrix[new_pos[0]][new_pos[1]] =  new_matrix[agent[0]][agent[1]]
        new_matrix[agent[0]][agent[1]] = 0

        empty_space.pop(empty_space.index(new_pos))
        empty_space.append(agent)

    return new_matrix


def run(N, density, f1, n_iter=100, tmin=0.1, tmax=0.9, measure_happiness=False, measure_r_values=False):

    matrix_    = make_matrix(N, density, f1)
    new_matrix = matrix_.copy()

    tol_matrix = get_tol_matrix(N, tmax=tmax, tmin=tmin)

    happiness = list()
    red_rs  = list()
    blu_rs  = list()

    print('', end='r')

    for i in range(n_iter):
        if i > 0:
            new_matrix = iteration(matrix_, dis_algo, tol_matrix)

        if new_matrix == "STOP":
            break

        if measure_happiness:
            happiness.append(get_happiness(matrix_, dis_algo, tol_matrix))
        if measure_r_values:
            red_rs.append([get_r(coord[0], coord[1], N) for coord in get_agents(matrix_, -1)])
            blu_rs.append([get_r(coord[0], coord[1], N) for coord in get_agents(matrix_,  1)])

        matrix = new_matrix.copy()

        print(str(i+1) + '/' + str(n_iter), end='\r')

    print('Finished!', end='\r')
    return {'matrix':matrix_, 'happiness':happiness, 'red_rs':red_rs, 'blu_rs':blu_rs}

def save_data(lista, filename):
    textfile = open(filename, "w")
    for el in lista:
        if (type(el)==list):
            textfile.write(str(el))
            textfile.write('\n')
        else:
            textfile.write(str(el))
        textfile.write('\n')
    textfile.close()
