import numpy as np


#Group Member:
#Bin Zhang (5660329599)
#Yihang Chen (6338254416)
#Hanzhi Zhang (4395561906)


# load the hmm-data.txt
def load_data(file_name):
    grids,tower_locations, noisy_distances = [],[],[]
    with open(file_name) as f:

        # read the grid  cell
        f.readline()
        f.readline()
        for i in range(10):
            grids.append([int(j) for j in f.readline().split()])

        #read tower location
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        for i in range(4):
            tower_locations.append([int(j) for j in  f.readline().split()[2:]])

        #read noisy distances
        f.readline()
        f.readline()
        f.readline()
        f.readline()
        for i in range(11):
            noisy_distances.append([float(j) for j in f.readline().split()])

    return grids,tower_locations,noisy_distances

# get the the inital probablity for the robot at each valid cell and its coordinate
def initial_state(grids):
    valid_num = 0
    locations = {}
    states_coordinates = {}

    for row_index,row_value in enumerate(grids):
        for column_index, column_value in enumerate(row_value):
            if column_value ==1:
                states_coordinates[valid_num] = [row_index,column_index]
                # states_coordinates {1: [0, 0], 2: [0, 1], 3: [0, 2] ...]
                valid_num+=1

    initial_prob  = np.ones(valid_num)/valid_num # shape  (87,) [0.01149425 0.01149425 ....] 1/87

    return initial_prob,states_coordinates


# calculate the probality of point1 to point2
def cal_prob(point1, point2, states_coordinates):
    # first : count how many points can p1 to jump
    left = [point1[0] - 1, point1[1]]
    right = [point1[0] + 1, point1[1]]
    top = [point1[0], point1[1] + 1]
    bot = [point1[0], point1[1] - 1]
    valid_cells = list(states_coordinates.values())
    access_points = 0
    if left in valid_cells:
        access_points += 1
    if right in valid_cells:
        access_points += 1
    if top in valid_cells:
        access_points += 1
    if bot in valid_cells:
        access_points += 1

    # print(point1,point2)
    # print("access_points:"+str(access_points))
    # second: determine whether p1 can access to p2
    if point2 == left or point2 == right or point2 == top or point2 == bot:
        return 1 / access_points
    else:
        return 0


# get the matrix recording the probability of xi to xj
def transition_matrix(states_coordinates):
    locations = list(states_coordinates.values())  # [[0, 0], [0, 1], [0, 2], [0, 3]...

    transition_matrix = np.zeros((len(locations), len(locations)))

    for i, point1 in enumerate(locations):
        for j, point2 in enumerate(locations):
            transition_matrix[i][j] = cal_prob(point1, point2, states_coordinates)

    # print(transition_matrix)
    return transition_matrix  ## 87*87

# euclidean_dist to cal 2 points
def euclidean_dist(p1, p2):
    p1_x, p1_y = p1[0], p1[1]
    p2_x, p2_y = p2[0], p2[1]

    return ((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2) ** (1 / 2)


# calculate distance of each point to 4 towers
def distance(states_coordinates, tower_locations):
    locations = list(states_coordinates.values())
    distance_matrix = np.zeros((len(locations), len(tower_locations)))
    for i, point1 in enumerate(locations):
        for j in range(len(tower_locations)):
            distance_matrix[i][j] = euclidean_dist(point1, tower_locations[j])

    return distance_matrix


# calculate the prob of the cell to the tower
def point_to_tower_prob(point_index, tower_locations, states_coordinates):
    prob = 1
    point_loc = states_coordinates[point_index]
    for tower in tower_locations:
        distance_to_tower = euclidean_dist(point_loc, tower)
        min_dis = np.round(0.7 * distance_to_tower, 1)
        max_dis = np.round(1.3 * distance_to_tower, 1)
        range_max_min = max_dis - min_dis
        prob = prob * 1/range_max_min

    return prob


# according to the observation, get the transmission matrix for each step
def transmission_matrix(distance_matrix,noisy_distances):
    transmission_matrix = np.zeros((len(noisy_distances),len(distance_matrix)))

    # get the possibility of each cell according to the noisy measurements
    for i,ei in enumerate(noisy_distances):
        for j,dj in enumerate(distance_matrix):
            min_d = np.round(0.7 * dj, 1)
            max_d = np.round(1.3 * dj, 1)
            result = isPossiblePoint(ei,min_d,max_d)
            if result == 1:
                transmission_matrix[i][j] =  point_to_tower_prob(j,tower_locations,states_coordinates)
            else:
                transmission_matrix[i][j] =  0

    # print(transmission_matrix)
    return transmission_matrix

def isPossiblePoint(noisy_measures,min_d,max_d):
    for i in range(len(noisy_measures)):
        if noisy_measures[i] < min_d[i] or noisy_measures[i] >max_d[i]:
            return 0
    return 1


def viterbi(initial_prob, transition_matrix, transmission_matrix):
    # using the viterbi algo to redue complexcity
    generate_state = np.zeros((87, 1))
    state_recording = []  # record each path state
    for i in range(10):
        if i == 0:
            initial_state = np.array(initial_prob).reshape(-1, 1)  # （87*1）
            generate_state = initial_state

        eliminate_e = generate_state * transmission_matrix[i].reshape(-1, 1)  # split e
        eliminate_x = eliminate_e * transition_matrix  # split x
        generate_state = np.max(eliminate_x, axis=0).reshape(-1, 1)

        state_recording.append(eliminate_x)

    # get the final state prob in each cell
    final_state_prob = generate_state * transmission_matrix[10].reshape(-1, 1)

    return state_recording, final_state_prob  # eliminate_e is the final state of the robot


def get_final_point(state_prob, states_coordinates):
    max_prob = np.max(state_prob)  # 3.39453441870283e-16
    result = np.where(state_prob == max_prob)
    final_points_index = list(result[0])  # [26 33]
    return max_prob, final_points_index  # [26 33]


def trace_back(final_point_index, state_recording):
    combin_prop = []
    path = []
    next_point_index = final_point_index
    state_length = len(state_recording)

    path.append(next_point_index)

    for i in range(state_length)[::-1]:
        last_max_prob = np.max(state_recording[i], axis=0)[next_point_index]
        combin_prop.append(last_max_prob)
        locations = np.where(state_recording == last_max_prob)
        next_point_index = list(locations[2]).index(next_point_index)
        last_point_index = list(locations[1])[next_point_index]

        path.append(last_point_index)

        next_point_index = last_point_index  # update the point for recursive tracking

    # reverse the path
    path = path[::-1]
    return path, combin_prop

def get_path_cor(path, states_coordinates):
    coordinates = []
    for point in path:
        coordinate = states_coordinates[point]
        coordinates.append(coordinate)

    return coordinates

if __name__ == '__main__':
    # read data from hmm-data.txt
    grids,tower_locations,noisy_distances  = load_data('hmm-data.txt')
    # get the initial states probs each valid cell location
    initial_prob, states_coordinates = initial_state(grids)
    # get the transition_matrix for xi to xj
    transition_matrix = transition_matrix(states_coordinates)
    # get the distance_matrix for xi to towers
    distance_matrix = distance(states_coordinates, tower_locations)
    #  get the transmission_matrix for ei to xi
    transmission_matrix = transmission_matrix(distance_matrix, noisy_distances)
    # use viterbi to get the final point cell and each max prob states
    state_recording, final_state_prob = viterbi(initial_prob, transition_matrix, transmission_matrix)
    # get final point locations
    max_prob, final_points_index = get_final_point(final_state_prob, states_coordinates)

    # for each possible final point , trace back its path
    for point in final_points_index: # final_points_index could be multiple ==> here is only one point according to the max prob

        path, combin_prop = trace_back(point, state_recording)
        coordinates = get_path_cor(path, states_coordinates)
        print("Most possible path:")

        for i in coordinates[:-1]:
            print("{}->".format(i),end="")

        print(coordinates[-1])