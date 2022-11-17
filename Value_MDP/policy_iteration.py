import Tag_env_mdp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plot

gama = 0.95
epsilon = 0.0001
action = {0: "move_south", 1: "move_north", 2: "move_east", 3: "move_west", 4: "pickup_passenger",
          5: "drop_off_passenger"}
startLocation = {0: "0,0", 1: "0,4", 2: "4,0", 3: "4,3"}
end_states = []
start_states = []


def explor_Tr(env):
    """
    :param env: the environment by the convention of the open AI
    :return: transition function of the domain, includes : reward, transition by action
    """
    Tr = {}
    for i in range(env.nS):
        Tr[i] = {}
        for a in range(env.nA):
            Tr[i][a] = {"s'": -1, "r": -100}

            # randering the env for the state
            env.reset()
            env.env.state = env._encode_state(i)
            # do action a at state s
            obz = env.step(a)
            Tr[i][a]["s'"] = obz[0]
            Tr[i][a]["r"] = obz[1]
            if obz[2] == True:  # done == end state
                end_states.append(obz[0])
    return Tr


def init_start_states(env):
    for s in range(env.env.nS):
        agent_pos, num_opp, opp_pos = env.decode_state(s)
        if agent_pos != opp_pos:  # the passenger is not in the taxi
            if s not in end_states:
                start_states.append(s)


def policy_eval(env, pai, V, Tr):
    # value calculation for V
    while True:
        vold = V.copy()
        for s in range(env.env.nS):
            if s not in end_states:
                V[s] = Tr[s][pai[s]]["r"] + gama * V[Tr[s][pai[s]]["s'"]]
        if LA.norm(V - vold, np.inf) <= epsilon:
            break


def policy_improve(env, pai, V, Tr):
    for s in range(env.env.nS):
        Ma = 0
        Mv = Tr[s][Ma]["r"] + gama * V[Tr[s][Ma]["s'"]]
        for a in range(1, 6):
            v = Tr[s][a]["r"] + gama * V[Tr[s][a]["s'"]]
            if v > Mv:
                Ma = a
                Mv = v
        pai[s] = Ma
    return pai


def plotV(iterV):
    optimal = [iterV[len(iterV) - 1] for i in range(len(iterV))]
    plot.plot(iterV, color="blue")
    plot.plot(optimal, color="green")
    plot.legend(["Policy Iteration Value", "Optimal Policy Value"])
    plot.title("Improvement over Iteration")
    plot.xlabel("kth Iteration")
    plot.ylabel("Value Policy")
    plot.show()


def pass_location(taxi_row, taxi_col, passenger_index):
    if passenger_index == 4:
        return str(taxi_row) + "," + str(taxi_col)
    else:
        return startLocation[passenger_index]


def simulate(env, optimal_pai):
    sum_of_reward = 0
    steps = 0
    s = env.reset()
    done = False
    counter = 1
    while not done:
        taxi_row, taxi_col, passenger_index, destination_index = env.decode(s)
        passenger_location = pass_location(taxi_row, taxi_col, passenger_index)
        pai_action = optimal_pai[s]
        s, r, done, x = env.env.step(pai_action)
        taxi_location = str(taxi_row) + "," + str(taxi_col)
        print_r = r
        if print_r > 0:
            print_r = "+" + str(print_r)
        print(str(counter) + ".", taxi_location, passenger_location, action[pai_action], print_r)
        steps += 1
        sum_of_reward += r
        counter += 1

    print("total steps:", steps)
    if sum_of_reward > 0:
        sum_of_reward = "+" + str(sum_of_reward)
    print("total rewards:", sum_of_reward, "\n")


def simulateRender(env, optimal_pai):
    env.reset()
    env.render()
    done = False
    while not done:
        obz = env.env.step(optimal_pai[env.env.s])
        done = obz[2]
        env.render()


def valueOpt(V, s):
    return V[s]


def print_value_opt(check_state):
    print('\n')
    for i in range(500):
        print(str(check_state[i]), " - ", valueOpt(V, i))
    print('\n')


def v_eval_overPi(env, pai, Tr, V):
    acc = 0
    v = V.copy()
    while True:
        vold = v.copy()
        policy_eval(env, pai, v, Tr)
        if LA.norm(v - vold, np.inf) <= epsilon:
            break

    for s in start_states:
        acc += v[s]
    return acc / len(start_states)


if __name__ == '__main__':
    # initialization
    env = Tag_env_mdp.TagEnv()
    Tr = explor_Tr(env)
    init_start_states(env)
    pai = np.asarray([env.action_space.sample() for i in range(env.env.nS)])
    V = np.zeros(env.env.nS)
    env.reset()
    dic_R_v0_mean = []

    # Policy Iteration until converges
    while True:
        policy_eval(env, pai, V, Tr)
        pai_tag = policy_improve(env, np.array(pai, copy=True), V, Tr)
        if LA.norm(pai - pai_tag, np.inf) == 0:
            break
        pai = pai_tag

        dic_R_v0_mean.append(v_eval_overPi(env, pai, Tr, V))

    plotV(dic_R_v0_mean)
    check_state = [i for i in range(500)]
    simulateRender(env, pai)
    print_value_opt(check_state)

    for i in range(3):
        print("-----------------------------------------------------")
        simulate(env, pai)

