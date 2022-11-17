import Tag_env_mdp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plot

gama = 0.95
epsilon = 0.0001
action = {0: "move_south", 1: "move_north", 2: "move_east", 3: "move_west", 4: "Tag"}
end_states = []
start_states = []


def legal_state_id(env):
    legal_state = []
    for i in range(env.nS):
        state = env.decode_state(i)
        if env.grid.is_inside(state.agent_pos) & env.grid.is_inside(state.opponent_pos[0]):
            legal_state.append(i)
    return legal_state

def explor_Tr(env, legal_states):
    """
    :param env: the environment by the convention of the open AI
    :return: transition function of the domain, includes : reward, transition by action
    """
    Tr = {}
    for i in legal_states:
        Tr[i] = {}
        for a in range(env.nA):
            Tr[i][a] = {"s'": -1, "r": -100}

            # randering the env for the state
            env.reset()
            env.state = env.decode_state(i)
            # do action a at state s
            obz = env.step(a)
            Tr[i][a]["s'"] = env.encode_state(obz[0])
            Tr[i][a]["r"] = obz[1]
            if obz[2] == True:  # done == end state
                end_states.append(obz[0])
    return Tr


def init_start_states(env, legal_states):
    for s in legal_states:
        tag_state = env.decode_state(s)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        if agent_pos != opp_pos:  # the passenger is not in the taxi
            if s not in end_states:
                start_states.append(s)


def policy_eval(pai, V, Tr, legal_states):
    # value calculation for V
    while True:
        vold = V.copy()
        for s in legal_states:
            if s not in end_states:
                V[s] = Tr[s][pai[s]]["r"] + gama * V[Tr[s][pai[s]]["s'"]]
        if LA.norm(V - vold, np.inf) <= epsilon:
            break


def policy_improve(pai, V, Tr, legal_states):
    for s in legal_states:
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



def simulate(env, optimal_pai):
    sum_of_reward = 0
    steps = 0
    s = env.reset()
    done = False
    counter = 1
    while not done:
        tag_state = env.decode(s)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        pai_action = optimal_pai[s]
        s, r, done, x = env.step(pai_action)
        agent_position = str(agent_pos.x) + "," + str(agent_pos.y)
        opp_position = str(opp_pos.x) + "," + str(opp_pos.y)
        print_r = r
        if print_r > 0:
            print_r = "+" + str(print_r)
        print(str(counter) + ".", agent_position, opp_position, action[pai_action], print_r)
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
        obz = env.step(optimal_pai[env.s])
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
    legal_states = legal_state_id(env)
    Tr = explor_Tr(env, legal_states)
    init_start_states(env, legal_states)
    pai = np.asarray([env.action_space.sample() for i in range(env.nS)])
    V = np.zeros(env.nS)
    env.reset()
    dic_R_v0_mean = []

    # Policy Iteration until converges
    while True:
        policy_eval(pai, V, Tr, legal_states)
        pai_tag = policy_improve(np.array(pai, copy=True), V, Tr, legal_states)
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

