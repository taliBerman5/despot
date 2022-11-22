import Tag_env_mdp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plot

gama = 0.95
epsilon = 0.0001
action = {0: "up", 1: "right", 2: "down", 3: "left", 4: "Tag"}
end_states = []
start_states = []

np.random.seed(1)


# reward are deterministic for state, transition is not
def explor_Tr(env):
    """
    :param env: the environment by the convention of the open AI
    :return: transition function of the domain, includes : reward, transition by action
    """
    Tr = {}
    for s in range(env.nS):
        Tr[s] = {}
        for a in range(env.nA):
            Tr[s][a] = {"s'": -1, "r": -100}

            # randering the env for the state
            env.reset()
            env.state = env.decode_state(s)
            # do action 'a' at state s
            obz = env.step(a)
            Tr[s][a]["s'"] = env.stateTransitionDistribution(s, a)
            Tr[s][a]["r"] = obz[1]
    return Tr


def init_start_states(env):
    for s in range(env.nS):
        tag_state = env.decode_state(s)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        if agent_pos != opp_pos:  # the passenger is not in the taxi
            start_states.append(s)


def policy_eval(env, pai, V, Tr):
    # value calculation for V
    while True:
        Vold = V.copy()
        for s in range(env.nS):
            V[s] = Tr[s][pai[s]]["r"] + gama * V_next(V, Tr, s, pai[s])
        if LA.norm(V - Vold, np.inf) <= epsilon:
            break


def V_next(V, Tr, state, action):
    return sum([V[s_next] * Tr[state][action]["s'"][s_next] for s_next in Tr[state][action]["s'"]])


def policy_improve(env, pai, V, Tr):
    for s in range(env.nS):
        Ma = 0
        Mv = Tr[s][Ma]["r"] + gama * V_next(V, Tr, s, Ma)
        for a in range(1, env.nA):
            v = Tr[s][a]["r"] + gama * V_next(V, Tr, s, a)
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
    tag_state = env.reset()
    done = False
    counter = 1
    while not done:
        s = env.encode_state(tag_state)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        pai_action = optimal_pai[s]
        tag_state, r, done = env.step(pai_action)
        agent_position = str(agent_pos.x) + "," + str(agent_pos.y)
        opp_position = str(opp_pos.x) + "," + str(opp_pos.y)
        print_r = r
        if print_r > 0:
            print_r = "+" + str(print_r)
        print(str(counter) + ". state id", s, agent_position, opp_position, action[pai_action], print_r)
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
        obz = env.step(optimal_pai[env.encode_state(env.state)])
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
    pai = np.asarray([env.action_space.sample() for i in range(env.nS)])
    V = np.zeros(env.nS)
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
    # simulateRender(env, pai)
    print_value_opt(check_state)

    for i in range(3):
        print("-----------------------------------------------------")
        simulate(env, pai)
