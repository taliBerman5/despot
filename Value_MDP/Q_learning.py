import Tag_env_mdp
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plot

gamma = 0.95
epsilon = 1
steps_amount = 1000000
steps_per_episode = 250
num_of_simulates = 100


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



def init_start_states(env, legal_states):
    for s in legal_states:
        tag_state = env.decode_state(s)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        if agent_pos != opp_pos:  # the passenger is not in the taxi
            if s not in end_states:
                start_states.append(s)


def epsilon_greedy(env, Q_table, state, eps):
    if np.random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state, :])


def Q_learning(env, eps, alpha):
    i = 0
    policy_value = []
    policy_steps_value = []
    Q_table = np.zeros((env.observation_space.n, env.action_space.n))

    while i <= steps_amount:
        state = env.reset()
        for j in range(steps_per_episode):
            action = epsilon_greedy(env, Q_table, state, eps)
            s_tag, r, done, prob = env.step(action)
            Q_table[state, action] += alpha *(r + (gamma * np.max(Q_table[s_tag, :])) - Q_table[state, action])

            i += 1

            if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
                policy_steps_value.append(i)
                policy_value.append(eval_policy(env, Q_table))

            if done:
                if r == 1:  # decay epsilon only if reached to the goal
                     eps = max(0.01, eps * 0.99)
                break
            state = s_tag

    return Q_table, policy_value, policy_steps_value

def eval_policy(env, Q_table):
    total_rewards = 0
    sum_success = 0
    for i in range(num_of_simulates):
        state = env.reset()
        path_len = 0
        r = 0
        discount_factor = []
        rewards = []
        curr_reward = 0
        for j in range(steps_per_episode):
            action = np.argmax(Q_table[state, :])
            state, r, done, prob = env.step(action)

            discount_factor.append(gamma ** path_len)
            rewards.append(r)
            path_len += 1
            if done:
                break
        for ri in range(path_len):
            curr_reward += discount_factor[ri] * rewards[ri]
        total_rewards += curr_reward

        sum_success += r

    return total_rewards / num_of_simulates



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

