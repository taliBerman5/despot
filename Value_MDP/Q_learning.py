import random
import json
import numpy as np
import matplotlib.pyplot as plt
import Tag_env_mdp


np.random.seed(1)
random.seed(1)

gamma = 0.95
epsilon = 1
steps_amount = 500000
steps_per_episode = 90
num_of_simulates = 100

action_map = {0: "up", 1: "right", 2: "down", 3: "left", 4: "Tag"}



def epsilon_greedy(env, Q_table, state, eps):
    if np.random.uniform(0, 1) < eps:
        return env.action_space.sample()
    else:
        return np.argmax(Q_table[state, :])


def Q_learning(env, eps, alpha):
    i = 0
    policy_value = []
    policy_steps_value = []
    Q_table = np.zeros((env.nS, env.nA))

    while i <= steps_amount:
        tag_state = env.reset()
        state = env.encode_state(tag_state)
        for j in range(steps_per_episode):
            action = epsilon_greedy(env, Q_table, state, eps)

            tag_s_tag, r, done = env.step(action)
            s_tag = env.encode_state(tag_s_tag)
            Q_table[state, action] += alpha * (r + (gamma * np.max(Q_table[s_tag, :])) - Q_table[state, action])

            i += 1

            if i == 1000 or i == 3000 or i == 5000 or i % 10000 == 0:
                print(i)
                policy_steps_value.append(i)
                policy_value.append(eval_policy(Q_table))

            if done:
                if r == 10:  # decay epsilon only if reached to the goal
                     eps = max(0.01, eps * 0.99)
                break
            state = s_tag

    return Q_table, policy_value, policy_steps_value





def eval_policy(Q_table):
    env = Tag_env_mdp.TagEnv()
    total_rewards = 0
    sum_success = 0
    for i in range(num_of_simulates):
        tag_state = env.reset()
        state = env.encode_state(tag_state)
        path_len = 0
        r = 0
        discount_factor = []
        rewards = []
        curr_reward = 0
        for j in range(steps_per_episode):
            action = np.argmax(Q_table[state, :])
            tag_state, r, done = env.step(action)
            state = env.encode_state(tag_state)

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


def simulate(env, Q_table):
    sum_of_reward = 0
    steps = 0
    tag_state = env.reset()
    done = False
    counter = 1
    while not done:
        s = env.encode_state(tag_state)
        agent_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos
        agent_position = str(agent_pos.x) + "," + str(agent_pos.y)
        opp_position = str(opp_pos.x) + "," + str(opp_pos.y)
        action = np.argmax(Q_table[s, :])
        state, r, done = env.step(action)
        sum_of_reward += r
        r = int(r)
        if r > 0:
            r = "+" + str(r)
        print(str(counter) + ". state id", s, agent_position, opp_position, action_map[action], r)
        steps += 1
        counter += 1

    print("total steps:", steps)
    if sum_of_reward > 0:
        sum_of_reward = "+" + str(sum_of_reward)
    print("total rewards:", sum_of_reward, "\n")


def simulateRender(env, Q_table):
    tag_state = env.reset()
    s_idx = env.encode_state(tag_state)
    env.render()
    done = False
    while not done:
        tag_state, r, done = env.step(np.argmax(Q_table[s_idx, :]))
        s_idx = env.encode_state(tag_state)
        env.render()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # initialization
    env = Tag_env_mdp.TagEnv()
    env.render()
    alpha_arr = [0.03, 0.05, 0.03, 0.1]
    lamda_arr = [0.2, 0.1, 0.05, 0.1]
    Q_table_saver = []
    policy_value_saver = []


    Q_table, policy_value, policy_steps_value = Q_learning(env, epsilon, alpha_arr[0])

    plt.figure()
    plt.plot(policy_steps_value, policy_value, label='q-learning')
    plt.legend()
    plt.title("Average Policy value of alpha:"+str(alpha_arr[0])+ "lambda:" + str(lamda_arr[0]))
    plt.xlabel("steps")
    plt.ylabel("policy value")
    plt.show()

    V_table = {s: np.max(Q_table[s]) for s in range(env.nS)}

    with open('V_from_Q_learning_Tag.txt', 'w') as convert_file:
        convert_file.write(json.dumps(V_table))

    for i in range(10):
        simulate(env, Q_table)


