from enum import Enum

import numpy as np
from gym import Env
from gym.spaces import Discrete
from coord import Coord, Moves, Grid



def action_to_str(action):
    if action == 0:
        return "up"
    elif action == 1:
        return "right"
    elif action == 2:
        return "down"
    elif action == 3:
        return "left"
    elif action == 4:
        return "tag"
    else:
        raise NotImplementedError()


class Action(Enum):
    NORTH = 0
    EAST = 1  # east
    SOUTH = 2
    WEST = 3  # west
    TAG = 4


class TagGrid(Grid):
    def __init__(self, board_size):
        super().__init__(*board_size)
        # self.build_board(0)
        self.build_board()

    def sample(self):
        return self.get_tag_coord(np.random.randint(0, 29))

    def is_inside(self, coord):
        if coord.y >= 2:
            return coord.x >= 5 and coord.x < 8 and coord.y < 5
        else:
            return coord.x >= 0 and coord.x < 10 and coord.y >= 0

    def get_tag_coord(self, idx):
        assert idx >= 0 and idx < self.n_tiles
        if idx < 20:
            return Coord(idx % 10, idx // 10)
        idx -= 20
        return Coord(idx % 3 + 5, idx // 3 + 2)

    def get_index(self, coord):
        # return self.x_size * coord.y + coord.x
        assert coord.x >= 0 and coord.x < 10
        assert coord.y >= 0 and coord.y < 5
        if coord.y < 2:
            return coord.y * 10 + coord.x
        assert coord.x >= 5 and coord.x < 8
        return 20 + (coord.y - 2) * 3 + coord.x - 5

    def is_corner(self, coord):
        if not self.is_inside(coord):
            return False
        if coord.y < 2:
            return coord.x == 0 or coord.x == 9
        else:
            return coord.y == 4 and (coord.x == 5 or coord.x == 7)

    @property
    def get_available_coord(self):
        return [self.get_tag_coord(idx) for idx in range(self.n_tiles)]


# grid = TagGrid((10, 5), obs_cells=29)


class TagEnv(Env):
    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_opponents=1, move_prob=.8, board_size=(10, 5)):
        self.move_prob = move_prob
        self._reward_range = 10
        self._discount = .95
        self.action_space = Discrete(len(Action))
        self.nA = len(Action)
        self.grid = TagGrid(board_size)
        self.time = 0
        self.done = False
        self.state = self._get_init_state(False)
        self.last_action = 4
        self.nS = 2500

    def reset(self):
        self.done = False
        self.time = 0
        self.last_action = 4
        self.state = self._get_init_state(False)
        return self.state  # get state

    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def step(self, action):  # state
        assert self.action_space.contains(action)  # action are idx of movements
        assert self.done is False
        # assert self.done == False

        reward = 0.
        self.time += 1
        self.last_action = action
        assert self.grid.is_inside(self.state.agent_pos)
        assert self.grid.is_inside(self.state.opponent_pos)

        if action == 4:
            tagged = False
            if self.state.opponent_pos == self.state.agent_pos:  # check if x==x_agent and y==y_agent
                reward = 10.
                self.done = True
            elif self.grid.is_inside(self.state.opponent_pos) and self.state.num_opp > 0:
                self.move_opponent()
            if not tagged:
                reward = -10.

        else:
            reward = -1.
            self.move_opponent()
            next_pos = self.state.agent_pos + Moves.get_coord(action)
            if self.grid.is_inside(next_pos):
                self.state.agent_pos = next_pos

        # p_ob = self._compute_prob(action, self.state, ob)
        return self.state, reward, self.done  # self._encode_state(self.state)}

    def render(self, mode="human", close=False):
        # return
        if close:
            return
        if mode == "human":
            agent_pos = self.grid.get_index(self.state.agent_pos)
            opponent_pos = self.grid.get_index(self.state.opponent_pos)
            msg = "State: " + str(self.encode_state(self.state)) + " Time: " + str(self.time) + " Action: " + action_to_str(
                self.last_action)
            print(f'agent pos {agent_pos}, opponent pos {opponent_pos}, msg {msg}')

    # def _encode_state(self, state):
    #     s = np.zeros(self.num_opponents + 1, dtype=np.int32) - 1
    #     s[0] = self.grid.get_index(state.agent_pos)
    #     for idx, opp in zip(range(1, len(s)), state.opponent_pos):
    #         opp_idx = self.grid.get_index(opp)
    #         s[idx] = opp_idx
    #     return s

    def encode_state(self, state):  #TB
        s = np.zeros(2)
        s[0] = self.grid.get_index(state.agent_pos)
        opp_idx = self.grid.get_index(state.opponent_pos)
        s[1] = opp_idx
        return s[0] * self.grid.n_tiles + s[1]

    def decode_state(self, state_id): #TB
        agent_idx = state_id // self.grid.n_tiles
        opp_idx = state_id % self.grid.n_tiles
        tag_state = TagState(self.grid.get_tag_coord(agent_idx), self.grid.get_tag_coord(opp_idx))
        return tag_state


    # def _decode_state(self, state):
    #     agent_idx = state[0]
    #     tag_state = TagState(self.grid.get_tag_coord(agent_idx))
    #     for opp_idx in state[1:]:
    #         if opp_idx > -1:
    #             tag_state.num_opp += 1
    #         opp_pos = self.grid.get_tag_coord(opp_idx)
    #         tag_state.opponent_pos.append(opp_pos)
    #
    #         # true_pos = [grid.get_index(pos) for pos in env.state.opponent_pos]
    #     # assert np.all(state[1:] == true_pos)
    #     # assert np.all(tag_state.opponent_pos == env.state.opponent_pos)
    #     return tag_state

    def _get_init_state(self, should_encode=False):

        agent_pos = self.grid.sample()
        assert self.grid.is_inside(agent_pos)
        opp_pos = self.grid.sample()
        assert self.grid.is_inside(opp_pos)

        tag_state = TagState(agent_pos, opp_pos)


        return tag_state if not should_encode else self._encode_state(tag_state)

    def _set_state(self, state):
        # self.reset()
        self.done = False
        # self.state = self._decode_state(state)
        self.state = state

    def move_opponent(self):
        opp_pos = self.state.opponent_pos
        actions = self._admissable_actions(self.state.agent_pos, opp_pos)
        if np.random.binomial(1, self.move_prob):
            move = np.random.choice(actions).value
            if self.grid.is_inside(opp_pos + move):
                self.state.opponent_pos = self.state.opponent_pos + move


    def _generate_legal(self):
        return list(range(self.action_space.n))



    # def _local_move(self, state, last_action, last_ob):
    #     if len(state.opponent_pos) > 0:
    #         opp = np.random.randint(len(state.opponent_pos))
    #     else:
    #         return False
    #
    #     if state.opponent_pos[opp] == Coord(-1, -1):
    #         return False
    #     state.opponent_pos[opp] = self.grid.sample()
    #     if last_ob != self.grid.get_index(state.agent_pos):
    #         state.agent_pos = self.grid.get_tag_coord(last_ob)
    #
    #     ob = self._sample_ob(state, last_action)
    #     return ob == last_ob

    def oppTransitionDistribution(self, state):
        distribution = {}
        tag_state = self.decode_state(state)
        rob_pos = tag_state.agent_pos
        opp_pos = tag_state.opponent_pos[0]
        if rob_pos.x == opp_pos.x:
            index = self.grid.get_index(opp_pos + Coord(1, 0)) if self.grid.is_inside(opp_pos + Coord(1, 0)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.2

            index = self.grid.get_index(opp_pos + Coord(-1, 0)) if self.grid.is_inside(opp_pos + Coord(-1, 0)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.2
        else:
            dx = 1 if opp_pos.x > rob_pos.x else -1
            index = self.grid.get_index(opp_pos + Coord(dx, 0)) if self.grid.is_inside(opp_pos + Coord(dx, 0)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.4

        if rob_pos.y == opp_pos.y:
            index = self.grid.get_index(opp_pos + Coord(0, 1)) if self.grid.is_inside(opp_pos + Coord(0, 1)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.2

            index = self.grid.get_index(opp_pos + Coord(0, -1)) if self.grid.is_inside(opp_pos + Coord(0, -1)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.2
        else:
            dy = 1 if opp_pos.y > rob_pos.y else -1
            index = self.grid.get_index(opp_pos + Coord(0, dy)) if self.grid.is_inside(opp_pos + Coord(0, dy)) else self.grid.get_index(opp_pos)
            distribution[index] += 0.4

        distribution[self.grid.get_index(opp_pos)] += 0.2

        return distribution

    def stateTransitionDistribution(self, state):

        opp_distribution = self.oppTransitionDistribution(state)



    @staticmethod
    def _admissable_actions(agent_pos, opp_pos):
        actions = []
        if opp_pos.x >= agent_pos.x:
            actions.append(Moves.EAST)
        if opp_pos.y >= agent_pos.y:
            actions.append(Moves.NORTH)
        if opp_pos.x <= agent_pos.x:
            actions.append(Moves.WEST)
        if opp_pos.y <= agent_pos.y:
            actions.append(Moves.SOUTH)
        if opp_pos.x == agent_pos.x and opp_pos.y > agent_pos.y:
            actions.append(Moves.NORTH)
        if opp_pos.y == agent_pos.y and opp_pos.x > agent_pos.x:
            actions.append(Moves.EAST)
        if opp_pos.x == agent_pos.x and opp_pos.y < agent_pos.y:
            actions.append(Moves.SOUTH)
        if opp_pos.y == agent_pos.y and opp_pos.x < agent_pos.x:
            actions.append(Moves.WEST)
        assert len(actions) > 0
        return actions

# add heuristcs to tag problem
class TagState(object):
    def __init__(self, coord, opp_coord):
        self.agent_pos = coord
        self.opponent_pos = opp_coord
        self.num_opp = 0


if __name__ == '__main__':
    env = TagEnv()
    # env.reset()
    # state = env.tag_state
    # gui = RobotGrid(start_coord=state.agent_pos, obj_coord=state.opponent_pos)
    # action = Action.sample()
    # env.step(action=action)
    # gui.render(action, env.tag_state)
    #


    legal_state = []
    for i in range(env.nS):
        state = env.decode_state(i)
        if env.grid.is_inside(state.agent_pos) & env.grid.is_inside(state.opponent_pos):
            legal_state.append(i)

    for i in legal_state:
        state = env.decode_state(i)
        ind = env.encode_state(state)
        if i != ind:
            print("noooo")



    env.render()
    done = False
    r = 0
    while not done:
        action = np.random.choice(env._generate_legal())
        state, rw, done = env.step(action)
        # ind = env.encode_state(state)
        # s = env.decode_state(ind)
        # print(f'real {state.agent_pos} : {state.opponent_pos}, me - {s.agent_pos} : {s.opponent_pos}')
        # if (state.agent_pos != s.agent_pos) | (state.opponent_pos[0] != s.opponent_pos[0]):
        #     print("noooooooooooooooooo")
        # env._set_state(info['state'])
        env.render()
        r += rw
    print('done, r {}'.format(r))
