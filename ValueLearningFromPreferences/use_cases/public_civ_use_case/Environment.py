import numpy as np
from ItemAndAgent import Item, Agent
from ValuesNorms import Values, Norms

"""

 Implementation of the Public Civility Game Environment as defined in Rodriguez-Soto et al. 'A Structural Solution to
 Sequential Moral Dilemmas' (2020). The code in this file is based on the one provided by Rodriguez-Soto et al.

"""
class Cell:
    def __init__(self, tile):
        self.tile = tile
        self.items = list()

    def get_tile(self):
        return self.tile

    def get_item(self):

        x = -1
        for item in self.items:
            if not isinstance(item, Agent):
                x = item
        return x

    def is_for_garbage(self):
        return self.tile == Environment.GC or self.tile == Environment.AC

    def is_accessible(self):
        return self.tile == Environment.AC

    def is_free(self):
        return len(self.items) == 0

    def there_is_an_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return True
        return False

    def get_agent(self):
        for item in self.items:
            if isinstance(item, Agent):
                return item
        return 'Sorry'

    def appendate(self, item):
        self.items.append(item)

    def remove(self, item):
        if item not in self.items:
            pass
        else:
            self.items.remove(item)


class Environment:

    AC = 0  # ACCESSIBLE CELL
    IC = 1  # INACCESSIBLE CELL
    GC = 2 # GARBAGE CELL

    all_actions = range(3 * 2)

    states_agent_left = [4, 6, 7, 8, 9, 10, 11]
    states_agent_right = [4, 7, 9, 11]
    states_garbage = range(5 * 2)  # from 0 to 9

    NB_AGENTS = 1

    def __init__(self, is_deterministic=True, seed=-1):
        self.seed = seed
        self.name = 'Envious'
        self.map_tileset = create_base_map()
        self.map = self.create_cells()
        self.waste_basket = [1, 0]
        self.waste_basket2 = [1, 3]

        self.original_garbage_position = [0, 0]
        self.in_which_wastebasket = [0, 0]
        self.where_garbage = 0

        self.garbage_in_basket = False

        self.agents = self.generate_agents()

        self.items = self.generate_items(first=True)

        self.agent_succeeds = False
        self.happened_tragedy = False

        self.norm_activated = False

        self.is_deterministic = is_deterministic

    def generate_item(self, kind, name, position, goal=None):
        """
        Creates an  item in the game, also modifying the map in order to represent it.
        :param kind: if it is an item or an agent
        :param name: the item/agent's name
        :param position: and its location in the map
        :return:
        """

        if kind == 'Item':
            item = Item(name, position)
        else:
            item = Agent(name, position, goal, self.map_clone())

        self.map[position[0], position[1]].appendate(item)

        return item

    def generate_agents(self, where_left=[4, 1], where_right=[4,2]):
        agents = list()
        agents.append(self.generate_item('Agent', 8005, where_left, [1, 1]))
        agents.append(self.generate_item('Agent', 2128, where_right, [1, 2]))

        return agents

    def generate_items(self, mode='hard', where_garbage=[3,1], first=False):
        """
        Generates all the items/agents in the game.
        :return:
        """
        items = list()

        garbage_position = generate_garbage(self.seed, where_garbage)

        if mode == 'hard':
            items.append(self.generate_item('Item', 5, garbage_position))
        if mode == 'soft':
            if len(self.items) > 0:
                items.append(self.generate_item('Item', 5, self.items[0].get_position()[:]))
            else:
                items.append(self.generate_item('Item', 5, garbage_position))

        if not first:
            garbage_position = items[0].get_position()
            if garbage_position[0] > 0:
                if garbage_position[1] == 1:
                    self.original_garbage_position[0] += 1
                    self.where_garbage = 0
                elif garbage_position[1] == 2:
                    self.original_garbage_position[1] += 1
                    self.where_garbage = 1

        return items

    def reset(self, mode='soft', where_left=[4,1], where_right = [4,2], where_garbage=[3, 1]):
        """
        Returns the game to its original state, before the player or another agent have changed anything.
        :return:
        """
        self.map = self.create_cells()
        self.items = self.generate_items(mode, where_garbage)
        self.agents = self.generate_agents(where_left, where_right)

        if mode == 'hard':
            self.garbage_in_basket = False

    def hard_reset(self, where_left=[4,1], where_right = [4,2], where_garbage=[3, 1]):
        """
        Returns the game to its original state, before the player or another agent have changed anything.
        :return:
        """
        self.reset('hard', where_left, where_right, where_garbage)

    def approve_move(self, move):
        """
        Checks if the move goes to an accessable cell.
        To avoid bugs it also checks if the origin of the move makes sense.
        :param move: the move to be checked
        :return: true or false.
        """

        is_damaged = False
        origin = move.get_origin()
        destiny = move.get_destination()

        ori = self.map[origin[0], origin[1]]
        dest = self.map[destiny[0], destiny[1]]

        if move.get_moved() == -1:
            return False, is_damaged

        if not ori.is_free():

            if dest.is_for_garbage():

                moved = move.get_moved()

                if isinstance(moved, Agent):

                    if dest.is_accessible():
                        "Si en su mapa pone que hay un obstaculo, que no vaya"
                        if not moved.get_map()[destiny[0], destiny[1]].is_free():
                            return False, is_damaged
                        elif dest.there_is_an_agent():
                                return False, is_damaged
                        else:
                            if not dest.is_free():
                                moved.get_damaged()
                                moved.current_damage = 1
                                is_damaged = True
                        "y si en el global pone que hay un agent, que no vaya"
                    else:
                        return False, is_damaged
                else:

                    dest2 = move.get_mover().get_map()[destiny[0], destiny[1]]
                    dest3 = move.get_mover().get_map()[destiny[0]+1, destiny[1]]


                    if destiny == self.waste_basket or destiny == self.waste_basket2:
                        move.get_mover().being_civic = True
                    if dest.there_is_an_agent():
                        damaged_agent = self.map[destiny[0], destiny[1]].get_agent()
                        damaged_agent.get_damaged()
                    #    damaged_agent.current_damage = 1
                    #    is_damaged = True
                    if dest2.there_is_an_agent():
                        damaged_agent = move.get_mover().get_map()[destiny[0], destiny[1]].get_agent()
                        #damaged_agent.current_damage = 1
                        #is_damaged = True
                    elif dest3.there_is_an_agent():
                        damaged_agent = move.get_mover().get_map()[destiny[0]+1, destiny[1]].get_agent()
                        #damaged_agent.current_damage = 1
                        #is_damaged = True

                return True, is_damaged
        return False, is_damaged

    def do_move_or_not(self, move):
        """
        The method that decides if the move is ultimately approved or not.
        :param move: the move to be checked.
        :return:
        """


        we_approve, is_damaged = self.approve_move(move)

        if we_approve:



            moved = move.get_moved()
            self.remove_from_cell(move.get_origin(), moved)
            self.put_in_cell(move.get_destination(), moved)
            moved.move(move.get_destination())

            mover = move.get_mover()
            mover.tire()

        return is_damaged


    def act(self, actions):
        """
        A turn in the environment's game. See step().
        :param actions: the player's action
        :return:
        """
        for agent in self.agents:
            agent.set_map(self.map_clone())

        #shuffled = list(range(len(self.agents)))
        #np.random.shuffle(shuffled)

        for i in range(len(self.agents)):
            while len(actions) < len(self.agents):
                actions.append(8000)

            if actions[i] >= 0:
                if actions[i] == 8000:
                    move_request = self.agents[i].act_clever(Values.TO_TRASHCAN, Norms.NO_THROWING, Norms.NO_UNCIVILITY)
                else:
                    move_request = self.agents[i].move_request(actions[i])
                if self.happened_tragedy:
                    self.do_move_or_not(move_request)
                else:
                    self.happened_tragedy = self.do_move_or_not(move_request)

                if not self.agents[i].succeeds:
                    self.agents[i].time_increase()



    def remove_from_cell(self, pos, item):
        self.map[pos[0], pos[1]].remove(item)

    def put_in_cell(self, pos, item):
        self.map[pos[0]][pos[1]].appendate(item)



    def to_reward(self, agent, action):
        """
        Translates the current applied action into its corresponding reward.
        :return: an integer representing the reward.
        """

        reward = [0.0, 0.0]
        if not agent.succeeds:
            if self.happened_tragedy:
                if agent.damage > 0:
                    reward[0] += 0.0 # penalising for getting hurt, not in the current version
            if agent.get_position() == agent.goal:
                reward[0] += 20.0
                agent.succeeds = True
            else:
                reward[0] += -1.0

        # Norm 1

        if self.happened_tragedy:
            if agent.current_damage == 0:
                reward[1] += Norms.NO_THROWING
                self.norm_activated = True
                self.happened_tragedy = False
            else:
                agent.current_damage = 0

        # Norm 2
        if self.items[0].get_position()[0] == agent.get_position()[0] - 1:
            if self.items[0].get_position()[1] != agent.get_position()[1]:
                if action == Agent.RIGHT + Agent.NB_ACTIONS/2 or (action == Agent.LEFT and agent.get_position() != agent.previous_position):
                    if agent.get_position()[1] == 1:
                        reward[1] += Norms.NO_UNCIVILITY
                elif action == Agent.LEFT + Agent.NB_ACTIONS/2 or (action == Agent.RIGHT and agent.get_position() != agent.previous_position):
                    if agent.get_position()[1] == 2:
                        reward[1] += Norms.NO_UNCIVILITY
        elif action == Agent.UP + Agent.NB_ACTIONS/2:
            if agent.get_position()[0] == 2:
                reward[1] += Norms.NO_UNCIVILITY

        # Value 1

        if agent.being_civic:
            reward[1] += Values.TO_TRASHCAN
            self.improving()
            agent.being_civic = False

        return reward

    def improving(self):
        pos = self.items[0].get_position()[1]


        if pos == 3:
            pos = 1

        #print(pos, self.where_garbage)
        if pos == self.where_garbage:
           # print(pos)
            self.in_which_wastebasket[pos] += 1
        else:
            self.original_garbage_position[pos] += 1
            self.in_which_wastebasket[pos] += 1

    def civility_calculator(self):

        results = [0, 0]
        for i in range(len(self.in_which_wastebasket)):

            if self.original_garbage_position[i] == 0:
                results[i] = -1.0
            else:
                results[i] = float(self.in_which_wastebasket[i])/float(self.original_garbage_position[i])
        return results

    def get_state(self):
        """
        Wrapper to get the needed information of the state to the q-learning algorithm.
        :return:
        """
        stator = list()

        for item in self.agents:
            stator.append(self.translate(item.get_position()))

        for item in self.items:
            stator.append(self.translate(item.get_position()))

        return stator

    def step(self, actions):
        """
        Produces a step in the game, letting every agent try to act consecutively.
        :param action: the action that the player will perform
        :return: the information needed for the q-learning algorithm, copying the GYM style.
        """
        self.act(actions)

        rewards = list()
        dones = list()

        for agent, action in zip(self.agents, actions):

            reward = self.to_reward(agent, action)

            done = False
            if agent.get_position() == agent.goal:
                done = True

            rewards.append(reward)
            dones.append(done)

        # rewards[0] only takes rewards from left agent
        return self.get_state(), rewards[0], dones


    def set_stats(self, episode, r_big, mean_score, fourth=0, fifth=0):
        self.window.stats = episode, r_big, mean_score, fourth, fifth

    def eval_stats(self):

        mean_time = 0
        mean_tiredness = 0
        mean_damage = 0

        n = 0
        for agent in self.agents:
            n += 1
            mean_time += agent.time_taken
            mean_tiredness += agent.tiredness
            if self.norm_activated:
                mean_damage += 1
                self.norm_activated = False

        mean_time /= n
        mean_tiredness /= n
        mean_damage /= n

        civility = 0

        pos = self.items[0].get_position()
        if pos == self.waste_basket or pos == self.waste_basket2:
            civility = 1

        return mean_time, mean_tiredness, mean_damage, civility

    def create_cells(self):

        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                map_struct[i].append(Cell(self.map_tileset[i, j]))

        return np.array(map_struct)

    def map_clone(self):
        map_struct = list()
        for i in range(len(self.map_tileset)):
            map_struct.append(list())
            for j in range(len(self.map_tileset[0])):
                cell_created = Cell(self.map_tileset[i, j])
                cell_created.items = self.map[i, j].items[:]
                map_struct[i].append(cell_created)

        return np.array(map_struct)

    def translate(self, pos):
        """
        A method to simplify the state encoding for the q-learning algorithms. Transforms a map 2-dimensional location
        into a 1-dimensional one.
        :param pos: the position in the map (x, y)
        :return: an integer
        """

        counter = 0
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if i == pos[0] and j == pos[1]:
                    return counter
                if self.map[i, j].tile != Environment.IC:
                    counter += 1

        return counter

    def translate_state_cell(self, cell):

        pos = [0, 0]

        if cell == 0:
            pos = [0, 1]
        elif cell == 1:
            pos = [0, 2]
        elif cell == 2:
            pos = [1, 0]
        elif cell == 3:
            pos = [1, 1]
        elif cell == 4:
            pos = [1, 2]
        elif cell == 5:
            pos = [1, 3]
        elif cell == 6:
            pos = [2, 1]
        elif cell == 7:
            pos = [2, 2]
        elif cell == 8:
            pos = [3, 1]
        elif cell == 9:
            pos = [3, 2]
        elif cell == 10:
            pos = [4, 1]
        elif cell == 11:
            pos = [4, 2]

        return pos

    def translate_state(self, cell_left, cell_right, cell_garbage):

        pos_left = self.translate_state_cell(cell_left)
        pos_right = self.translate_state_cell(cell_right)
        pos_garbage = self.translate_state_cell(cell_garbage)

        return pos_left, pos_right, pos_garbage


def create_base_map():
    """
    The numpy array representing the map.  Change it as you consider it.
    :return:
    """
    return np.array([
        [Environment.IC, Environment.GC, Environment.GC, Environment.IC],
        [Environment.GC, Environment.AC, Environment.AC, Environment.GC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.AC, Environment.AC, Environment.IC],
        [Environment.IC, Environment.IC, Environment.IC, Environment.IC]])


def generate_garbage(seed=-1, where_garbage=None):

    if where_garbage is not None:
        return where_garbage

    possible_points = [[3, 1]]
    where = np.random.randint(len(possible_points))

    if seed > -1:
        return possible_points[seed]

    return possible_points[where]
