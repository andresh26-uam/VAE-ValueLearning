import numpy as np


class Move:
    def __init__(self, mover, moved, origin, destination):
        """
        Moves have a mover agent ID, a moved item ID, an origin and a destination.
        The moved item is NOT declared here but in another method.
        :param mover:  the Agent ID (int) that will do the move
        :param origin: The original location of the moved item.
        :param destination: The destination of the moved item.
        """
        self.mover = mover
        self.moved = moved
        self.origin = origin
        self.destination = destination

    def get_mover(self):
        """
        :return: the mover agent ID
        """
        return self.mover

    def get_moved(self):
        """

        :return: the moved item ID.
        """
        return self.moved

    def get_origin(self):
        """

        :return: the moved item location
        """
        return self.origin

    def get_destination(self):
        """
        :return: the moved item desired destination.
        """
        return self.destination

    def check_validity(self, mat):
        """
        Simple method to avoid code problems later. Simply checks if the destination coordinates actually exists.
        Doesn't bother checking if the destination is an unreachable cell.
        :param mat: the matrix representing the map
        :return: true or false according to the analysis
        """

        limits = mat.shape

        for i in range(len(self.destination)):
            if self.destination[i] < 0 or self.destination[i] > (limits[i] - 1):
                return False

        return True


class Item:

    def __init__(self, name, pos):
        """
        Items are very basic classes. They simply have a name/ID (integer) and a position that may change.
        :param name: an integer (be wary of not repeating it, very important).
        :param pos: the position, two integers.
        """
        self.name = name
        self.position = pos
        self.previous_position = pos
        self.origin = pos

    def move(self, pos):
        self.previous_position = self.position
        self.position = pos

    def get_position(self):
        return self.position

    def get_name(self):
        return self.name


class Agent(Item):
    """
    Agent inherits from its parent  Item, but now it can also perform actions.
    """

    RIGHT = 0
    UP = 1
    LEFT = 2
    #DOWN = 3

    NB_ACTIONS = 6

    def __init__(self, name, pos, goal, mat):
        Item.__init__(self, name, pos)
        self.goal = goal
        self.map = mat

        self.damage = 0
        self.tiredness = 0
        self.time_taken = 0
        self.succeeds = False
        self.being_civic = False
        self.current_damage = 0

    def set_map(self, mat):
        self.map = mat

    def get_map(self):
        return self.map

    def move_request(self, direction_agent):
        """
        Receives a direction (an integer) and interprets the associated movement.

        :param direction_agent: an integer from 0 to 7
        :return: a move class instance
        """

        origin = self.position[:]
        moved = self

        if direction_agent > Agent.NB_ACTIONS/2 - 1:
            moved = self.map[origin[0] - 1, origin[1]].get_item()
            origin[0] -= 1

        destination = origin[:]

        if direction_agent % (Agent.NB_ACTIONS/2) == Agent.LEFT:
            "Left"
            destination[1] -= 1
        elif direction_agent % (Agent.NB_ACTIONS/2) == Agent.RIGHT:
            "Right"
            destination[1] += 1
        elif direction_agent % (Agent.NB_ACTIONS/2) == Agent.UP:
            "Up"
            destination[0] -= 1
        else:
            print('this ought to never happen')

        move = Move(self, moved, origin, destination)

        if move.check_validity(self.map):
            return move
        else:
            return Move(self, moved, origin, origin)

    def act(self):
        """
        A method to randomly choose a possible actino.
        :return: a move request.
        """

        direction_chosen = np.random.randint(Agent.NB_ACTIONS)
        return self.move_request(direction_chosen)

    def act_clever(self, moral_value, norm_value, norm_civility):
        """
        An heuristic method to choose the best action according to the map and location.
        :return: a move request.
        """

        pos = self.position
        possible_actions = [Agent.UP]
        direction_chosen = Agent.UP

        if not self.map[pos[0] - 1, pos[1]].is_free():
            if moral_value == 0 and norm_civility == 0:
                if self.map[pos[0] - 1, pos[1] + 1].is_for_garbage():
                    possible_actions.append(Agent.RIGHT + Agent.NB_ACTIONS/2)
                if self.map[pos[0] - 1, pos[1] - 1].is_for_garbage():
                    possible_actions.append(Agent.LEFT + Agent.NB_ACTIONS / 2)
                if self.map[pos[0] - 2, pos[1]].is_for_garbage():
                    possible_actions.append(Agent.UP + Agent.NB_ACTIONS / 2)

                if norm_value < 0:

                    if not self.map[pos[0], pos[1] + 1].is_free() or not self.map[pos[0] - 1, pos[1] + 1].is_free():
                        if Agent.RIGHT + Agent.NB_ACTIONS/2 in possible_actions:
                            possible_actions.remove(Agent.RIGHT + Agent.NB_ACTIONS/2)
                    if not self.map[pos[0], pos[1] - 1].is_free() or not self.map[pos[0] - 1, pos[1] - 1].is_free():
                        if Agent.LEFT + Agent.NB_ACTIONS/2 in possible_actions:
                            possible_actions.remove(Agent.LEFT + Agent.NB_ACTIONS/2)

                if len(possible_actions) > 2:
                    possible_actions.remove(Agent.UP)
                randomly = np.random.randint(len(possible_actions))

                direction_chosen = possible_actions[randomly]

            else:
                if self.map[pos[0] - 2, pos[1]].is_accessible():
                    direction_chosen = Agent.UP + Agent.NB_ACTIONS/2
                else:
                    if self.map[pos[0] -1, pos[1] + 1].is_for_garbage() and not self.map[pos[0] - 1, pos[1] + 1].is_accessible():
                        direction_chosen = Agent.RIGHT + Agent.NB_ACTIONS / 2
                    else:
                        direction_chosen = Agent.LEFT + Agent.NB_ACTIONS / 2

        return self.move_request(direction_chosen)

    def tire(self):
        self.tiredness += 1

    def get_damaged(self):
        self.damage += 1

    def time_increase(self):
        self.time_taken += 1

    def reset(self):
        self.position = self.origin
        self.succeeds = False
        self.being_civic = False
