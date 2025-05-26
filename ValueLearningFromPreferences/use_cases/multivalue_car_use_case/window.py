import pygame
from pygame import __rect_constructor as rc
import sys

RIGHT = 0
UP = 1
LEFT = 2

PUSH_RIGHT = 3
PUSH_UP = 4
PUSH_LEFT = 5

afasdada = True

def arrow(x, y, direction, factor, screen):
    if direction % 3 == RIGHT:
        pygame.draw.polygon(screen, (0, 0, 0), [[10 * factor + x, 10 * factor + y], [10 * factor + x, 40 * factor + y], [30 * factor + x, 25 * factor + y]], direction - direction % 3)
    elif direction % 3 == UP:
        pygame.draw.polygon(screen, (0, 0, 0), [[10 * factor + x, 30 * factor + y], [40 * factor + x, 30 * factor + y], [25 * factor + x, 10 * factor + y]], direction - direction % 3)
    elif direction % 3 == LEFT:
        pygame.draw.polygon(screen, (0, 0, 0), [[30 * factor + x, 10 * factor + y], [30 * factor + x, 40 * factor + y], [10 * factor + x, 25 * factor + y]], direction - direction % 3)


def square_sil(screen, x, y):
    """
    Creates a square silhouette.

    :param screen: where to draw it
    :param x: x-axis origin of the square
    :param y: y-axis origin of the square
    :return: Nothing, the results are immediately seen.
    """
    color = (0, 0, 0)  # green

    # draw a rectangle
    pygame.draw.rect(screen, color, pygame.Rect(10 + x, 10 + y, 50, 5))  # UP
    pygame.draw.rect(screen, color, pygame.Rect(10 + x, 10 + y, 5, 50))  # LEFT
    pygame.draw.rect(screen, color, pygame.Rect(55 + x, 10 + y, 5, 50))  # RIGHT
    pygame.draw.rect(screen, color, pygame.Rect(10 + x, 55 + y, 50, 5))  # DOWN


def to_color(nb):
    """
    Assings some colours to concrete integer numbers.
    :param nb: An integer number.
    :return: the colour in RGB format.
    """
    if nb == 0: # Accessible
        return 128, 128, 128
    elif nb == 1: # Inaccessible
        return 0, 0, 0
    elif nb == 4: # Everyone cell (paso de cebra)
        return 0, 0, 128
    elif nb == 2000: # Goal left
        return 128, 128, 128
    elif nb == 3000: # Wastebasket
        return 0, 128, 0
    else:
        return 128, 0, 0


class Window:

    AC = 0  # ACCESSIBLE CELL (for cars and garbage)
    IC = 1  # INACCESSIBLE CELL
    GC = 2 # GARBAGE CELL (only for garbage)
    PC = 3 # PEDESTRIAN CELL (only for pedestrian)
    EC = 4 # EVERYONE CELL (for everyone)

    def __init__(self, info):
        """
        Initializes all the important variables needed.

        :param info: all the necessary information extracted from an Environment Class Instance.
        """

        self.width = 640
        self.height = 640
        self.map, self.player, self.waste_basket, self.waste_basket2, self.goal, self.goal2 = info


        self.map[self.goal[0], self.goal[1]] = 2000
        self.map[self.goal2[0], self.goal2[1]] = 2000

        # Waste_basket part
        #self.map[self.waste_basket[0], self.waste_basket[1]] = 3000
        #self.map[self.waste_basket2[0], self.waste_basket2[1]] = 3000

        self.items = list()
        self.stats = 0, 0, 0, 0
        self.paused = False

        self.image = pygame.image.load('wastebasket.png')
        self.image = pygame.transform.scale(self.image, (40, 40))


    def update(self, info):
        """
        Updates all the variables that may change in the game.
        :param info: all the necessary information extracted from an Environment Class Instance.
        :return: -
        """
        self.player = info[0]
        self.items = info[1:]

    def controls(self):
        """
        All the controls of the game the player can access.
        :return:
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_ESCAPE:
                    self.paused = True
                    sys.exit()

            if event.type == pygame.QUIT:
                self.paused = True
                sys.exit()

    def message(self, screen, mode='Training'):
        """
        All the text you can read in the window is created here.
        :param screen: Screen where it needs to draw. Created in the method Window.create().
        :param mode: training or evaluating.
        :return:
        """
        fuente = pygame.font.Font(None, 20)

        if mode == 'Training':
            text = 'Episode {}. Total Reward: {}. Mean Reward: {:.3f}'.format(
                self.stats[0], self.stats[1], self.stats[2])
        else:
            text = '        EPISODE {}.'.format(self.stats[0])

        mensaje = fuente.render(text, 1, (255, 255, 255))

        screen.blit(mensaje, (15, 5))

        pygame.draw.line(screen, (255, 255, 255), (0, 25), (640, 25), 2)

        if self.paused:
            fuente = pygame.font.Font(None, 50)
            mensaje = fuente.render("FAST FORWARD", 1, (255, 255, 255))
            screen.blit(mensaje, (130 + 2 * 55, 43 + 2 * 55))

    def draw_char(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), (135 + self.player[1] * 55, 55 + self.player[0] * 55), 15)
        fuente = pygame.font.Font(None, 25)
        mensaje = fuente.render("C", 1, (255, 255, 255))
        screen.blit(mensaje, (130 + self.player[1] * 55, 43 + self.player[0] * 55))

    def draw_field(self, screen):
        """
        Draws in the window the field and the main character/"player".
        :param screen: Screen where it needs to draw. Created in the method Window.create().
        :return:
        """

        for i in range(self.map.shape[1]):
            for j in range(self.map.shape[0]):

                if self.map[j][i] != Window.IC:
                    pygame.draw.rect(screen, to_color(self.map[j][i]),
                                     rc(110 + i * 55, 30 + j * 55, 50, 50))
                    square_sil(screen, 100 + i * 55, 20 + j * 55)

        fuente = pygame.font.Font(None, 40)
        mensaje2 = fuente.render("X", 1, (0, 0, 0))
        screen.blit(mensaje2, (125 + self.goal[1]*55, 43 + self.goal[0]*55))

        mensaje2 = fuente.render("X", 1, (0, 0, 0))
        screen.blit(mensaje2, (125 + self.goal2[1]*55, 43 + self.goal2[0]*55))




    def draw_items(self, screen):
        """
        Draws all the items of the game except the player.
        :param screen: Screen where it needs to draw. Created in the method Window.create().
        :return:
        """
        for item in self.items:
            if item[2] == 5:
                pygame.draw.rect(screen, (100, 40, 60), rc(122 + item[1] * 55, 45 + item[0] * 55, 25, 25))
            else:
                pygame.draw.circle(screen, (0, 0, 0), (135 + item[1] * 55, 55 + item[0] * 55), 15)
                fuente = pygame.font.Font(None, 25)
                mensaje = fuente.render("P", 1, (255, 255, 255))
                screen.blit(mensaje, (130 + item[1] * 55, 43 + item[0] * 55))

        # Wastebasket
        #screen.blit(self.image, (115 + self.waste_basket[1]*55, 35 + self.waste_basket[0]*55))
        #screen.blit(self.image, (115 + self.waste_basket2[1]*55, 35 + self.waste_basket2[0]*55))

    def create(self, mode='Training'):
        """
        Starts the window. Until this method is called the output can only be seen in the console.
        :param mode: training or evaluating
        :return:
        """
        pygame.init()

        screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Learning with Q-Learning")

        while True:

            self.controls()
            screen.fill((25, 145, 255))
            self.message(screen, mode)
            if not self.paused:
                pygame.time.wait(10)

                self.draw_field(screen)


                #if afasdada:
                #    arrow(165, 85 + 1 * 55, UP, 1.0, screen)
                #    arrow(165, 85 + 2 * 55, UP, 1.0, screen)
                #    arrow(165, 85 + 3 * 55, UP, 1.0, screen)
                #else:
                #    arrow(165, 85 + 1 * 55, PUSH_LEFT, 1.0, screen)
                #    arrow(165, 85 + 2 * 55, PUSH_UP, 1.0, screen)
                #    arrow(165, 85 + 3 * 55, PUSH_UP, 1.0, screen)

                self.draw_char(screen)
                self.draw_items(screen)

            pygame.display.flip()
