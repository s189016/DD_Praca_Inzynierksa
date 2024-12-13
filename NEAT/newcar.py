import math
import sys
import neat
import pygame
import pickle

# Constants
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 188.2 / 3  # Car width
CAR_SIZE_Y = 230 / 3    # Car length

target_x = 1600  # X-coordinate of the target
target_y = 500   # Y-coordinate of the target

stat_x = 250     # Initial X position of the car
stat_y = 100     # Initial Y position of the car

BORDER_COLOR = (255, 255, 255, 255)  # Color of obstacles for collision detection

current_generation = 0  # Generation counter

class Car:
    def __init__(self):
        self.sprite = pygame.image.load('car.png').convert()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite

        self.distance_to_target = 9999
        self.previous_distance_to_target = 9999
        self.position = [stat_x, stat_y]
        self.angle = 0
        self.speed = 20
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []
        self.alive = True
        self.collision_penalized = False
        self.angle_history = []  # Track angle changes
        self.spinning_penalized = False  # Track if already penalized for spinning

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def check_collision(self, game_map):
        self.alive = True
        extra_points = []
        for i in range(1, 5):
            ratio = i / 5.0
            extra_points.extend([
                [
                    self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * CAR_SIZE_X * ratio,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * CAR_SIZE_Y * ratio
                ],
                [
                    self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * CAR_SIZE_X * ratio,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * CAR_SIZE_Y * ratio
                ],
                [
                    self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * CAR_SIZE_X * ratio,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * CAR_SIZE_Y * ratio
                ],
                [
                    self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * CAR_SIZE_X * ratio,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * CAR_SIZE_Y * ratio
                ]
            ])
        for point in self.corners + extra_points:
            x, y = int(point[0]), int(point[1])
            if 0 <= x < WIDTH and 0 <= y < HEIGHT:
                if game_map.get_at((x, y)) == BORDER_COLOR:
                    self.alive = False
                    break

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
        y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        while not game_map.get_at((x, y)) == BORDER_COLOR and length < 300:
            length += 1
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)

        if length < 300:
            dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
            self.radars.append([(x, y), dist])

    def update(self, game_map):
        self.rotated_sprite = self.rotate_center(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * self.speed
        self.position[0] = max(self.position[0], 20)
        self.position[0] = min(self.position[0], WIDTH - 120)

        self.position[1] += math.sin(math.radians(360 - self.angle)) * self.speed
        self.position[1] = max(self.position[1], 20)
        self.position[1] = min(self.position[1], HEIGHT - 120)

        self.center = [int(self.position[0]) + CAR_SIZE_X / 2, int(self.position[1]) + CAR_SIZE_Y / 2]

        length = 0.5 * CAR_SIZE_X
        left_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 30))) * length,
                    self.center[1] + math.sin(math.radians(360 - (self.angle + 30))) * length]
        right_top = [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
                     self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length]
        left_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 210))) * length,
                       self.center[1] + math.sin(math.radians(360 - (self.angle + 210))) * length]
        right_bottom = [self.center[0] + math.cos(math.radians(360 - (self.angle + 330))) * length,
                        self.center[1] + math.sin(math.radians(360 - (self.angle + 330))) * length]
        self.corners = [left_top, right_top, left_bottom, right_bottom]

        self.check_collision(game_map)
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        # Track angle changes to detect spinning
        self.angle_history.append(self.angle)
        if len(self.angle_history) > 30:  # Keep the last 30 frames of angles
            self.angle_history.pop(0)

    def is_spinning(self):
        # Check if the car is spinning by looking at recent angle changes
        if len(self.angle_history) >= 30:
            angle_change = sum(
                abs(self.angle_history[i] - self.angle_history[i - 1]) for i in range(1, len(self.angle_history)))
            if angle_change > 720:  # Threshold for spinning, adjust as needed
                return True
        return False

    def get_data(self):
        return_values = [0] * 6
        for i, radar in enumerate(self.radars):
            if i < 5:
                return_values[i] = int(radar[1] / 30)

        self.previous_distance_to_target = self.distance_to_target
        self.distance_to_target = int(math.sqrt((target_x - self.center[0]) ** 2 + (target_y - self.center[1]) ** 2))
        return_values[5] = self.distance_to_target

        return return_values

    def is_alive(self):
        return self.alive

    def rotate_center(self, image, angle):
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rectangle = rotated_image.get_rect(center=image.get_rect().center)
        return rotated_image

def calculate_fitness(car, genome):
    target_reached_reward = 100000
    proximity_reward = 18
    proximity_penalty = -10
    collision_penalty = -650
    time_penalty = -15
    safety_margin_penalty = -60
    spinning_penalty = -300000  # Penalty for spinning in circles

    if car.distance_to_target < 60:
        genome.fitness += target_reached_reward
        print("Target reached!")
        car.alive = False
        return

    if car.distance_to_target < car.previous_distance_to_target:
        genome.fitness += proximity_reward
    else:
        genome.fitness += proximity_penalty

    if not car.is_alive():
        genome.fitness += collision_penalty

    for radar in car.radars:
        if radar[1] < 50:
            genome.fitness += safety_margin_penalty

    if car.is_spinning() and not car.spinning_penalized:
        car.spinning_penalized = True
        genome.fitness += spinning_penalty

    genome.fitness += time_penalty

def run_simulation(genomes, config):
    nets = []
    cars = []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    # genomes = list(population.population.items())[-10:]
    # #wypisz genomes wraz z wartosciami fitness
    # for i, g in genomes:
    #     print("Genome: ", g.key, "Fitness: ", g.fitness)
    #     #zapisz genom z najwieksza wartoscia fitness
    #         with open("best_genome2.pkl", "wb") as f:
    #             pickle.dump(g, f)
    #             print("Best genome saved as best_genome2.pkl")


    for i, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        g.fitness = 0
        cars.append(Car())

    clock = pygame.time.Clock()
    generation_font = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map = pygame.image.load('map4.jpg').convert()

    global current_generation
    current_generation += 1

    start_time = pygame.time.get_ticks()  # Get the start time in milliseconds

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        for i, car in enumerate(cars):
            output = nets[i].activate(car.get_data())
            choice = output.index(max(output))
            if choice == 0:
                car.angle += 10
            elif choice == 1:
                car.angle -= 10
            elif choice == 2 and car.speed - 2 >= 12:
                car.speed -= 2
            else:
                car.speed += 2

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                calculate_fitness(car, genomes[i][1])
            elif not car.collision_penalized:
                car.collision_penalized = True
                genomes[i][1].fitness -= 200

        # End the simulation if no cars are alive
        if still_alive == 0:
            break

        # Check if 10 seconds have elapsed
        elapsed_time = pygame.time.get_ticks() - start_time
        if elapsed_time > 20000:  # 20,000 milliseconds = 20 seconds
            break

        screen.blit(game_map, (0, 0))
        pygame.draw.circle(screen, (0, 0, 255, 255), (target_x, target_y), 20)

        for car in cars:
            if car.is_alive():
                car.draw(screen)

        text = generation_font.render("Generation: " + str(current_generation), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 450)
        screen.blit(text, text_rect)

        text = alive_font.render("Still Alive: " + str(still_alive), True, (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (900, 490)
        screen.blit(text, text_rect)

        pygame.display.flip()
        clock.tick(60)



if __name__ == "__main__":
    config_path = "config2.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    #population = neat.Population(config)
    population = neat.Checkpointer.restore_checkpoint('genome-278')
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    checkpointer = neat.Checkpointer(generation_interval=30, filename_prefix="genome-")
    population.add_reporter(checkpointer)


    best_genome = population.run(run_simulation, 1000)

