import pygame
import math
import pickle
import neat
import sys

# Constants
WIDTH = 1920
HEIGHT = 1080

CAR_SIZE_X = 188.2 / 3  # Car width
CAR_SIZE_Y = 230 / 3    # Car length

target_x = 2000  # X-coordinate of the target
target_y = 400   # Y-coordinate of the target

stat_x = 150     # Initial X position of the car
stat_y = 500     # Initial Y position of the car

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
        self.angle = -90
        self.speed = 20
        self.center = [self.position[0] + CAR_SIZE_X / 2, self.position[1] + CAR_SIZE_Y / 2]

        self.radars = []
        self.alive = True
        self.collision_penalized = False
        self.angle_history = []  # Track angle changes
        self.spinning_penalized = False  # Track if already penalized for spinning
        # Start time for tracking how long the car has been alive
        self.start_time = pygame.time.get_ticks()
        self.angle_history = []  # Track recent angles
        self.spinning_penalized = False  # Flag to avoid multiple penalties for the same spin
        self.no_turn_counter = 0  # Counter for frames without significant turns
        self.head_on_collision_penalized = False  # Flag for penalizing head-on collisions
        self.start_time = pygame.time.get_ticks()  # Track when the car starts driving
        self.speed = 20  # Example initial speed, adjust as necessary
        self.previous_speed = self.speed  # To track speed changes

        self.path = []

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        self.draw_radar(screen)
        self.draw_path(screen)

    def draw_radar(self, screen):
        for radar in self.radars:
            position = radar[0]
            pygame.draw.line(screen, (0, 255, 0), self.center, position, 1)
            pygame.draw.circle(screen, (0, 255, 0), position, 5)

    def draw_path(self, screen):
        # Rysowanie śladu przejazdu
        if len(self.path) > 1:
            pygame.draw.lines(screen, (255, 0, 0), False, self.path, 3)  # Ślad w kolorze czerwonym

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
        self.path.append((int(self.center[0]), int(self.center[1])))
        self.radars.clear()

        for d in range(-90, 120, 45):
            self.check_radar(d, game_map)

        # Track angle changes to detect spinning
        self.angle_history.append(self.angle)
        if len(self.angle_history) > 30:  # Keep the last 30 frames of angles
            self.angle_history.pop(0)

            # Track if the car is making a turn
            if abs(self.angle_history[-1] - self.angle) < 5:  # Small angle change means "no turn"
                self.no_turn_counter += 1
            else:
                self.no_turn_counter = 0  # Reset counter if a turn is made

            # Update angle history as before
            self.angle_history.append(self.angle)
            if len(self.angle_history) > 30:
                self.angle_history.pop(0)
            # Track if the car is making a turn
            if abs(self.angle_history[-1] - self.angle) < 5:  # Small angle change means "no turn"
                self.no_turn_counter += 1
            else:
                self.no_turn_counter = 0  # Reset counter if a turn is made

            # Update angle history as before
            self.angle_history.append(self.angle)
            if len(self.angle_history) > 30:
                self.angle_history.pop(0)

    def is_spinning(self):
        # Check if the car is spinning by looking at recent angle changes
        if len(self.angle_history) >= 45:
            angle_change = sum(
                abs(self.angle_history[i] - self.angle_history[i - 1]) for i in range(1, len(self.angle_history)))
            if angle_change > 540:  # Threshold for spinning, adjust as needed
                print("Spinning detected!")
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
    # Parametry
    max_simulation_time = 30  # Maksymalny czas symulacji w sekundach
    max_speed = 20  # Maksymalna prędkość pojazdu
    proximity_penalty_threshold = 100  # Odległość, poniżej której naliczana jest kara za bliskość przeszkód
    proximity_penalty_factor = -5  # Współczynnik kary za bliskość przeszkód
    slow_down_reward = 20  # Nagroda za zmniejszenie prędkości w pobliżu przeszkód
    direction_reward_factor = 0.05  # Współczynnik nagrody za kierowanie się do celu
    target_reward = 500000  # Nagroda za osiągnięcie celu
    distance_reward_factor = 50000  # Współczynnik nagrody za zbliżanie się do celu

    # Oblicz dystans pokonany w tej iteracji
    current_distance = car.distance_to_target
    distance_traveled = car.previous_distance_to_target - current_distance


    if car.distance_to_target < 1000 and car.distance_to_target < car.previous_distance_to_target:
        genome.fitness += 10
    if car.distance_to_target < 500 and car.distance_to_target < car.previous_distance_to_target:
        genome.fitness += 20
    if car.distance_to_target < 250 and car.distance_to_target < car.previous_distance_to_target:
        genome.fitness += 50
    if car.distance_to_target < 100 and car.distance_to_target < car.previous_distance_to_target:
        genome.fitness += 100


    # Nagroda za osiągnięcie celu
    if car.distance_to_target < 100:  # Cel osiągnięty
        genome.fitness += target_reward
        print("Cel osiągnięty!")
        car.alive = False  # Zatrzymanie samochodu
        return

    # Nagroda za zbliżanie się do celu
    if current_distance > 0:  # Unikaj dzielenia przez zero
        genome.fitness += distance_reward_factor / current_distance


    # Kara za bliskość przeszkód
    for radar in car.radars:
        if radar[1] < proximity_penalty_threshold:  # Jeśli odległość od przeszkody mniejsza niż próg
            genome.fitness += proximity_penalty_factor * (proximity_penalty_threshold - radar[1])
            # Nagroda za zmniejszenie prędkości w pobliżu przeszkód
            if car.speed < max_speed / 2:  # Jeśli prędkość jest niższa niż połowa maksymalnej
                genome.fitness += slow_down_reward

    # Nagroda za kierowanie się do celu
    car_to_target_vector = (target_x - car.center[0], target_y - car.center[1])
    car_direction_vector = (
        math.cos(math.radians(360 - car.angle)),
        math.sin(math.radians(360 - car.angle))
    )
    dot_product = (car_to_target_vector[0] * car_direction_vector[0] +
                   car_to_target_vector[1] * car_direction_vector[1])
    magnitude_car_to_target = math.sqrt(car_to_target_vector[0]**2 + car_to_target_vector[1]**2)
    magnitude_car_direction = math.sqrt(car_direction_vector[0]**2 + car_direction_vector[1]**2)

    if magnitude_car_to_target > 0 and magnitude_car_direction > 0:  # Unikaj dzielenia przez zero
        cos_theta = dot_product / (magnitude_car_to_target * magnitude_car_direction)
        direction_reward = direction_reward_factor * cos_theta * current_distance
        genome.fitness += direction_reward
    # Kara za kolizję
    if not car.is_alive() and not car.collision_penalized:
        genome.fitness -= 10000  # Kara za kolizję
        car.collision_penalized = True

def run_simulation(genome):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    game_map = pygame.image.load('map3.jpg').convert()

    car = Car()
    car.alive = True

    clock = pygame.time.Clock()

    while car.is_alive():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)

        car_data = car.get_data()  # Zawiera dane z radarów i odległość do celu
        # Dodaj współrzędne robota i celu do danych wejściowych
        robot_x, robot_y = car.center[0], car.center[1]
        input_data = car_data + [robot_x, robot_y, target_x, target_y]

        output = net.activate(input_data)
        choice = output.index(max(output))
        if choice == 0:
            car.angle += 10
        elif choice == 1:
            car.angle -= 10
        elif choice == 2 and car.speed - 2 >= 12:
            car.speed -= 2
        else:
            car.speed += 2

        car.update(game_map)

        calculate_fitness(car, genome)

        distance_to_target = car.distance_to_target
        if distance_to_target < 300:  # You can adjust the threshold as needed
            print("Car reached the target!")
            car.speed = 0
            #wstrzymaj symulację do czasu naciesniecia esc
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        sys.exit(0)
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            sys.exit(0)
                pygame.display.flip()
                clock.tick(60)





        screen.blit(game_map, (0, 0))
        pygame.draw.circle(screen, (0, 0, 255, 255), (target_x, target_y), 20)
        car.draw(screen)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    # Load the saved best genome
    with open("best_genomeAAA.pkl", "rb") as f:
        best_genome = pickle.load(f)

    config_path = "config2A.txt"
    config = neat.config.Config(neat.DefaultGenome,
                                neat.DefaultReproduction,
                                neat.DefaultSpeciesSet,
                                neat.DefaultStagnation,
                                config_path)

    run_simulation(best_genome)
