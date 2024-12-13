import time
import numpy as np
import matplotlib.pyplot as plt
from queue import PriorityQueue
import networkx as nx
import os

def save_results_to_file(base_filename, results):
    i = 1
    while os.path.exists(f"{base_filename}-{i}.txt"):
        i += 1
    filename = f"{base_filename}-{i}.txt"
    with open(filename, 'w') as file:
        file.write(results)
    print(f"Results saved to {filename}")
def calculate_path_length(path):
    if path is None or len(path) < 2:
        return 0
    length = 0
    for i in range(len(path) - 1):
        length += np.linalg.norm(np.array(path[i]) - np.array(path[i + 1]))
    return length

np.random.seed(0)
points = np.random.rand(100, 3) * 10

obstacles = points

def plot_points(ax, points):
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', marker='o')

def check_collision(start, end, obstacles, threshold=0.5):
    for obs in obstacles:
        dist = np.linalg.norm(np.cross(end - start, start - obs)) / np.linalg.norm(end - start)
        if dist < threshold:
            return True
    return False

class PRM:
    def __init__(self, start, goal, num_points=150, k=10, max_distance=2.5):
        self.start = start
        self.goal = goal
        self.num_points = num_points
        self.k = k
        self.max_distance = max_distance
        self.nodes = [start, goal]
        self.edges = []

    def generate_random_points(self):
        return np.random.rand(self.num_points, 3) * 10

    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def build_graph(self):
        graph = nx.Graph()
        graph.add_node(tuple(self.start))
        graph.add_node(tuple(self.goal))

        random_points = self.generate_random_points()
        for point in random_points:
            self.nodes.append(point)
            graph.add_node(tuple(point))

        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i < j and self.distance(node1, node2) <= self.max_distance:
                    graph.add_edge(tuple(node1), tuple(node2))

        return graph

    def find_path(self, graph):
        start_node = tuple(self.start)
        goal_node = tuple(self.goal)
        try:
            path = nx.shortest_path(graph, source=start_node, target=goal_node)
        except nx.NetworkXNoPath:
            path = None
        return path

class RRT:
    def __init__(self, start, goal, max_iter=1000, step_size=10.0, obstacles=obstacles):
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = [start]
        self.edges = []
        self.obstacles = obstacles if obstacles is not None else []

    def get_nearest_node(self, point):
        distances = np.linalg.norm(np.array(self.nodes) - point, axis=1)
        return np.argmin(distances)

    def step(self, node, goal):
        direction = np.array(goal) - np.array(node)
        norm = np.linalg.norm(direction)
        if norm > self.step_size:
            direction = direction / norm * self.step_size
        new_node = node + direction
        return new_node

    def run(self, max_nodes=150):
        for _ in range(self.max_iter):
            if len(self.nodes) >= max_nodes:
                break

            random_point = np.random.rand(3) * 10
            nearest_node_idx = self.get_nearest_node(random_point)
            nearest_node = self.nodes[nearest_node_idx]
            new_node = self.step(nearest_node, random_point)

            if not check_collision(nearest_node, new_node, self.obstacles):
                self.nodes.append(new_node)
                self.edges.append((nearest_node, new_node))

                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < self.step_size:
                    self.nodes.append(self.goal)
                    self.edges.append((new_node, self.goal))
                    break

        return self.nodes, self.edges

class BiRRT:
    def __init__(self, start, goal, max_iter=1000, step_size=4.0, obstacles=obstacles):
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.start_nodes = [start]
        self.goal_nodes = [goal]
        self.start_edges = []
        self.goal_edges = []
        self.obstacles = obstacles if obstacles is not None else []

    def get_nearest_node(self, point, tree_nodes):
        distances = np.linalg.norm(np.array(tree_nodes) - point, axis=1)
        return np.argmin(distances)

    def step(self, node, goal):
        direction = np.array(goal) - np.array(node)
        norm = np.linalg.norm(direction)
        if norm > self.step_size:
            direction = direction / norm * self.step_size
        new_node = node + direction
        return new_node

    def run(self):
        for _ in range(self.max_iter):
            random_point = np.random.rand(3) * 10
            nearest_start_idx = self.get_nearest_node(random_point, self.start_nodes)
            nearest_goal_idx = self.get_nearest_node(random_point, self.goal_nodes)

            new_start_node = self.step(self.start_nodes[nearest_start_idx], random_point)
            new_goal_node = self.step(self.goal_nodes[nearest_goal_idx], random_point)

            if not check_collision(self.start_nodes[nearest_start_idx], new_start_node, self.obstacles):
                self.start_nodes.append(new_start_node)
                self.start_edges.append((self.start_nodes[nearest_start_idx], new_start_node))

            if not check_collision(self.goal_nodes[nearest_goal_idx], new_goal_node, self.obstacles):
                self.goal_nodes.append(new_goal_node)
                self.goal_edges.append((self.goal_nodes[nearest_goal_idx], new_goal_node))

            if np.linalg.norm(np.array(new_start_node) - np.array(new_goal_node)) < self.step_size:
                self.start_nodes.append(new_goal_node)
                self.start_edges.append((new_start_node, new_goal_node))
                break

        return self.start_nodes, self.start_edges, self.goal_nodes, self.goal_edges

class RRTStar:
    def __init__(self, start, goal, max_iter=1000, step_size=10.0, obstacles=obstacles, radius=10):
        self.start = start
        self.goal = goal
        self.max_iter = max_iter
        self.step_size = step_size
        self.nodes = [start]
        self.edges = []
        self.obstacles = obstacles if obstacles is not None else []
        self.radius = radius

    def get_nearest_node(self, point):
        distances = np.linalg.norm(np.array(self.nodes) - point, axis=1)
        return np.argmin(distances)

    def step(self, node, goal):
        direction = np.array(goal) - np.array(node)
        norm = np.linalg.norm(direction)
        if norm > self.step_size:
            direction = direction / norm * self.step_size
        new_node = node + direction
        return new_node

    def rewire(self, new_node):
        for node in self.nodes:
            if np.linalg.norm(np.array(new_node) - np.array(node)) <= self.radius:
                if not check_collision(new_node, node, self.obstacles):
                    self.edges.append((new_node, node))

    def run(self):
        for _ in range(self.max_iter):
            random_point = np.random.rand(3) * 10
            nearest_node_idx = self.get_nearest_node(random_point)
            nearest_node = self.nodes[nearest_node_idx]
            new_node = self.step(nearest_node, random_point)

            if not check_collision(nearest_node, new_node, self.obstacles):
                self.nodes.append(new_node)
                self.edges.append((nearest_node, new_node))
                self.rewire(new_node)

                if np.linalg.norm(np.array(new_node) - np.array(self.goal)) < self.step_size:
                    self.nodes.append(self.goal)
                    self.edges.append((new_node, self.goal))
                    break

        return self.nodes, self.edges

def plot_trajectory(ax, path, color='red'):
    if path:
        for i in range(len(path) - 1):
            node1 = np.array(path[i])
            node2 = np.array(path[i + 1])
            ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]], color=color)

class AStar:
    def __init__(self, start, goal, obstacles, grid_size=10, resolution=1.0):
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.obstacles = {tuple(np.round(o / resolution) * resolution) for o in obstacles}
        self.grid_size = grid_size
        self.resolution = resolution

    def heuristic(self, node):
        return np.linalg.norm(np.array(node) - np.array(self.goal))

    def get_neighbors(self, node):
        directions = [
            (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1), (1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (-1, 0, -1), (0, 1, 1), (0, -1, -1),
            (1, -1, 0), (-1, 1, 0), (0, 1, -1), (0, -1, 1),
            (1, 1, 1), (-1, -1, -1), (1, -1, -1), (-1, 1, 1),
            (1, -1, 1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1)
        ]
        neighbors = []
        for direction in directions:
            neighbor = tuple(np.round(np.array(node) + np.array(direction) * self.resolution, decimals=2))
            if all(0 <= n < self.grid_size for n in neighbor) and neighbor not in self.obstacles:
                neighbors.append(neighbor)
        return neighbors

    def run(self):
        open_set = PriorityQueue()
        open_set.put((0, self.start))
        came_from = {}
        g_score = {self.start: 0}
        f_score = {self.start: self.heuristic(self.start)}

        while not open_set.empty():
            _, current = open_set.get()
            if current == self.goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(self.start)
                return path[::-1]

            for neighbor in self.get_neighbors(current):
                tentative_g_score = g_score[current] + np.linalg.norm(np.array(current) - np.array(neighbor))
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor)
                    open_set.put((f_score[neighbor], neighbor))

        return None


start = np.array([0, 0, 0])
goal = np.array([9, 9, 9])

obstacles = points

start_time = time.time()
prm = PRM(start, goal)
graph = prm.build_graph()
path_prm = prm.find_path(graph)
end_time = time.time()
time_prm = (end_time - start_time) * 1000
length_prm = calculate_path_length(path_prm)


start_time = time.time()
rrt = RRT(start, goal, obstacles=obstacles)
nodes_rrt, edges_rrt = rrt.run()
end_time = time.time()
time_rrt = (end_time - start_time) * 1000
length_rrt = calculate_path_length(nodes_rrt)

start_time = time.time()
rrt_star = RRTStar(start, goal, obstacles=obstacles)
nodes_rrt_star, edges_rrt_star = rrt_star.run()
end_time = time.time()
time_rrt_star = (end_time - start_time) * 1000
length_rrt_star = calculate_path_length(nodes_rrt_star)

start_time = time.time()
birrt = BiRRT(start, goal, obstacles=obstacles)
nodes_birrt_start, edges_birrt_start, nodes_birrt_goal, edges_birrt_goal = birrt.run()
end_time = time.time()
time_birrt = (end_time - start_time) * 1000
length_birrt = calculate_path_length(nodes_birrt_start) + calculate_path_length(nodes_birrt_goal)

start_time = time.time()
astar = AStar(start, goal, obstacles)
path_astar = astar.run()
end_time = time.time()
time_astar = (end_time - start_time) * 1000
length_astar = calculate_path_length(path_astar)

fig = plt.figure(figsize=(11, 15))

ax_prm = fig.add_subplot(231, projection='3d')
plot_points(ax_prm, points)
plot_trajectory(ax_prm, path_prm, color='blue')
ax_prm.scatter(start[0], start[1], start[2], color='green', label='Start')
ax_prm.scatter(goal[0], goal[1], goal[2], color='orange', label='Goal')
ax_prm.set_xlabel('X')
ax_prm.set_ylabel('Y')
ax_prm.set_zlabel('Z')
ax_prm.set_title('PRM Path')
ax_prm.legend()

ax_rrt = fig.add_subplot(232, projection='3d')
plot_points(ax_rrt, points)
plot_trajectory(ax_rrt, nodes_rrt, color='red')
ax_rrt.scatter(start[0], start[1], start[2], color='green', label='Start')
ax_rrt.scatter(goal[0], goal[1], goal[2], color='orange', label='Goal')
ax_rrt.set_xlabel('X')
ax_rrt.set_ylabel('Y')
ax_rrt.set_zlabel('Z')
ax_rrt.set_title('RRT Path')
ax_rrt.legend()

ax_rrt_star = fig.add_subplot(233, projection='3d')
plot_points(ax_rrt_star, points)
plot_trajectory(ax_rrt_star, nodes_rrt_star, color='green')
ax_rrt_star.scatter(start[0], start[1], start[2], color='green', label='Start')
ax_rrt_star.scatter(goal[0], goal[1], goal[2], color='orange', label='Goal')
ax_rrt_star.set_xlabel('X')
ax_rrt_star.set_ylabel('Y')
ax_rrt_star.set_zlabel('Z')
ax_rrt_star.set_title('RRT* Path')
ax_rrt_star.legend()


ax_birrt = fig.add_subplot(234, projection='3d')
plot_points(ax_birrt, points)
plot_trajectory(ax_birrt, nodes_birrt_start, color='blue')
plot_trajectory(ax_birrt, nodes_birrt_goal, color='green')
ax_birrt.scatter(start[0], start[1], start[2], color='green', label='Start')
ax_birrt.scatter(goal[0], goal[1], goal[2], color='orange', label='Goal')
ax_birrt.set_xlabel('X')
ax_birrt.set_ylabel('Y')
ax_birrt.set_zlabel('Z')
ax_birrt.set_title('BiRRT Path')
ax_birrt.legend()

ax_astar = fig.add_subplot(235, projection='3d')
plot_points(ax_astar, points)
if path_astar:
    plot_trajectory(ax_astar, path_astar, color='purple')
ax_astar.scatter(start[0], start[1], start[2], color='green', label='Start')
ax_astar.scatter(goal[0], goal[1], goal[2], color='orange', label='Goal')
ax_astar.set_xlabel('X')
ax_astar.set_ylabel('Y')
ax_astar.set_zlabel('Z')
ax_astar.set_title('A* Path')
ax_astar.legend()

plt.tight_layout()
plt.show()

results = (
    f"PRM: Time = {time_prm:.2f} ms, Path Length = {length_prm:.2f}\n"
    f"RRT: Time = {time_rrt:.2f} ms, Path Length = {length_rrt:.2f}\n"
    f"RRT*: Time = {time_rrt_star:.2f} ms, Path Length = {length_rrt_star:.2f}\n"
    f"BiRRT: Time = {time_birrt:.2f} ms, Path Length = {length_birrt:.2f}\n"
    f"A*: Time = {time_astar:.2f} ms, Path Length = {length_astar:.2f}\n"
)

save_results_to_file("wyniki-", results)