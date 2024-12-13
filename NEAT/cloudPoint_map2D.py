import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# Ścieżka do pliku .ply z siatką trójkątów
mesh_path = "scena_test_3.PLY"  # Zmień na ścieżkę do swojego pliku .ply

# Wczytywanie siatki trójkątów
try:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    print("Wczytano siatkę trójkątów z pliku:", mesh_path)
except Exception as e:
    print("Nie udało się wczytać siatki. Błąd:", e)
    mesh = None

if mesh is not None:
    # Obliczanie normalnych dla siatki
    mesh.compute_vertex_normals()

    # Próba wygenerowania chmury punktów metodą Poissona
    pcl = mesh.sample_points_poisson_disk(number_of_points=20000)

    # Konwersja chmury punktów na format numpy
    points = np.asarray(pcl.points)

    # Filtracja poziomu podłogi i ograniczenie do dolnych 10% wysokości chmury punktów
    min_z = np.min(points[:, 2])  # Wysokość podłogi (minimalna wartość Z)
    max_z = np.max(points[:, 2])  # Maksymalna wysokość chmury punktów
    floor_offset = 0.2  # Offset nad poziomem podłogi (regulacja w metrach lub jednostkach pliku)
    height_threshold = min_z + floor_offset + 0.1 * (max_z - min_z)  # Górna granica dla dolnych 10% wysokości

    # Filtracja punktów znajdujących się w zakresie dolnych 10% wysokości, z offsetem od podłogi
    filtered_points = points[(points[:, 2] > min_z + floor_offset) & (points[:, 2] <= height_threshold)]

    # Wyświetlenie chmury punktów po filtracji
    pcl_filtered = o3d.geometry.PointCloud()
    pcl_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    o3d.visualization.draw_geometries([pcl_filtered], window_name="Filtered Point Cloud", width=1920, height=1080)

    # Przykładowy zapis chmury punktów do pliku
    o3d.io.write_point_cloud("filtered_cloud.ply", pcl_filtered)
    print("Przefiltrowana chmura punktów zapisana jako 'filtered_cloud.ply'.")

    # Konfiguracja obrazu .jpg
    image_size = (1080, 1920)  # Rozdzielczość obrazu
    goal_position = (int(image_size[1] * 0.85), int(image_size[0] * 0.45))  # Pozycja celu na obrazie
    goal_radius = 10  # Rozmiar celu podróży

    # Normalizacja i skalowanie punktów 2D do rozmiaru obrazu
    x_points = ((filtered_points[:, 0] - np.min(filtered_points[:, 0])) /
                (np.max(filtered_points[:, 0]) - np.min(filtered_points[:, 0])) * (image_size[1] - 1)).astype(int)
    y_points = ((filtered_points[:, 1] - np.min(filtered_points[:, 1])) /
                (np.max(filtered_points[:, 1]) - np.min(filtered_points[:, 1])) * (image_size[0] - 1)).astype(int)

    # Generowanie czarnego tła
    image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Wzór pikseli, który przypomina krzyż z przerwami
    pattern_offsets = [
        (0, 0),   # Centralny punkt
        (-2, 0),  # Lewo
        (2, 0),   # Prawo
        (0, -2),  # Góra
        (0, 2),   # Dół
        (-1, -1), # Skos lewy-górny
        (-1, 1),  # Skos lewy-dolny
        (1, -1),  # Skos prawy-górny
        (1, 1)    # Skos prawy-dolny
    ]

    # Rysowanie przeszkód z użyciem wzoru
    for x, y in zip(x_points, y_points):
        for dx, dy in pattern_offsets:
            px, py = x + dx, y + dy
            if 0 <= py < image_size[0] and 0 <= px < image_size[1]:
                image[py, px] = [255, 255, 255]

    # Wyszukiwanie najbliższych sąsiadów dla połączeń między punktami
    points_2d = np.column_stack((x_points, y_points))
    tree = cKDTree(points_2d)
    pairs = tree.query_pairs(r=40)  # Maksymalny zasięg połączenia między punktami (regulowany w pikselach)

    # Rysowanie linii między najbliższymi punktami
    for i, j in pairs:
        x1, y1 = points_2d[i]
        x2, y2 = points_2d[j]
        rr, cc = np.linspace(y1, y2, num=100, dtype=int), np.linspace(x1, x2, num=100, dtype=int)
        image[rr, cc] = [255, 255, 255]

    # Dodanie celu podróży jako niebieskiego punktu
    for y in range(image_size[0]):
        for x in range(image_size[1]):
            if (x - goal_position[0]) ** 2 + (y - goal_position[1]) ** 2 <= goal_radius ** 2:
                image[y, x] = [0, 0, 255]

    # Zapisanie obrazu jako .jpg
    plt.imsave("room_projection_with_goal.jpg", image)
    print("Obraz zapisany jako 'room_projection_with_goal.jpg'.")

    # Wyświetlenie obrazu
    plt.imshow(image)
    plt.axis('off')
    plt.show()

else:
    print("Operacja zakończona, brak wczytanej siatki.")
