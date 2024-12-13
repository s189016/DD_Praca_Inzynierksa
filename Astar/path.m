clear;

% Wczytanie danych mapy 3D
mapData = load("dMapCityBlock.mat");
omap = mapData.omap;

% Inflatowanie mapy
inflate(omap, 1);

% Wczytanie danych zajętości z omap (do siatki 3D)
occupancyMatrix = readOccupancy(omap);

% Start i cel w 3D
start = [40, 180, 25];
goal = [150, 33, 35];

% Rozdzielczość mapy
resolution = omap.Resolution;

% Konwersja współrzędnych na indeksy siatki
startIdx = round(start * resolution);
goalIdx = round(goal * resolution);

% Algorytm A*
[path, pathFound] = aStar3D(occupancyMatrix, startIdx, goalIdx);

% Wyświetlanie wyników
if pathFound
    figure;
    show(omap);
    hold on;
    plot3(path(:, 1) / resolution, path(:, 2) / resolution, path(:, 3) / resolution, ...
        'r-', 'LineWidth', 2);
    scatter3(start(1), start(2), start(3), "g", "filled"); % Start state
    scatter3(goal(1), goal(2), goal(3), "r", "filled");   % Goal state
    title('3D A* Path Planning');
    xlabel('X (m)');
    ylabel('Y (m)');
    zlabel('Z (m)');
    hold off;
else
    error("No path found using A* in 3D space.");
end

function [path, found] = aStar3D(occupancyMatrix, startIdx, goalIdx)
    % Funkcja A* dla przestrzeni 3D

    % Inicjalizacja
    gridSize = size(occupancyMatrix);
    openSet = [startIdx, 0, heuristic(startIdx, goalIdx)]; % [x, y, z, g, f]
    cameFrom = containers.Map;
    gScore = inf(gridSize);
    gScore(startIdx(1), startIdx(2), startIdx(3)) = 0;

    % Ruchy 3D (sąsiedzi w 6 kierunkach)
    neighbors = [-1 0 0; 1 0 0; 0 -1 0; 0 1 0; 0 0 -1; 0 0 1];

    while ~isempty(openSet)
        % Wybierz węzeł z najniższym f-score
        [~, idx] = min(openSet(:, 5));
        current = openSet(idx, 1:3);
        openSet(idx, :) = [];

        % Sprawdź, czy osiągnięto cel
        if isequal(current, goalIdx)
            path = reconstructPath(cameFrom, current);
            found = true;
            return;
        end

        % Przeglądaj sąsiadów
        for i = 1:size(neighbors, 1)
            neighbor = current + neighbors(i, :);
            if any(neighbor < 1 | neighbor > gridSize) || ...
                    occupancyMatrix(neighbor(1), neighbor(2), neighbor(3)) > 0.5
                continue;
            end

            tentativeGScore = gScore(current(1), current(2), current(3)) + 1;
            if tentativeGScore < gScore(neighbor(1), neighbor(2), neighbor(3))
                gScore(neighbor(1), neighbor(2), neighbor(3)) = tentativeGScore;
                fScore = tentativeGScore + heuristic(neighbor, goalIdx);
                openSet = [openSet; neighbor, tentativeGScore, fScore]; %#ok<AGROW>
                cameFrom(mat2str(neighbor)) = current;
            end
        end
    end

    % Jeśli ścieżka nie została znaleziona
    path = [];
    found = false;
end

function h = heuristic(node, goal)
    % Heurystyka: odległość Manhattan w 3D
    h = sum(abs(node - goal));
end

function path = reconstructPath(cameFrom, current)
    % Rekonstrukcja ścieżki
    path = current;
    while cameFrom.isKey(mat2str(current))
        current = cameFrom(mat2str(current));
        path = [current; path]; %#ok<AGROW>
    end
end
