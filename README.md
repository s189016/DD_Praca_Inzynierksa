# DD_Praca_Inzynierksa

## Instrukcja uruchomienia oraz opis rozwiązań

### Algorytm NEAT

Aby uruchomić program trenowania sieci neuronowej do wyznaczania trajektorii, należy upewnić się, że na komputerze zainstalowano Python w wersji co najmniej 3.8 oraz wymagane biblioteki: `Pygame`, `neat-python` i `pickle`. Struktura plików powinna obejmować trzy kluczowe elementy: `car.png` (obraz samochodu), `map4.jpg` (mapa symulacji) oraz `config2.txt` (plik konfiguracyjny dla algorytmu NEAT). Skrypt uruchamia się poleceniem `python newcar.py`, gdzie `newcar.py` to nazwa pliku zawierającego kod. Jeśli w katalogu znajdują się zapisane stany populacji NEAT, takie jak `genome-278`, program automatycznie wznowi symulację od zapisanego punktu.

Kod zawiera klasę `Car`, która reprezentuje pojazd poruszający się po mapie. Obiekt pojazdu posiada atrybuty określające jego pozycję, prędkość, kąt obrotu oraz stan (czy wciąż jest aktywny). Klasa zarządza również radarami, które mierzą odległości od przeszkód, oraz historią zmian kąta obrotu, co pozwala na wykrywanie niepożądanego obracania się w miejscu. Samochód rysowany jest na ekranie w aktualnej pozycji, z uwzględnieniem obrotu, a jego sensory wizualizowane są jako linie wychodzące z jego środka.

Funkcja `calculate_fitness` odpowiada za ocenę jakości działania każdego genomu. Oceniane są między innymi: zmniejszanie dystansu do celu, unikanie kolizji, zachowanie minimalnej odległości od przeszkód oraz brak objawów kręcenia się w miejscu. Jeżeli pojazd osiągnie cel, genom otrzymuje znaczną nagrodę, a symulacja dla tego pojazdu zostaje zakończona. Kolizje i nieefektywne zachowanie karane są obniżeniem wyniku fitness.

Pętla symulacji została zaimplementowana w funkcji `run_simulation`. W jej ramach tworzona jest wizualizacja, a następnie modele sterujące pojazdami są aktywowane w celu podejmowania decyzji. Sieci neuronowe sterują pojazdem, dostarczając mu polecenia skrętu, przyspieszania lub zwalniania na podstawie danych wejściowych, takich jak odczyty z radarów oraz odległość od celu. Symulacja kończy się, gdy wszystkie pojazdy przestają być aktywne lub po upływie określonego czasu (domyślnie 20 sekund).

Konfiguracja algorytmu NEAT znajduje się w pliku `config2.txt`. Plik ten określa liczbę wejść i wyjść dla sieci neuronowej, parametry mutacji oraz inne ustawienia wpływające na proces ewolucji. Możesz zmieniać jego zawartość, aby dostosować działanie algorytmu do swoich potrzeb, np. modyfikując maksymalną liczbę generacji lub zakres możliwych mutacji.

Do wznawiania treningu wykorzystywane są pliki checkpoint, zapisywane automatycznie co 30 generacji. Najlepsze genomy można zapisać w pliku `.pkl`, co pozwala na ich późniejsze użycie do testowania na innych mapach.

Zmiany w zachowaniu symulacji można wprowadzać modyfikując funkcję `calculate_fitness`, aby eksperymentować z różnymi nagrodami i karami. Możesz także edytować klasę `Car`, aby dodać nowe sensory lub zmienić sposób wykrywania kolizji. Wszelkie testy debugowe warto przeprowadzać, wstawiając dodatkowe komunikaty w miejscach takich jak `calculate_fitness` czy `run_simulation`.

Wizualizacja w Pygame pozwala na monitorowanie stanu symulacji w czasie rzeczywistym. Na ekranie widoczne są samochody, cel zaznaczony na niebiesko, a także liczniki generacji i liczby aktywnych pojazdów.

Projekt został zaprojektowany w sposób modularny, umożliwiając łatwe dodawanie nowych funkcjonalności lub modyfikację istniejących.

Aby odtworzyć najlepszy przejazd po nauczaniu, należy w taki sam sposób uruchomić plik `best.py`, który zwizualizuje najlepszy przejazd, wczytując wcześniej zapisany genom. Wizualizacja polega na odtworzeniu najlepszego przejazdu oraz narysowaniu za nim jego trajektorii, co pozwala na lepsze zrozumienie zachowań modelu i jego wydajności w realnych warunkach.

### Algorytm HyA*

Aby uruchomić program `path.py`, konieczne jest posiadanie zainstalowanego MATLAB-a z dodatkami `Automated Driving Toolbox` oraz `Computer Vision Toolbox`. Plik `parkingLotPCMapPoints.mat`, zawierający chmurę punktów, pochodzi z wbudowanego pakietu MATLAB-a i musi znajdować się w tym samym folderze co kod źródłowy. Kod należy zapisać jako plik `.m`, a następnie uruchomić go bezpośrednio w MATLAB-ie.

Do edycji parametrów, takich jak punkty startowe i końcowe, należy zmodyfikować zmienne `startPose` i `goalPose` w odpowiednich sekcjach kodu. Fragment odpowiedzialny za wyznaczanie tras znajduje się od linii z inicjalizacją obiektu `plannerHybridAStar`. Jeśli mapa kosztów wymaga dostosowania, należy zmienić parametry w sekcji `vehicleCostmap`, takie jak `cellSize` lub `InflationRadius`.

Całość programu wizualizuje poszczególne etapy analizy danych, od wstępnej filtracji, poprzez tworzenie mapy NDT i mapy kosztów, aż po generowanie tras za pomocą algorytmu Hybrid A*. Wszystkie wyniki są prezentowane w oddzielnych oknach graficznych.

### Porównanie algorytmów w labiryncie 3D

Program `porownanie.py` wymaga zainstalowania bibliotek Python: `numpy`, `networkx`, `matplotlib`, `open3d` oraz `time`. Dodatkowo wymagany jest plik `scena_test.PLY`, który zawiera chmurę punktów reprezentującą przeszkody w przestrzeni 3D. Uruchomienie programu odbywa się za pomocą polecenia `python porownanie.py`. Po poprawnym wykonaniu programu zostanie wygenerowana wizualizacja porównująca wyniki działania algorytmów PRM (Probabilistic Roadmap) oraz A* (A-star). Wizualizacja zawiera dwa panele 3D, które przedstawiają trajektorie generowane przez oba algorytmy wraz z przeszkodami i zaznaczonymi punktami startowym oraz końcowym.

Program został podzielony na dwie główne klasy: `DiscretePRM` i `AStar`, które realizują odpowiednio algorytmy PRM i A*. Klasa `DiscretePRM` pozwala na budowę grafu probabilistycznego w przestrzeni trójwymiarowej. Kluczowymi parametrami tej klasy są `start`, `goal`, `obstacles`, `resolution`, `num_points` i `max_distance`. Parametr `start` definiuje współrzędne punktu początkowego, `goal` odnosi się do punktu końcowego, a `obstacles` zawiera przeszkody. Do kluczowych metod klasy `DiscretePRM` należy `build_graph`, która generuje graf, sprawdzając możliwe połączenia między węzłami bez kolizji, oraz `find_path`, odpowiedzialna za znalezienie najkrótszej ścieżki w grafie. Metoda `scale_back_path` umożliwia skalowanie współrzędnych ścieżki z grafu z powrotem do rzeczywistego układu współrzędnych.

Klasa `AStar` implementuje algorytm A* w przestrzeni 3D. Analogiczne parametry `start`, `goal`, `obstacles` oraz `resolution` kontrolują pozycję punktów, przeszkody oraz dokładność dyskretnej reprezentacji przestrzeni. Kluczowe metody to `get_neighbors`, która generuje sąsiadujące węzły w przestrzeni, oraz `run`, która wykonuje logikę algorytmu A*, wykorzystując heurystykę bazującą na odległości euklidesowej do celu. Metoda `scale_back_path` pozwala na przeskalowanie współrzędnych ścieżki do rzeczywistego układu współrzędnych.

Wizualizacja wyników realizowana jest za pomocą biblioteki `matplotlib`. Dodatkowo, w pliku `scena_test.PLY` wprowadza się dane dotyczące przeszkód w przestrzeni trójwymiarowej, które są następnie wyświetlane na wykresie.

### Testowanie i optymalizacja

Testowanie każdego z algorytmów odbywa się na różnych zestawach danych, które reprezentują różne trudności w poruszaniu się po labiryncie 3D. Na podstawie wyników analizy czasu wykonywania oraz jakości znalezionych trajektorii, możliwe jest dobranie odpowiednich algorytmów do różnych zastosowań. Optymalizacja kodu może obejmować poprawę metod generowania sąsiadów w przestrzeni, optymalizację algorytmu A* oraz przyspieszenie procesu budowy grafu w PRM.
