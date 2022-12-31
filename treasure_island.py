# %% [markdown]
# # Treasure Island

# %% [markdown]
# ## Import packages

# %%
import numpy as np
from numpy import typing as npt
from scipy.ndimage.morphology import binary_dilation

from typing import List, Tuple

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

from collections import deque

import random
import math
from string import digits
import copy
import csv

from datetime import datetime

# %%
# ### Set a Random Number Generator

# %%
rng = np.random.RandomState(70)

# %%
class UserInterface:
    pass

# %% [markdown]
# ## Map Generator

# %%
class MapGenerator:
    def __init__(
        self,
        rows: int,
        cols: int
    ) -> None:
        self.rows = rows
        self.cols = cols
        avg = (rows + cols)/2
        self.number_of_region = int(0.000966*avg**2 + 0.31935*avg + 5/3)
        self.number_of_prison = int(-0.0008345*avg**2 + 0.1467*avg + 1.06)

        self.init_land_chance = 0.1                     #Probability of cells to be seeded as 'land terrain', range 0 to 1
        self.init_sea_chance = 0.04                     #Probability of cells to be seeded as 'sea terrain', range 0 to 1
        self.init_mountain_chance = 0.04                #Probability of 'land terrain' cells to be seeded as 'mountain terrain', range 0 to 1
        self.init_region_chance = 0.0007                #Probability of 'land terrain' cells to be seeded as a region seed, range 0 to 1

        self.land_chance = 0.055                        #Probability of cells to become 'land terrain' for each adjacent 'land terrain' cell, range 0 to 1
        self.sea_chance = 0.022                         #Probability of cells to become 'sea terrain' for each adjacent 'sea terrain' cell, range 0 to 1
        self.border_sea_chance = 0.08                   #Probability of cells on the edges of the map to become 'sea terrain', range -1 to 1
        self.mountain_chance = 0.01                     #Probability of 'land terrain' cells to become 'mountain terrain' for each adjacent 'mountain terrain' cell, range 0 to 1
        self.sea_mountain_chance = -0.01                #Probability of 'land terrain' cells to become 'mountain terrain' for each adjacent 'sea terrain' cell, range -1 to 1
        self.mountain_amplifier = 12                    #Increase to have larger mountain ranges, range >= 0 
        self.Map = [['.']*self.rows for _ in range(self.cols)]
        self.region_map = [[0]*self.rows for _ in range(self.cols)]
        self.mountain_map = [[0]*self.rows for _ in range(self.cols)]

    def neighbors(self, a, radius, row_number, column_number):
     return [[a[i][j] if  i >= 0 and i < len(a) and j >= 0 and j < len(a[0]) else '|'
                for j in range(column_number-radius, column_number+radius+1)]
                    for i in range(row_number-radius, row_number+radius+1)]
    
    def map_print(self):
        for coord_x, row in enumerate(self.Map):
            for coord_y, terrain in enumerate(row):
                cur = self.Map[coord_x][coord_y]
                spaces = 3 - len(str(cur))
                symbol = ' ' * spaces + str(cur)
                if cur == '_' or cur in range(1, self.number_of_region + 1):                
                    print('\025[92m', symbol, end='')
                elif cur == 0:                
                    print('\033[96m', symbol, end='')
                elif cur == 'M':                
                    print('\033[91m', symbol, end='')
                elif cur == 'p':                
                    print('\033[93m', symbol, end='')
                else:
                    print('\033[97m', symbol, end='')
            print()
        print('\033[97m')
            
    def get_neighbour_terrain(self, area):
        land = 0
        sea = 0
        for coord_x, row in enumerate(area):
            for coord_y, terrain in enumerate(row):
                if area[coord_x][coord_y] == '_':
                    land += self.land_chance
                elif area[coord_x][coord_y] == 0:
                    sea += self.sea_chance
                elif(area[coord_x][coord_y] == '|'):
                    sea += self.border_sea_chance
        
        chance = rng.uniform(0,1)
        if chance <= land:
            return '_'
        elif chance >= 1 - sea:
            return 0
        else:
            return '.'
    
    def get_neighbour_mountain(self, area):
        mountain = 0
        for coord_x, row in enumerate(area):
            for coord_y, terrain in enumerate(row):
                if area[coord_x][coord_y] == 'M':
                    mountain += self.mountain_chance
                if area[coord_x][coord_y] == 0:
                    mountain += self.sea_mountain_chance
        
        chance = rng.uniform(0,1)
        if chance <= mountain:
            return 'M'
        else:
            return area[1][1]

    def get_neighbour_region(self, area):
        region = [0 for i in range(self.number_of_region)]
        for coord_x, row in enumerate(area):
            for coord_y, terrain in enumerate(row):
                if area[coord_x][coord_y] in range(1, self.number_of_region + 1):
                    num = area[coord_x][coord_y]
                    region[num - 1] += 0.1
                
        
        chance = rng.uniform(0,1)
        for r, c in enumerate(region):
            chance -= c
            if(chance <= 0):
                return r + 1

        return '_'

    def generate(self):
        Map = [['.']*self.rows for _ in range(self.cols)]

        for coord_x, row in enumerate(Map):
            for coord_y, terrain in enumerate(row):
                chance = rng.uniform(0,1)
                if chance <= self.init_land_chance:
                    Map[coord_x][coord_y] = '_'
                elif rng.uniform(0,1) >= 1 - self.init_sea_chance:
                    Map[coord_x][coord_y] = 0

        isFull = False

        while(isFull == False):
            isFull = True
            for coord_x, row in enumerate(Map):
                for coord_y, terrain in enumerate(row):
                    if Map[coord_x][coord_y] == '.':          
                        isFull = False
                        area = self.neighbors(Map, 1, coord_x, coord_y).copy()
                        Map[coord_x][coord_y] = self.get_neighbour_terrain(area)

        count = 1
        while(count <= self.number_of_region):
            for coord_x, row in enumerate(Map):
                for coord_y, terrain in enumerate(row):
                    chance = rng.uniform(0,1)
                    if Map[coord_x][coord_y] == '_' and chance <= self.init_region_chance and count <= self.number_of_region:
                        Map[coord_x][coord_y] = count
                        count += 1

        for i in range(0,(self.rows + self.cols) + 20):
            for coord_x, row in enumerate(Map):
                for coord_y, terrain in enumerate(row):     
                    if Map[coord_x][coord_y] == '_':
                        area = self.neighbors(Map, 1, coord_x, coord_y).copy()
                        Map[coord_x][coord_y] = self.get_neighbour_region(area)

        for coord_x, row in enumerate(Map):
            for coord_y, terrain in enumerate(row):     
                if Map[coord_x][coord_y] == '_':
                    Map[coord_x][coord_y] = 0
        
        for coord_x, row in enumerate(Map):
            for coord_y, terrain in enumerate(row):     
                if Map[coord_x][coord_y] == '_':
                    Map[coord_x][coord_y] = 0
        
        self.region_map = copy.deepcopy(list(map(list, zip(*Map))))

        for coord_x, row in enumerate(Map):
            for coord_y, terrain in enumerate(row):
                chance = rng.uniform(0,1)
                if Map[coord_x][coord_y] in range(1, self.number_of_region + 1) and chance <= self.init_mountain_chance:
                    Map[coord_x][coord_y] = 'M'

        for i in range(0, self.mountain_amplifier):
            for coord_x, row in enumerate(Map): 
                for coord_y, terrain in enumerate(row):     
                    if Map[coord_x][coord_y] in range(1, self.number_of_region + 1):
                        area = self.neighbors(Map, 1, coord_x, coord_y).copy()
                        Map[coord_x][coord_y] = self.get_neighbour_mountain(area)  

        count = 1

        while(count <= self.number_of_prison):
            for coord_x, row in enumerate(Map):
                for coord_y, terrain in enumerate(row):
                    chance = rng.uniform(0,1)
                    if Map[coord_x][coord_y] in range(1,self.number_of_region + 1) and chance <= 0.001 and count <= self.number_of_prison:
                        Map[coord_x][coord_y] = 'p'
                        count += 1

        for coord_x, row in enumerate(Map): 
                for coord_y, terrain in enumerate(row):
                    if Map[coord_x][coord_y] == 'M':
                        self.mountain_map[coord_x][coord_y] = 1
        self.mountain_map = list(map(list, zip(*self.mountain_map))).copy()                 
        self.Map = list(map(list, zip(*Map))).copy()
    
    def place_pirate(self):
        
        while(True):
            for coord_x, row in enumerate(self.Map):
                for coord_y, terrain in enumerate(row):
                    if self.Map[coord_x][coord_y] == 'p' and rng.uniform(0,1) <= 1/self.number_of_prison:
                        return coord_x, coord_y

    def place_agent(self):
        while(True):
            for coord_x, row in enumerate(self.Map):
                for coord_y, terrain in enumerate(row):
                    if self.Map[coord_x][coord_y] in range(1, self.number_of_region + 1) and rng.uniform(0,1) <= 0.001:
                        return coord_x, coord_y

    def place_treasure(self):
        while(True):
            for coord_x, row in enumerate(self.Map):
                for coord_y, terrain in enumerate(row):
                    if self.Map[coord_x][coord_y] in range(1, self.number_of_region + 1) and rng.uniform(0,1) <= 0.001:
                        return coord_x, coord_y

# %% [markdown]
# ## Map


# %%
class Agent:
    def __init__(self, coord: Tuple[int, int]) -> None:
        self.coord = coord
    

# %%
class JackSparrow(Agent):
    def __init__(self, coord: Tuple[int, int]) -> None:
        super().__init__(coord)

# %%
class Pirate(Agent):
    def __init__(self, coord: Tuple[int, int]) -> None:
        super().__init__(coord)

class Node:
    def __init__(self, pt: Tuple[int, int], step: int):
        self.pt = pt
        self.step = step

# %%
class Map:
    def __init__(self, map: MapGenerator):
        self.total_tile = map.rows * map.cols
        self.shape = (map.rows, map.cols)
        self.avg_size = (map.rows + map.cols) / 2
        self.total_region = map.number_of_region

        map.generate()

        self.value = np.array(map.Map, dtype=str)
        self.value[self.value == '0'] = '~'
        self.region = np.array(map.region_map, dtype=int)
        is_mountain = np.array(map.mountain_map, dtype=bool)
        self.mountain = np.unique(self.region[is_mountain])

        self.potential= np.ones((map.rows, map.cols), dtype=bool)
        self.potential[(self.region == 0) | (is_mountain)] = False

        self.jacksparrow = JackSparrow(map.place_agent())
        # self.value[self.jacksparrow.coord] = 'A'

        self.pirate = Pirate(map.place_pirate())
        # self.value[self.pirate.coord] = 'Pi'

        self.treasure = map.place_treasure()
        # self.value[self.treasure] = 'T'

        self.n_turns = 1

        self.logs = [[]]

        self.hint_list = []

        self.is_win = False
        self.is_lose = False
        self.is_teleported = False

        self.veri_important = {"1", "3", "5", "8"}

        free_turn = int(0.0032552083 * self.avg_size**2 + 0.15625 * self.avg_size + 0.68)
        self.free_turn = rng.randint(free_turn, free_turn + int(0.5 * free_turn))

        self.reveal_turn = self.free_turn // 2

        self.is_free = False
        self.logs[0].append("Game start")
        self.logs[0].append(f"Agent appears at {self.jacksparrow.coord}")

        self.logs[0].append("The pirate’s prison is going to reveal the coordinate")
        self.logs[0].append(f"at the beginning of turn number {self.reveal_turn}")
        self.logs[0].append(f"The pirate is free at the beginning of turn number {self.free_turn}")

        steps, pirate_path = self.shortest_path(self.pirate.coord, self.treasure)
        self.pirate_path = deque(pirate_path)

        # Map generate hints function to string
        self.hints = {"1": self.generate_hint_1, "2": self.generate_hint_2, "3": self.generate_hint_3, "4": self.generate_hint_4,
                      "5": self.generate_hint_5, "6": self.generate_hint_6, "7": self.generate_hint_7, "8": self.generate_hint_8,
                      "9": self.generate_hint_9, "10": self.generate_hint_10, "11": self.generate_hint_11, "12": self.generate_hint_12,
                      "13": self.generate_hint_13, "14": self.generate_hint_14, "15": self.generate_hint_15}

    def place_pirate(self):
        while(True):
            for coord_x, row in enumerate(self.value):
                for coord_y, terrain in enumerate(row):
                    if self.value[coord_x][coord_y] == 'p' and rng.uniform(0,1) <= 1/self.total_prison:
                        return coord_x, coord_y

    def place_agent(self):
        str_idx = [str(i) for i in range(1, self.total_region + 1)]
        while(True):
            for coord_x, row in enumerate(self.value):
                for coord_y, terrain in enumerate(row):
                    if self.value[coord_x][coord_y] in str_idx and rng.uniform(0,1) <= 0.001:
                        return coord_x, coord_y

    def import_map(self, path: str):
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter=';')
            read_map = []
            for row in reader:
                read_map.append(row)

            print(read_map)

        self.shape = int(read_map[0][0][0]), int(read_map[0][0][2])
        self.total_tile = self.shape[0] * self.shape[1]

        print("shape:", self.shape)

        self.avg_size = (self.shape[0] * self.shape[1]) / 2
        self.total_region = int(read_map[3][0]) - 1

        rows = int(read_map[0][0][0])

        raw_map = read_map[5:5 + rows]
        for rows in raw_map:
            for i, value in enumerate(rows):
                rows[i] = value.replace(' ','')

        self.value = np.empty(self.shape, dtype=str)
        self.region = np.empty(self.shape, dtype=int)
        self.mountain = np.empty(self.shape, dtype=int)
        
        self.mountain = set()
        self.total_prison = 0

        for i, rows in enumerate(raw_map):
            for j, value in enumerate(rows):
                if len(value) == 1:
                    self.value[i, j] = value 
                    self.region[i, j] = int(value)
                else:
                    region = int(value[:-1])
                    self.value[i, j] = value[-1]

                    if value[-1] == 'M':
                        self.mountain.add(region)
                    elif value[-1] == 'P':
                        self.value[i, j] = value[-1].lower()
                        self.total_prison += 1

                    self.region[i, j] = region


        self.value[self.value == '0'] = '~'
        is_mountain = self.value == 'M'

        self.potential= np.ones(self.shape, dtype=bool)
        self.potential[(self.region == 0) | (is_mountain)] = False

        print("value ", self.value)
        print("region ", self.region)
        print(self.total_region)
        print(self.total_prison)

        self.jacksparrow = JackSparrow(self.place_agent())

        self.pirate = Pirate(self.place_pirate())

        self.treasure = (int(read_map[4][0][0]), int(read_map[4][0][2]))

        self.n_turns = 1


        self.logs = []
        self.hint_list = []
   
        self.is_win = False
        self.is_lose = False
        self.is_teleported = False
   
        self.veri_important = {"1", "3", "5", "8"}
  
        self.reveal_turn = read_map[1]
        self.free_turn = read_map[2]
        self.is_free = False
   
        steps, pirate_path = self.shortest_path(self.pirate.coord, self.treasure)
        self.pirate_path = deque(pirate_path)

    def export_map(self, export_folder: str):
        with open(export_folder + f'map_export_{str(datetime.now()).replace(":", "")}.txt', 'w') as f:
            f.write(f"{self.shape[0]} {self.shape[1]}\n")
            f.write(f"{self.reveal_turn}\n")
            f.write(f"{self.free_turn}\n")
            f.write(f"{self.total_region + 1}\n")
            f.write(f"{self.treasure[0]} {self.treasure[1]}\n")
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    if self.value[i, j] in ['p', 'M']:
                        f.write(f'{self.region[i, j]}{self.value[i, j].upper()}; ')
                    else:
                        f.write(f'{self.region[i, j]}; ')
                f.write('\n')

    def map_print(self):
        str_regions =  [str(i) for i in range(1, self.total_region + 1)]
        for coord_x, row in enumerate(self.value):
            for coord_y, terrain in enumerate(row):
                cur = self.value[coord_x][coord_y]
                spaces = 3 - len(str(cur))
                symbol = ' ' * spaces + str(cur)
                if cur == '_' or cur in str_regions:
                    print('\033[92m', symbol, end='')
                elif cur == "0":                
                    print('\033[96m', symbol, end='')
                elif cur == 'M':                
                    print('\033[91m', symbol, end='')
                elif cur == 'p':                
                    print('\033[93m', symbol, end='')
                else:
                    print('\033[97m', symbol, end='')
            print()
        print('\033[97m')

    def ravel_index(self, index: Tuple[int, int]) -> int:
        H, W = self.shape
        return W * index[0] + index[1]

    def get_boundary(self, region1, region2):
        bound1 = np.isin(self.region, region1)
        bound2 = np.isin(self.region, region2)

        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        bound1 = binary_dilation(bound1==0, k) & bound1
        bound2 = binary_dilation(bound2==0, k) & bound2

        top = np.roll(bound2, -1, axis=0)
        top[-1,] = 0

        bottom = np.roll(bound2, 1, axis=0)
        bottom[0,] = 0
        
        left = np.roll(bound2, -1, axis=1)
        left[:,-1] = 0

        right = np.roll(bound2, 1, axis=1)
        right[:,0] = 0

        bound1 = bound1 * top + bound1 * bottom + bound1 * left + bound1 * right
        bound2 = np.roll(bound1 * top, 1, axis=0) + np.roll(bound1 * bottom, -1, axis=0) + np.roll(bound1 * left, 1, axis=1) + np.roll(bound1 * right, -1, axis=1)
        res = bound1 + bound2

        return res

    def generate_hint_1(self) -> Tuple[str, bool, np.ndarray, str]:
        # A list of random tiles that doesn't contain the treasure (1 to 12)

        # -----> VERY IMPORTANT (Not Trueness)!!!! <------

        # trueness of this hint
        trueness = True

        # get random tiles doest not contain the treasure
        no_tiles = rng.randint(2, int(self.avg_size * 0.75))
        rand_tiles = rng.choice(np.arange(self.total_tile), size=no_tiles, replace=False)

        # get tile that overlaps with the treasure
        overlap = rand_tiles == self.ravel_index(self.treasure)

        # convert those tiles to tuple index
        tile_coords = np.unravel_index(rand_tiles, self.shape)

        masked_tiles = np.zeros(self.shape, dtype=bool)
        masked_tiles[tile_coords] = True

        # if one of them contain the treasure
        if np.any(overlap):
            trueness = False

        hinted_tiles = list(zip(tile_coords[0], tile_coords[1]))

        log = f"{hinted_tiles} do not contain treasure"
                        
        return "1", trueness, masked_tiles, log
    
    def generate_hint_2(self) -> Tuple[str, bool, np.ndarray, str]:
        # 2-5 regions that 1 of them has the treasure.

        # trueness of this hint
        trueness = False

        # number of regions
        no_reg = rng.randint(2, int(self.avg_size * 0.375))
        rand_regions = rng.choice(np.arange(1, self.total_region + 1), size=no_reg, replace=False)

        # get region that overlaps with the treasure's region
        overlap = rand_regions == self.region[self.treasure]

        # get mask of titles of those regions
        masked_tiles = np.isin(self.region, rand_regions)

        # if random region consist of a region that has the treasure
        if np.any(overlap):
            trueness = True

        hinted_regions = list(rand_regions)
        
        log = f"One of these regions contain the treasure: {hinted_regions}"
            
        return "2", trueness, masked_tiles, log

    def generate_hint_3(self) -> Tuple[str, bool, np.ndarray, str]:
        # 1-3 regions that do not contain the treasure.

        # -----> VERY IMPORTANT (Not Trueness)!!!! <------

        # trueness of this hint
        trueness = True

        # number of regions
        no_reg = rng.randint(1, int(self.avg_size * 0.1875))
        rand_regions = rng.choice(np.arange(1, self.total_region + 1), size=no_reg, replace=False)

        # get region that overlaps with the treasure's region
        overlap = rand_regions == self.region[self.treasure]

        # get mask of titles of those regions
        masked_tiles = np.isin(self.region, rand_regions)

        # if random region consist of a region that has the treasure
        if np.any(overlap):
            trueness = False

        hinted_regions = list(rand_regions)
        
        log = f"These regions do not contain the treasure: {hinted_regions}"

        return "3", trueness, masked_tiles, log

    def generate_hint_4(self) -> Tuple[str, bool, np.ndarray, str]:
        # A large rectangle area that has the treasure

        trueness = False

        h_size = int(rng.uniform(0.5, 0.8) * self.shape[0])
        w_size = int(rng.uniform(0.5, 0.8) * self.shape[1])
        
        start_point_x = rng.randint(0, self.shape[0] - h_size + 1)
        start_point_y = rng.randint(0, self.shape[1] - w_size + 1)
        
        end_point_x = start_point_x + h_size - 1
        end_point_y = start_point_y + w_size - 1
        
        # get mask of those tiles
        masked_tiles = np.zeros(self.shape, dtype=bool)
        masked_tiles[start_point_x:end_point_x + 1, start_point_y:end_point_y + 1] = True

        if start_point_x <= self.treasure[0] <= end_point_x and start_point_y <= self.treasure[1] <= end_point_y:
            trueness = True
        
        log = f"Large rectangle area has the treasure: [{start_point_x}, {start_point_y}, {end_point_x}, {end_point_y}]"

        return "4", trueness, masked_tiles, log
        
    def generate_hint_5(self) -> Tuple[str, bool, np.ndarray, str]:
        # A small rectangle area that doesn't has the treasure.

        # -----> VERY IMPORTANT (Not Trueness)!!!! <------

        trueness = False
        h_size = int(rng.uniform(0.2, 0.5) * self.shape[0])
        w_size = int(rng.uniform(0.2, 0.5) * self.shape[1])
        
        start_point_x = rng.randint(0, self.shape[0] - h_size + 1)
        start_point_y = rng.randint(0, self.shape[1] - w_size + 1)
        
        end_point_x = start_point_x + h_size - 1
        end_point_y = start_point_y + w_size - 1
        
        if not (start_point_x <= self.treasure[0] <= end_point_x and start_point_y <= self.treasure[1] <= end_point_y):
            trueness = True
            
        # get mask of those tiles
        masked_tiles = np.zeros(self.shape, dtype=bool)
        masked_tiles[start_point_x:end_point_x + 1, start_point_y:end_point_y + 1] = True
        
        log = f"Small rectangle area has no treasure: [{start_point_x}, {start_point_y}, {end_point_x}, {end_point_y}]"
        
        return "5", trueness, masked_tiles, log

    def generate_hint_6(self) -> Tuple[str, bool, None, str]:
        # You are the nearest person to the treasure

        # calculate the distances
        agent_treasure = (self.jacksparrow.coord[0] - self.treasure[0]) ** 2 + (self.jacksparrow.coord[1] - self.treasure[1]) ** 2
        pirate_treasure = (self.pirate.coord[0] - self.treasure[0]) ** 2 + (self.pirate.coord[1] - self.treasure[1]) ** 2

        # trueness of this hint
        trueness = agent_treasure < pirate_treasure

        log = "You are the nearest person to the treasure"

        return "6", trueness, None, log

    def generate_hint_7(self) -> Tuple[str, bool, np.ndarray, str]:
        # A column and/or a row that contain the treasure (rare)
        trueness = False

        no_row = rng.randint(self.shape[0])
        no_col = rng.randint(self.shape[1])
        no_type = rng.randint(3)
        log = ""

        masked_tiles = np.zeros(self.shape, dtype=bool)

        if no_type in [0, 2]: 
            masked_tiles[no_row, :] = True
            if self.treasure[0] == no_row:
                trueness = True

        if no_type in [1, 2]: 
            masked_tiles[:, no_col] = True
            if self.treasure[1] == no_col:
                trueness = True

        if no_type == 0:
            log = "Row {} contains the treasure".format(no_row)
        elif no_type == 1:
            log = "Column {} contains the treasure".format(no_col)
        else:
            log = "Row {} or column {} contain the treasure".format(no_row, no_col)

        return "7", trueness, masked_tiles, log

    def generate_hint_8(self) -> Tuple[str, bool, np.ndarray, str]:
        # A column and/or a row that do not contain the treasure
        trueness = True 

        # -----> VERY IMPORTANT (Not Trueness)!!!! <------

        no_row = rng.randint(self.shape[0])
        no_col = rng.randint(self.shape[1])
        no_type = rng.randint(3)
        log = ""

        masked_tiles = np.zeros(self.shape, dtype=bool)

        if no_type in [0, 2]: 
            masked_tiles[no_row, :] = True
            if self.treasure[0] == no_row:
                trueness = False

        if no_type in [1, 2]:
            masked_tiles[:, no_col] = True
            if self.treasure[1] == no_col:
                trueness = False

        if no_type == 0:
            log = "Row {} does not contain the treasure".format(no_row)
        elif no_type == 1:
            log = "Column {} does not contain the treasure".format(no_col)
        else:
            log = "Row {} or column {} do not contain the treasure".format(no_row, no_col)

        return "8", trueness, masked_tiles, log

    def generate_hint_9(self) -> Tuple[str, bool, np.ndarray, str]:
        # 2 regions that the treasure is somewhere in their boundary
        trueness = False

        #random two regions
        rand_regions = rng.choice(np.arange(1, self.total_region + 1), size=2, replace=False)
                
        # get region that overlaps with the treasure's region
        # overlap = rand_regions == self.region[self.treasure]

        # if random region consist of a region that has the treasure
        masked_tiles = self.get_boundary(rand_regions[0], rand_regions[1])

        if masked_tiles[self.treasure]:
            trueness = True
        
        log = "The treasure is somewhere in the boundary of region {} and region {}".format(rand_regions[0], rand_regions[1])
        return "9", trueness, masked_tiles, log
    
    def generate_hint_10(self) -> Tuple[str, bool, np.ndarray, str]:
        # The treasure is somewhere in a boundary of 2 regions 
        trueness = False

        #random two regions
        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        masked_tiles = np.zeros(self.shape, dtype=bool)

        for i in range(self.total_region):
            bound = np.isin(self.region, i)
            bound = binary_dilation(bound==0, k) & bound
            masked_tiles += bound
            
        if masked_tiles[self.treasure]:
            trueness = True

        log = "The treasure is somewhere in a boundary of 2 regions"
        return "10", trueness, masked_tiles, log

    def generate_hint_11(self) -> Tuple[str, bool, np.ndarray, str]:
        # The treasure is somewhere in an area bounded by 2-3 tiles from sea
        no_tiles = rng.randint(2, 3)

        # trueness of this hint
        trueness = False

        k = np.zeros((3,3),dtype=int); k[1] = 1; k[:,1] = 1 # for 8-connected
        bound = np.isin(self.region, 0)
        masked_sea = ~bound

        masked_titles = np.zeros(self.shape, dtype=bool)
        for _ in range(no_tiles):
            top = np.roll(bound, -1, axis=0)
            top[-1,] = 0

            bottom = np.roll(bound, 1, axis=0)
            bottom[0,] = 0
            
            left = np.roll(bound, -1, axis=1)
            left[:,-1] = 0

            right = np.roll(bound, 1, axis=1)
            right[:,0] = 0

            bound = top + bottom + left + right
            masked_titles += bound
        
        masked_titles &= masked_sea
        if masked_titles[self.treasure]:
            trueness = True

        log = "The treasure is somewhere in an area bounded by {} tiles from sea".format(no_tiles)

        return "11", trueness, masked_titles, log

    def generate_hint_12(self) -> Tuple[str, bool, np.ndarray, str]:
        # A half of the map without treasure

        # -----> VERY IMPORTANT (True, False for negation)!!!! <------

        # trueness of this hint
        trueness = False

        # random part of the map (0: left, 2: top, 3: bottom, 4: right)
        parts = ["left", "top", "bottom", "right"]
        part = rng.randint(4)

        masked_tiles = np.zeros(self.shape, dtype=bool)

        match part:
            case 0:
                vertical_middle_axis = (self.shape[1] - 1) // 2 + 1

                masked_tiles[:, vertical_middle_axis:] = True

                # if the treasure is not in the left part
                if self.treasure[1] >= vertical_middle_axis:
                    trueness = True

            case 1:
                horizontal_middle_axis = (self.shape[0] - 1) // 2 + 1

                masked_tiles[horizontal_middle_axis:] = True

                # if the treasure is not in the top part
                if self.treasure[0] >= horizontal_middle_axis:
                    trueness = True

            case 2:
                horizontal_middle_axis = (self.shape[0] - 1) // 2 + 1

                masked_tiles[:horizontal_middle_axis] = True

                # if the treasure is not in the bottom part
                if self.treasure[0] < horizontal_middle_axis:
                    trueness = True

            case 3:
                vertical_middle_axis = (self.shape[1] - 1) // 2 + 1

                masked_tiles[:, :vertical_middle_axis] = True

                # if the treasure is in the right part
                if self.treasure[1] < vertical_middle_axis:
                    trueness = True
        
        log = f"{parts[part]} part of the map does not contain the treasure."

        return "12", trueness, masked_tiles, log

    def generate_hint_13(self) -> Tuple[str, bool, np.ndarray, str]:
        # From the center of the map/from the prison that he's staying, he tells
        # you a direction that has the treasure (W, E, N, S or SE, SW, NE, NW)
        trueness = False
        direction = ['North', 'North West', 'West', 'South West', 'South', 'South East', 'East', 'North East']
        
        center_X = (self.shape[0]-1) // 2
        center_Y = (self.shape[1]-1) // 2
        pos_X = center_X - self.treasure[0]
        pos_Y = center_Y - self.treasure[1]

        dir = rng.randint(8)
        masked_tiles = np.zeros(self.shape, dtype=bool)
        match dir:
            case 0:
                masked_tiles[:center_X, center_Y] = True
                if (pos_X == 0) and (pos_Y > 0):
                    trueness = True
            case 1:
                masked_tiles[:center_X+1, :center_Y+1] = True
                if (pos_X > 0) and (pos_Y > 0):
                    trueness = True
            case 2:
                masked_tiles[center_X, :center_Y+1] = True
                if (pos_X > 0) and (pos_Y == 0):
                    trueness = True
            case 3:
                masked_tiles[:center_X+1, center_Y:] = True
                if (pos_X > 0) and (pos_Y < 0):
                    trueness = True
            case 4:
                masked_tiles[center_X:, center_Y] = True
                if (pos_X == 0) and (pos_Y < 0):
                    trueness = True
            case 5:
                masked_tiles[center_X:, center_Y:] = True
                if (pos_X < 0) and (pos_Y < 0):
                    trueness = True
            case 6:
                masked_tiles[center_X, center_Y:] = True
                if (pos_X < 0) and (pos_Y == 0):
                    trueness = True
            case 7:
                masked_tiles[center_X:, :center_Y+1] = True
                if (pos_X < 0) and (pos_Y > 0):
                    trueness = True
        
        log = "The treasure is in the {} of the center of the map".format(direction[dir])

        return "13", trueness, masked_tiles, log

    def generate_hint_14(self) -> Tuple[str, bool, np.ndarray, str]: 
        # 2 squares that are different in size, the small one is placed inside the
        # bigger one, the treasure is somewhere inside the gap between 2 squares

        # trueness of this hint
        trueness = False
    
        # define ratio of the square
        big_ratio = rng.uniform(0.5, 0.8)
        small_ratio = rng.uniform(0.1, big_ratio)
        
        # average value of W and H
        avg_size = sum(self.shape) / 2

        # big rectangle 
        big_size = int(big_ratio * avg_size)
        
        # small rectangle
        small_size = int(small_ratio * avg_size)

        # top-left point of small square
        big_start_x = rng.randint(self.shape[0] - big_size + 1)
        big_start_y = rng.randint(self.shape[1] - big_size + 1)

        big_top_left = big_start_x, big_start_y

        # bottom-right point of big square
        big_end_x = big_start_x + big_size - 1
        big_end_y = big_start_y + big_size - 1

        big_bottom_right = big_end_x, big_end_y

        # top-left point of small square
        small_start_x = rng.randint(big_start_x, self.shape[0] - small_size + 1)
        small_start_y = rng.randint(big_start_y, self.shape[1] - small_size + 1)

        small_top_left = small_start_x, small_start_y

        # bottom-right point of small square
        small_end_x = small_start_x + small_size - 1
        small_end_y = small_start_y + small_size - 1

        small_bottom_right = small_end_x, small_end_y

        masked_tiles = np.zeros(self.shape, dtype=bool)

        # masked true for big square
        masked_tiles[big_start_x:big_end_x + 1, big_start_y:big_end_y + 1] = True

        # masked false for small square
        masked_tiles[small_start_x:small_end_x + 1, small_start_y:small_end_y + 1] = False

        if masked_tiles[self.treasure]:
            trueness = True

        log = f"The treasure is in between 2 squares: S1=[{big_top_left},{big_bottom_right}], S2=[{small_top_left},{small_bottom_right}]"
            
        return "14", trueness, masked_tiles, log

    def generate_hint_15(self) -> Tuple[str, bool, np.ndarray, str]:
        # The treasure is in a region that has mountain

        # trueness of this hint
        trueness = False

        # if list of mountain region contain the region of the treasure
        overlap = self.mountain == self.region[self.treasure]

        # get all tiles that is in mountain region
        masked_titles = np.isin(self.region, self.mountain)

        # if the treasure is in region mountain        
        if np.any(overlap):
            trueness = True

        log = f"The treasure is in a region that has mountain"

        return "15", trueness, masked_titles, log

    def move_jack(self, direction: str, steps: int) -> None:
        
        coord = list(self.jacksparrow.coord)

        if direction == 'EAST':
            coord[1] += steps
        elif direction == 'WEST':
            coord[1] -= steps
        elif direction == 'NORTH':
            coord[0] -= steps
        else:
            coord[0] += steps

        self.logs[self.n_turns].append(f"The agent move {steps} steps to the {direction}")

        self.jacksparrow.coord = tuple(coord)
    
    def teleport(self, coord: Tuple[int, int]) -> None:
        self.jacksparrow.coord = coord
        self.logs[self.n_turns].append(f"The agent teleports to {coord}")
    
    def move_pirate(self, direction: str, steps: int) -> None:
        
        coord = list(self.pirate.coord)

        if direction == 'EAST':
            coord[1] += steps
        elif direction == 'WEST':
            coord[1] -= steps
        elif direction == 'NORTH':
            coord[0] -= steps
        else:
            coord[0] += steps

        self.logs[self.n_turns].append(f"The pirate move {steps} steps to the {direction}")

        self.pirate.coord = tuple(coord)

    def scan(self, size: int):
        start_row = max(self.jacksparrow.coord[0] - size // 2, 0)

        end_row = min(self.jacksparrow.coord[0] + size // 2, self.shape[0] - 1)

        start_col = max(self.jacksparrow.coord[1] - size // 2, 0)

        end_col = min(self.jacksparrow.coord[1] + size // 2, self.shape[1] - 1)
        
        self.logs[self.n_turns].append(f"AGENT PERFOMED A [{size} x {size}] SCAN")

        self.potential[start_row: end_row + 1, start_col: end_col + 1] = False

        if start_row <= self.treasure[0] <= end_row and start_col <= self.treasure[1] <= end_col:
            self.logs[self.n_turns].append("AGENT WIN")
            self.is_win = True
            return True

        return False

    def gen_1st_hint(self):
        while True:
            hint_type = str(rng.randint(1, 16))
            gen_hint = self.hints[hint_type]
            _, trueness, masked_tiles, log = gen_hint()

            if trueness:
                self.logs[self.n_turns].append(log)
                self.logs[self.n_turns].append("ADD HINT1 TO HINT LIST")

                self.verify_hint(1, hint_type, trueness, masked_tiles, log)

                break
                        
    def hint_generator(self):
        hint_type = str(rng.randint(1, 16))
        
        hint_type, trueness, masked_tiles, log = self.hints[hint_type]()

        self.hint_list.append((self.n_turns, hint_type, trueness, masked_tiles, log))

        self.logs[self.n_turns].append(f"HINT{self.n_turns}: The agent receives a hint:")
        self.logs[self.n_turns].append(log)
        self.logs[self.n_turns].append(f"ADD HINT{self.n_turns} TO HINT LIST")

    def apply_masked_tiles(self, trueness: bool, masked_tiles: np.ndarray) -> None:
        # The treasure is somewhere in a boundary of 2 regions 

        if trueness:
            self.potential &= masked_tiles
        else:
            self.potential[masked_tiles] = False

    def verify_hint(self, hint_number, hint_type: str, trueness: bool, masked_tiles: np.ndarray, log: str):
        if hint_type == "6":
            return

        self.logs[self.n_turns].append(f"HINT{hint_number}: is_verified = TRUE, is_truth = {trueness}")
        self.logs[self.n_turns].append(log)
        
        if hint_type in self.veri_important:
            trueness = not trueness
        elif hint_type == "12":
            if not trueness:
                masked_tiles = ~masked_tiles
            trueness = True
        
        self.apply_masked_tiles(trueness, masked_tiles)

    # def choose_direction(self) -> str:
    #     x_coord, y_coord = self.jacksparrow.coord

    #     dir = ["N", "W", "S", "E"]
        
    #     north = self.potential[:x_coord + 1, :].sum()
    #     west = self.potential[:, :y_coord + 1].sum()
    #     south = self.potential[x_coord:, :].sum()
    #     east = self.potential[:, y_coord:].sum()

    #     direction = np.array([north, west, south, east])
    #     return dir[direction.argmax()]

    def isValid(self, row: int, col: int):
            return (row >= 0) and (row < self.shape[0]) and (col >= 0) and (col < self.shape[1])
        
    def BFS(self, board, src: Tuple[int, int], dest: Tuple[int, int]):
        rowDir = [-1, 0, 0, 1]
        colDir = [0, -1, 1, 0]

        if not board[src[0]][src[1]].isdigit() and board[src[0]][src[1]] != 'p':
            return [], np.inf
        
        visited = {}
        
        visited[src] = (-1, -1)
        
        q = deque()
        
        s = Node(src, 0)
        q.append(s)

        while q:
            front = q.popleft()
            
            pt = front.pt
            if pt == dest:
                path = []
                while pt[0] != -1:
                    path.append(pt)
                    pt = visited[pt]
                path.reverse()
                return path, front.step
            
            for i in range(4):
                row = pt[0] + rowDir[i]
                col = pt[1] + colDir[i]
                
                if self.isValid(row, col) and (board[row][col].isdigit() or board[row][col] == 'p') and not (row, col) in visited:
                    visited[(row, col)] = pt
                    neighbor = Node((row, col), front.step + 1)
                    q.append(neighbor)
        
        return [], np.inf

    def decode(self, path):
        if not path:
            return []

        tmp = []
        direction = {
            (-1, 0): 'NORTH',
            (0, 1): 'EAST',
            (0, -1): 'WEST',
            (1, 0): 'SOUTH'
        }
        n = len(path)
        for i in range(1, n):
            tmp_x, tmp_y = path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]
            tmp.append(direction[(tmp_x, tmp_y)])

        dirArray = []
        countArray = []
        count = 1
        
        if not len(tmp):
            return []

        for j in range(len(tmp) - 1):
            if tmp[j] != tmp[j + 1]:
                dirArray.append(tmp[j])
                countArray.append(count)
                count = 1
        
            else:
                count += 1
        
        dirArray.append(tmp[-1])
        countArray.append(count)

        res = []
        for x, y in zip(dirArray, countArray):
            res.append((x, y))

        return res
    
    def shortest_path(self, source, dest):
        path, step = self.BFS(self.value, source,dest)

        if step == np.inf:
            path = []
        
        return step, self.decode(path)
    
    def nearest_path(self, n_clusters: int) -> List[Tuple[str, int]]:
        centers = self.kmeans_center(n_clusters)

        min_steps = np.inf
        min_path = []

        for center in centers:
            if center == self.jacksparrow.coord:
                continue

            n_steps, path = self.shortest_path(self.jacksparrow.coord, center)
            if n_steps < min_steps:
                min_steps = n_steps
                min_path = path

        if not min_path:
            idx_tiles = np.where(self.potential)
            coords = list(zip(*idx_tiles))

            rand_idx = rng.randint(len(coords), size=len(coords))

            for i in rand_idx:
                n_steps, path = self.shortest_path(self.jacksparrow.coord, coords[i])

                if path:
                    return path

        return min_path

    def kmeans_center(self, n_clusters) -> List[Tuple[int, int]]:
        true_index = np.where(self.potential)
        points = list(zip(true_index[0], true_index[1]))

        if len(points) > n_clusters:
            kmeans = KMeans(n_clusters)
            kmeans.fit(points)

            closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, points)

            return [points[i] for i in closest]

        return points

    def first_turn(self) -> None:
        self.logs[self.n_turns].append("START TURN 1")

        # generate first hint
        self.gen_1st_hint()

        n_clusters = int(self.potential.sum() ** (1/5))
        path = self.nearest_path(n_clusters=max(2, n_clusters))
        direction = ''
        n_steps = 0

        if path:
            direction, n_steps = path[0]
        else:        
            direction = rng.choice(['NORTH', 'EAST', 'WEST', 'SOUTH'])
            n_steps = rng.randint(1, 5)

        # first action
        self.scan(5)

        if self.is_win:
            return

        # second action
        move_steps = min(n_steps, 4)
        self.move_jack(direction, move_steps)
        
        if move_steps < 3:
            self.scan(3)

            if self.is_win:
                return

    def normal_turn(self) -> None:
        n_actions = 2

        if rng.rand() > 0.7:
            rand_hint = rng.randint(len(self.hint_list)) 
            self.verify_hint(*self.hint_list[rand_hint])
            n_actions -= 1
        
        n_clusters = int(self.potential.sum() ** (1/3))
        path = self.nearest_path(n_clusters=max(2, n_clusters))
        direction = ''
        n_steps = 0

        if path:
            direction, n_steps = path[0]
        else:        
            direction = rng.choice(['NORTH', 'EAST', 'WEST', 'SOUTH'])
            n_steps = rng.randint(1, 5)

        if n_actions == 2:
            self.scan(5)

            if self.is_win:
                return

            self.move_jack(direction, min(n_steps, 2))
            self.scan(3)

            if self.is_win:
                return
        else:
            action_choice = rng.choice(np.arange(3)) 

            if action_choice == 0:
                self.move_jack(direction, min(n_steps, 2))
                self.scan(3)

                if self.is_win:
                    return
            elif action_choice == 1:
                self.scan(5)

                if self.is_win:
                    return
            else:
                move_steps = min(n_steps, 4)
                self.move_jack(direction, move_steps)
                
                if move_steps < 3:
                    self.scan(3)

                    if self.is_win:
                        return

    def pirate_action(self) -> None:
        direction = None
        if self.pirate_path:
            direction, n_steps = self.pirate_path[0]
            steps = min(n_steps, 2)
            self.move_pirate(direction, min(n_steps, 2))

            if self.pirate.coord == self.treasure:
                self.logs[self.n_turns].append("AGENT LOSE, THE PIRATE FOUND THE TREASURE")
                self.is_lose = True
                self.n_turns += 1

            n_steps -= steps

            if n_steps == 0:
                self.pirate_path.popleft()
            else:
                self.pirate_path[0] = direction, n_steps

        if not self.is_lose and not self.is_teleported:
            if rng.rand() > self.avg_size / (self.avg_size + 5):
                self.is_teleported = True
                self.teleport(self.teleport_coord(direction))

    def teleport_coord(self, direction):
        masked_tiles = np.ones(self.shape, dtype=bool)

        match direction:
            case "North":
                masked_tiles[self.pirate.coord[0]:,] = False
            case "West":
                masked_tiles[:, self.pirate.coord[1]:] = False
            case "South":
                masked_tiles[: self.pirate.coord[0],] = False
            case "East":
                masked_tiles[:, :self.pirate.coord[1]] = False
        
        masked_tiles &= self.potential

        idx_tiles = np.where(masked_tiles)

        coords = list(zip(*idx_tiles))
        choices = np.random.choice(np.arange(len(coords)))

        return coords[choices]
