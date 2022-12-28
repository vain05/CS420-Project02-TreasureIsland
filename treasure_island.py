# %% [markdown]
# # Treasure Island

# %% [markdown]
# ## Import packages

# %%
import numpy as np
from numpy import typing as npt
from scipy.ndimage.morphology import binary_dilation

from typing import List, Tuple

import random
import math
from string import digits
import copy

# %% [markdown]
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
    
    def take_action(self):
        raise NotImplementedError()
    

# %%
class JackSparrow(Agent):
    def __init__(self, coord: Tuple[int, int]) -> None:
        super().__init__(coord)
    
    def take_action(self, potential: np.ndarray, scanned: np.ndarray, hints_list):
        pass
    
    def move(self, steps: int, direction: str) -> None:
        
        coord = list(self.coord)

        if direction == 'E':
            coord[1] += steps
        elif direction == 'W':
            coord[1] -= steps
        elif direction == 'N':
            coord[0] -= steps
        else:
            coord[0] += steps
        
        self.coord = tuple(coord)

# %%
class Pirate(Agent):
    def __init__(self, coord: Tuple[int, int]) -> None:
        super().__init__(coord)
    
    def take_action(self):
        pass

# %%
class Map:
    def __init__(self, map: MapGenerator):
        self.total_tile = map.rows * map.cols
        self.shape = (map.rows, map.cols)
        self.total_region = map.number_of_region

        map.generate()

        self.value = np.array(map.Map, dtype=str)
        self.region = np.array(map.region_map, dtype=int)
        self.mountain = np.array(map.mountain_map, dtype=int)

        self.scanned = np.zeros((map.rows, map.cols), dtype=bool)
        self.potential= np.ones((map.rows, map.cols), dtype=bool)

        self.jacksparrow = JackSparrow(map.place_agent())
        # self.value[self.jacksparrow.coord] = 'A'

        self.pirate = Pirate(map.place_pirate())
        # self.value[self.pirate.coord] = 'Pi'

        self.treasure = map.place_treasure()
        # self.value[self.treasure] = 'T'

        self.logs = []

        self.hint_list = []

        self.veri_important = ["1", "3", "5", "8"]

        # Map generate hints function to string
        self.hints = {"1": self.generate_hint_1, "2": self.generate_hint_2, "3": self.generate_hint_3, "4": self.generate_hint_4,
                      "5": self.generate_hint_5, "6": self.generate_hint_6, "7": self.generate_hint_7, "8": self.generate_hint_8,
                      "9": self.generate_hint_9, "10": self.generate_hint_10, "11": self.generate_hint_11, "12": self.generate_hint_12,
                      "13": self.generate_hint_13, "14": self.generate_hint_14, "15": self.generate_hint_15}

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
        no_tiles = rng.randint(1, 13)
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

        log = f"These tiles {hinted_tiles} do not contain the treasure"
                        
        return "1", trueness, masked_tiles, log
    
    def generate_hint_2(self) -> Tuple[str, bool, np.ndarray, str]:
        # 2-5 regions that 1 of them has the treasure.

        # trueness of this hint
        trueness = False

        # number of regions
        no_reg = rng.randint(2, 6)
        rand_regions = rng.choice(np.arange(1, self.total_region + 1), size=no_reg, replace=False)
        print(rand_regions)

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
        no_reg = rng.randint(1, 3)
        rand_regions = rng.choice(np.arange(1, self.total_region + 1), size=no_reg, replace=False)
        print(rand_regions)

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
        
        log = f"Large rectangle area has the treasure. Top-Left-Bottom-Right = [{start_point_x}, {start_point_y}, {end_point_x}, {end_point_y}]"

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
        
        log = f"Small rectangle area doesn't the treasure. Top-Left-Bottom-Right = [{start_point_x}, {start_point_y}, {end_point_x}, {end_point_y}]"
        
        return "5", trueness, masked_tiles, log

    def generate_hint_6(self) -> Tuple[str, bool, None, str]:
        # You are the nearest person to the treasure

        # calculate the distances
        agent_treasure = (self.jacksparrow.coord[0] - self.treasure[0]) ** 2 + (self.jacksparrow.coord[1] - self.treasure[1]) ** 2
        pirate_treasure = (self.pirate.coord[0] - self.treasure[0]) ** 2 + (self.pirate.coord[1] - self.treasure[1]) ** 2

        # trueness of this hint
        trueness = agent_treasure < pirate_treasure
        print(trueness)

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
        print(rand_regions)
                
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

        log = f"The treasure is somewhere in the gap between 2 squares: S1 = [{big_top_left}, {big_bottom_right}], S2 = [{small_top_left}, {small_bottom_right}]"
            
        return "14", trueness, masked_tiles, log

    def generate_hint_15(self) -> Tuple[str, bool, np.ndarray, str]:
        # The treasure is in a region that has mountain

        # trueness of this hint
        trueness = False

        # if list of mountain region contain the region of the treasure
        overlap = self.mountain == self.region[self.treasure]
        print(overlap)

        # get all tiles that is in mountain region
        masked_titles = np.isin(self.region, self.mountain)

        # if the treasure is in region mountain        
        if np.any(overlap):
            trueness = True

        log = f"The treasure is in a region that has mountain"

        return "15", trueness, masked_titles, log
    
    def scan(self, size: int):
        start_row = max(self.jacksparrow.coord[0] - size // 2, 0)

        end_row = min(self.jacksparrow.coord[0] + size // 2, self.shape[0] - 1)

        start_col = max(self.jacksparrow.coord[1] - size // 2, 0)

        end_col = min(self.jacksparrow.coord[1] + size // 2, self.shape[1] - 1)

        if start_row <= self.treasure[0] <= end_row and start_col <= self.treasure[1] <= end_col:
            return True
        else:
            self.scanned[start_row: end_row + 1, start_col: end_col + 1] = True

        return False

    def gen_1st_hint(self):
        while True:
            for hint_type, gen_hint in self.hints.items():
                _, trueness, masked_tiles, log = gen_hint()

                if trueness:
                    self.logs.append("ADD HINT1 TO HINT LIST")
                    self.logs.append(log)
                    self.logs.append("HINT1: is_verified = TRUE, is_truth = TRUE")

                    self.verify_hint(hint_type, trueness, masked_tiles)

                    break
                        
    def hint_generator(self, n_turn: int):
        key = str(rng.randint(1, 16))
        
        trueness, masked_tiles, log = self.hints[key]()

        self.hint_list.append((key, trueness, masked_tiles))

        self.logs.append(f"HINT{n_turn}: The agent receives a hint:" + f"{log}")
        self.logs.append(f"ADD HINT{n_turn} TO HINT LIST")

    def apply_masked_tiles(self, trueness: bool, masked_tiles: np.ndarray) -> None:
        # The treasure is somewhere in a boundary of 2 regions 

        if trueness:
            self.potential &= masked_tiles
        else:
            self.potential[masked_tiles] = False

    def verify_hint(self, hint_type: str, trueness: bool, masked_tiles: np.ndarray):
        if hint_type == "6":
            return

        if hint_type in self.veri_important:
            trueness = not trueness
        elif hint_type == "12":
            if not trueness:
                masked_tiles = ~masked_tiles
            trueness = True
        
        self.apply_masked_tiles(trueness, masked_tiles)

    def operate(self) -> None:
        self.logs.append("Game start")
        self.logs.append(f"Agent appears at {self.jacksparrow.coord}")

        reveal_turn = rng.randint(2, 10)
        free_turn = rng.randint(reveal_turn + 1, 5)
        
        self.logs.append(f"The pirate’s prison is going to reveal the at the beginning of turn number {reveal_turn}")
        self.logs.append(f"The pirate is free at the beginning of turn number {free_turn}")
        
        n_turn = 1

        # start first turn 
        self.gen_1st_hint()

        while self.jacksparrow != self.treasure:
            self.logs.append(f"START TURN {n_turn}")

            # the first hint is supposed to be true
            self.hint_generator(n_turn)

            # actions of agent
            
            # actions of pirate


# %%
# map_gen = MapGenerator(16, 18)
# m = Map(map_gen)

# # %%
# m.map_print()
# print(f"Agent coord: {m.jacksparrow.coord}")
# print(f"Pirate coord: {m.pirate}")
# print(f"Treasure coord: {m.treasure}")
# print(f"Treasure's region: {m.region[m.treasure]}")
# print()

# trueness, data, log = m.generate_hint_10()
# print(trueness)
# print(data)
# print(log)
# print()

# print(m.scanned.astype(int))
# print()

# print(m.potential.astype(int))
# print()


