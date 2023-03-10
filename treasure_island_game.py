import pygame as pg
from sys import exit
import os
import time
import numpy as np
from copy import deepcopy

import treasure_island as mg

from datetime import datetime
import csv

###########################################################################################
HEIGTH, WIDTH = 16, 16
###########################################################################################

#Colors
default_tile_color = 'azure'
tile_text_color = 'grey1'
sea_color = '#54cff7'
land_color = 'forest green'
mountain_color = 'dark slate gray'
prison_color = 'indian red'
scanned_color = '#20c200'
# potential_color = 'LightGoldenrod1'

main_color = 'NavajoWhite2'
secondary_color = 'burlywood2'
button_color = 'burlywood3'
button_text_color = 'tan4'
#Sizes
icon_size =  (760/max(HEIGTH, WIDTH))/512
tile_size = 900*0.9/max(HEIGTH, WIDTH)        
gap_size = 900*0.1/(max(HEIGTH, WIDTH) - 1)
tile_font_size = int(360/max(HEIGTH, WIDTH))

###########################################################################################

def get_center(surface, parent_surface):
    return (parent_surface.surface.get_width() - surface.width) // 2, (parent_surface.surface.get_height() - surface.height) // 2

def get_horizontal_center(surface, parent_surface):
    return (parent_surface.surface.get_width() - surface.width) // 2

def get_vertical_center(surface, parent_surface):
    return (parent_surface.surface.get_height() - surface.height) // 2
class ScreenSurface:
    def __init__(
        self,
        scr_w: float,
        scr_h: float,
        color: str,
    ) -> None:
        self.surface = pg.display.set_mode((scr_w,scr_h))
        self.surface.fill(color)
        self.width, self.height = self.surface.get_size()
        self.rect = self.surface.get_rect()
        self.parent_surface = None
        self.x, self.y = 0, 0

    def get_parent_pos(self,parent_surface):
        return 0, 0    
        
    def get_center(self, parent_surface):
        return 0,0
    
    def get_horizontal_center(self, parent_surface):
        return 0

    def get_vertical_center(self, parent_surface):
        return 0
class ColoredSurface:
    def __init__(
        self,
        width: float, 
        height: float, 
        color: str
    ) -> None:
        self.surface = pg.Surface((width,height))
        self.surface.fill(color)
        self.width = width
        self.height = height
        self.color = color
        self.rect = self.surface.get_rect()
        self.parent_surface = None

    def get_parent_pos(self,parent_surface):
        return parent_surface.x, parent_surface.y

    def get_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2, (parent_surface.surface.get_height() - self.height) // 2
    
    def get_horizontal_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2

    def get_vertical_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_height() - self.height) // 2

    

    def draw_center(self, parent_surface) -> None:
        parent_surface.surface.blit(self.surface, self.get_center(parent_surface))
        a = self.get_center(parent_surface)
        b = parent_surface.rect.topleft
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.rect.left + self.get_horizontal_center(parent_surface), parent_surface.rect.top + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + y)

class ImageSurface:
    def __init__(
        self,
        image: str,
        scale: float
    ) -> None:
        self.image = pg.image.load(image)
        self.surface = pg.transform.scale(self.image, (self.image.get_width() * scale, self.image.get_height() * scale))
        self.width, self.height = self.surface.get_size()
        self.rect = self.surface.get_rect()
        self.x, self.y = self.rect.topleft

    def get_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2, (parent_surface.surface.get_height() - self.height) // 2
    
    def get_horizontal_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2

    def get_vertical_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_height() - self.height) // 2

    def draw_center(self, parent_surface) -> None:
        parent_surface.surface.blit(self.surface, self.get_center(parent_surface))
        a = self.get_center(parent_surface)
        b = parent_surface.rect.topleft
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.rect.left + self.get_horizontal_center(parent_surface), parent_surface.rect.top + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + y)

font_list = []    

class Text:
    def __init__(
        self,
        text: str,
        size: int,
        color: str
    ) -> None:
        self.surface = font_list[size].render(text, True, color)
        self.rect = self.surface.get_rect()
        self.width, self.height = self.surface.get_size()
        self.parent_surface = None
        self.x, self.y = self.rect.topleft

    def get_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2, (parent_surface.surface.get_height() - self.height) // 2

    def get_horizontal_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2

    def get_vertical_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_height() - self.height) // 2

    def draw_center(self, parent_surface) -> None:
        parent_surface.surface.blit(self.surface, self.get_center(parent_surface))
        a = self.get_center(parent_surface)
        b = parent_surface.rect.topleft
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.rect.left + self.get_horizontal_center(parent_surface), parent_surface.rect.top + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + y)

class Button:
    def __init__(
        self,
        width: float,
        height: float,
        button_color: str,
        text: str,
        text_size: int,
        text_color: str,
    ) -> None:
        self.box = ColoredSurface(width, height, button_color)
        self.text = Text(text, text_size, text_color)
        self.rect = self.box.surface.get_rect()
        self.surface = self.box.surface
        self.surface.blit(self.text.surface, self.text.get_center(self.box))
        self.width, self.height = self.surface.get_size()
        self.parent_surface = None
        self.x, self.y = self.rect.topleft
        self.text_size = text_size
        self.text_color = text_color

    # def change_text(self, new_text):
    #     self.text = Text(new_text, self.text_size, self.text_color)
    #     self.surface.blit(self.text.surface, self.text.get_center(self.box))

    def get_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2, (parent_surface.surface.get_height() - self.height) // 2
    
    def get_horizontal_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_width() - self.width) // 2

    def get_vertical_center(self, parent_surface):
        self.parent_surface = parent_surface
        return (parent_surface.surface.get_height() - self.height) // 2

    def draw_center(self, parent_surface) -> None:
        parent_surface.surface.blit(self.surface, self.get_center(parent_surface))
        a = self.get_center(parent_surface)
        b = parent_surface.rect.topleft
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.rect.left + self.get_horizontal_center(parent_surface), parent_surface.rect.top + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.rect.left + x, parent_surface.rect.top + y)       
    def is_clicked(self) -> bool:
        
        if pg.mouse.get_pressed()[0]:
            if self.rect.collidepoint(pg.mouse.get_pos()):
                return True
        return False
###########################################################################################

pg.init()

###########################################################################################
color_list = []

color_list.append('#B5D5C5')
color_list.append('#B08BBB')
color_list.append('#F1F7B5')
color_list.append('#FFD4B2')
color_list.append('#FFDDD2')
color_list.append('#8B7E74')
color_list.append('#B6E2A1')
color_list.append('#CDFCF6')
color_list.append('#665A48')
color_list.append('#DEBACE')
color_list.append('#BB6464')
color_list.append('#FD8A8A')

###########################################################################################

font1 = pg.font.Font('font/BlackRose.ttf', tile_font_size)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 25)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 35)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 45)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 55)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 80)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 100)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 220)
font_list.append(font1)
font1 = pg.font.Font('font/BlackRose.ttf', 16)
font_list.append(font1)

###########################################################################################


pg.display.set_caption('Treasure Island')

icon = pg.image.load('asset/treasure_island.png')
pg.display.set_icon(icon)

clock = pg.time.Clock()

screen = ScreenSurface(1600,1000,'cyan')
background = ImageSurface('asset/background.jpg',1.1848)
background_loading = ImageSurface('asset/background.jpg',1.1848)
########### MENU ###########
menu_box = ColoredSurface(550, 800, main_color)

start_button = Button(400, 100, button_color, 'Start', 5, button_text_color)
import_button = Button(400, 100, button_color, 'Import', 5, button_text_color)
setting_button = Button(400, 100, button_color, 'Setting', 5, button_text_color)
quit_button = Button(400, 100, button_color, 'Quit', 5, button_text_color)

loading = Button(1200, 750, button_color, 'Loading...', 6, button_text_color)
loading.draw_center(background_loading)

########### SETTING ###########

setting_box = ColoredSurface(550, 800, main_color)
x16_button = Button(120,80, button_color, '16x16', 3, button_text_color)
x32_button = Button(120,80, button_color, '32x32', 3, button_text_color)
x64_button = Button(120,80, button_color, '64x64', 3, button_text_color)
custom_height = Button(150,150, button_text_color, str(HEIGTH), 5, main_color)
custom_width = Button(150,150, button_text_color, str(WIDTH), 5, main_color)
h_minus_button = Button(150, 150, main_color, '<', 6, button_text_color)
h_plus_button = Button(150, 150, main_color, '>', 6, button_text_color)
w_minus_button = Button(150, 150, main_color, '<', 6, button_text_color)
w_plus_button = Button(150, 150, main_color, '>', 6, button_text_color)
back_setting_button = Button(400, 100, button_color, 'Back', 5, button_text_color)
########### GAME ###########
game_box = ColoredSurface(950, 950, main_color)
game_inner_box = ColoredSurface(925, 925, button_color)
info_box = ColoredSurface(550, 950, secondary_color)
log_box = ColoredSurface(500, 600, 'wheat1')
log_title = Text('Log', 2, tile_text_color)

play_button = Button(500, 50, button_color, 'Play', 4, button_text_color)
regenerate_button = Button(500, 50, button_color, 'New Map', 4, button_text_color)
back_button = Button(500, 50, button_color, 'Back', 4, button_text_color)

previous_button = Button(80, 50, button_color, '<<', 1, button_text_color)
next_button = Button(80, 50, button_color, '>>', 1, button_text_color)

export_button = Button(80, 50, button_color, 'Export', 1, button_text_color)
next_step_button = Button(80, 50, button_color, 'Next', 1, button_text_color)

agent_icon = ImageSurface('asset/agent.png', icon_size)
pirate_icon = ImageSurface('asset/pirate.png', icon_size)
treasure_icon = ImageSurface('asset/treasure.png', icon_size)
both_icon = ImageSurface('asset/both.png', icon_size)

potential_states = []
agent_positions = []
pirate_positions = []
state_index = 1

stage = 0
import_init = 0
update = 1
init = 1
is_clicked = False
frame = 0
map_gen = mg.MapGenerator(HEIGTH, WIDTH)
m = mg.Map(map_gen)
running = 0

def output_log(folder_path, is_win, is_lose, n_turns, logs):
    log_name = 'LOG_' + str(len(os.listdir(folder_path)))
    with open(folder_path + log_name + '.txt', mode='w') as f:
        total_lines = 0
        for log in logs:
            total_lines += len(log)
        
        f.write(str(total_lines) + '\n')

        if is_win and is_lose:
            f.write('THE AGENT CANNOT WIN\n')
        elif is_win:
            f.write('WIN\n')

        elif is_lose:
            f.write('LOSE\n')

        f.write(str(n_turns) + '\n')
        
        for log in logs:
            for line in log:
                f.writelines(line + '\n')

while True:
    is_clicked = False
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                is_clicked = True
        

    ########### MENU ###########
    if stage == 0:
        if init == 1:
            init = 0
            background.draw_center(screen)

            menu_box.draw_center(screen)

            start_button.draw_center_horizontal(menu_box,80)    
            import_button.draw_center_horizontal(menu_box,260)  
            setting_button.draw_center_horizontal(menu_box,440)
            quit_button.draw_center_horizontal(menu_box,620)

        menu_box.draw_center(screen)
        
        if is_clicked:
            if quit_button.rect.collidepoint(pg.mouse.get_pos()):
                pg.quit()
                exit()

            if setting_button.rect.collidepoint(pg.mouse.get_pos()):
                stage = 1
                init = 1

            if start_button.rect.collidepoint(pg.mouse.get_pos()):
                stage = 2
                init = 1
                background_loading.draw_center(screen)
            
            if import_button.rect.collidepoint(pg.mouse.get_pos()):
                stage = 2
                import_init = 1
                

    ########### SETTING ###########
    elif stage == 1:
        if init == 1:
            init = 0

            background.draw_center(screen)
            setting_box.draw_center(screen)

            x16_button.draw(setting_box, 50, 100)
            x32_button.draw(setting_box, 215, 100)
            x64_button.draw(setting_box, 380, 100)
            
            h_minus_button.draw(setting_box, 50, 250)
            custom_height.draw(setting_box, 200, 250)
            h_plus_button.draw(setting_box, 350, 250)

            w_minus_button.draw(setting_box, 50, 450)
            custom_width.draw(setting_box, 200, 450)
            w_plus_button.draw(setting_box, 350, 450)

            back_setting_button.draw_center_horizontal(setting_box, 650)

        setting_box.draw_center(screen)

        if is_clicked:
            if back_setting_button.rect.collidepoint(pg.mouse.get_pos()):
                stage = 0
                init = 1
            if x16_button.rect.collidepoint(pg.mouse.get_pos()):
                HEIGTH = 16
                WIDTH = 16
            if x32_button.rect.collidepoint(pg.mouse.get_pos()):
                HEIGTH = 32
                WIDTH = 32
            if x64_button.rect.collidepoint(pg.mouse.get_pos()):
                HEIGTH = 64
                WIDTH = 64
            if h_minus_button.rect.collidepoint(pg.mouse.get_pos()):
                if HEIGTH > 5:
                    HEIGTH -= 1
            if h_plus_button.rect.collidepoint(pg.mouse.get_pos()):
                HEIGTH += 1
            if w_minus_button.rect.collidepoint(pg.mouse.get_pos()):
                if WIDTH > 5:
                    WIDTH -= 1
            if w_plus_button.rect.collidepoint(pg.mouse.get_pos()):
                WIDTH += 1

            custom_height = Button(150,150, button_text_color, str(HEIGTH), 5, main_color)
            custom_height.draw(setting_box, 200, 250)
            custom_width = Button(150,150, button_text_color, str(WIDTH), 5, main_color)
            custom_width.draw(setting_box, 200, 450)

    ########### GAME ###########
    elif stage == 2:
        if import_init == 1:
            import_init = 0
            m.import_map('./input_map.txt')

            HEIGTH, WIDTH = m.shape
            icon_size =  (760/max(HEIGTH, WIDTH))/512
            tile_size = 900*0.9/max(HEIGTH, WIDTH)        
            gap_size = 900*0.1/(max(HEIGTH, WIDTH) - 1)
            tile_font_size = int(360/max(HEIGTH, WIDTH))
            font_list[0] = pg.font.Font('font/BlackRose.ttf', tile_font_size)

            agent_icon = ImageSurface('asset/agent.png', icon_size)
            pirate_icon = ImageSurface('asset/pirate.png', icon_size)
            treasure_icon = ImageSurface('asset/treasure.png', icon_size)
            both_icon = ImageSurface('asset/both.png', icon_size)

            update = 1
            background_loading.draw_center(screen)
            rows, cols = m.shape
            Map = m.value

            potential_states = []
            agent_positions = []
            pirate_positions = []
            state_index = 1

            potential_states.append(m.potential.copy())
            agent_positions.append(m.jacksparrow.coord)
            pirate_positions.append(m.pirate.coord)
            update = 1

            m.map_print()
            print(f"Agent coord: {m.jacksparrow.coord}")
            print(f"Pirate coord: {m.pirate.coord}")
            print(f"Treasure coord: {m.treasure}")
            print()

            log_box = ColoredSurface(500, 600, 'wheat1')
            log_title.draw(log_box, 12, 5)
            log_box.draw_center_horizontal(info_box, 25)

            game_box = ColoredSurface(950, 950, main_color)
            game_inner_box = ColoredSurface(925, 925, button_color)
            info_box = ColoredSurface(550, 950, secondary_color)
            log_box = ColoredSurface(500, 600, 'wheat1')
            log_title = Text('Log', 2, tile_text_color)

            background.draw_center(screen)
            
            info_box.draw_center_vertical(screen, 1025)

            log_box.draw_center_horizontal(info_box, 25)
            log_title.draw(log_box, 12, 5)

            play_button.draw_center_horizontal(info_box, 725)
            regenerate_button.draw_center_horizontal(info_box, 800)
            back_button.draw_center_horizontal(info_box, 875)

            previous_button.draw(info_box, 25 ,650)
            next_button.draw(info_box, 130,650)

            export_button.draw(info_box, 340 ,650)
            next_step_button.draw(info_box, 445 ,650)

            game_box.draw_center_vertical(screen, 25)
            game_inner_box.draw_center(game_box)    

            str_regions =  [str(i) for i in range(1, m.total_region + 1)]     
            for i, r in enumerate(Map):
                for j, value in enumerate(r):
                    tile = Button(tile_size, tile_size, default_tile_color, '', 0, tile_text_color)
                    
                    if value == '~':
                        tile = Button(tile_size, tile_size, sea_color, '', 0, tile_text_color)
                    elif value in str_regions:
                        # if m.scanned[i][j] == 1:
                        #     tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        if m.potential[i][j] == 0:
                            tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        else: 
                            tile = Button(tile_size, tile_size, color_list[int(value)%12], str(value), 0, tile_text_color)
                    elif value == 'M':
                        tile = Button(tile_size, tile_size, mountain_color, str(value), 0, tile_text_color)
                    elif value == 'p':
                        tile = Button(tile_size, tile_size, prison_color, str(value), 0, tile_text_color)
                    
                    tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    
                    if (i,j) == m.jacksparrow.coord and (i,j) == m.pirate.coord and state_index >= m.reveal_turn:
                        both_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    else:
                        if (i,j) == m.jacksparrow.coord:                    
                            agent_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                        if (i,j) == m.pirate.coord:
                            pirate_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    if (i,j) == m.treasure:
                        treasure_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    

            

        
        if init == 1:
            init = 0
            update = 1

            icon_size =  (760/max(HEIGTH, WIDTH))/512
            tile_size = 900*0.9/max(HEIGTH, WIDTH)        
            gap_size = 900*0.1/(max(HEIGTH, WIDTH) - 1)
            tile_font_size = int(360/max(HEIGTH, WIDTH))
            font_list[0] = pg.font.Font('font/BlackRose.ttf', tile_font_size)

            agent_icon = ImageSurface('asset/agent.png', icon_size)
            pirate_icon = ImageSurface('asset/pirate.png', icon_size)
            treasure_icon = ImageSurface('asset/treasure.png', icon_size)
            both_icon = ImageSurface('asset/both.png', icon_size)


            background_loading.draw_center(screen)
            map_gen = mg.MapGenerator(HEIGTH, WIDTH)
            m = mg.Map(map_gen)
            rows, cols = m.shape
            Map = m.value

            potential_states = []
            agent_positions = []
            pirate_positions = []
            state_index = 1

            potential_states.append(m.potential.copy())
            agent_positions.append(m.jacksparrow.coord)
            pirate_positions.append(m.pirate.coord)
            update = 1

            m.map_print()
            print(f"Agent coord: {m.jacksparrow.coord}")
            print(f"Pirate coord: {m.pirate.coord}")
            print(f"Treasure coord: {m.treasure}")
            print()

            log_box = ColoredSurface(500, 600, 'wheat1')
            log_title.draw(log_box, 12, 5)
            log_box.draw_center_horizontal(info_box, 25)

            game_box = ColoredSurface(950, 950, main_color)
            game_inner_box = ColoredSurface(925, 925, button_color)
            info_box = ColoredSurface(550, 950, secondary_color)
            log_box = ColoredSurface(500, 600, 'wheat1')
            log_title = Text('Log', 2, tile_text_color)

            background.draw_center(screen)
            
            info_box.draw_center_vertical(screen, 1025)

            log_box.draw_center_horizontal(info_box, 25)
            log_title.draw(log_box, 12, 5)

            play_button.draw_center_horizontal(info_box, 725)
            regenerate_button.draw_center_horizontal(info_box, 800)
            back_button.draw_center_horizontal(info_box, 875)

            previous_button.draw(info_box, 25 ,650)
            next_button.draw(info_box, 130,650)

            export_button.draw(info_box, 340 ,650)
            next_step_button.draw(info_box, 445 ,650)

            game_box.draw_center_vertical(screen, 25)
            game_inner_box.draw_center(game_box)    

            str_regions =  [str(i) for i in range(1, m.total_region + 1)]     
            for i, r in enumerate(Map):
                for j, value in enumerate(r):
                    tile = Button(tile_size, tile_size, default_tile_color, '', 0, tile_text_color)
                    
                    if value == '~':
                        tile = Button(tile_size, tile_size, sea_color, '', 0, tile_text_color)
                    elif value == '_' or value in str_regions:
                        # if m.scanned[i][j] == 1:
                        #     tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        if m.potential[i][j] == 0:
                            tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        else: 
                            tile = Button(tile_size, tile_size, color_list[int(value)%12], str(value), 0, tile_text_color)
                    elif value == 'M':
                        tile = Button(tile_size, tile_size, mountain_color, str(value), 0, tile_text_color)
                    elif value == 'p':
                        tile = Button(tile_size, tile_size, prison_color, str(value), 0, tile_text_color)
                    
                    tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    
                    if (i,j) == m.jacksparrow.coord and (i,j) == m.pirate.coord and state_index >= m.reveal_turn:
                        both_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    else:
                        if (i,j) == m.jacksparrow.coord:                    
                            agent_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                        if (i,j) == m.pirate.coord:
                            pirate_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    if (i,j) == m.treasure:
                        treasure_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    

        background.draw_center(screen)
        info_box.draw_center_vertical(screen, 1025)
        log_box.draw_center_horizontal(info_box, 25)
        game_box.draw_center_vertical(screen, 25)
        game_inner_box.draw_center(game_box) 

        if update == 1:
            update = 0

            log_box = ColoredSurface(500, 600, 'wheat1')
            log_box.draw_center_horizontal(info_box, 25)
            log_title.draw(log_box, 12, 5)

            for i, log in  enumerate(m.logs[state_index - 1]):
                log_card = Button(480, 30, button_color, log, 8, 'grey10')
                log_card.draw_center_horizontal(log_box, 40 + 40*i)    

            str_regions =  [str(i) for i in range(1, m.total_region + 1)]     
            for i, r in enumerate(Map):
                for j, value in enumerate(r):
                    tile = Button(tile_size, tile_size, default_tile_color, '', 0, tile_text_color)
                    
                    if value == '~':
                        tile = Button(tile_size, tile_size, sea_color, '', 0, tile_text_color)
                    elif value in str_regions:
                        # if m.scanned[i][j] == 1:
                        #     tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        if m.potential[i][j] == 0:
                            tile = Button(tile_size, tile_size, scanned_color, str(value), 0, tile_text_color)
                        else: 
                            tile = Button(tile_size, tile_size, color_list[int(value)%12], str(value), 0, tile_text_color)
                    elif value == 'M':
                        tile = Button(tile_size, tile_size, mountain_color, str(value), 0, tile_text_color)
                    elif value == 'p':
                        tile = Button(tile_size, tile_size, prison_color, str(value), 0, tile_text_color)
                    
                    tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    
                    if (i,j) == m.jacksparrow.coord == m.pirate.coord and state_index >= m.reveal_turn:
                        both_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    else:
                        if (i,j) == m.jacksparrow.coord:                    
                            agent_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                        if (i,j) == m.pirate.coord and state_index >= m.reveal_turn:
                            pirate_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                    if (i,j) == m.treasure and (m.potential[i][j] == 0 or m.treasure == m.pirate.coord):
                        treasure_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))

        if running == 1:              
            if m.is_lose or m.is_win:
                running = 0
            else:
                m.logs.append([])

                if state_index != m.n_turns:
                    state_index = m.n_turns

                    m.potential = potential_states[state_index - 1]
                    m.jacksparrow.coord = agent_positions[state_index - 1]
                    m.pirate.coord = pirate_positions[state_index - 1]

                if m.n_turns != 1:
                    m.logs[m.n_turns].append(f"START TURN {m.n_turns}")

                    # generate a hint at the beginning of turn
                    m.hint_generator()

                if m.n_turns == m.reveal_turn:
                    m.logs[m.n_turns].append(f"The pirate is at the {m.pirate.coord} prison")
                
                if m.n_turns >= m.free_turn:
                    if m.n_turns == m.free_turn:
                        m.logs[m.n_turns].append(f"The pirate is free")
                    m.pirate_turn()

                if m.n_turns == 1:
                    print(m.logs[0], '\n')
                    m.first_turn()
                    m.n_turns += 1

                elif not m.is_lose:
                    m.normal_turn()
                    m.n_turns += 1

                update = 1
                print(m.logs[m.n_turns - 1], '\n')

                potential_states.append(m.potential.copy())
                agent_positions.append(m.jacksparrow.coord)
                pirate_positions.append(m.pirate.coord)
                state_index += 1

                if m.is_lose or m.is_win:
                    running = 0

                    output_log('./outputs_logs/', m.is_win, m.is_lose, m.n_turns - 1, m.logs)

            update = 1

        if is_clicked:
            if play_button.rect.collidepoint(pg.mouse.get_pos()):
                running = 1

            if regenerate_button.rect.collidepoint(pg.mouse.get_pos()):
                background_loading.draw_center(screen)
                init = 1
                # background_loading.draw_center(screen)
                # map_gen = mg.MapGenerator(HEIGTH, WIDTH)
                # m = mg.Map(map_gen)
                # rows, cols = m.shape
                # Map = m.value
                # n_turns = 1

                # m.map_print()
                # print(f"Agent coord: {m.jacksparrow.coord}")
                # print(f"Pirate coord: {m.pirate.coord}")
                # print(f"Treasure coord: {m.treasure}")
                # print()
                # log_list = []
                # log_box = ColoredSurface(500, 600, 'wheat1')
                # log_box.draw_center_horizontal(info_box, 25)
                # log_title.draw(log_box, 12, 5)
                
                # update = 1

            if back_button.rect.collidepoint(pg.mouse.get_pos()):
                stage = 0
                init = 1
                print()

            if previous_button.rect.collidepoint(pg.mouse.get_pos()):
                if state_index > 1:
                    state_index -= 1
                    update = 1

                    m.potential = potential_states[state_index - 1]
                    m.jacksparrow.coord = agent_positions[state_index - 1]
                    m.pirate.coord = pirate_positions[state_index - 1]
                    
            if next_button.rect.collidepoint(pg.mouse.get_pos()):
                if state_index < m.n_turns:
                    state_index += 1
                    update = 1

                    m.potential = potential_states[state_index - 1]
                    m.jacksparrow.coord = agent_positions[state_index - 1]
                    m.pirate.coord = pirate_positions[state_index - 1]


            if export_button.rect.collidepoint(pg.mouse.get_pos()):
                m.export_map('./exported_maps/')
                update = 1
                    
            if next_step_button.rect.collidepoint(pg.mouse.get_pos()):
                if not m.is_lose and not m.is_win:
                    m.logs.append([])

                    if state_index != m.n_turns:
                        state_index = m.n_turns

                        m.potential = potential_states[state_index - 1]
                        m.jacksparrow.coord = agent_positions[state_index - 1]
                        m.pirate.coord = pirate_positions[state_index - 1]

                    if m.n_turns != 1:
                        m.logs[m.n_turns].append(f"START TURN {m.n_turns}")

                        # generate a hint at the beginning of turn
                        m.hint_generator()

                    if m.n_turns == m.reveal_turn:
                        m.logs[m.n_turns].append(f"The pirate is at the {m.pirate.coord} prison")
                    
                    if m.n_turns >= m.free_turn:
                        if m.n_turns == m.free_turn:
                            m.logs[m.n_turns].append(f"The pirate is free")
                        m.pirate_turn()

                    if m.n_turns == 1:
                        print(m.logs[0], '\n')
                        m.first_turn()
                        m.n_turns += 1

                    elif not m.is_lose:
                        m.normal_turn()
                        m.n_turns += 1

                    update = 1
                    print(m.logs[m.n_turns - 1], '\n')

                    potential_states.append(m.potential.copy())
                    agent_positions.append(m.jacksparrow.coord)
                    pirate_positions.append(m.pirate.coord)
                    state_index += 1

                    if m.is_lose or m.is_win:
                        running = 0

                        output_log('./outputs_logs/', m.is_win, m.is_lose, m.n_turns, m.logs)

                update = 1

    pg.display.update() 
    clock.tick(60)
