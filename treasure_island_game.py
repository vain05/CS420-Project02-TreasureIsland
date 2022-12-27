import pygame as pg
from sys import exit
from typing import Tuple
import os
import time

import map_generator as mg

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
        
class Text:
    def __init__(
        self,
        text: str,
        size: int,
        color: str
    ) -> None:
        self.font = pg.font.Font('font/BlackRose.ttf', size)
        self.surface = self.font.render(text, True, color)
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

    def assign_parrent(self, parent_surface):
        self.parent_surface = parent_surface

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
        mouse_pos = pg.mouse.get_pos()
        
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

###########################################################################################

map_gen = mg.MapGenerator(16, 16)
m = mg.Map(map_gen)
rows, cols = m.shape
Map = m.value

m.map_print()
print(f"Agent coord: {m.jacksparrow.coord}")
print(f"Pirate coord: {m.pirate.coord}")
print(f"Treasure coord: {m.treasure}")
print()

###########################################################################################

#Colors
default_tile_color = 'azure'
tile_text_color = 'grey1'
sea_color = 'SteelBlue1'
land_color = 'forest green'
mountain_color = 'dark slate gray'
prison_color = 'khaki1'

main_color = 'NavajoWhite2'
secondary_color = 'burlywood2'
button_color = 'burlywood3'

#Sizes
icon_size =  (760/max(rows,cols))/512
tile_size = 900*0.9/max(rows,cols)        
gap_size = 900*0.1/(max(rows,cols) - 1)
tile_font_size = int(360/max(rows,cols))

###########################################################################################
pg.init()


pg.display.set_caption('Treasure Island')

icon = pg.image.load('asset/treasure_island.png')
pg.display.set_icon(icon)

clock = pg.time.Clock()

screen = ScreenSurface(1600,1000,'cyan')
background = ImageSurface('asset/background.jpg',1.1848)

########### MENU ###########
menu_box = ColoredSurface(550, 800, main_color)

start_font = pg.font.Font('font/BlackRose.ttf', 80)
start_text = start_font.render('Start', True, 'tan4')
setting_text = start_font.render('Setting', True, 'tan4')
quit_text = start_font.render('Quit', True, 'tan4')

start_button = Button(400, 100, button_color, 'Start', 80, 'tan4')
setting_button = Button(400, 100, button_color, 'Setting', 80, 'tan4')
quit_button = Button(400, 100, button_color, 'Quit', 80, 'tan4')

########### GAME ###########
game_box = ColoredSurface(950, 950, main_color)
game_inner_box = ColoredSurface(925, 925, button_color)
info_box = ColoredSurface(550, 950, secondary_color)
log_box = ColoredSurface(500, 600, 'wheat1')
log_title = Text('Log', 35, tile_text_color)

play_button = Button(500, 50, button_color, 'Play', 55, 'tan4')
regenerate_button = Button(500, 50, button_color, 'New Map', 55, 'tan4')
back_button = Button(500, 50, button_color, 'Back', 55, 'tan4')

up_button = Button(80, 50, button_color, 'UP', 30, 'tan4')
down_button = Button(80, 50, button_color, 'DOWN', 30, 'tan4')
left_button = Button(80, 50, button_color, 'LEFT', 30, 'tan4')
right_button = Button(80, 50, button_color, 'RIGHT', 30, 'tan4')

stage = 0

while True:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
            exit()

    ########### MENU ###########
    if stage == 0:
        background.draw_center(screen)

        menu_box.draw_center(screen)

        start_button.draw_center_horizontal(menu_box,150)    
        setting_button.draw_center_horizontal(menu_box,350)  
        quit_button.draw_center_horizontal(menu_box,550)

        if(quit_button.is_clicked()):
            pg.quit()
            exit()

        if(start_button.is_clicked()):
            stage = 1

    ########### GAME ###########
    elif stage == 1:
        background.draw_center(screen)
        
        info_box.draw_center_vertical(screen, 1025)

        log_box.draw_center_horizontal(info_box, 25)
        log_title.draw(log_box, 15, 10)
        play_button.draw_center_horizontal(info_box, 725)
        regenerate_button.draw_center_horizontal(info_box, 800)
        back_button.draw_center_horizontal(info_box, 875)

        up_button.draw(info_box, 25 ,650)
        down_button.draw(info_box, 165,650)
        left_button.draw(info_box, 305,650)
        right_button.draw(info_box, 445 ,650)

        if(play_button.is_clicked()):
            pass

        if(regenerate_button.is_clicked()):
            pass

        if(back_button.is_clicked()):
            stage = 0

        if(up_button.is_clicked()):
            m.jacksparrow.coord = (m.jacksparrow.coord[0] - 1, m.jacksparrow.coord[1])
        if(down_button.is_clicked()):
            m.jacksparrow.coord = (m.jacksparrow.coord[0] + 1, m.jacksparrow.coord[1])
        if(left_button.is_clicked()):
            m.jacksparrow.coord = (m.jacksparrow.coord[0], m.jacksparrow.coord[1] - 1)
        if(right_button.is_clicked()):
            m.jacksparrow.coord = (m.jacksparrow.coord[0], m.jacksparrow.coord[1] + 1)

        game_box.draw_center_vertical(screen, 25)
        game_inner_box.draw_center(game_box)        

        str_regions =  [str(i) for i in range(1, m.total_region + 1)]     
        for i, r in enumerate(Map):
            for j, value in enumerate(r):
                tile = Button(tile_size, tile_size, default_tile_color, '', tile_font_size, tile_text_color)
                
                if value == '0':
                    tile = Button(tile_size, tile_size, sea_color, '', tile_font_size, tile_text_color)
                elif value == '_' or value in str_regions:
                    tile = Button(tile_size, tile_size, land_color, str(value), tile_font_size, tile_text_color)
                elif value == 'M':
                    tile = Button(tile_size, tile_size, mountain_color, str(value), tile_font_size, tile_text_color)
                elif value == 'p':
                    tile = Button(tile_size, tile_size, prison_color, str(value), tile_font_size, tile_text_color)
                
                tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                
                if (i,j) == m.pirate.coord:
                    pirate_icon = ImageSurface('asset/pirate.png', icon_size)
                    pirate_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                if (i,j) == m.treasure:
                    treasure_icon = ImageSurface('asset/treasure.png', icon_size)
                    treasure_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
                if (i,j) == m.jacksparrow.coord:                    
                    agent_icon = ImageSurface('asset/agent.png', icon_size)
                    agent_icon.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))

                j += 1
            i += 1
    
    pg.display.update() 
    clock.tick(60)
