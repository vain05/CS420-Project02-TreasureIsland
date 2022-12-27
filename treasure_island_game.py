import pygame as pg
from sys import exit
from typing import Tuple
import os
import time

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
        self.parent_surface = None

    
        
    def get_center(self, parent_surface):
        return (self.width) // 2, (self.height) // 2
    
    def get_horizontal_center(self, parent_surface):
        return (self.width) // 2

    def get_vertical_center(self, parent_surface):
        return (self.height) // 2
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
        b = parent_surface.get_center(parent_surface.parent_surface)
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + self.get_horizontal_center(parent_surface), parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + y)

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
        b = parent_surface.get_center(parent_surface.parent_surface)
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + self.get_horizontal_center(parent_surface), parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + self.get_vertical_center(parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + y)

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
        b = parent_surface.get_center(parent_surface.parent_surface)
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (self.get_horizontal_center(parent_surface), y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + self.get_horizontal_center(parent_surface), parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, self.get_vertical_center(parent_surface)))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + self.get_vertical_center(parent_surface))
    
    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
class Button:
    def __init__(
        self,
        width: float,
        height: float,
        button_color: str,
        text: str,
        text_size: float,
        text_color: str,
    ) -> None:
        self.box = ColoredSurface(width, height, button_color)
        self.text = Text(text, text_size, text_color)
        self.rect = self.box.surface.get_rect()
        self.surface = self.box.surface
        self.surface.blit(self.text.surface, self.text.get_center(self.box))

    def draw_center(self, parent_surface) -> None:
        parent_surface.surface.blit(self.surface, get_center(self.rect, parent_surface))
        a = get_center(self.rect, parent_surface)
        b = parent_surface.get_center(parent_surface.parent_surface)
        self.rect.topleft = (a[0] + b[0], a[1] + b[1])
        

    def draw_center_horizontal(self, parent_surface, y) -> None:
        parent_surface.surface.blit(self.surface, (get_horizontal_center(self.rect, parent_surface), y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + get_horizontal_center(self.rect, parent_surface), parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
        

    def draw_center_vertical(self, parent_surface, x) -> None:
        parent_surface.surface.blit(self.surface, (x, get_vertical_center(self.rect, parent_surface)))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + get_vertical_center(self.rect, parent_surface))

    def draw(self, parent_surface, x, y):
        parent_surface.surface.blit(self.surface, (x,y))
        self.rect.topleft = (parent_surface.get_horizontal_center(parent_surface.parent_surface) + x, parent_surface.get_vertical_center(parent_surface.parent_surface) + y)
        
    def is_clicked(self) -> bool:
        mouse_pos = pg.mouse.get_pos()
        
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(mouse_pos):
                return True
        return False

###########################################################################################


rows = 16
cols = 16
Map = [['~', '~', '~', '~', '~', '~', '~', '~', '~', '~'], ['~', 4, 4, 4, '~', '~', '~', '~', 1, '~'], ['~', 4, 4, 4, 4, 3, '~', 1, 1, 1], ['~', 4, 4, 2, 4, 4, 1, 1, 1, 1], ['~', '~', 2, 2, 1, 1, 1, 1, 1, 1], ['~', 'M', 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, '~', 1, 1, 1, 1, 1, 1, 1], ['~', '~', '~', '~', 1, 1, '~', 1, 1, 1], ['_', '~', '~', '~', 1, 1, 1, 1, 1, '~'], ['~', '_', '~', 1, 1, 1, 1, '~', 1, 1]]

###########################################################################################

def draw_grid(Map):
    game_box = ColoredSurface(950, 950, 'NavajoWhite2')
    game_inner_box = ColoredSurface(925, 925, 'burlywood3')

    game_box.draw_center_vertical(screen, 25)
    game_inner_box.draw_center(game_box)
    tile_size = 900*0.9/max(rows,cols)        
    gap_size = 900*0.1/(max(rows,cols) - 1)
    font_size = int(360/max(rows,cols))
                
    for i, value in enumerate(Map):
        for j, value in enumerate(Map):
            if value == '~':
                tile = Button(tile_size, tile_size, 'SteelBlue1', value, font_size, 'grey1')
                tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
            elif value in range(1,13):
                tile = Button(tile_size, tile_size, 'forest green', str(value), font_size, 'grey1')
                tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
            elif value == 'M':
                tile = Button(tile_size, tile_size, 'dark olive green', str(value), font_size, 'grey1')
                tile.draw(game_inner_box, 12.5 + j * (tile_size+gap_size), 12.5 + i * (tile_size+gap_size))
            j += 1
        i += 1
###########################################################################################
pg.init()


pg.display.set_caption('Treasure Island')

icon = pg.image.load('asset/treasure_island.png')
pg.display.set_icon(icon)

clock = pg.time.Clock()

screen = ScreenSurface(1600,1000,'cyan')
background = ImageSurface('asset/background.jpg',1.1848)

########### MENU ###########
menu_box = ColoredSurface(550, 800, 'NavajoWhite2')

start_font = pg.font.Font('font/BlackRose.ttf', 80)
start_text = start_font.render('Start', True, 'tan4')
setting_text = start_font.render('Setting', True, 'tan4')
quit_text = start_font.render('Quit', True, 'tan4')

start_button = Button(400, 100, 'burlywood3', 'Start', 80, 'tan4')
setting_button = Button(400, 100, 'burlywood3', 'Setting', 80, 'tan4')
quit_button = Button(400, 100, 'burlywood3', 'Quit', 80, 'tan4')

########### GAME ###########

info_box = ColoredSurface(550, 950, 'burlywood2')


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

        draw_grid(Map)


    pg.display.update() 
    clock.tick(60)
