from os import path

import pygame.draw
from mlgame.utils.enum import get_ai_name
from mlgame.view.view_model import create_asset_init_data, create_image_view_data, create_rect_view_data

from .env import LEFT_CMD, RIGHT_CMD, FORWARD_CMD, BACKWARD_CMD, SHOOT, SHOOT_COOLDOWN, \
    IMAGE_DIR, ORANGE, BLUE

Vec = pygame.math.Vector2


class Player(pygame.sprite.Sprite):
    def __init__(self, construction, **kwargs):
        super().__init__()
        """
        初始化玩家資料
        construction可直接由TiledMap打包地圖資訊後傳入
        :param construction:
        :param kwargs:
        """
        self.id = construction["_id"]
        self.no = construction["_no"]
        self.rect = pygame.Rect(construction["_init_pos"], construction["_init_size"])
        self.play_rect_area = kwargs["play_rect_area"]
        self.origin_xy = self.rect.topleft
        self.origin_center = self.rect.center
        self.origin_size = (self.rect.width, self.rect.height)
        self.surface = pygame.Surface(self.origin_size)
        self.angle = 0
        self.score = 0
        self.used_frame = 0
        self.last_shoot_frame = 0
        self.lives = 3
        self.power = 10
        self.vel = Vec(0, 0)

        self.speed = 8
        # TODO refactor use vel
        self.move = {"left_up": Vec(-self.speed, -self.speed), "right_up": Vec(self.speed, -self.speed),
                     "left_down": Vec(-self.speed, self.speed), "right_down": Vec(self.speed, self.speed),
                     "left": Vec(-self.speed, 0), "right": Vec(self.speed, 0), "up": Vec(0, -self.speed),
                     "down": Vec(0, self.speed)}
        self.rot = 0
        self.last_shoot_frame = self.used_frame
        self.last_turn_frame = self.used_frame
        self.rot_speed = 45
        self.oil = 100
        self.is_alive = True
        self.is_shoot = False
        self.is_forward = False
        self.is_backward = False
        self.is_turn_right = False
        self.is_turn_left = False
        self.act_cd = kwargs["act_cd"]
        if self.rect.x >= self.play_rect_area.width // 2 and self.rect.y < (self.play_rect_area.height - 100) // 2:
            self.quadrant = 1
        elif self.rect.x < self.play_rect_area.width // 2 and self.rect.y < (self.play_rect_area.height - 100) // 2:
            self.quadrant = 2
        elif self.rect.x < self.play_rect_area.width // 2 and self.rect.y >= (self.play_rect_area.height - 100) // 2:
            self.quadrant = 3
        else:
            self.quadrant = 4
    def update(self, command: dict):
        self.used_frame += 1
        if self.is_alive:
            self.act(command[get_ai_name(self.no - 1)])
        if self.lives <= 0:
            self.is_alive = False
            self.out()

        self.rotate()

        if not self.act_cd:
            self.is_turn_right = False
            self.is_turn_left = False
        elif self.used_frame - self.last_turn_frame > self.act_cd:
            self.is_turn_right = False
            self.is_turn_left = False

        if self.rect.right > self.play_rect_area.right \
                or self.rect.left < self.play_rect_area.left \
                or self.rect.bottom > self.play_rect_area.bottom \
                or self.rect.top < self.play_rect_area.top:
            self.collide_with_walls()

    def rotate(self):
        new_sur = pygame.transform.rotate(self.surface, self.rot)
        self.rot = self.rot % 360
        self.angle = 3.14 / 180 * self.rot
        origin_center = self.rect.center
        self.rect = new_sur.get_rect()
        self.rect.center = origin_center

    def act(self, commands: list):
        if not commands:
            return None
        if self.power and SHOOT in commands:
            self.shoot()
        if self.oil <= 0:
            self.oil = 0
            return
        if LEFT_CMD in commands:
            self.oil -= 0.1
            self.turn_left()
        elif RIGHT_CMD in commands:
            self.oil -= 0.1
            self.turn_right()
        elif FORWARD_CMD in commands and BACKWARD_CMD not in commands:
            self.oil -= 0.1
            self.forward()
            self.is_forward = True
            self.is_backward = False
        elif BACKWARD_CMD in commands and FORWARD_CMD not in commands:
            self.oil -= 0.1
            self.backward()
            self.is_backward = True
            self.is_forward = False

    def shoot(self):
        if self.act_cd and self.used_frame - self.last_shoot_frame > SHOOT_COOLDOWN:
            self.last_shoot_frame = self.used_frame
            self.power -= 1
            self.is_shoot = True
        elif not self.act_cd:
            self.power -= 1
            self.is_shoot = True

    def forward(self):
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        else:
            rot = self.rot
        if rot == 0:
            self.rect.center += self.move["left"]
        elif rot == 315:
            self.rect.center += self.move["left_up"]
        elif rot == 270:
            self.rect.center += self.move["up"]
        elif rot == 225:
            self.rect.center += self.move["right_up"]
        elif rot == 180:
            self.rect.center += self.move["right"]
        elif rot == 135:
            self.rect.center += self.move["right_down"]
        elif rot == 90:
            self.rect.center += self.move["down"]
        elif rot == 45:
            self.rect.center += self.move["left_down"]

    def backward(self):
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        else:
            rot = self.rot
        if rot == 0:
            self.rect.center += self.move["right"]
        elif rot == 315:
            self.rect.center += self.move["right_down"]
        elif rot == 270:
            self.rect.center += self.move["down"]
        elif rot == 225:
            self.rect.center += self.move["left_down"]
        elif rot == 180:
            self.rect.center += self.move["left"]
        elif rot == 135:
            self.rect.center += self.move["left_up"]
        elif rot == 90:
            self.rect.center += self.move["up"]
        elif rot == 45:
            self.rect.center += self.move["right_up"]

    def turn_left(self):
        if not self.is_turn_left:
            self.last_turn_frame = self.used_frame
            self.rot += self.rot_speed
            self.is_turn_left = True

    def turn_right(self):
        if not self.is_turn_right:
            self.last_turn_frame = self.used_frame
            self.rot -= self.rot_speed
            self.is_turn_right = True

    def collide_with_walls(self):
        if self.is_turn_left:
            self.turn_right()
        elif self.is_turn_right:
            self.turn_left()
        if self.is_forward:
            self.backward()
        elif self.is_backward:
            self.forward()

    def collide_with_bullets(self):
        self.lives -= 1

    def get_power(self, power: int):
        self.power += power
        if self.power > 10:
            self.power = 10
        elif self.power < 0:
            self.power = 0

    def get_oil(self, oil: int):
        self.oil += oil
        if self.oil > 100:
            self.oil = 100
        elif self.oil < 0:
            self.oil = 0

    def get_rot(self):
        if self.id == 2:
            return (self.rot + 180) % 360
        return self.rot

    def get_data_from_obj_to_game(self) -> dict:
        rot = self.rot
        if self.id != 1:
            rot = self.rot + 180
            if rot >= 360:
                rot -= 360
        info = {"id": f"{self.no}P",
                "x": self.rect.x,
                "y": self.rect.y,
                "speed": self.speed,
                "score": self.score,
                "power": self.power,
                "oil": self.oil,
                "lives": self.lives,
                "angle": rot
                }
        return info

    def get_obj_progress_data(self) -> dict:
        image_data = create_image_view_data(f"{self.id}P", *self.rect.topleft, *self.origin_size, self.angle)
        return image_data

    def get_obj_toggle_data(self) -> list:
        image_data = []
        team_id = "team_a"
        # 初始位置為中點 + 5，再根據 1P～3P 依序往右
        x = self.play_rect_area.midbottom[0] + 5 + (self.no-1) * 50
        y = self.play_rect_area.height + 65
        if self.no > 3:
            team_id = "team_b"
            # 初始位置為中點，再根據 4P～6P 依序往左
            x = self.play_rect_area.midbottom[0] - (self.no-1) * 50 + 100
        # lives A_x: 520, B_x: 620
        for live in range(self.lives):
            image_data.append(create_image_view_data(f"{team_id}_lives", x, y, 30, 30))
            x += 5
            y -= 5
        # oil A_x: 650, B_x: 360
        y = self.play_rect_area.height + 60
        if self.no > 3:
            x = self.play_rect_area.midbottom[0] - 65 - (self.no-1) * 80 + 75
            image_data.append(create_rect_view_data(f"{team_id}_oil", x, y, int(self.oil*0.75), 10, ORANGE))
        else:
            x = self.play_rect_area.midbottom[0] + 150 + (self.no-1) * 80
            image_data.append(create_rect_view_data(f"{team_id}_oil", x, y, int(self.oil*0.75), 10, ORANGE))
        y = self.play_rect_area.height + 80
        if self.no > 3:
            for power in range(self.power):
                image_data.append(create_rect_view_data(f"{team_id}_power", x+67, y, 5, 10, BLUE))
                x -= 7
        else:
            for power in range(self.power):
                image_data.append(create_rect_view_data(f"{team_id}_power", x+2, y, 5, 10, BLUE))
                x += 7
        # power A_x: 650, B_x: 360

        return image_data

    def get_obj_init_data(self) -> list:
        img_data = {"1P": "https://raw.githubusercontent.com/Jesse-Jumbo/TankMan/main/asset/image/1P.svg",
                    "2P": "https://raw.githubusercontent.com/Jesse-Jumbo/TankMan/main/asset/image/2P.svg"}
        image_init_data = []
        for id, url in img_data.items():
            image_init_data.append(create_asset_init_data(id, self.origin_size[0], self.origin_size[1],
                                                          path.join(IMAGE_DIR, f"{id}.png"), url))
        return image_init_data

    def get_info_to_game_result(self) -> dict:
        team_id = "a"
        if self.no > 3:
            team_id = "b"
        info = {"no": f"{team_id}_{self.no}P"
            , "x": self.rect.x
            , "y": self.rect.y
            , "score": self.score
            , "lives": self.lives
                }
        return info

    def out(self):
        self.rect.topleft = ((self.play_rect_area.width // 2 - 100) + 50 * self.no, -50)
        self.rot = 0
