import cv2
from gym import spaces
import numpy as np


class Maze:
    AGENT_SIZE = 0.1

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)

    DIRECTIONS = 4
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, config=None):
        self.size = config["size"]
        self.agent_radius = self.AGENT_SIZE * 0.5
        self.start = config["start"]
        self.position = self.start
        self.goal = config["goal"]
        self.walls = config["walls"]
        self.observation_space = spaces.Box(low=0., high=self.size, shape=(2, ), dtype=np.float32)
        self.action_space = spaces.Discrete(self.DIRECTIONS)

        self.rooms = []
        config["rooms"].sort(key=lambda r: r["lvl"])
        for room_config in config["rooms"]:
            room_walls = []
            for wall in config["walls"]:
                if room_config["area"].overlaps_rectangle(wall):
                    room_walls.append(wall)
            room = Room(room_config["area"], room_config["entrypoints"], room_walls)
            self.rooms.append(room)

    def step(self, action):
        next_position = self.move(action)

        for wall in self.walls:
            if wall.inside(next_position):
                next_position = self.position

        self.position = next_position
        return self.position

    def move(self, action):
        if action == self.UP:
            direction = Point(0, self.AGENT_SIZE)
        elif action == self.RIGHT:
            direction = Point(self.AGENT_SIZE, 0)
        elif action == self.DOWN:
            direction = Point(0, -self.AGENT_SIZE)
        elif action == self.LEFT:
            direction = Point(-self.AGENT_SIZE, 0)

        next_position = self.position.add(direction)

        if next_position.x + self.agent_radius >= self.size or next_position.y + self.agent_radius >= self.size:
            next_position = self.position

        if next_position.x - self.agent_radius <= 0 or next_position.y - self.agent_radius <= 0:
            next_position = self.position

        return next_position

    def reset(self):
        self.position = self.start
        return self.position

    def draw(self, magnification=500, save=False):
        image = np.zeros([int(magnification * self.size), int(magnification * self.size), 3], dtype=np.uint8)
        bottom_left = (0, 0)
        top_right = (magnification * self.size, magnification * self.size)
        cv2.rectangle(image, bottom_left, top_right, self.WHITE, thickness=cv2.FILLED)
        cv2.rectangle(image, bottom_left, top_right, self.BLACK, thickness=int(magnification * 0.02))

        for wall in self.walls:
            wall_bottom_left = wall.bottom_left.scale(magnification)
            wall_top_right = wall.top_right.scale(magnification)
            cv2.rectangle(image, wall_bottom_left.pt(), wall_top_right.pt(), self.BLACK, thickness=cv2.FILLED)

        agent_centre = self.position.scale(magnification)
        agent_radius = int(self.agent_radius * magnification)
        agent_colour = self.BLACK
        cv2.circle(image, agent_centre.pt(), agent_radius, agent_colour, cv2.FILLED)

        goal_centre = self.goal.scale(magnification)
        goal_radius = int(self.agent_radius * magnification)
        goal_colour = self.GREEN
        cv2.circle(image, goal_centre.pt(), goal_radius, goal_colour, cv2.FILLED)

        # Show the image
        cv2.imshow("Environment", image)
        if save:
            cv2.imwrite("Policy.png", image)
        # This line is necessary to give time for the image to be rendered on the screen
        cv2.waitKey(1)


class MazeModel:

    def __init__(self, room_models):
        self.room_models = room_models

    def predict(self, position):
        for room, room_model in self.room_models:
            if room.inside(position):
                action = room_model.get_greedy_action(position.to_numpy())
                return action


class Room:

    def __init__(self, area, entrypoints, walls):
        self.area = area
        self.entrypoints = entrypoints
        self.walls = walls

    def inside(self, position, radius=0):
        for wall in self.walls:
            if wall.collides_with_agent(position, radius):
                return False

        return self.area.inside(position)

    def reset(self):
        return self.entrypoints[np.random.choice(len(self.entrypoints))]


class Area:

    def __init__(self, rectangles):
        self.rectangles = rectangles

    def inside(self, position):
        for rect in self.rectangles:
            if rect.inside(position):
                return True
        return False

    def overlaps_rectangle(self, rectangle):
        for rect in self.rectangles:
            if rect.overlaps(rectangle):
                return True
        return False

    def overlaps_area(self, other_area):
        for other_rect in other_area.rectangles:
            if self.overlaps_rectangle(other_rect):
                return True
        return False


class Rectangle:

    def __init__(self, bottom_left, top_right):
        self.bottom_left = bottom_left
        self.top_right = top_right

    def inside(self, position):
        return self.bottom_left.x <= position.x <= self.top_right.x and self.bottom_left.y <= position.y <= self.top_right.y

    def overlaps(self, other_rectangle):
        return self.inside(other_rectangle.bottom_left) or other_rectangle.inside(self.bottom_left)

    def collides_with_agent(self, agent_position, radius):
        hitbox_bottom_left = Point(agent_position.x - radius, agent_position.y + radius)
        hitbox_top_right = Point(agent_position.x + radius, agent_position.y + radius)
        hitbox = Rectangle(hitbox_bottom_left, hitbox_top_right)
        return self.overlaps(hitbox)


class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    @staticmethod
    def from_numpy(np_array):
        return Point(np_array[0], np_array[1])

    def add(self, point):
        return Point(self.x + point.x, self.y + point.y)

    def scale(self, factor):
        return Point(self.x * factor, self.y * factor)

    def pt(self):
        return (int(self.x), int(self.y))

    def to_numpy(self):
        return np.array([self.x, self.y])
