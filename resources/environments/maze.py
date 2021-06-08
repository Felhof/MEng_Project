import cv2
from gym import spaces
import numpy as np


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
PINK = (230, 50, 210)

class Maze:
    AGENT_SIZE = 0.05

    DIRECTIONS = 4
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def __init__(self, config=None):
        self.size = config["size"]
        self.agent_radius = self.AGENT_SIZE * 0.45
        self.start = config["start"]
        self.position = self.start
        self.goal = config["goal"]
        self.walls = config["walls"]
        self.observation_space = spaces.Box(low=0., high=self.size, shape=(2, ), dtype=np.float32)
        self.action_space = spaces.Discrete(self.DIRECTIONS)

        self.rooms = {}
        for room_config in config["rooms"]:
            room_walls = []
            for wall in config["walls"]:
                if room_config["area"].overlaps_rectangle(wall):
                    room_walls.append(wall)
            room = Room(room_config["area"], room_config["entrypoints"], room_walls, room_config["policy color"],
                        room_config["time"])
            rooms = self.rooms.get(room_config["lvl"], [])
            rooms.append(room)
            self.rooms[room_config["lvl"]] = rooms

    def step(self, action):
        next_position = self.move(action)

        for wall in self.walls:
            if wall.inside(next_position):
                next_position = self.position

        self.position = next_position

        reward = self.size - np.linalg.norm(self.goal.to_numpy() - self.position.to_numpy())

        return self.position.to_numpy(), reward

    def action_to_direction(self, action):
        if action == self.UP:
            direction = Point(0, self.AGENT_SIZE)
        elif action == self.RIGHT:
            direction = Point(self.AGENT_SIZE, 0)
        elif action == self.DOWN:
            direction = Point(0, -self.AGENT_SIZE)
        elif action == self.LEFT:
            direction = Point(-self.AGENT_SIZE, 0)

        return direction

    def move(self, action):
        direction = self.action_to_direction(action)

        next_position = self.position.add(direction)

        if next_position.x + self.agent_radius >= self.size or next_position.y + self.agent_radius >= self.size:
            next_position = self.position

        if next_position.x - self.agent_radius <= 0 or next_position.y - self.agent_radius <= 0:
            next_position = self.position

        return next_position

    def reset(self):
        self.position = self.start
        return self.position.to_numpy()

    def render(self, show_policy=False, model=None):
        image = self.draw()
        if show_policy:
            image = self.draw_policy(image, model, self.position)
        self.show_image(image)

    def show_image(self, image, save=False):
        cv2.imshow("Environment", image)
        cv2.waitKey(1)

    def save_policy(self, image, title):
        cv2.imshow("Environment", image)
        cv2.imwrite("../img/{}_policy.png".format(title), image)

    def draw(self, magnification=500):
        image = np.zeros([int(magnification * self.size), int(magnification * self.size), 3], dtype=np.uint8)
        bottom_left = (0, 0)
        top_right = (int(magnification * self.size), int(magnification * self.size))
        cv2.rectangle(image, bottom_left, top_right, WHITE, thickness=cv2.FILLED)
        cv2.rectangle(image, bottom_left, top_right, BLACK, thickness=int(magnification * 0.02))

        for wall in self.walls:
            wall_bottom_left = wall.bottom_left.scale(magnification)
            wall_top_right = wall.top_right.scale(magnification)
            cv2.rectangle(image, wall_bottom_left.pt(), wall_top_right.pt(), BLACK, thickness=cv2.FILLED)

        agent_centre = self.position.scale(magnification)
        agent_radius = int(self.agent_radius * magnification)
        agent_colour = BLACK
        cv2.circle(image, agent_centre.pt(), agent_radius, agent_colour, cv2.FILLED)

        goal_centre = self.goal.scale(magnification)
        goal_radius = int(self.agent_radius * magnification)
        goal_colour = GREEN
        cv2.circle(image, goal_centre.pt(), goal_radius, goal_colour, cv2.FILLED)

        return image

    def draw_policy(self, image, model, start, magnification=500, alpha=1., room=None):
        state = start
        overlay = image.copy()
        for _ in range(100):
            arrow_color = model.get_room_color(state)
            action = model.predict(state.to_numpy())
            if action is None:
                break
            direction = self.action_to_direction(action)
            next_state = state.add(direction)
            start = state.add(direction.scale(0.1))
            end = state.add(direction.scale(0.9))
            line_start = (int(magnification * start.x), int(magnification * start.y))
            line_end = (int(magnification * end.x), int(magnification * end.y))
            cv2.arrowedLine(overlay, line_start, line_end, arrow_color, thickness=3, tipLength=0.5)
            for wall in self.walls:
                if wall.inside(next_state):
                    break
            if room is not None:
                if not room.inside(next_state):
                    break
            state = next_state
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        return image

    def is_goal(self, point):
        return self.goal.x == point[0] and self.goal.y == point[1]


class MazeModel:

    def __init__(self, room_models):
        self.room_models = room_models

    def predict(self, obs):
        position = Point(obs[0], obs[1])
        for room, room_model in self.room_models:
            if room.inside(position):
                action = room_model.predict(obs)
                return action

    def get_room_color(self, position):
        for room, room_model in self.room_models:
            if room.inside(position):
                return room.color


class Room:

    def __init__(self, area, entrypoints, walls, color, training_time):
        self.area = area
        self.entrypoints = entrypoints
        self.walls = walls
        self.color = {
            "red": RED,
            "blue": BLUE,
            "pink": PINK
            }[color]
        self.training_time = training_time

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
        return self.inside(other_rectangle.bottom_left) or self.inside(other_rectangle.top_right) or other_rectangle.inside(self.bottom_left)

    def collides_with_agent(self, agent_position, radius):
        hitbox_bottom_left = Point(agent_position.x - radius, agent_position.y - radius)
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
