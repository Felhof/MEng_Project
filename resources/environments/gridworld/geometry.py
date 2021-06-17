import numpy as np


RED = (255, 0, 0)
BLUE = (0, 0, 255)
PINK = (230, 50, 210)


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
