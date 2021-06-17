from resources.environments.gridworld.geometry import Point


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
