from resources.environments.room_environment import RoomEnvironment
from resources.environments.maze import Maze, MazeModel, Point, Rectangle, Area
from resources.environments.sac_agent import SACAgent

import time


def main():
    maze = Maze(simple_maze_config)
    maze_model = train_maze_model(maze)

    for n in range(2000):
        if n % 300 == 0:
            obs = maze.reset()
        action = maze_model.predict(obs)
        obs = maze.step(action)
        maze.draw()


def train_maze_model(maze):
    room_models = []

    for room in maze.rooms:
        room_model, environment = train_room_model(maze=maze, room=room, lower_lvls=room_models)
        room_models.append((room, room_model))

        state = environment.reset()
        for _ in range(1000):
            action = room_model.get_greedy_action(state)
            next_state, reward, done, info = environment.step(action)
            if done:
                state = environment.reset()
            else:
                state = next_state
            environment.render()
            time.sleep(0.01)

    maze_model = MazeModel(room_models)

    return maze_model


def train_room_model(maze=None, room=None, lower_lvls=None, training_episodes=50, display_on=False):

    environment = RoomEnvironment(maze=maze, room=room, lower_levels=lower_lvls)

    room_model = SACAgent()

    state = environment.reset()

    reward_per_episode = []
    episode_reward = 0

    current_episode = 1
    while current_episode <= training_episodes:
        print("Episode {0}/{1}".format(current_episode, training_episodes), end='\r')
        action = room_model.get_next_action(state)
        next_state, reward, done, info = environment.step(action)
        episode_reward += reward
        room_model.set_next_state_and_reward(next_state, reward)
        state = next_state
        if done:
            reward_per_episode.append(episode_reward)
            episode_reward = 0
            state = environment.reset()
            current_episode += 1
        # Optionally, show the environment
        if display_on:
            environment.render()
            time.sleep(0.01)

    return room_model, environment


simple_maze_config = {
    "size": 1,
    "start": Point(0.1, 0.1),
    "goal": Point(0.1, 0.8),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0., 0.), Point(1., 1.))]),
            "entrypoints": [Point(0.1, 0.1)]
        }
    ],
    "walls": [Rectangle(Point(0., 0.4), Point(0.4, 0.5))]
}

example_maze_config = {
    "size": 10,
    "start": Point(1, 9),
    "goal": Point(9, 1),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0, 0), Point(10, 4))]),
            "entrypoints": [Point(0.5, 3.5), Point(1.5, 3.5)]
        },
        {
            "lvl": 1,
            "area": Area([Rectangle(Point(0, 4), Point(10, 7))]),
            "entrypoints": [Point(8.5, 6.5), Point(9.5, 6.5)]
        },
        {
            "lvl": 2,
            "area": Area([Rectangle(Point(0, 7), Point(10, 10))]),
            "entrypoints": [Point(1, 9)]
        }
    ],
    "walls": [Rectangle(Point(2, 3), Point(10, 4)), Rectangle(Point(0, 6), Point(8, 7))]
}


if __name__ == "__main__":
    main()
