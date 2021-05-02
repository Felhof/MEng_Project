from resources.environments.room_environment import RoomEnvironment
from resources.environments.maze import Maze, MazeModel, Point, Rectangle, Area
from resources.sac_agent import SACAgent

import numpy as np

import time
import matplotlib.pyplot as plt

import torch


def main(config, iterations=4):
    maze = Maze(config)
    maze_models = []

    training_steps = config["training_steps"]
    iteration_rewards = []
    times_goal_reached = 0

    for _ in range(iterations):
        maze_model = train_maze_model(maze, training_steps, config["name"])
        iteration_reward = 0
        reached_goal = False
        obs = maze.reset()
        for n in range(300):
            action = maze_model.predict(obs)
            obs, reward = maze.step(action)
            reached_goal = reached_goal or maze.is_goal(obs)
            iteration_reward += reward
            #maze.render()
            #time.sleep(0.01)

        maze_models.append(maze_model)
        iteration_rewards.append(np.mean(iteration_reward))

    alpha = 1/iterations
    position = maze.reset()
    position = Point(position[0], position[1])
    image = maze.draw()

    for maze_model in maze_models:
        image = maze.draw_policy(image, maze_model, position, alpha=alpha)

    maze.save_policy(image, config["name"] + "_average")
    print("{0} reward mean: {1}".format(config["name"], np.mean(iteration_rewards)))
    print("{0} reward variance: {1}".format(config["name"], np.var(iteration_rewards)))
    print("{0} reached goal {1} times".format(config["name"], times_goal_reached))


def train_maze_model(maze, training_steps, name, show=False):
    room_models = []

    for idx, (room, steps) in enumerate(zip(maze.rooms, training_steps)):
        title = "{0}_Room_Model_Lvl_{1}".format(name, idx)
        room_model, environment = train_room_model(maze=maze, room=room, lower_lvls=room_models, training_steps=steps, title=title)
        room_models.append((room, room_model))

        state = environment.reset()
        if show:
            for _ in range(100):
                action = room_model.predict(state)
                next_state, reward, done, info = environment.step(action)
                if done:
                    state = environment.reset()
                else:
                    state = next_state
                environment.render(show_policy=True, model=room_model)
                time.sleep(0.01)

    maze_model = MazeModel(room_models)

    return maze_model


def train_room_model(maze=None, room=None, lower_lvls=None, training_steps=10000, episode_length=400,
                     backup_frequency=1, title="Room_Model", display_on=False, plot=False):

    environment = RoomEnvironment(maze=maze, room=room, lower_levels=lower_lvls, episode_length=episode_length)

    room_model = SACAgent(episode_length=episode_length)

    state = environment.reset()

    reward_per_episode = []
    current_episode_reward = 0

    backup_model = None
    best_evaluation = -10**5

    episode = 1

    for n in range(training_steps):
        print("Episode {0}/{1}".format(n + 1, training_steps), end='\r')
        action = room_model.get_next_action(state)
        next_state, reward, done, info = environment.step(action)
        current_episode_reward += reward
        room_model.set_next_state_and_reward(next_state, reward)
        state = next_state
        if done:
            if episode % backup_frequency == 0:
                evaluation_score = evaluate_model(room_model, environment)
                if evaluation_score > best_evaluation:
                    best_evaluation = evaluation_score
                    backup_model = room_model.create_backup_model()
            episode += 1
            reward_per_episode.append(current_episode_reward)
            current_episode_reward = 0
            state = environment.reset()
        # Optionally, show the environment
        if display_on:
            environment.render(show_policy=False, model=room_model)
            time.sleep(0.02)

    evaluation_score = evaluate_model(room_model, environment)
    if evaluation_score > best_evaluation:
        backup_model = room_model.create_backup_model()

    if plot:
        position = environment.reset()
        position = Point(position[0], position[1])
        image = maze.draw()
        image = maze.draw_policy(image, room_model, position, room=environment.room)
        maze.save_policy(image, title)

        plt.clf()
        plt.figure(figsize=(20, 10))
        plt.title(title)
        plt.plot(range(len(reward_per_episode)), reward_per_episode, "b", label="Reward")
        plt.legend()
        plt.xlabel("episode")
        plt.ylabel("reward")
        plt.axhline(y=0, color='r', linestyle='-')
        plt.savefig("../img/{}_learning_curve".format(title))

    return backup_model, environment


def evaluate_model(model, environment, steps=150):
    state = environment.reset()
    total_reward = 0
    for _ in range(steps):
        action = model.predict(state)
        next_state, reward, done, info = environment.step(action)
        total_reward += reward
        if done:
            break
        state = next_state
    return total_reward

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
    "size": 1,
    "start": Point(0.1, 0.9),
    "goal": Point(0.9, 0.1),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0., 0.), Point(1., 0.4))]),
            "entrypoints": [Point(0.25, 0.35), Point(0.35, 0.35)]
        },
        {
            "lvl": 1,
            "area": Area([Rectangle(Point(0., 0.4), Point(1., 0.7))]),
            "entrypoints": [Point(0.55, 0.65), Point(0.65, 0.65)]
        },
        {
            "lvl": 2,
            "area": Area([Rectangle(Point(0., 0.7), Point(1., 1.))]),
            "entrypoints": [Point(0.1, 0.9)]
        }
    ],
    "walls": [Rectangle(Point(0., 0.3), Point(0.2, 0.4)), Rectangle(Point(0.4, 0.3), Point(1., 0.4)),
              Rectangle(Point(0., 0.6), Point(0.5, 0.7)), Rectangle(Point(0.7, 0.6), Point(1., 0.7))]
}

two_rooms_two_areas_maze_config = {
    "name": "2-rooms-2-areas-b",
    "training_steps": [5000, 10000],
    "size": 1,
    "start": Point(0.4, 0.8),
    "goal": Point(0.9, 0.1),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0., 0.), Point(1., 0.5))]),
            "entrypoints": [Point(0.35, 0.45), Point(0.45, 0.45)],
            "policy color": "blue"
        },
        {
            "lvl": 1,
            "area": Area([Rectangle(Point(0., 0.5), Point(1., 1.))]),
            "entrypoints": [Point(0.4, 0.8)],
            "policy color": "red"
        }
    ],
    "walls": [Rectangle(Point(0., 0.4), Point(0.3, 0.5)), Rectangle(Point(0.5, 0.4), Point(1., 0.5))]
}

two_rooms_one_area_maze_config = {
    "name": "2-rooms-1-area-b",
    "training_steps": [30000],
    "size": 1,
    "start": Point(0.4, 0.8),
    "goal": Point(0.9, 0.1),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0., 0.), Point(1., 1.))]),
            "entrypoints": [Point(0.4, 0.8)]
        }
    ],
    "walls": [Rectangle(Point(0., 0.4), Point(0.3, 0.5)), Rectangle(Point(0.5, 0.4), Point(1., 0.5))]
}

four_rooms_four_areas_maze_config = {
    "name": "4-rooms-4-areas",
    "training_steps": [20000, 20000, 20000, 20000],
    "size": 2,
    "start": Point(0.15, 0.15),
    "goal": Point(1.8, 1.8),
    "rooms": [
        {
            # top right
            "lvl": 0,
            "area": Area([Rectangle(Point(1.0, 0.9), Point(2., 2.))]),
            "entrypoints": [Point(1.55, 0.95), Point(1.65, 0.95),
                            Point(1.05, 1.45), Point(1.05, 1.55)],
            "policy color": "red"
        },
        {
            # top left
            "lvl": 1,
            "area": Area([Rectangle(Point(0.0, 0.9), Point(1.0, 2.0))]),
            "entrypoints": [Point(0.35, 0.95), Point(0.45, 0.95)],
            "policy color": "blue"
        },
        {
            # bottom right
            "lvl": 2,
            "area": Area([Rectangle(Point(1.0, 0.0), Point(2.0, 0.9))]),
            "entrypoints": [Point(1.05, 0.35), Point(1.05, 0.35)],
            "policy color": "blue"
        },
        {
            # bottom left
            "lvl": 3,
            "area": Area([Rectangle(Point(0., 0.), Point(1.0, 0.9))]),
            "entrypoints": [Point(0.15, 0.15)],
            "policy color": "pink"
        }
    ],
    "walls": [Rectangle(Point(0.0, 0.9), Point(0.3, 1.0)), Rectangle(Point(0.5, 0.9), Point(1.5, 1.0)),
              Rectangle(Point(1.0, 0.0), Point(1.1, 0.3)), Rectangle(Point(1.0, 0.5), Point(1.1, 1.4)),
              Rectangle(Point(1.7, 0.9), Point(2.0, 1.0)), Rectangle(Point(1.0, 1.6), Point(1.1, 2.0))]
}

four_rooms_one_area_maze_config = {
    "name": "4-rooms-1-area",
    "training_steps": [80000],
    "size": 2,
    "start": Point(0.15, 0.15),
    "goal": Point(1.8, 1.8),
    "rooms": [
        {
            # bottom left
            "lvl": 1,
            "area": Area([Rectangle(Point(0., 0.), Point(2.0, 2.0))]),
            "entrypoints": [Point(0.15, 0.15)],
            "policy color": "blue"
        }
    ],
    "walls": [Rectangle(Point(0.0, 0.9), Point(0.3, 1.0)), Rectangle(Point(0.5, 0.9), Point(1.5, 1.0)),
              Rectangle(Point(1.0, 0.0), Point(1.1, 0.3)), Rectangle(Point(1.0, 0.5), Point(1.1, 1.4)),
              Rectangle(Point(1.7, 0.9), Point(2.0, 1.0)), Rectangle(Point(1.0, 1.6), Point(1.1, 2.0))]
}

two_paths_maze_config = {
    "name": "2_paths_4_areas",
    "training_steps": [20000, 50000, 50000, 50000, 20000],
    "size": 2.5,
    "start": Point(0.1, 0.2),
    "goal": Point(2.3, 1.0),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(1.7, 0.6), Point(2.5, 1.4))]),
            "entrypoints": [Point(2.05, 0.65), Point(2.15, 0.65),
                            Point(2.05, 1.35), Point(2.15, 1.35)],
            "policy color": "red"
        },
        {
            "lvl": 1,
            "area": Area([Rectangle(Point(0.8, 0.), Point(2.5, 0.6))]),
            "entrypoints": [Point(0.85, 0.15), Point(0.85, 0.25)],
            "policy color": "blue"
        },
        {
            "lvl": 2,
            "area": Area([Rectangle(Point(0.8, 1.3), Point(2.5, 2.5))]),
            "entrypoints": [Point(1.35, 1.35), Point(1.45, 1.35)],
            "policy color": "blue"
        },
        {
            "lvl": 3,
            "area": Area([Rectangle(Point(0.8, 0.6), Point(1.7, 1.3))]),
            "entrypoints": [Point(0.85, 0.75), Point(0.85, 0.85)],
            "policy color": "pink"
        },
        {
            "lvl": 4,
            "area": Area([Rectangle(Point(0.0, 0.0), Point(0.8, 2.5))]),
            "entrypoints": [Point(0.1, 0.2)],
            "policy color": "red"
        },
    ],
    "walls": [Rectangle(Point(0.8, 0.0), Point(0.9, 0.1)), Rectangle(Point(0.8, 0.3), Point(0.9, 0.7)),
              Rectangle(Point(0.8, 0.6), Point(2.0, 0.7)), Rectangle(Point(2.2, 0.6), Point(2.5, 0.7)),
              Rectangle(Point(0.8, 0.9), Point(0.9, 2.5)), Rectangle(Point(0.9, 1.3), Point(1.3, 1.4)),
              Rectangle(Point(1.7, 0.6), Point(1.8, 1.4)), Rectangle(Point(1.5, 1.3), Point(2.0, 1.4)),
              Rectangle(Point(2.2, 1.3), Point(2.5, 1.4))]
}

u_turn_maze_config = {
    "name": "u_turn",
    "training_steps": [20000, 40000, 40000],
    "size": 2.5,
    "start": Point(0.85, 0.75),
    "goal": Point(2.3, 1.0),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(1.7, 0.6), Point(2.5, 1.4))]),
            "entrypoints": [Point(2.05, 0.65), Point(2.15, 0.65),
                            Point(2.05, 1.35), Point(2.15, 1.35)],
            "policy color": "red"
        },
        {
            "lvl": 2,
            "area": Area([Rectangle(Point(0.8, 1.3), Point(2.5, 2.5))]),
            "entrypoints": [Point(1.35, 1.35), Point(1.45, 1.35)],
            "policy color": "blue"
        },
        {
            "lvl": 3,
            "area": Area([Rectangle(Point(0.8, 0.6), Point(1.7, 1.3))]),
            "entrypoints": [Point(0.85, 0.75)],
            "policy color": "pink"
        }
    ],
    "walls": [Rectangle(Point(0.8, 0.0), Point(0.9, 0.1)), Rectangle(Point(0.8, 0.3), Point(0.9, 0.7)),
              Rectangle(Point(0.8, 0.6), Point(2.0, 0.7)), Rectangle(Point(2.2, 0.6), Point(2.5, 0.7)),
              Rectangle(Point(0.8, 0.9), Point(0.9, 2.5)), Rectangle(Point(0.9, 1.3), Point(1.3, 1.4)),
              Rectangle(Point(1.7, 0.6), Point(1.8, 1.4)), Rectangle(Point(1.5, 1.3), Point(2.0, 1.4)),
              Rectangle(Point(2.2, 1.3), Point(2.5, 1.4))]
}


hard_turn_maze_config = {
    "name": "hard_turn",
    "training_steps": [20000, 40000],
    "size": 2.5,
    "start": Point(1.35, 1.35),
    "goal": Point(2.3, 1.0),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(1.7, 0.6), Point(2.5, 1.4))]),
            "entrypoints": [Point(2.05, 1.35), Point(2.15, 1.35)],
            "policy color": "red"
        },
        {
            "lvl": 2,
            "area": Area([Rectangle(Point(0.8, 1.3), Point(2.5, 2.5))]),
            "entrypoints": [Point(1.35, 1.35), Point(1.45, 1.35)],
            "policy color": "blue"
        }
    ],
    "walls": [Rectangle(Point(0.8, 0.0), Point(0.9, 0.1)), Rectangle(Point(0.8, 0.3), Point(0.9, 0.7)),
              Rectangle(Point(0.8, 0.6), Point(2.0, 0.7)), Rectangle(Point(2.2, 0.6), Point(2.5, 0.7)),
              Rectangle(Point(0.8, 0.9), Point(0.9, 2.5)), Rectangle(Point(0.9, 1.3), Point(1.3, 1.4)),
              Rectangle(Point(1.7, 0.6), Point(1.8, 1.4)), Rectangle(Point(1.5, 1.3), Point(2.0, 1.4)),
              Rectangle(Point(2.2, 1.3), Point(2.5, 1.4)), Rectangle(Point(1.3, 1.2), Point(1.5, 1.3)),
              Rectangle(Point(2.0, 0.5), Point(2.2, 0.6))]
}


if __name__ == "__main__":
    main(two_rooms_two_areas_maze_config)
    main(hard_turn_maze_config)
    #main(four_rooms_four_areas_maze_config)
