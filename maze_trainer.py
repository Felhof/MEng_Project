from resources.environments.gridworld.room_environment import RoomEnvironment
from resources.environments.gridworld.maze_environment import MazeEnvironment
from resources.environments.gridworld.geometry import Area, Rectangle, Point
from resources.environments.gridworld.maze_model import MazeModel
from resources.sac_agent import SACAgent

import multiprocessing as mp
import time
import matplotlib.pyplot as plt


SECONDS_PER_MINUTE = 60
MAX_WORKERS = 4


def main(config, iterations=15):
    maze = MazeEnvironment(config)
    maze_models = []

    for _ in range(iterations):
        maze_model = train_maze_model(maze, config["name"])
        obs = maze.reset()
        for n in range(300):
            action = maze_model.predict(obs)
            obs, reward = maze.step(action)

        maze_models.append(maze_model)

    alpha = 1/iterations
    position = maze.reset()
    position = Point(position[0], position[1])
    image = maze.draw()

    for maze_model in maze_models:
        image = maze.draw_policy(image, maze_model, position, alpha=alpha)

    maze.save_policy(image, config["name"] + "_average")


def train_maze_model(maze, name, show=False):
    num_workers = min(mp.cpu_count(), MAX_WORKERS)

    room_models = []
    lvl = 0
    while lvl in maze.rooms:
        pool = mp.Pool(num_workers)
        rooms = maze.rooms[lvl]
        room_model_results = []
        for idx, room in enumerate(rooms):
            title = "{0}_Room_Model_Lvl_{1}_Room_{2}".format(name, lvl, idx)
            room_model_results.append((room, pool.apply_async(train_room_model, (),
                                                              {"maze": maze,
                                                               "room": room,
                                                               "lower_lvls": room_models,
                                                               "title": title})))

        for room_model_result in room_model_results:
            room = room_model_result[0]
            room_model, environment = room_model_result[1].get()
            room_models.append((room, room_model))

            if show:
                state = environment.reset()
                for _ in range(100):
                    action = room_model.predict(state)
                    next_state, reward, done, info = environment.step(action)
                    if done:
                        state = environment.reset()
                    else:
                        state = next_state
                    environment.render(show_policy=True, model=room_model)
                    time.sleep(0.01)

        lvl += 1

    maze_model = MazeModel(room_models)

    return maze_model


def train_room_model(maze=None, room=None, lower_lvls=None, target_network_update_interval=25, title="Room_Model",
                     display_on=False, plot=False):

    environment = RoomEnvironment(maze=maze, room=room, lower_levels=lower_lvls)

    room_model = SACAgent()

    state = environment.reset()

    reward_per_episode = []
    episode_reward = 0

    start_time = time.time()
    end_time = start_time + room.training_time

    num_steps_taken = 0

    while time.time() < end_time:
        time_passed = round(time.time() - start_time, 2)
        print("{0}/{1}".format(time_passed, room.training_time), end='\r')
        action = room_model.get_next_action(state)
        next_state, reward, done, info = environment.step(action)
        episode_reward += reward
        room_model.train_on_transition(state, action, next_state, reward)
        if num_steps_taken % target_network_update_interval == 0:
            room_model.update_target_networks()
        state = next_state
        if done:
            reward_per_episode.append(episode_reward)
            episode_reward = 0
            state = environment.reset()
        if display_on:
            environment.render(show_policy=False, model=room_model)
            time.sleep(0.02)
        num_steps_taken += 1

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

    return room_model, environment


three_rooms_one_area_maze_config = {
    "name": "3-rooms-1-area",
    "size": 1,
    "start": Point(0.15, 0.75),
    "goal": Point(0.9, 0.5),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0.0, 0.0), Point(1., 1.))]),
            "entrypoints": [Point(0.15, 0.75)],
            "policy color": "red",
            "time": 10*SECONDS_PER_MINUTE
        }
    ],
    "walls": [Rectangle(Point(0.3, 0.0), Point(0.4, 0.3)), Rectangle(Point(0.3, 0.5), Point(0.4, 1.0)),
              Rectangle(Point(0.6, 0.0), Point(0.7, 0.6)), Rectangle(Point(0.6, 0.8), Point(0.7, 1.0))]
}



three_rooms_three_areas_maze_config = {
    "name": "3-rooms-3-areas",
    "size": 1,
    "start": Point(0.15, 0.75),
    "goal": Point(0.9, 0.5),
    "rooms": [
        {
            # left
            "lvl": 0,
            "area": Area([Rectangle(Point(0.6, 0.0), Point(1., 1.))]),
            "entrypoints": [Point(0.65, 0.65), Point(0.65, 0.75)],
            "policy color": "red",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # middle
            "lvl": 1,
            "area": Area([Rectangle(Point(0.3, 0.0), Point(0.6, 1.0))]),
            "entrypoints": [Point(0.35, 0.35), Point(0.35, 0.45)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # right
            "lvl": 2,
            "area": Area([Rectangle(Point(0.0, 0.0), Point(0.3, 1.0))]),
            "entrypoints": [Point(0.15, 0.75)],
            "policy color": "pink",
            "time": 10*SECONDS_PER_MINUTE/3
        }
    ],
    "walls": [Rectangle(Point(0.3, 0.0), Point(0.4, 0.3)), Rectangle(Point(0.3, 0.5), Point(0.4, 1.0)),
              Rectangle(Point(0.6, 0.0), Point(0.7, 0.6)), Rectangle(Point(0.6, 0.8), Point(0.7, 1.0))]
}

four_rooms_four_areas_maze_config = {
    "name": "4-rooms-4-areas",
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
            "policy color": "red",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # top left
            "lvl": 1,
            "area": Area([Rectangle(Point(0.0, 0.9), Point(1.0, 2.0))]),
            "entrypoints": [Point(0.35, 0.95), Point(0.45, 0.95)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # bottom right
            "lvl": 1,
            "area": Area([Rectangle(Point(1.0, 0.0), Point(2.0, 0.9))]),
            "entrypoints": [Point(1.05, 0.35), Point(1.05, 0.35)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # bottom left
            "lvl": 2,
            "area": Area([Rectangle(Point(0., 0.), Point(1.0, 0.9))]),
            "entrypoints": [Point(0.15, 0.15)],
            "policy color": "pink",
            "time": 10*SECONDS_PER_MINUTE/3
        }
    ],
    "walls": [Rectangle(Point(0.0, 0.9), Point(0.3, 1.0)), Rectangle(Point(0.5, 0.9), Point(1.5, 1.0)),
              Rectangle(Point(1.0, 0.0), Point(1.1, 0.3)), Rectangle(Point(1.0, 0.5), Point(1.1, 1.4)),
              Rectangle(Point(1.7, 0.9), Point(2.0, 1.0)), Rectangle(Point(1.0, 1.6), Point(1.1, 2.0))]
}

four_rooms_one_area_maze_config = {
    "name": "4-rooms-1-area",
    "size": 2,
    "start": Point(0.15, 0.15),
    "goal": Point(1.8, 1.8),
    "rooms": [
        {
            "lvl": 0,
            "area": Area([Rectangle(Point(0., 0.), Point(2.0, 2.0))]),
            "entrypoints": [Point(0.15, 0.15)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE
        }
    ],
    "walls": [Rectangle(Point(0.0, 0.9), Point(0.3, 1.0)), Rectangle(Point(0.5, 0.9), Point(1.5, 1.0)),
              Rectangle(Point(1.0, 0.0), Point(1.1, 0.3)), Rectangle(Point(1.0, 0.5), Point(1.1, 1.4)),
              Rectangle(Point(1.7, 0.9), Point(2.0, 1.0)), Rectangle(Point(1.0, 1.6), Point(1.1, 2.0))]
}

maze_with_local_maxmimum_4_areas = {
    "name": "maze_with_local_maxmimum_4_areas",
    "size": 2,
    "start": Point(0.95, 1.65),
    "goal": Point(1.5, 0.3),
    "rooms": [
        {
            # top right
            "lvl": 0,
            "area": Area([Rectangle(Point(0.9, 0.0), Point(2., 0.7))]),
            "entrypoints": [Point(0.95, 0.25), Point(0.95, 0.25)],
            "policy color": "red",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # top left
            "lvl": 1,
            "area": Area([Rectangle(Point(0.0, 0.0), Point(0.9, 1.4))]),
            "entrypoints": [Point(0.55, 1.35), Point(0.65, 1.35)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # middle right
            "lvl": 1,
            "area": Area([Rectangle(Point(0.9, 0.7), Point(2.0, 1.4))]),
            "entrypoints": [Point(1.25, 1.35), Point(1.35, 1.35)],
            "policy color": "blue",
            "time": 10*SECONDS_PER_MINUTE/3
        },
        {
            # bottom
            "lvl": 2,
            "area": Area([Rectangle(Point(0.0, 1.4), Point(2.0, 2.0))]),
            "entrypoints": [Point(0.95, 1.65)],
            "policy color": "pink",
            "time": 10*SECONDS_PER_MINUTE/3
        }
    ],
    "walls": [Rectangle(Point(0.9, 0.0), Point(1.0, 0.2)), Rectangle(Point(0.9, 0.4), Point(1.0, 1.4)),
              Rectangle(Point(0.9, 0.6), Point(2.0, 0.7)), Rectangle(Point(0.0, 1.3), Point(0.5, 1.4)),
              Rectangle(Point(0.7, 1.3), Point(1.2, 1.4)), Rectangle(Point(1.4, 1.3), Point(2.0, 1.4))]
}


maze_with_local_maxmimum_1_area = {
    "name": "maze_with_local_maxmimum_1_area",
    "size": 2,
    "start": Point(0.95, 1.65),
    "goal": Point(1.5, 0.3),
    "rooms": [
        {
            # top right
            "lvl": 0,
            "area": Area([Rectangle(Point(0.0, 0.0), Point(2., 2.))]),
            "entrypoints": [Point(0.95, 1.65)],
            "policy color": "red",
            "time": 10*SECONDS_PER_MINUTE
        }
    ],
    "walls": [Rectangle(Point(0.9, 0.0), Point(1.0, 0.2)), Rectangle(Point(0.9, 0.4), Point(1.0, 1.4)),
              Rectangle(Point(0.9, 0.6), Point(2.0, 0.7)), Rectangle(Point(0.0, 1.3), Point(0.5, 1.4)),
              Rectangle(Point(0.7, 1.3), Point(1.2, 1.4)), Rectangle(Point(1.4, 1.3), Point(2.0, 1.4))]
}


if __name__ == "__main__":
    main(three_rooms_three_areas_maze_config)
