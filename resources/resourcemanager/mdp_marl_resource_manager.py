from resources.resourcemanager.base_resource_manager import BaseResourceManager

from stable_baselines3 import A2C
from stable_baselines3.common.cmd_util import make_vec_env

from resources.callbacks import ProgressBarManager
from resources.rap_environment import MDPResourceAllocationEnvironment, RestrictedMDPResourceAllocationEnvironment
from resources.MDP import MDPBuilder, RestrictedMDP
from resources import MTA
from resources.multistage_model import MultiStageActorCritic
import torch


class MDPMultiAgentResourceManager(BaseResourceManager):

    def __init__(self, rap, training_steps=50000, steps_per_episode=500, log_dir="/tmp/gym"):
        super(MDPMultiAgentResourceManager, self).__init__(rap, log_dir=log_dir)

        self.rap_mdp = MDPBuilder(self.ra_problem).build_mdp()
        self.rap_mdp.transform(3)
        state_idx = list(self.rap_mdp.idx_to_state.keys())
        state_dllst = MTA.DLLst(state_idx)

        MTA.mta_for_scc_and_levels(state_dllst, self.rap_mdp)
        scc_lst = MTA.SCC_lst
        scc_lst.sort(key=lambda scc: scc.lvl)
        self.levels = []
        current_level = []
        level = 0
        for scc in scc_lst:
            if scc.lvl != level:
                self.levels.append(current_level)
                level += 1
                current_level = []
            current_level.append(scc)
        self.levels.append(current_level)

        self.training_steps = training_steps
        self.steps_per_episode = steps_per_episode

    def train_model(self):
        models = []

        lower_level_values = {}
        for lvl_id, level in enumerate(self.levels):
            current_level_values = {}
            for scc_id, scc in enumerate(level):
                state_idxs = scc.get_state_idxs()
                restricted_mdp = RestrictedMDP(self.rap_mdp, state_idxs, lower_level_values)
                environment = RestrictedMDPResourceAllocationEnvironment(self.ra_problem, restricted_mdp)
                vector_environment = make_vec_env(lambda: environment, n_envs=1, monitor_dir=self.log_dir)
                model = A2C('MlpPolicy', vector_environment, verbose=1, tensorboard_log=self.log_dir)

                with ProgressBarManager(self.training_steps) as progress_callback:
                    model.learn(total_timesteps=self.training_steps, callback=progress_callback)

                self.plot_training_results(title="SubAgent {0} {1}".format(lvl_id + 1, scc_id + 1))
                self.environment = environment
                self.model = model
                self.run_model()

                models.append(model)

                states = torch.tensor([restricted_mdp.idx_to_state_list(idx) for idx in state_idxs])
                _, values, _ = model.policy.forward(states)

                current_level_values.update({idx: value.item() for idx, value in zip(state_idxs, values)})
            lower_level_values = current_level_values

        whole_environment = MDPResourceAllocationEnvironment(self.ra_problem, self.steps_per_episode)
        whole_vector_environment = make_vec_env(lambda: whole_environment, n_envs=1, monitor_dir=self.log_dir)
        policy_kwargs = {"stage1_models": models}
        multistage_model = A2C(MultiStageActorCritic, whole_vector_environment, verbose=1, tensorboard_log=self.log_dir,
                               policy_kwargs=policy_kwargs)
        with ProgressBarManager(self.training_steps) as progress_callback:
            multistage_model.learn(total_timesteps=self.training_steps, callback=progress_callback)

        self.environment = whole_vector_environment
        self.model = multistage_model