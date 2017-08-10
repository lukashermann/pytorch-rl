from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from copy import deepcopy
from gym.spaces.box import Box
import inspect
import cv2
from sklearn.utils.extmath import cartesian

from utils.helpers import Experience            # NOTE: here state0 is always "None"
from utils.helpers import preprocessMujocoRgb, preprocessMujocoRgbd, preprocessMujocoRgbdLow
from core.env import Env

class MujocoEnv(Env):  # pixel-level inputs, Discrete
    def __init__(self, args, env_ind=0):
        super(MujocoEnv, self).__init__(args, env_ind)

        assert self.env_type == "mujoco"
        try: import gym
        except ImportError as e: self.logger.warning("WARNING: gym not found")

        self.env = gym.make(self.game)
        self.env.seed(self.seed)    # NOTE: so each env would be different

        # continuous space
        if args.agent_type == "a3c":
            self.enable_continuous = args.enable_continuous
        else:
            self.enable_continuous = False

        # action space setup
        if self.enable_continuous:
            self.actions = range(self.action_dim)
        else:
            self.actions = self._setup_actions()
        self.logger.warning("Action Space: %s", self.actions)
        # state space setup
        self.hei_state = args.hei_state
        self.wid_state = args.wid_state
        self.preprocess_mode = args.preprocess_mode if not None else 0 # 0 RGB | 1 RGBD | 2 RGBD + Low Level
        assert self.hei_state == self.wid_state
        self.logger.warning("State  Space: (" + str(self.state_shape) + " * " + str(self.state_shape) + ")")

    @property
    def action_dim(self):
        if self.enable_continuous:
            return self.env.action_space.shape[0]
        else:
            return len(self.actions)

    # discretize like in sim-to-real paper
    def _setup_actions(self):
        # discretize continuous action space
        self.dof = self.env.action_space.shape[0]
        discr_steps = 3
        actions = range(discr_steps)
        self.continuous_actions = np.array([0,self.env.action_space.low[0]*0.1,self.env.action_space.high[0]*0.1])
        self.enable_mjc_dis = True
        print(self.continuous_actions)
        return actions

    def _setup_actions2(self):
        # discretize continuous action space
        dof =self.env.action_space.shape[0]
        discr_steps = 11
        assert discr_steps % 2 == 1 and discr_steps >= 3
        actions = range(discr_steps**dof)
        possible_actions = [0]
        low = self.env.action_space.low[0]
        high = self.env.action_space.high[0]
        for i in range(int((discr_steps - 1) / 2)):
            possible_actions.append(high - i*(high/((discr_steps-1)/2)))
            possible_actions.append(low - i*(low/((discr_steps-1)/2)))
        self.continuous_actions = np.array(cartesian([possible_actions]*dof))
        print(self.continuous_actions)
        return actions

    def _discrete_to_continuous(self, action_index):
        return self.continuous_actions[action_index]

    def _preprocessState(self, state):
        if self.preprocess_mode == 0:   # RGB
            state = preprocessMujocoRgb(state,self.hei_state, self.wid_state)
        elif self.preprocess_mode == 1: # RGBD
            state = preprocessMujocoRgbd(state,self.hei_state, self.wid_state)
        else: # RGBD + Low Level observation
            state = preprocessMujocoRgbdLow(state,self.hei_state, self.wid_state)
        return state#.reshape(self.hei_state * self.wid_state)

    @property
    def state_shape(self):
        return self.hei_state

    def render(self):
        return self.env.render()

    def visual(self):
        if self.visualize:
            self.win_state1 = self.vis.image(np.transpose(self.exp_state1[0], (2, 0, 1)), env=self.refs, win=self.win_state1, opts=dict(title="state1"))
        if self.mode == 2:
            frame_name = self.img_dir + "frame_%04d.jpg" % self.frame_ind
            self.imsave(frame_name, self.exp_state1[0])
            self.logger.warning("Saved  Frame    @ Step: " + str(self.frame_ind) + " To: " + frame_name)
            self.frame_ind += 1

    def sample_random_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self._reset_experience()
        self.exp_state1 = self.env.reset()
        return self._get_experience()

    def step(self, action_index):
        if self.enable_continuous:
            self.exp_action = action_index
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        else:
            self.exp_action = self._discrete_to_continuous(action_index)
            self.exp_state1, self.exp_reward, self.exp_terminal1, _ = self.env.step(self.exp_action)
        return self._get_experience()
