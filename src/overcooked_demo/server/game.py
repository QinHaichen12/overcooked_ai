import json
import math
import os
import pickle
import random
from abc import ABC, abstractmethod
from collections import deque
from queue import Empty, Full, LifoQueue, Queue
from threading import Lock, Thread
from time import time

try:
    import ray
except ImportError:
    ray = None
from utils import DOCKER_VOLUME, create_dirs
from overcooked_ai_py.agents.agent import RandomAgent
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import (
    NO_COUNTERS_PARAMS,
    MotionPlanner,
)

# Relative path to where all static pre-trained agents are stored on server
AGENT_DIR = None

# Maximum allowable game time (in seconds)
MAX_GAME_TIME = None


def _configure(max_game_time, agent_dir):
    global AGENT_DIR, MAX_GAME_TIME
    MAX_GAME_TIME = max_game_time
    AGENT_DIR = agent_dir


def fix_bc_path(path):
    """
    Loading a PPO agent trained with a BC agent requires loading the BC model as well when restoring the trainer, even though the BC model is not used in game
    For now the solution is to include the saved BC model and fix the relative path to the model in the config.pkl file
    """

    import dill

    # the path is the agents/Rllib.*/agent directory
    agent_path = os.path.abspath(os.path.dirname(path))
    with open(os.path.join(agent_path, "config.pkl"), "rb") as f:
        data = dill.load(f)
    bc_model_dir = data["bc_params"]["bc_config"]["model_dir"]
    last_dir = os.path.basename(bc_model_dir)
    bc_model_dir = os.path.abspath(
        os.path.join(agent_path, "bc_params", last_dir)
    )
    data["bc_params"]["bc_config"]["model_dir"] = bc_model_dir
    with open(os.path.join(agent_path, "config.pkl"), "wb") as f:
        dill.dump(data, f)


class Game(ABC):

    """
    Class representing a game object. Coordinates the simultaneous actions of arbitrary
    number of players. Override this base class in order to use.

    Players can post actions to a `pending_actions` queue, and driver code can call `tick` to apply these actions.


    It should be noted that most operations in this class are not on their own thread safe. Thus, client code should
    acquire `self.lock` before making any modifications to the instance.

    One important exception to the above rule is `enqueue_actions` which is thread safe out of the box
    """

    # Possible TODO: create a static list of IDs used by the class so far to verify id uniqueness
    # This would need to be serialized, however, which might cause too great a performance hit to
    # be worth it

    EMPTY = "EMPTY"

    class Status:
        DONE = "done"
        ACTIVE = "active"
        RESET = "reset"
        INACTIVE = "inactive"
        ERROR = "error"

    def __init__(self, *args, **kwargs):
        """
        players (list): List of IDs of players currently in the game
        spectators (set): Collection of IDs of players that are not allowed to enqueue actions but are currently watching the game
        id (int):   Unique identifier for this game
        pending_actions List[(Queue)]: Buffer of (player_id, action) pairs have submitted that haven't been commited yet
        lock (Lock):    Used to serialize updates to the game state
        is_active(bool): Whether the game is currently being played or not
        """
        self.players = []
        self.spectators = set()
        self.pending_actions = []
        self.id = kwargs.get("id", id(self))
        self.lock = Lock()
        self._is_active = False

    @abstractmethod
    def is_full(self):
        """
        Returns whether there is room for additional players to join or not
        """
        pass

    @abstractmethod
    def apply_action(self, player_idx, action):
        """
        Updates the game state by applying a single (player_idx, action) tuple. Subclasses should try to override this method
        if possible
        """
        pass

    @abstractmethod
    def is_finished(self):
        """
        Returns whether the game has concluded or not
        """
        pass

    def is_ready(self):
        """
        Returns whether the game can be started. Defaults to having enough players
        """
        return self.is_full()

    @property
    def is_active(self):
        """
        Whether the game is currently being played
        """
        return self._is_active

    @property
    def reset_timeout(self):
        """
        Number of milliseconds to pause game on reset
        """
        return 3000

    def apply_actions(self):
        """
        Updates the game state by applying each of the pending actions in the buffer. Is called by the tick method. Subclasses
        should override this method if joint actions are necessary. If actions can be serialized, overriding `apply_action` is
        preferred
        """
        for i in range(len(self.players)):
            try:
                while True:
                    action = self.pending_actions[i].get(block=False)
                    self.apply_action(i, action)
            except Empty:
                pass

    def activate(self):
        """
        Activates the game to let server know real-time updates should start. Provides little functionality but useful as
        a check for debugging
        """
        self._is_active = True

    def deactivate(self):
        """
        Deactives the game such that subsequent calls to `tick` will be no-ops. Used to handle case where game ends but
        there is still a buffer of client pings to handle
        """
        self._is_active = False

    def reset(self):
        """
        Restarts the game while keeping all active players by resetting game stats and temporarily disabling `tick`
        """
        if not self.is_active:
            raise ValueError("Inactive Games cannot be reset")
        if self.is_finished():
            return self.Status.DONE
        self.deactivate()
        self.activate()
        return self.Status.RESET

    def needs_reset(self):
        """
        Returns whether the game should be reset on the next call to `tick`
        """
        return False

    def tick(self):
        """
        Updates the game state by applying each of the pending actions. This is done so that players cannot directly modify
        the game state, offering an additional level of safety and thread security.

        One can think of "enqueue_action" like calling "git add" and "tick" like calling "git commit"

        Subclasses should try to override `apply_actions` if possible. Only override this method if necessary
        """
        if not self.is_active:
            return self.Status.INACTIVE
        if self.needs_reset():
            self.reset()
            return self.Status.RESET

        self.apply_actions()
        return self.Status.DONE if self.is_finished() else self.Status.ACTIVE

    def enqueue_action(self, player_id, action):
        """
        Add (player_id, action) pair to the pending action queue, without modifying underlying game state

        Note: This function IS thread safe
        """
        if not self.is_active:
            # Could run into issues with is_active not being thread safe
            return
        if player_id not in self.players:
            # Only players actively in game are allowed to enqueue actions
            return
        try:
            player_idx = self.players.index(player_id)
            self.pending_actions[player_idx].put(action)
        except Full:
            pass

    def get_state(self):
        """
        Return a JSON compatible serialized state of the game. Note that this should be as minimalistic as possible
        as the size of the game state will be the most important factor in game performance. This is sent to the client
        every frame update.
        """
        return {"players": self.players}

    def to_json(self):
        """
        Return a JSON compatible serialized state of the game. Contains all information about the game, does not need to
        be minimalistic. This is sent to the client only once, upon game creation
        """
        return self.get_state()

    def is_empty(self):
        """
        Return whether it is safe to garbage collect this game instance
        """
        return not self.num_players

    def add_player(self, player_id, idx=None, buff_size=-1):
        """
        Add player_id to the game
        """
        if self.is_full():
            raise ValueError("Cannot add players to full game")
        if self.is_active:
            raise ValueError("Cannot add players to active games")
        if not idx and self.EMPTY in self.players:
            idx = self.players.index(self.EMPTY)
        elif not idx:
            idx = len(self.players)

        padding = max(0, idx - len(self.players) + 1)
        for _ in range(padding):
            self.players.append(self.EMPTY)
            self.pending_actions.append(self.EMPTY)

        self.players[idx] = player_id
        self.pending_actions[idx] = Queue(maxsize=buff_size)

    def add_spectator(self, spectator_id):
        """
        Add spectator_id to list of spectators for this game
        """
        if spectator_id in self.players:
            raise ValueError("Cannot spectate and play at same time")
        self.spectators.add(spectator_id)

    def remove_player(self, player_id):
        """
        Remove player_id from the game
        """
        try:
            idx = self.players.index(player_id)
            self.players[idx] = self.EMPTY
            self.pending_actions[idx] = self.EMPTY
        except ValueError:
            return False
        else:
            return True

    def remove_spectator(self, spectator_id):
        """
        Removes spectator_id if they are in list of spectators. Returns True if spectator successfully removed, False otherwise
        """
        try:
            self.spectators.remove(spectator_id)
        except ValueError:
            return False
        else:
            return True

    def clear_pending_actions(self):
        """
        Remove all queued actions for all players
        """
        for i, player in enumerate(self.players):
            if player != self.EMPTY:
                queue = self.pending_actions[i]
                queue.queue.clear()

    @property
    def num_players(self):
        return len([player for player in self.players if player != self.EMPTY])

    def get_data(self):
        """
        Return any game metadata to server driver.
        """
        return {}


class DummyGame(Game):

    """
    Standin class used to test basic server logic
    """

    def __init__(self, **kwargs):
        super(DummyGame, self).__init__(**kwargs)
        self.counter = 0

    def is_full(self):
        return self.num_players == 2

    def apply_action(self, idx, action):
        pass

    def apply_actions(self):
        self.counter += 1

    def is_finished(self):
        return self.counter >= 100

    def get_state(self):
        state = super(DummyGame, self).get_state()
        state["count"] = self.counter
        return state


class DummyInteractiveGame(Game):

    """
    Standing class used to test interactive components of the server logic
    """

    def __init__(self, **kwargs):
        super(DummyInteractiveGame, self).__init__(**kwargs)
        self.max_players = int(
            kwargs.get("playerZero", "human") == "human"
        ) + int(kwargs.get("playerOne", "human") == "human")
        self.max_count = kwargs.get("max_count", 30)
        self.counter = 0
        self.counts = [0] * self.max_players

    def is_full(self):
        return self.num_players == self.max_players

    def is_finished(self):
        return max(self.counts) >= self.max_count

    def apply_action(self, player_idx, action):
        if action.upper() == Direction.NORTH:
            self.counts[player_idx] += 1
        if action.upper() == Direction.SOUTH:
            self.counts[player_idx] -= 1

    def apply_actions(self):
        super(DummyInteractiveGame, self).apply_actions()
        self.counter += 1

    def get_state(self):
        state = super(DummyInteractiveGame, self).get_state()
        state["count"] = self.counter
        for i in range(self.num_players):
            state["player_{}_count".format(i)] = self.counts[i]
        return state


class OvercookedGame(Game):
    """
    Class for bridging the gap between Overcooked_Env and the Game interface

    Instance variable:
        - max_players (int): Maximum number of players that can be in the game at once
        - mdp (OvercookedGridworld): Controls the underlying Overcooked game logic
        - score (int): Current reward acheived by all players
        - max_time (int): Number of seconds the game should last
        - npc_policies (dict): Maps user_id to policy (Agent) for each AI player
        - npc_state_queues (dict): Mapping of NPC user_ids to LIFO queues for the policy to process
        - curr_tick (int): How many times the game server has called this instance's `tick` method
        - ticker_per_ai_action (int): How many frames should pass in between NPC policy forward passes.
            Note that this is a lower bound; if the policy is computationally expensive the actual frames
            per forward pass can be higher
        - action_to_overcooked_action (dict): Maps action names returned by client to action names used by OvercookedGridworld
            Note that this is an instance variable and not a static variable for efficiency reasons
        - human_players (set(str)): Collection of all player IDs that correspond to humans
        - npc_players (set(str)): Collection of all player IDs that correspond to AI
        - randomized (boolean): Whether the order of the layouts should be randomized

    Methods:
        - npc_policy_consumer: Background process that asynchronously computes NPC policy forward passes. One thread
            spawned for each NPC
        - _curr_game_over: Determines whether the game on the current mdp has ended
    """

    def __init__(
        self,
        layouts=["cramped_room"],
        mdp_params={},
        num_players=2,
        gameTime=30,
        playerZero="human",
        playerOne="human",
        showPotential=False,
        randomized=False,
        ticks_per_ai_action=1,
        **kwargs
    ):
        super(OvercookedGame, self).__init__(**kwargs)
        self.show_potential = showPotential
        self.mdp_params = mdp_params
        self.layouts = layouts
        self.max_players = int(num_players)
        self.mdp = None
        self.mp = None
        self.score = 0
        self.phi = 0
        self.max_time = min(int(gameTime), MAX_GAME_TIME)
        self.npc_policies = {}
        self.npc_state_queues = {}
        self.action_to_overcooked_action = {
            "STAY": Action.STAY,
            "UP": Direction.NORTH,
            "DOWN": Direction.SOUTH,
            "LEFT": Direction.WEST,
            "RIGHT": Direction.EAST,
            "SPACE": Action.INTERACT,
        }
        self.ticks_per_ai_action = ticks_per_ai_action
        self.curr_tick = 0
        self.human_players = set()
        self.npc_players = set()

        if randomized:
            random.shuffle(self.layouts)

        if playerZero != "human":
            player_zero_id = playerZero + "_0"
            self.add_player(player_zero_id, idx=0, buff_size=1, is_human=False)
            self.npc_policies[player_zero_id] = self.get_policy(
                playerZero, idx=0
            )
            self.npc_state_queues[player_zero_id] = LifoQueue()

        if playerOne != "human":
            player_one_id = playerOne + "_1"
            self.add_player(player_one_id, idx=1, buff_size=1, is_human=False)
            self.npc_policies[player_one_id] = self.get_policy(
                playerOne, idx=1
            )
            self.npc_state_queues[player_one_id] = LifoQueue()
        # Always kill ray after loading agent, otherwise, ray will crash once process exits
        # Only kill ray after loading both agents to avoid having to restart ray during loading
        if ray is not None and ray.is_initialized():
            ray.shutdown()

        if kwargs["dataCollection"]:
            self.write_data = True
            self.write_config = kwargs["collection_config"]
        else:
            self.write_data = False

        self.trajectory = []

    def _curr_game_over(self):
        return time() - self.start_time >= self.max_time

    def needs_reset(self):
        return self._curr_game_over() and not self.is_finished()

    def add_player(self, player_id, idx=None, buff_size=-1, is_human=True):
        super(OvercookedGame, self).add_player(
            player_id, idx=idx, buff_size=buff_size
        )
        if is_human:
            self.human_players.add(player_id)
        else:
            self.npc_players.add(player_id)

    def remove_player(self, player_id):
        removed = super(OvercookedGame, self).remove_player(player_id)
        if removed:
            if player_id in self.human_players:
                self.human_players.remove(player_id)
            elif player_id in self.npc_players:
                self.npc_players.remove(player_id)
            else:
                raise ValueError("Inconsistent state")

    def npc_policy_consumer(self, policy_id):
        queue = self.npc_state_queues[policy_id]
        policy = self.npc_policies[policy_id]
        while self._is_active:
            state = queue.get()
            npc_action, _ = policy.action(state)
            super(OvercookedGame, self).enqueue_action(policy_id, npc_action)

    def is_full(self):
        return self.num_players >= self.max_players

    def is_finished(self):
        val = not self.layouts and self._curr_game_over()
        return val

    def is_empty(self):
        """
        Game is considered safe to scrap if there are no active players or if there are no humans (spectating or playing)
        """
        return (
            super(OvercookedGame, self).is_empty()
            or not self.spectators
            and not self.human_players
        )

    def is_ready(self):
        """
        Game is ready to be activated if there are a sufficient number of players and at least one human (spectator or player)
        """
        return super(OvercookedGame, self).is_ready() and not self.is_empty()

    def apply_action(self, player_id, action):
        pass

    def apply_actions(self):
        # Default joint action, as NPC policies and clients probably don't enqueue actions fast
        # enough to produce one at every tick
        joint_action = [Action.STAY] * len(self.players)

        # Synchronize individual player actions into a joint-action as required by overcooked logic
        for i in range(len(self.players)):
            # if this is a human, don't block and inject
            if self.players[i] in self.human_players:
                try:
                    # we don't block here in case humans want to Stay
                    joint_action[i] = self.pending_actions[i].get(block=False)
                except Empty:
                    pass
            else:
                # we block on agent actions to ensure that the agent gets to do one action per state
                joint_action[i] = self.pending_actions[i].get(block=True)

        # Apply overcooked game logic to get state transition
        prev_state = self.state
        self.state, info = self.mdp.get_state_transition(
            prev_state, joint_action
        )
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                prev_state, self.mp, gamma=0.99
            )

        # Send next state to all background consumers if needed
        if self.curr_tick % self.ticks_per_ai_action == 0:
            for npc_id in self.npc_policies:
                self.npc_state_queues[npc_id].put(self.state, block=False)

        # Update score based on soup deliveries that might have occured
        curr_reward = sum(info["sparse_reward_by_agent"])
        self.score += curr_reward

        transition = {
            "state": json.dumps(prev_state.to_dict()),
            "joint_action": json.dumps(joint_action),
            "reward": curr_reward,
            "time_left": max(self.max_time - (time() - self.start_time), 0),
            "score": self.score,
            "time_elapsed": time() - self.start_time,
            "cur_gameloop": self.curr_tick,
            "layout": json.dumps(self.mdp.terrain_mtx),
            "layout_name": self.curr_layout,
            "trial_id": str(self.start_time),
            "player_0_id": self.players[0],
            "player_1_id": self.players[1],
            "player_0_is_human": self.players[0] in self.human_players,
            "player_1_is_human": self.players[1] in self.human_players,
        }

        self.trajectory.append(transition)

        # Return about the current transition
        return prev_state, joint_action, info

    def enqueue_action(self, player_id, action):
        overcooked_action = self.action_to_overcooked_action[action]
        super(OvercookedGame, self).enqueue_action(
            player_id, overcooked_action
        )

    def reset(self):
        status = super(OvercookedGame, self).reset()
        if status == self.Status.RESET:
            # Hacky way of making sure game timer doesn't "start" until after reset timeout has passed
            self.start_time += self.reset_timeout / 1000

    def tick(self):
        self.curr_tick += 1
        return super(OvercookedGame, self).tick()

    def activate(self):
        super(OvercookedGame, self).activate()

        # Sanity check at start of each game
        if not self.npc_players.union(self.human_players) == set(self.players):
            raise ValueError("Inconsistent State")

        self.curr_layout = self.layouts.pop()
        self.mdp = OvercookedGridworld.from_layout_name(
            self.curr_layout, **self.mdp_params
        )
        if self.show_potential:
            self.mp = MotionPlanner.from_pickle_or_compute(
                self.mdp, counter_goals=NO_COUNTERS_PARAMS
            )
        self.state = self.mdp.get_standard_start_state()
        if self.show_potential:
            self.phi = self.mdp.potential_function(
                self.state, self.mp, gamma=0.99
            )
        self.start_time = time()
        self.curr_tick = 0
        self.score = 0
        self.threads = []
        for npc_policy in self.npc_policies:
            self.npc_policies[npc_policy].reset()
            self.npc_state_queues[npc_policy].put(self.state)
            t = Thread(target=self.npc_policy_consumer, args=(npc_policy,))
            self.threads.append(t)
            t.start()

    def deactivate(self):
        super(OvercookedGame, self).deactivate()
        # Ensure the background consumers do not hang
        for npc_policy in self.npc_policies:
            self.npc_state_queues[npc_policy].put(self.state)

        # Wait for all background threads to exit
        for t in self.threads:
            t.join()

        # Clear all action queues
        self.clear_pending_actions()

    def get_state(self):
        state_dict = {}
        state_dict["potential"] = self.phi if self.show_potential else None
        state_dict["state"] = self.state.to_dict()
        state_dict["score"] = self.score
        state_dict["time_left"] = max(
            self.max_time - (time() - self.start_time), 0
        )
        return state_dict

    def to_json(self):
        obj_dict = {}
        obj_dict["terrain"] = self.mdp.terrain_mtx if self._is_active else None
        obj_dict["state"] = self.get_state() if self._is_active else None
        return obj_dict

    def get_policy(self, npc_id, idx=0):
        if npc_id.lower() in [
            "bayesianbeliefai",
            "bayesbeliefai",
            "bayesianhelperai",
        ]:
            return BayesianBeliefAI(
                agent_index=idx, mdp_getter=lambda: self.mdp
            )

        if npc_id.lower().startswith("rllib"):
            try:
                from human_aware_rl.rllib.rllib import load_agent

                # Loading rllib agents requires additional helpers
                fpath = os.path.abspath(
                    os.path.join(AGENT_DIR, npc_id, "agent")
                )
                fix_bc_path(fpath)
                agent = load_agent(fpath, agent_index=idx)
                return agent
            except Exception as e:
                print(
                    "Warning: failed to load {0} ({1}). Falling back to RandAI.".format(
                        npc_id, e
                    )
                )
                return RandomAgent(all_actions=Action.ALL_ACTIONS)
        else:
            try:
                fpath = os.path.abspath(
                    os.path.join(AGENT_DIR, npc_id, "agent.pickle")
                )
                with open(fpath, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                raise IOError("Error loading agent\n{}".format(e.__repr__()))

    def get_data(self):
        """
        Returns and then clears the accumulated trajectory
        """
        data = {
            "uid": str(time()),
            "trajectory": self.trajectory,
        }
        self.trajectory = []
        # if we want to store the data and there is data to store
        if self.write_data and len(data["trajectory"]) > 0:
            configs = self.write_config
            # create necessary dirs
            data_path = create_dirs(configs, self.curr_layout)
            # the 3-layer-directory structure should be able to uniquely define any experiment
            with open(os.path.join(data_path, "result.pkl"), "wb") as f:
                pickle.dump(data, f)
        return data


class OvercookedTutorial(OvercookedGame):

    """
    Wrapper on OvercookedGame that includes additional data for tutorial mechanics, most notably the introduction of tutorial "phases"

    Instance Variables:
        - curr_phase (int): Indicates what tutorial phase we are currently on
        - phase_two_score (float): The exact sparse reward the user must obtain to advance past phase 2
    """

    def __init__(
        self,
        layouts=["tutorial_0"],
        mdp_params={},
        playerZero="human",
        playerOne="AI",
        phaseTwoScore=15,
        **kwargs
    ):
        super(OvercookedTutorial, self).__init__(
            layouts=layouts,
            mdp_params=mdp_params,
            playerZero=playerZero,
            playerOne=playerOne,
            showPotential=False,
            **kwargs
        )
        self.phase_two_score = phaseTwoScore
        self.phase_two_finished = False
        self.max_time = 0
        self.max_players = 2
        self.ticks_per_ai_action = 1
        self.curr_phase = 0
        # we don't collect tutorial data
        self.write_data = False

    @property
    def reset_timeout(self):
        return 1

    def needs_reset(self):
        if self.curr_phase == 0:
            return self.score > 0
        elif self.curr_phase == 1:
            return self.score > 0
        elif self.curr_phase == 2:
            return self.phase_two_finished
        return False

    def is_finished(self):
        return not self.layouts and self.score >= float("inf")

    def reset(self):
        super(OvercookedTutorial, self).reset()
        self.curr_phase += 1

    def get_policy(self, *args, **kwargs):
        return TutorialAI()

    def apply_actions(self):
        """
        Apply regular MDP logic with retroactive score adjustment tutorial purposes
        """
        _, _, info = super(OvercookedTutorial, self).apply_actions()

        human_reward, ai_reward = info["sparse_reward_by_agent"]

        # We only want to keep track of the human's score in the tutorial
        self.score -= ai_reward

        # Phase two requires a specific reward to complete
        if self.curr_phase == 2:
            self.score = 0
            if human_reward == self.phase_two_score:
                self.phase_two_finished = True


class DummyOvercookedGame(OvercookedGame):
    """
    Class that hardcodes the AI to be random. Used for debugging
    """

    def __init__(self, layouts=["cramped_room"], **kwargs):
        super(DummyOvercookedGame, self).__init__(layouts, **kwargs)

    def get_policy(self, *args, **kwargs):
        return DummyAI()


class BayesianBeliefAI:
    """
    A lightweight cooperative agent that maintains a Bayesian belief over
    the human partner's intent and picks complementary tasks.

    This implementation is intentionally simple and fast enough for live demo
    play without RLlib dependencies.
    """

    INTENTS = [
        "get_onion",
        "get_tomato",
        "get_dish",
        "put_in_pot",
        "start_cooking",
        "pickup_soup",
        "deliver_soup",
    ]

    def __init__(
        self,
        agent_index=1,
        mdp_getter=None,
        model_path=None,
        use_model_likelihood=True,
        use_transition_prior=True,
        use_commitment=True,
        use_yield=True,
    ):
        self.agent_index = agent_index
        self.mdp_getter = mdp_getter
        self.model = self._load_model(model_path)
        self.use_model_likelihood = use_model_likelihood
        self.use_transition_prior = use_transition_prior
        self.use_commitment = use_commitment
        self.use_yield = use_yield
        self.reset()

    def reset(self):
        p = 1.0 / len(self.INTENTS)
        self.intent_belief = {intent: p for intent in self.INTENTS}
        self.intent_history = deque(maxlen=8)
        self._prev_human_obs = None
        self.handoff_role = "none"

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index

    def set_mdp(self, mdp):
        self._eval_mdp = mdp
        if self.mdp_getter is None:
            self.mdp_getter = lambda: self._eval_mdp

    def action(self, state):
        mdp = self.mdp_getter() if self.mdp_getter else None
        if mdp is None:
            return Action.STAY, None

        layout_name = mdp.mdp_params.get("layout_name", "")
        human_idx = 1 - self.agent_index
        self._update_intent_belief(state, mdp, human_idx, layout_name)
        intent = max(self.intent_belief, key=self.intent_belief.get)
        self.intent_history.append(intent)

        commitment = self._commitment_strength()
        if self.use_commitment and commitment >= 0.7 and len(self.intent_history) >= 4:
            counts = {}
            for i in self.intent_history:
                counts[i] = counts.get(i, 0) + 1
            intent = max(counts, key=counts.get)

        action = self._choose_cooperative_action(
            state, mdp, intent, dict(self.intent_belief)
        )
        info = {
            "belief": dict(self.intent_belief),
            "inferred_intent": intent,
            "intent_commitment": commitment,
            "handoff_role": self.handoff_role,
        }
        return action, info

    def _update_intent_belief(self, state, mdp, human_idx, layout_name):
        human = state.players[human_idx]
        observed_action = self._infer_observed_action(self._prev_human_obs, human)
        likelihood = self._heuristic_likelihood(
            state,
            mdp,
            human,
            observed_action=observed_action,
            prev_obs=self._prev_human_obs,
        )

        if self.model and self.use_model_likelihood:
            model_likelihood = self._model_likelihood(
                state,
                mdp,
                human,
                layout_name,
                observed_action,
                prev_obs=self._prev_human_obs,
            )
            for intent in self.INTENTS:
                likelihood[intent] = (
                    0.65 * model_likelihood[intent] + 0.35 * likelihood[intent]
                )

        prior = self._transition_prior(layout_name)

        eps = 1e-6
        unnorm = {}
        for intent in self.INTENTS:
            unnorm[intent] = prior[intent] * (likelihood[intent] + eps)
        z = sum(unnorm.values())
        if z <= 0:
            p = 1.0 / len(self.INTENTS)
            self.intent_belief = {intent: p for intent in self.INTENTS}
            return

        posterior = {intent: unnorm[intent] / z for intent in self.INTENTS}
        u = 1.0 / len(self.INTENTS)
        self.intent_belief = {
            intent: 0.9 * posterior[intent] + 0.1 * u for intent in self.INTENTS
        }
        self._prev_human_obs = self._snapshot_player_state(human)

    def _heuristic_likelihood(
        self, state, mdp, human, observed_action=None, prev_obs=None
    ):
        likelihood = {intent: 0.10 for intent in self.INTENTS}
        pot_states = mdp.get_pot_states(state)
        has_ready = bool(pot_states["ready"])
        has_cooking = bool(pot_states["cooking"])
        has_ready_or_cooking = bool(has_ready or has_cooking)
        has_ready_to_cook = bool(pot_states.get("3_items"))
        context_bucket = self._context_bucket(state, mdp)
        progress_bucket = self._progress_bucket(
            human, mdp, observed_action, prev_obs=prev_obs
        )

        if human.has_object():
            obj_name = human.get_object().name
            if obj_name in ["onion", "tomato"]:
                likelihood["put_in_pot"] = 0.75
                if observed_action == Action.INTERACT and self._is_near_any(
                    human.position, mdp.get_pot_locations()
                ):
                    likelihood["put_in_pot"] = 0.95
            elif obj_name == "dish":
                likelihood["pickup_soup"] = 0.70
                if observed_action == Action.INTERACT and self._is_near_any(
                    human.position, mdp.get_pot_locations()
                ):
                    likelihood["pickup_soup"] = 0.96
            elif obj_name == "soup":
                likelihood["deliver_soup"] = 0.85
                if observed_action == Action.INTERACT and self._is_near_any(
                    human.position, mdp.get_serving_locations()
                ):
                    likelihood["deliver_soup"] = 0.98
        else:
            if self._is_near_any(human.position, mdp.get_onion_dispenser_locations()):
                likelihood["get_onion"] = 0.65
            if self._is_near_any(human.position, mdp.get_tomato_dispenser_locations()):
                likelihood["get_tomato"] = 0.65
            if self._is_near_any(human.position, mdp.get_dish_dispenser_locations()):
                likelihood["get_dish"] = 0.72 if has_ready_or_cooking else 0.65
            if has_ready_or_cooking and self._is_near_any(
                human.position, mdp.get_pot_locations()
            ):
                likelihood["pickup_soup"] = max(likelihood["pickup_soup"], 0.52)

        if progress_bucket == "toward_onion":
            likelihood["get_onion"] = max(likelihood["get_onion"], 0.60)
        elif progress_bucket == "toward_tomato":
            likelihood["get_tomato"] = max(likelihood["get_tomato"], 0.60)
        elif progress_bucket == "toward_dish":
            likelihood["get_dish"] = max(
                likelihood["get_dish"], 0.72 if has_ready_or_cooking else 0.58
            )
        elif progress_bucket == "toward_pot":
            if human.has_object() and human.get_object().name in ["onion", "tomato"]:
                likelihood["put_in_pot"] = max(likelihood["put_in_pot"], 0.82)
            elif has_ready_to_cook:
                likelihood["start_cooking"] = max(likelihood["start_cooking"], 0.78)
        elif progress_bucket == "toward_serve":
            likelihood["deliver_soup"] = max(likelihood["deliver_soup"], 0.58)

        if observed_action == Action.INTERACT and self._is_near_any(
            human.position, mdp.get_pot_locations()
        ):
            if human.has_object() and human.get_object().name in ["onion", "tomato"]:
                likelihood["put_in_pot"] = 0.96
            elif human.has_object() and human.get_object().name == "dish":
                likelihood["pickup_soup"] = 0.96
            elif has_ready_to_cook:
                likelihood["start_cooking"] = 0.92

        if context_bucket == "ready":
            likelihood["get_dish"] = max(likelihood["get_dish"], 0.68)
            likelihood["pickup_soup"] = max(likelihood["pickup_soup"], 0.62)
        elif context_bucket == "cooking":
            likelihood["get_dish"] = max(likelihood["get_dish"], 0.58)
        elif context_bucket == "ready_to_cook":
            likelihood["start_cooking"] = max(likelihood["start_cooking"], 0.74)
        elif context_bucket == "needs_ingredient":
            likelihood["get_onion"] = max(likelihood["get_onion"], 0.52)
            likelihood["get_tomato"] = max(likelihood["get_tomato"], 0.52)
        return likelihood

    def _transition_prior(self, layout_name):
        if not self.model or not self.use_transition_prior:
            return dict(self.intent_belief)

        layout_priors = self.model.get("layout_priors", {}).get(layout_name, {})
        layout_transitions = self.model.get("layout_transitions", {}).get(layout_name, {})
        if layout_priors and layout_transitions:
            prior_source = layout_priors
            transition_source = layout_transitions
        else:
            group = self._layout_group(layout_name)
            prior_source = self.model.get("priors", {}).get(group, {})
            transition_source = self.model.get("transitions", {}).get(group, {})

        prior = {}
        for dst in self.INTENTS:
            t_prob = 0.0
            for src in self.INTENTS:
                row = transition_source.get(src, {})
                t_prob += self.intent_belief[src] * row.get(dst, 0.0)
            prior[dst] = 0.5 * prior_source.get(dst, 1.0 / len(self.INTENTS)) + 0.5 * t_prob

        z = sum(prior.values())
        if z <= 0:
            p = 1.0 / len(self.INTENTS)
            return {intent: p for intent in self.INTENTS}
        return {k: v / z for k, v in prior.items()}

    def _model_likelihood(
        self, state, mdp, human, layout_name, observed_action=None, prev_obs=None
    ):
        emissions = self.model.get("emissions", {}) if self.model else {}
        layout_emissions = self.model.get("layout_emissions", {}) if self.model else {}
        held_probs = self._resolve_emission_table(
            emissions, layout_emissions, "held", layout_name
        )
        prox_probs = self._resolve_emission_table(
            emissions, layout_emissions, "proximity", layout_name
        )
        action_probs = self._resolve_emission_table(
            emissions, layout_emissions, "action", layout_name
        )
        context_probs = self._resolve_emission_table(
            emissions, layout_emissions, "context", layout_name
        )
        progress_probs = self._resolve_emission_table(
            emissions, layout_emissions, "progress", layout_name
        )

        held_bucket = self._held_bucket(human)
        prox_bucket = self._proximity_bucket(human, mdp)
        context_bucket = self._context_bucket(state, mdp)
        progress_bucket = self._progress_bucket(
            human, mdp, observed_action, prev_obs=prev_obs
        )
        action_key = str(observed_action) if observed_action is not None else None

        likelihood = {}
        for intent in self.INTENTS:
            p_held = held_probs.get(intent, {}).get(held_bucket, 1e-3)
            p_prox = prox_probs.get(intent, {}).get(prox_bucket, 1e-3)
            p_context = context_probs.get(intent, {}).get(context_bucket, 1.0)
            p_progress = progress_probs.get(intent, {}).get(progress_bucket, 1.0)
            p_action = 1.0
            if action_key is not None:
                p_action = action_probs.get(intent, {}).get(action_key, 1e-3)
            likelihood[intent] = max(
                1e-6,
                p_held
                * p_prox
                * math.sqrt(p_action)
                * math.sqrt(p_context)
                * math.sqrt(p_progress),
            )

        z = sum(likelihood.values())
        if z <= 0:
            p = 1.0 / len(self.INTENTS)
            return {intent: p for intent in self.INTENTS}
        return {k: v / z for k, v in likelihood.items()}

    def _held_bucket(self, player):
        if not player.has_object():
            return "none"
        obj_name = player.get_object().name
        if obj_name in ["onion", "tomato", "dish", "soup"]:
            return obj_name
        return "other"

    def _context_bucket(self, state, mdp):
        pot_states = mdp.get_pot_states(state)
        if pot_states["ready"]:
            return "ready"
        if pot_states["cooking"]:
            return "cooking"
        if pot_states.get("3_items"):
            return "ready_to_cook"

        open_pots = (
            list(pot_states["empty"])
            + list(pot_states["1_items"])
            + list(pot_states["2_items"])
        )
        if open_pots:
            return "needs_ingredient"
        return "other"

    def _progress_bucket(self, player, mdp, observed_action, prev_obs=None):
        if observed_action not in Action.MOTION_ACTIONS:
            return "none"

        start_pos = prev_obs["position"] if prev_obs is not None else player.position
        end_pos = (
            player.position
            if prev_obs is not None
            else Action.move_in_direction(player.position, observed_action)
        )
        feature_targets = [
            ("toward_onion", mdp.get_onion_dispenser_locations()),
            ("toward_tomato", mdp.get_tomato_dispenser_locations()),
            ("toward_dish", mdp.get_dish_dispenser_locations()),
            ("toward_pot", mdp.get_pot_locations()),
            ("toward_serve", mdp.get_serving_locations()),
        ]

        best_bucket = "none"
        best_improvement = 0
        for bucket, targets in feature_targets:
            current_dist = self._min_distance(start_pos, targets)
            next_dist = self._min_distance(end_pos, targets)
            if current_dist is None or next_dist is None:
                continue
            improvement = current_dist - next_dist
            if improvement > best_improvement:
                best_improvement = improvement
                best_bucket = bucket

        return best_bucket

    def _resolve_emission_table(
        self, emissions, layout_emissions, emission_name, layout_name
    ):
        if layout_name:
            layout_specific = layout_emissions.get(emission_name, {}).get(layout_name)
            if layout_specific:
                return layout_specific
        return emissions.get(emission_name, {})

    def _snapshot_player_state(self, player):
        held_name = None
        if player.has_object():
            held_name = player.get_object().name
        return {
            "position": player.position,
            "orientation": player.orientation,
            "held": held_name,
        }

    def _infer_observed_action(self, prev_obs, player):
        if prev_obs is None:
            return None

        curr_pos = player.position
        curr_orient = player.orientation
        curr_held = None if not player.has_object() else player.get_object().name

        prev_pos = prev_obs["position"]
        prev_orient = prev_obs["orientation"]
        prev_held = prev_obs["held"]

        if curr_pos != prev_pos:
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            step = (dx, dy)
            if step in Direction.ALL_DIRECTIONS:
                return step

        if curr_orient != prev_orient and curr_orient in Direction.ALL_DIRECTIONS:
            return curr_orient

        if curr_held != prev_held:
            return Action.INTERACT

        return Action.STAY

    def _proximity_bucket(self, player, mdp):
        pos = player.position
        if self._is_near_any(pos, mdp.get_onion_dispenser_locations()):
            return "onion"
        if self._is_near_any(pos, mdp.get_tomato_dispenser_locations()):
            return "tomato"
        if self._is_near_any(pos, mdp.get_dish_dispenser_locations()):
            return "dish"
        if self._is_near_any(pos, mdp.get_pot_locations()):
            return "pot"
        if self._is_near_any(pos, mdp.get_serving_locations()):
            return "serve"
        return "none"

    def _layout_group(self, layout_name):
        lname = layout_name.lower()
        for token in [
            "cramped",
            "corridor",
            "forced",
            "counter_circuit",
            "you_shall_not_pass",
            "bottleneck",
        ]:
            if token in lname:
                return "constrained"
        return "open"

    @staticmethod
    def _min_distance(pos, targets):
        if not targets:
            return None
        return min(abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets)

    def _commitment_strength(self):
        if not self.intent_history:
            return 0.0
        counts = {}
        for intent in self.intent_history:
            counts[intent] = counts.get(intent, 0) + 1
        return max(counts.values()) / float(len(self.intent_history))

    def _load_model(self, model_path):
        candidates = []
        if model_path:
            candidates.append(model_path)

        env_path = os.getenv("OVERCOOKED_BAYESIAN_MODEL")
        if env_path:
            candidates.append(env_path)

        if AGENT_DIR:
            candidates.append(
                os.path.join(AGENT_DIR, "BayesianBeliefAI", "model.pkl")
            )

        for path in candidates:
            try:
                if not path:
                    continue
                abs_path = os.path.abspath(path)
                if not os.path.exists(abs_path):
                    continue
                with open(abs_path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "intents" in data:
                    print("Loaded Bayesian model from {}".format(abs_path))
                    return data
            except Exception as e:
                print("Warning: failed to load Bayesian model {} ({})".format(path, e))

        return None

    def _choose_cooperative_action(self, state, mdp, human_intent, intent_belief=None):
        me = state.players[self.agent_index]
        human = state.players[1 - self.agent_index]
        layout_name = mdp.mdp_params.get("layout_name", "")
        pot_states = mdp.get_pot_states(state)
        ready_pots = list(pot_states["ready"])
        cooking_pots = list(pot_states["cooking"])
        ready_or_cooking = ready_pots + cooking_pots
        full_not_cooking = pot_states.get("3_items", [])
        intent_belief = intent_belief or {}
        human_serving = self._human_on_serving_duty(human, human_intent, mdp)
        dish_urgent = self._should_prepare_dish_now(
            state,
            mdp,
            me,
            human,
            human_intent,
            ready_pots,
            cooking_pots,
        )

        unblock_action = self._unblock_human_deadend(mdp, me, human)
        if unblock_action is not None:
            return unblock_action

        serving_unblock_action = self._unblock_human_serving_lane(state, mdp, me, human)
        if serving_unblock_action is not None:
            return serving_unblock_action

        handoff_action = self._constrained_handoff_action(
            state, mdp, me, human, layout_name
        )
        if handoff_action is not None:
            return handoff_action

        ingredient_prob = sum(
            intent_belief.get(k, 0.0)
            for k in ["get_onion", "get_tomato", "put_in_pot", "start_cooking"]
        )
        delivery_prob = sum(
            intent_belief.get(k, 0.0)
            for k in ["get_dish", "pickup_soup", "deliver_soup"]
        )

        if ready_or_cooking:
            if me.has_object():
                held = me.get_object().name
                if held == "soup":
                    return self._go_interact_with(mdp, me, mdp.get_serving_locations())
                if held == "dish":
                    if ready_pots:
                        return self._go_interact_with(mdp, me, ready_pots)
                    if dish_urgent and (not human_serving):
                        return self._go_interact_with(mdp, me, cooking_pots)
                    return self._drop_held_object_for_handoff(state, mdp, me, human)
                if held in ["onion", "tomato"]:
                    return self._drop_held_object_for_handoff(state, mdp, me, human)

            if ready_pots:
                if human_serving and len(ready_pots) <= 1:
                    needed = self._needed_ingredient_for_best_pot(state, mdp)
                    if needed == "tomato":
                        return self._go_interact_with(
                            mdp, me, mdp.get_tomato_dispenser_locations()
                        )
                    return self._go_interact_with(
                        mdp, me, mdp.get_onion_dispenser_locations()
                    )
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

            if dish_urgent and (not human_serving):
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

            needed = self._needed_ingredient_for_best_pot(state, mdp)
            if needed == "tomato":
                return self._go_interact_with(mdp, me, mdp.get_tomato_dispenser_locations())
            return self._go_interact_with(mdp, me, mdp.get_onion_dispenser_locations())

        if full_not_cooking:
            pot = min(full_not_cooking, key=lambda p: self._manhattan(me.position, p))
            d_me = self._manhattan(me.position, pot)
            d_human = self._manhattan(human.position, pot)
            if (not me.has_object()) and d_me <= d_human:
                return self._go_interact_with(mdp, me, [pot])
            if me.has_object() and me.get_object().name in ["onion", "tomato"]:
                return self._drop_held_object_for_handoff(state, mdp, me, human)

        if self.use_yield and self._should_yield_to_human(
            state, me, human, human_intent, mdp
        ):
            return Action.STAY

        if me.has_object():
            held = me.get_object().name
            if held in ["onion", "tomato"]:
                pots = self._candidate_pots_for_ingredient(state, mdp, held)
                if not pots:
                    return self._drop_held_object_for_handoff(state, mdp, me, human)
                return self._go_interact_with(mdp, me, pots)
            if held == "dish":
                if ready_or_cooking:
                    return self._go_interact_with(mdp, me, ready_or_cooking)
                return self._drop_held_object_for_handoff(state, mdp, me, human)
            if held == "soup":
                return self._go_interact_with(mdp, me, mdp.get_serving_locations())

        if ingredient_prob >= delivery_prob + 0.05:
            if ready_pots:
                if (not human_serving) or len(ready_pots) > 1:
                    return self._go_interact_with(
                        mdp, me, mdp.get_dish_dispenser_locations()
                    )
            elif cooking_pots and dish_urgent and (not human_serving):
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

            needed = self._needed_ingredient_for_best_pot(state, mdp)
            if needed == "tomato":
                return self._go_interact_with(mdp, me, mdp.get_tomato_dispenser_locations())
            return self._go_interact_with(mdp, me, mdp.get_onion_dispenser_locations())

        if delivery_prob > ingredient_prob + 0.05:
            if full_not_cooking:
                return self._go_interact_with(mdp, me, full_not_cooking)
            needed = self._needed_ingredient_for_best_pot(state, mdp)
            if needed == "tomato":
                return self._go_interact_with(mdp, me, mdp.get_tomato_dispenser_locations())
            return self._go_interact_with(mdp, me, mdp.get_onion_dispenser_locations())

        if full_not_cooking:
            return self._go_interact_with(mdp, me, full_not_cooking)
        if ready_pots:
            if (not human_serving) or len(ready_pots) > 1:
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())
        elif cooking_pots and dish_urgent and (not human_serving):
            return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

        return self._go_interact_with(mdp, me, mdp.get_onion_dispenser_locations())

    def _human_on_serving_duty(self, human, human_intent, mdp=None):
        if human.has_object():
            held = human.get_object().name
            if held in ["dish", "soup"]:
                return True

        if human_intent not in ["get_dish", "pickup_soup", "deliver_soup"]:
            return False

        if mdp is None:
            return True

        targets = (
            mdp.get_dish_dispenser_locations()
            + mdp.get_pot_locations()
            + mdp.get_serving_locations()
        )
        if not targets:
            return False

        d_human = min(self._manhattan(human.position, t) for t in targets)
        return d_human <= 2

    def _should_prepare_dish_now(
        self,
        state,
        mdp,
        me,
        human,
        human_intent,
        ready_pots,
        cooking_pots,
    ):
        if ready_pots:
            return True
        if not cooking_pots:
            return False

        soonest_ready = None
        for pot in cooking_pots:
            if not state.has_object(pot):
                continue
            soup = state.get_object(pot)
            if not getattr(soup, "is_cooking", False):
                continue
            remaining = getattr(soup, "cook_time_remaining", None)
            if remaining is None:
                continue
            if soonest_ready is None or remaining < soonest_ready:
                soonest_ready = remaining

        if soonest_ready is None:
            return False

        dish_dist = self._min_interaction_distance(
            mdp, me.position, mdp.get_dish_dispenser_locations()
        )
        pot_dist = self._min_interaction_distance(mdp, me.position, cooking_pots)
        if dish_dist is None or pot_dist is None:
            travel_est = 8
        else:
            travel_est = dish_dist + pot_dist

        if self._human_on_serving_duty(human, human_intent, mdp) and soonest_ready > 1:
            return False

        return soonest_ready <= max(4, min(12, travel_est + 3))

    def _constrained_handoff_action(self, state, mdp, me, human, layout_name):
        if self._layout_group(layout_name) != "constrained":
            self.handoff_role = "none"
            return None

        role = self._infer_handoff_role(mdp, me)
        if role == "none":
            self.handoff_role = "none"
            return None

        self.handoff_role = role
        shared_counters = self._shared_handoff_counters(state, mdp, me, human)
        if not shared_counters:
            return None

        pot_states = mdp.get_pot_states(state)
        ready_or_cooking = pot_states["ready"] + pot_states["cooking"]
        full_not_cooking = pot_states.get("3_items", [])

        counter_objs = mdp.get_counter_objects_dict(state, counter_subset=shared_counters)
        empty_shared = [c for c in shared_counters if not state.has_object(c)]
        needed_ing = self._needed_ingredient_for_best_pot(state, mdp)

        if role == "giver":
            if me.has_object():
                held = me.get_object().name
                if held in ["onion", "tomato", "dish"]:
                    if empty_shared:
                        return self._go_interact_with(mdp, me, empty_shared)
                    return self._go_near_features(mdp, me, shared_counters)
                if held == "soup":
                    return self._go_interact_with(mdp, me, mdp.get_serving_locations())
                if empty_shared:
                    return self._go_interact_with(mdp, me, empty_shared)
                return Action.STAY

            need_dish = bool(ready_or_cooking) and not counter_objs.get("dish")
            if need_dish and self._can_interact_with_any(
                mdp, me, mdp.get_dish_dispenser_locations()
            ):
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

            source = (
                mdp.get_tomato_dispenser_locations()
                if needed_ing == "tomato"
                else mdp.get_onion_dispenser_locations()
            )
            if self._can_interact_with_any(mdp, me, source):
                return self._go_interact_with(mdp, me, source)

            if self._can_interact_with_any(mdp, me, mdp.get_dish_dispenser_locations()):
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())

            return self._go_near_features(mdp, me, shared_counters)

        if me.has_object():
            held = me.get_object().name
            if held == "soup":
                return self._go_interact_with(mdp, me, mdp.get_serving_locations())
            if held == "dish":
                if ready_or_cooking:
                    return self._go_interact_with(mdp, me, ready_or_cooking)
                if full_not_cooking:
                    return self._go_interact_with(mdp, me, full_not_cooking)
                return self._go_near_features(mdp, me, shared_counters)
            if held in ["onion", "tomato"]:
                pots = self._candidate_pots_for_ingredient(state, mdp, held)
                if pots:
                    return self._go_interact_with(mdp, me, pots)
                if full_not_cooking:
                    return self._go_interact_with(mdp, me, full_not_cooking)
                if empty_shared:
                    return self._go_interact_with(mdp, me, empty_shared)
                return Action.STAY

        if full_not_cooking:
            return self._go_interact_with(mdp, me, full_not_cooking)

        if ready_or_cooking:
            dish_positions = list(counter_objs.get("dish", []))
            if dish_positions:
                return self._go_interact_with(mdp, me, dish_positions)
            if self._can_interact_with_any(mdp, me, mdp.get_dish_dispenser_locations()):
                return self._go_interact_with(mdp, me, mdp.get_dish_dispenser_locations())
            return self._go_near_features(mdp, me, shared_counters)

        preferred_obj = "tomato" if needed_ing == "tomato" else "onion"
        preferred_positions = list(counter_objs.get(preferred_obj, []))
        if preferred_positions:
            return self._go_interact_with(mdp, me, preferred_positions)

        alt_obj = "onion" if preferred_obj == "tomato" else "tomato"
        alt_positions = list(counter_objs.get(alt_obj, []))
        if alt_positions:
            return self._go_interact_with(mdp, me, alt_positions)

        source = (
            mdp.get_tomato_dispenser_locations()
            if preferred_obj == "tomato"
            else mdp.get_onion_dispenser_locations()
        )
        if self._can_interact_with_any(mdp, me, source):
            return self._go_interact_with(mdp, me, source)

        return self._go_near_features(mdp, me, shared_counters)

    def _infer_handoff_role(self, mdp, player):
        can_onion = self._can_interact_with_any(
            mdp, player, mdp.get_onion_dispenser_locations()
        )
        can_tomato = self._can_interact_with_any(
            mdp, player, mdp.get_tomato_dispenser_locations()
        )
        can_dish = self._can_interact_with_any(
            mdp, player, mdp.get_dish_dispenser_locations()
        )
        can_pot = self._can_interact_with_any(mdp, player, mdp.get_pot_locations())
        can_serve = self._can_interact_with_any(
            mdp, player, mdp.get_serving_locations()
        )

        supply_score = int(can_onion) + int(can_tomato) + int(can_dish)
        cook_score = int(can_pot) + int(can_serve)

        if supply_score > 0 and cook_score == 0:
            return "giver"
        if cook_score > 0 and supply_score == 0:
            return "receiver"
        return "none"

    def _shared_handoff_counters(self, state, mdp, me, human):
        counters = mdp.get_counter_locations()
        ranked = []
        for c in counters:
            if state.has_object(c):
                me_d = self._min_interaction_distance(mdp, me.position, [c])
                human_d = self._min_interaction_distance(mdp, human.position, [c])
                if me_d is None or human_d is None:
                    continue
                ranked.append((me_d + human_d - 0.25, c))
                continue

            me_d = self._min_interaction_distance(mdp, me.position, [c])
            human_d = self._min_interaction_distance(mdp, human.position, [c])
            if me_d is None or human_d is None:
                continue
            ranked.append((me_d + human_d, c))

        ranked.sort(key=lambda x: x[0])
        return [c for _, c in ranked]

    def _unblock_human_deadend(self, mdp, me, human):
        valid = set(mdp.get_valid_player_positions())
        human_pos = human.position
        if human_pos not in valid:
            return None

        neighbors = []
        for direction in Direction.ALL_DIRECTIONS:
            nxt = Action.move_in_direction(human_pos, direction)
            if nxt in valid:
                neighbors.append(nxt)

        if len(neighbors) != 1:
            return None

        exit_tile = neighbors[0]
        if me.position != exit_tile:
            return None

        best = None
        for action in Direction.ALL_DIRECTIONS:
            nxt, _ = mdp._move_if_direction(me.position, me.orientation, action)
            if nxt == me.position or nxt == human_pos:
                continue

            score = self._manhattan(nxt, human_pos)
            if best is None or score > best[0]:
                best = (score, action)

        return None if best is None else best[1]

    def _unblock_human_serving_lane(self, state, mdp, me, human):
        if not human.has_object() or human.get_object().name != "soup":
            return None

        if me.has_object() and me.get_object().name == "soup":
            return None

        serving_locations = mdp.get_serving_locations()
        if not serving_locations:
            return None

        human_goal = self._closest_reachable_interaction_goal(
            mdp, human.position, serving_locations
        )
        if human_goal is None:
            return None

        _, human_target_pos, _ = human_goal
        human_next_action = self._first_action_on_shortest_path(
            mdp, human.position, human_target_pos
        )
        human_next_pos = (
            Action.move_in_direction(human.position, human_next_action)
            if human_next_action in Direction.ALL_DIRECTIONS
            else human.position
        )

        blocking_now = me.position == human_target_pos or me.position == human_next_pos
        if not blocking_now:
            return None

        serving_goal_positions = {
            pos for pos, _ in self._interaction_goals(mdp, serving_locations)
        }

        best = None
        for action in Direction.ALL_DIRECTIONS:
            nxt, _ = mdp._move_if_direction(me.position, me.orientation, action)
            if nxt == me.position or nxt == human.position:
                continue
            if nxt == human_next_pos:
                continue

            score = self._manhattan(nxt, human.position)
            if nxt not in serving_goal_positions:
                score += 3
            if best is None or score > best[0]:
                best = (score, action)

        return None if best is None else best[1]

    def _can_interact_with_any(self, mdp, player, feature_positions):
        return (
            self._closest_reachable_interaction_goal(
                mdp, player.position, feature_positions
            )
            is not None
        )

    def _min_interaction_distance(self, mdp, start_pos, feature_positions):
        best = self._closest_reachable_interaction_goal(
            mdp, start_pos, feature_positions
        )
        return None if best is None else best[0]

    def _closest_reachable_interaction_goal(self, mdp, start_pos, feature_positions):
        goals = self._interaction_goals(mdp, feature_positions)
        if not goals:
            return None

        best = None
        for target_pos, target_or in goals:
            dist = self._shortest_path_distance(mdp, start_pos, target_pos)
            if dist is None:
                continue
            if best is None or dist < best[0]:
                best = (dist, target_pos, target_or)
        return best

    def _go_near_features(self, mdp, player, feature_positions):
        best = self._closest_reachable_interaction_goal(
            mdp, player.position, feature_positions
        )
        if best is None:
            return Action.STAY

        _, target_pos, target_or = best
        if player.position == target_pos:
            if player.orientation == target_or:
                return Action.STAY
            return target_or

        return self._step_towards(mdp, player.position, player.orientation, target_pos)

    def _drop_held_object_for_handoff(self, state, mdp, me, human):
        counters = mdp.get_empty_counter_locations(state)
        if not counters:
            return Action.STAY

        near_human = [
            c for c in counters if self._manhattan(c, human.position) <= 2
        ]
        targets = near_human if near_human else counters
        return self._go_interact_with(mdp, me, targets)

    def _should_yield_to_human(self, state, me, human, human_intent, mdp):
        if not self.use_commitment:
            return False

        if self._commitment_strength() < 0.75:
            return False

        if self._manhattan(me.position, human.position) > 2:
            return False

        pot_states = mdp.get_pot_states(state)
        open_pots = (
            list(pot_states["empty"])
            + list(pot_states["1_items"])
            + list(pot_states["2_items"])
        )
        if open_pots:
            if (not me.has_object()) or (
                me.has_object() and me.get_object().name in ["onion", "tomato"]
            ):
                return False

        if human_intent == "get_dish":
            targets = mdp.get_dish_dispenser_locations()
        elif human_intent in ["deliver_soup", "pickup_soup"]:
            targets = mdp.get_serving_locations() + mdp.get_pot_locations()
        else:
            targets = mdp.get_pot_locations()

        if not targets:
            return False

        d_me = min(self._manhattan(me.position, t) for t in targets)
        d_human = min(self._manhattan(human.position, t) for t in targets)
        return d_human < d_me and d_human <= 2 and d_me <= 3

    def _needed_ingredient_for_best_pot(self, state, mdp):
        target_order = self._select_target_recipe(state, mdp)
        if target_order is None:
            return "onion"

        target = list(target_order.ingredients)
        target_onion = target.count("onion")
        target_tomato = target.count("tomato")

        pot_positions = mdp.get_pot_locations()
        best_missing = None
        best_score = -math.inf

        for pos in pot_positions:
            onion_count = 0
            tomato_count = 0
            if state.has_object(pos):
                soup = state.get_object(pos)
                if soup.is_cooking or soup.is_ready:
                    continue
                onion_count = soup.ingredients.count("onion")
                tomato_count = soup.ingredients.count("tomato")

            missing_onion = max(0, target_onion - onion_count)
            missing_tomato = max(0, target_tomato - tomato_count)
            overflow = max(0, onion_count - target_onion) + max(
                0, tomato_count - target_tomato
            )
            fill_score = onion_count + tomato_count - 2 * overflow

            if missing_onion <= 0 and missing_tomato <= 0:
                continue

            if fill_score > best_score:
                best_score = fill_score
                best_missing = "onion" if missing_onion >= missing_tomato else "tomato"

        return best_missing or ("onion" if target_onion >= target_tomato else "tomato")

    def _candidate_pots_for_ingredient(self, state, mdp, ingredient_name):
        target_order = self._select_target_recipe(state, mdp)
        target_onion = None
        target_tomato = None
        if target_order is not None:
            target = list(target_order.ingredients)
            target_onion = target.count("onion")
            target_tomato = target.count("tomato")

        pots = []
        fallback = []
        for pos in mdp.get_pot_locations():
            if not state.has_object(pos):
                pots.append(pos)
                fallback.append(pos)
                continue
            soup = state.get_object(pos)
            if soup.is_cooking or soup.is_ready or soup.is_full:
                continue

            fallback.append(pos)

            if target_order is None:
                pots.append(pos)
                continue

            onion_count = soup.ingredients.count("onion")
            tomato_count = soup.ingredients.count("tomato")

            if onion_count > target_onion or tomato_count > target_tomato:
                continue

            if ingredient_name == "onion" and onion_count >= target_onion:
                continue
            if ingredient_name == "tomato" and tomato_count >= target_tomato:
                continue

            pots.append(pos)

        return pots if pots else fallback

    def _select_target_recipe(self, state, mdp):
        recipes = list(state.all_orders)
        if not recipes:
            return None

        pot_positions = mdp.get_pot_locations()
        best_recipe = None
        best_score = -math.inf

        for recipe in recipes:
            ingredients = list(recipe.ingredients)
            target_onion = ingredients.count("onion")
            target_tomato = ingredients.count("tomato")
            target_total = target_onion + target_tomato

            reward = float(mdp.get_recipe_value(state, recipe))
            cook_time = float(getattr(recipe, "time", 20) or 20)
            throughput = reward / max(1.0, cook_time)

            best_pot_fit = -float(target_total)
            for pos in pot_positions:
                if not state.has_object(pos):
                    pot_fit = 0.0
                else:
                    soup = state.get_object(pos)
                    if soup.is_cooking or soup.is_ready:
                        pot_fit = 3.0 if soup.recipe == recipe else -2.0
                    else:
                        onion_count = soup.ingredients.count("onion")
                        tomato_count = soup.ingredients.count("tomato")
                        matching = min(onion_count, target_onion) + min(
                            tomato_count, target_tomato
                        )
                        extra = max(0, onion_count - target_onion) + max(
                            0, tomato_count - target_tomato
                        )
                        missing = target_total - matching
                        pot_fit = 2.0 * matching - 2.5 * extra - 0.5 * missing

                if pot_fit > best_pot_fit:
                    best_pot_fit = pot_fit

            score = 2.0 * throughput + 0.1 * reward + best_pot_fit
            if recipe in state.bonus_orders:
                score += 0.5

            if score > best_score:
                best_score = score
                best_recipe = recipe

        return best_recipe

    def _go_interact_with(self, mdp, player, feature_positions):
        best = self._closest_reachable_interaction_goal(
            mdp, player.position, feature_positions
        )
        if best is None:
            return Action.STAY

        _, target_pos, target_or = best

        if player.position == target_pos:
            if player.orientation == target_or:
                return Action.INTERACT
            return target_or

        return self._step_towards(mdp, player.position, player.orientation, target_pos)

    def _shortest_path_distance(self, mdp, start_pos, target_pos):
        if start_pos == target_pos:
            return 0

        valid = set(mdp.get_valid_player_positions())
        if start_pos not in valid or target_pos not in valid:
            return None

        frontier = deque([(start_pos, 0)])
        seen = {start_pos}

        while frontier:
            curr, dist = frontier.popleft()
            for action in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(curr, action)
                if nxt not in valid or nxt in seen:
                    continue
                if nxt == target_pos:
                    return dist + 1
                seen.add(nxt)
                frontier.append((nxt, dist + 1))

        return None

    def _step_towards(self, mdp, pos, orient, target_pos):
        bfs_action = self._first_action_on_shortest_path(mdp, pos, target_pos)
        if bfs_action is not None:
            return bfs_action

        best_action = Action.STAY
        best_dist = self._manhattan(pos, target_pos)
        for action in Direction.ALL_DIRECTIONS:
            next_pos, _ = mdp._move_if_direction(pos, orient, action)
            dist = self._manhattan(next_pos, target_pos)
            if dist < best_dist:
                best_dist = dist
                best_action = action
        return best_action

    def _first_action_on_shortest_path(self, mdp, start_pos, target_pos):
        if start_pos == target_pos:
            return Action.STAY

        valid = set(mdp.get_valid_player_positions())
        if start_pos not in valid or target_pos not in valid:
            return None

        frontier = deque([start_pos])
        parent = {start_pos: None}

        while frontier:
            curr = frontier.popleft()
            if curr == target_pos:
                break
            for action in Direction.ALL_DIRECTIONS:
                nxt = Action.move_in_direction(curr, action)
                if nxt not in valid or nxt in parent:
                    continue
                parent[nxt] = (curr, action)
                frontier.append(nxt)

        if target_pos not in parent:
            return None

        node = target_pos
        first_action = None
        while parent[node] is not None:
            prev, action = parent[node]
            first_action = action
            node = prev
        return first_action

    def _interaction_goals(self, mdp, feature_positions):
        valid = set(mdp.get_valid_player_positions())
        goals = []
        for fx, fy in feature_positions:
            for d in Direction.ALL_DIRECTIONS:
                px, py = Action.move_in_direction((fx, fy), d)
                if (px, py) in valid:
                    goals.append(((px, py), Direction.OPPOSITE_DIRECTIONS[d]))
        return goals

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _is_near_any(pos, targets, dist=1):
        for t in targets:
            if abs(pos[0] - t[0]) + abs(pos[1] - t[1]) <= dist:
                return True
        return False


class DummyAI:
    """
    Randomly samples actions. Used for debugging
    """

    def action(self, state):
        [action] = random.sample(
            [
                Action.STAY,
                Direction.NORTH,
                Direction.SOUTH,
                Direction.WEST,
                Direction.EAST,
                Action.INTERACT,
            ],
            1,
        )
        return action, None

    def reset(self):
        pass


class DummyComputeAI(DummyAI):
    """
    Performs simulated compute before randomly sampling actions. Used for debugging
    """

    def __init__(self, compute_unit_iters=1e5):
        """
        compute_unit_iters (int): Number of for loop cycles in one "unit" of compute. Number of
                                    units performed each time is randomly sampled
        """
        super(DummyComputeAI, self).__init__()
        self.compute_unit_iters = int(compute_unit_iters)

    def action(self, state):
        # Randomly sample amount of time to busy wait
        iters = random.randint(1, 10) * self.compute_unit_iters

        # Actually compute something (can't sleep) to avoid scheduling optimizations
        val = 0
        for i in range(iters):
            # Avoid branch prediction optimizations
            if i % 2 == 0:
                val += 1
            else:
                val += 2

        # Return randomly sampled action
        return super(DummyComputeAI, self).action(state)


class StayAI:
    """
    Always returns "stay" action. Used for debugging
    """

    def action(self, state):
        return Action.STAY, None

    def reset(self):
        pass


class TutorialAI:
    COOK_SOUP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab second onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Grab third onion
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.NORTH,
        Action.INTERACT,
        # Cook soup
        Action.INTERACT,
        # Grab plate
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        Direction.WEST,
        Direction.NORTH,
        # Deliver soup
        Action.INTERACT,
        Direction.EAST,
        Direction.EAST,
        Direction.EAST,
        Action.INTERACT,
        Direction.WEST,
    ]

    COOK_SOUP_COOP_LOOP = [
        # Grab first onion
        Direction.WEST,
        Direction.WEST,
        Direction.WEST,
        Action.INTERACT,
        # Place onion in pot
        Direction.EAST,
        Direction.SOUTH,
        Action.INTERACT,
        # Move to start so this loops
        Direction.EAST,
        Direction.EAST,
        # Pause to make cooperation more real time
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
        Action.STAY,
    ]

    def __init__(self):
        self.curr_phase = -1
        self.curr_tick = -1

    def action(self, state):
        self.curr_tick += 1
        if self.curr_phase == 0:
            return (
                self.COOK_SOUP_LOOP[self.curr_tick % len(self.COOK_SOUP_LOOP)],
                None,
            )
        elif self.curr_phase == 2:
            return (
                self.COOK_SOUP_COOP_LOOP[
                    self.curr_tick % len(self.COOK_SOUP_COOP_LOOP)
                ],
                None,
            )
        return Action.STAY, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
