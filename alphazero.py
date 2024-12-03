import numpy as np
import torch.nn.functional as F
import random
import torch
from datetime import datetime
import os
import pickle

from mcts import MCTSParallel
from self_eval import testAgentVSAgent, Agent
from models import PlaceCellResNet


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def selfPlay(self):
        return_memory = []
        player = 1
        spGames = [SPG(self.game) for spg in range(self.args["num_parallel_games"])]

        while len(spGames) > 0:
            states = np.stack([spg.state for spg in spGames])
            neutral_states = self.game.change_perspective(states, player)

            self.mcts.search(neutral_states, spGames)

            for i in range(len(spGames))[::-1]:
                spg = spGames[i]

                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (
                    1 / self.args["temperature"]
                )
                temperature_action_probs /= temperature_action_probs.sum()
                action = np.random.choice(
                    self.game.action_size, p=temperature_action_probs
                )

                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(
                    spg.state, action
                )

                if is_terminal:
                    for (
                        hist_neutral_state,
                        hist_action_probs,
                        hist_player,
                    ) in spg.memory:
                        hist_outcome = (
                            value
                            if hist_player == player
                            else self.game.get_opponent_value(value)
                        )
                        return_memory.append(
                            (
                                self.game.get_encoded_state(hist_neutral_state),
                                hist_action_probs,
                                hist_outcome,
                            )
                        )
                    del spGames[i]

            player = self.game.get_opponent(player)

        return return_memory

    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            sample = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])
            ]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets, dtype=torch.float32, device=self.model.device
            )
            value_targets = torch.tensor(
                value_targets, dtype=torch.float32, device=self.model.device
            )

            loss = 0
            modelOutput = self.model(state)
            if len(modelOutput) == 3:
                out_policy, out_value, out_latent = modelOutput
            else:
                out_policy, out_value, out_place, out_latent = modelOutput
                place_targets = self.model.placeCells(out_latent)
                loss += F.cross_entropy(out_place, place_targets)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss += policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _getLatents(self, memory):
        if not isinstance(self.model, PlaceCellResNet):
            raise Exception(
                "Cannot use this function if model does not have place cell."
            )
        latents = []
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            sample = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])
            ]
            state, _, _ = zip(*sample)
            state = np.array(state)
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            latents.append(
                torch.reshape(
                    self.model(state)[-1], (-1, self.model.placeCells.cellDim)
                )
            )

        return torch.concat(latents)

    def _countPlaceCellFrequencies(self, latents: torch.Tensor):
        if not isinstance(self.model, PlaceCellResNet):
            raise Exception(
                "Cannot use this function if model does not have place cell."
            )

        batchSize = 1000
        for i in range(0, len(latents), batchSize):
            actualBatchSize = (
                batchSize if i + batchSize < len(latents) else len(latents) - i
            )
            batch = latents[i : i + actualBatchSize]
            self.model.placeCells.countFrequencies(batch)

    def learn(self):
        startingPoint = 0
        experimentName = self.args["exp_name"]
        if "prev_version" in self.args and self.args["prev_version"]:
            startingPoint = self.args["prev_version"] + 1
        for iteration in range(startingPoint, self.args["num_iterations"]):
            print(datetime.now())
            print(f"CURRENT ITERATION OUT OF {self.args['num_iterations']}:", iteration)
            memory = None
            if memoryVersion := self.args["memory"]:
                with open(
                    f"results/{experimentName}/version_{memoryVersion}/memory.pkl",
                    "rb",
                ) as file:
                    memory = pickle.load(file)
            if not memory:
                memory = []

                self.model.eval()
                parallelIterations = (
                    self.args["num_selfPlay_iterations"]
                    // self.args["num_parallel_games"]
                )

                print(datetime.now())
                for selfPlay_iteration in range(parallelIterations):
                    print(
                        f"CURRENT SELF-PLAY ITERATION OUT OF {parallelIterations}:",
                        selfPlay_iteration,
                    )
                    memory += self.selfPlay()

            if isinstance(self.model, PlaceCellResNet):
                print(datetime.now())
                print("ALIGNING Place Cells' distribution")
                self.model.eval()
                latents = self._getLatents(memory)
                print(
                    f"Distances before alignment: {self.model.placeCells.getTotalDistance(latents) / len(latents)}"
                )

                def getShuffled(latents: torch.Tensor):
                    permutation = torch.randperm(latents.size(0))
                    return latents[permutation]

                alignmentBatchSize = self.args["num_selfPlay_iterations"] * 2
                for batchIdx in range(100):
                    latents = getShuffled(latents)
                    for batchIdx in range(0, len(latents), alignmentBatchSize):
                        batch = latents[
                            batchIdx : min(
                                len(latents) - 1, batchIdx + alignmentBatchSize
                            )
                        ]
                        self.model.placeCells.learn(batch)
                print(
                    f"Distances after alignment: {self.model.placeCells.getTotalDistance(latents) / len(latents)}"
                )

            self.model.train()
            print(datetime.now())
            for epoch in range(self.args["num_epochs"]):
                if isinstance(self.model, PlaceCellResNet):
                    self.model.placeCells.resetFireFrequency()
                print(f"CURRENT EPOCH OUT OF {self.args['num_epochs']}:", epoch)
                self.train(memory)

            print(datetime.now())
            folderPath = f"results/{experimentName}/version_{iteration}"
            if not os.path.exists(folderPath):
                os.makedirs(folderPath)
            torch.save(self.model.state_dict(), folderPath + "/model.pt")
            torch.save(self.optimizer.state_dict(), folderPath + "/optimizer.pt")
            with open(folderPath + "/memory.pkl", "wb") as file:
                pickle.dump(memory, file)

            testAgentVSAgent(
                Agent(experimentName, iteration, self.model), numberOfGamesToPlay=400
            )
            self.model.train()


class SPG:
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
