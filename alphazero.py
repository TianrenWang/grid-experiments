import numpy as np
import torch.nn.functional as F
import random
import torch
from datetime import datetime
import os
import pickle
import math

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

    def _negativeEntropyLoss(self, activations: torch.Tensor):
        """
        Compute the negative entropy loss for a batch of activation patterns.
        Encourages activation pattern to be sharp for a few cells per state.

        Args:
            activations (torch.Tensor): The output of the softmax layer,
                                        shape (batch_size, num_cells).
        Returns:
            torch.Tensor: The negative entropy loss.
        """

        if not isinstance(self.model, PlaceCellResNet):
            raise Exception("Can only be used if model is a PlaceCellResNet")

        epsilon = 1e-12
        log_probs = torch.log(activations + epsilon)
        entropy = -torch.sum(activations * log_probs) / activations.size(0)
        return entropy

    def _activationBiasLoss(self, activations: torch.Tensor):
        """
        Computes loss on how disproportionately a few cells fire across
        a batch. Discourages over reliance on a firing of few cells.

        Args:
            activations (torch.Tensor): The output of the softmax layer,
                                        shape (batch_size, num_cells).
        Returns:
            torch.Tensor: The activation bias loss.
        """

        if not isinstance(self.model, PlaceCellResNet):
            raise Exception("Can only be used if model is a PlaceCellResNet")

        totalActivationAcrossBatch = torch.sum(activations, dim=0)
        biasAcrossBatch = torch.sqrt(torch.sum(totalActivationAcrossBatch**2))
        batchSize = activations.size(0)
        floorActivations = math.floor(batchSize / self.model.numCells)
        ceilActivations = math.ceil(batchSize / self.model.numCells)
        remainingActivations = batchSize % self.model.numCells
        standardization = torch.sqrt(
            torch.scalar_tensor(
                remainingActivations * ceilActivations**2
                + (self.model.numCells - remainingActivations) * floorActivations**2
            )
        )
        return torch.abs(biasAcrossBatch - standardization) / activations.size(0)

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

    def _processTrainingSamples(self, samples):
        states, policy_targets, value_targets = zip(*samples)
        states, policy_targets, value_targets = (
            np.array(states),
            np.array(policy_targets),
            np.array(value_targets).reshape(-1, 1),
        )
        states = torch.tensor(states, dtype=torch.float32, device=self.model.device)
        policy_targets = torch.tensor(
            policy_targets, dtype=torch.float32, device=self.model.device
        )
        value_targets = torch.tensor(
            value_targets, dtype=torch.float32, device=self.model.device
        )
        return states, policy_targets, value_targets

    def train(self, memory):
        random.shuffle(memory)
        evaluationMemory = memory[:1024]
        memory = memory[1024:]
        for batchIdx in range(0, len(memory), self.args["batch_size"]):
            # Change to memory[batchIdx:batchIdx+self.args['batch_size']] in case of an error
            samples = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.args["batch_size"])
            ]
            states, policy_targets, value_targets = self._processTrainingSamples(
                samples
            )
            loss = 0
            modelOutput = self.model(states)
            if isinstance(self.model, PlaceCellResNet):
                out_policy, out_value, out_place, out_latent = modelOutput
                firingLoss = self._negativeEntropyLoss(out_place)
                biasLoss = self._activationBiasLoss(out_place)
                loss += firingLoss + biasLoss
            else:
                out_policy, out_value, out_latent = modelOutput
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss += policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        states, _, __ = self._processTrainingSamples(evaluationMemory)
        modelOutput = self.model(states)
        if isinstance(self.model, PlaceCellResNet):
            _, __, out_place, ___ = modelOutput
            firingLoss = self._negativeEntropyLoss(out_place)
            biasLoss = self._activationBiasLoss(out_place)
            print("Place Cell Evaluation:")
            print("Firing Loss: ", firingLoss.item())
            print("Bias Loss: ", biasLoss.item())

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

            if isinstance(self.model, PlaceCellResNet) and False:
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

                for _ in range(self.args["num_cell_alignments"]):
                    self.model.placeCells.resetFireFrequency()
                    self._countPlaceCellFrequencies(latents)

                    for _ in range(20):
                        self.model.placeCells.calibrate()

                    latents = getShuffled(latents)
                    for batchIdx in range(0, len(latents), self.args["batch_size"]):
                        batch = latents[
                            batchIdx : min(
                                len(latents) - 1, batchIdx + self.args["batch_size"]
                            )
                        ]
                        self.model.placeCells.learn(batch)

                for batchIdx in range(100):
                    latents = getShuffled(latents)
                    for batchIdx in range(0, len(latents), self.args["batch_size"]):
                        batch = latents[
                            batchIdx : min(
                                len(latents) - 1, batchIdx + self.args["batch_size"]
                            )
                        ]
                        self.model.placeCells.learn(batch)
                print(
                    f"Distances after alignment: {self.model.placeCells.getTotalDistance(latents) / len(latents)}"
                )

            self.model.train()
            print(datetime.now())
            for epoch in range(self.args["num_epochs"]):
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
