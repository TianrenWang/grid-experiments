import torch
import math

import numpy as np

print(np.__version__)


class Node:
    def __init__(
        self, game, args, state, parent=None, action_taken=None, prior=0, visit_count=0
    ):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []

        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self, exploration: int = 0):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child, exploration)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child, exploration: int = 0):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return (
            q_value
            + exploration
            * (math.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)

                child = Node(self.game, self.args, child_state, self, action, prob)
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
            self.parent.backpropagate(value)


class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        modelOutput = self.model(
            torch.tensor(
                self.game.get_encoded_state(state), device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(modelOutput[0], axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet([self.args["dirichlet_alpha"]] * self.game.action_size)

        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)

        for search in range(self.args["num_searches"]):
            node = root

            while node.is_fully_expanded():
                node = node.select(self.args["C"])

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                modelOutput = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(node.state),
                        device=self.model.device,
                    ).unsqueeze(0)
                )

                policy = torch.softmax(modelOutput[0], axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = modelOutput[1].item()

                node.expand(policy)

            node.backpropagate(value)

        if self.args["num_searches"] > 0:
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            return action_probs, modelOutput[len(modelOutput) - 1]
        else:
            return policy, modelOutput[len(modelOutput) - 1]


class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, spGames):
        modelOutput = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(modelOutput[0], axis=1).cpu().numpy()
        policy = (1 - self.args["dirichlet_epsilon"]) * policy + self.args[
            "dirichlet_epsilon"
        ] * np.random.dirichlet(
            [self.args["dirichlet_alpha"]] * self.game.action_size, size=policy.shape[0]
        )

        for i, spg in enumerate(spGames):
            spg_policy = policy[i]
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy)

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)

        for search in range(self.args["num_searches"]):
            for spg in spGames:
                spg.node = None
                node = spg.root

                while node.is_fully_expanded():
                    node = node.select(spg.exploration)

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken
                )
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    node.backpropagate(value)

                else:
                    spg.node = node

            expandable_spGames = [
                mappingIdx
                for mappingIdx in range(len(spGames))
                if spGames[mappingIdx].node is not None
            ]

            if len(expandable_spGames) > 0:
                states = np.stack(
                    [
                        spGames[mappingIdx].node.state
                        for mappingIdx in expandable_spGames
                    ]
                )

                modelOutput = self.model(
                    torch.tensor(
                        self.game.get_encoded_state(states), device=self.model.device
                    )
                )
                policy = torch.softmax(modelOutput[0], axis=1).cpu().numpy()
                value = modelOutput[1].cpu().numpy()

            for i, mappingIdx in enumerate(expandable_spGames):
                node = spGames[mappingIdx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)

                node.expand(spg_policy)
                node.backpropagate(spg_value)
