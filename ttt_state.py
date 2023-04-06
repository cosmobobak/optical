
from typing import Tuple
import numpy as np
from random import choice
from functools import reduce

class TTT:
    X, O = 1, -1
    ACTION_SPACE_SIZE = 9

    def __init__(self, input_node=None) -> None:
        if input_node is not None:
            self.node = input_node
        else:
            self.node: np.ndarray = np.zeros((2, 9))
        self.move_count = int(self.node.sum())
        self.stack = []

    def __eq__(self, other: "TTT") -> bool:
        return (self.node == other.node).all()

    def __hash__(self) -> int:
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        side_prime = 29
        side_x = reduce(lambda x, y: x * y, (primes[i] for i in range(9) if self.node[0][i] == 1), 1)
        side_o = reduce(lambda x, y: x * y, (primes[i] for i in range(9) if self.node[1][i] == 1), 1)
        product = side_x * side_o * side_prime
        return hash(product)

    def reset(self) -> None:
        self.move_count = 0
        self.stack = []
        self.node: np.ndarray = np.zeros((2, 9))

    def set_starting_position(self) -> None:
        self.reset()

    def get_turn(self) -> int:
        return self.O if (self.move_count & 1) else self.X

    def get_turn_as_str(self) -> str:
        return 'O' if (self.move_count & 1) else 'X'

    def get_move_counter(self) -> int:
        return self.move_count

    def pos_filled(self, i) -> bool:
        return self.node[0][i] != 0 or self.node[1][i] != 0

    # only valid to use if self.pos_filled() returns True:
    def player_at(self, i) -> bool:
        return self.node[0][i] != 0

    def probe_spot(self, i: int) -> bool:
        # tests the bit of the most recently played side
        return self.node[(self.move_count + 1) & 1][i] == 1

    def is_full(self) -> bool:
        return all((self.pos_filled(i) for i in range(9)))

    def symbol(self, xy: "tuple[int, int]") -> str:
        x, y = xy
        return ('X' if self.player_at(x * 3 + y) else 'O') if self.pos_filled(x * 3 + y) else '.'

    def __repr__(self) -> str:
        def gs(x): return self.symbol(x)
        pairs = [
            [(0, 0), (0, 1), (0, 2)],
            [(1, 0), (1, 1), (1, 2)],
            [(2, 0), (2, 1), (2, 2)]
        ]
        return '\n'.join([' '.join(map(gs, pairline)) for pairline in pairs])

    def play(self, i) -> None:
        self.node[self.move_count & 1][i] = 1
        self.move_count += 1
        self.stack.append(i)

    def unplay(self) -> None:
        assert self.move_count > 0
        i = self.stack.pop()
        self.move_count -= 1
        self.node[self.move_count & 1][i] = 0

    def push(self, i) -> None:
        self.play(i)

    def push_ret(self, i) -> "TTT":
        self.push(i)
        return self

    def pop(self) -> None:
        self.unplay()

    def evaluate(self) -> int:
        # check first diagonal
        if (self.probe_spot(0) and self.probe_spot(4) and self.probe_spot(8)):
            return -self.get_turn()

        # check second diagonal
        if (self.probe_spot(2) and self.probe_spot(4) and self.probe_spot(6)):
            return -self.get_turn()

        # check rows
        for i in range(3):
            if (self.probe_spot(i * 3) and self.probe_spot(i * 3 + 1) and self.probe_spot(i * 3 + 2)):
                return -self.get_turn()

        # check columns
        for i in range(3):
            if (self.probe_spot(i) and self.probe_spot(i + 3) and self.probe_spot(i + 6)):
                return -self.get_turn()

        return 0

    def is_terminal(self) -> bool:
        return self.is_full() or (self.evaluate() != 0)

    def legal_moves(self) -> "list[int]":
        return [m for m in range(9) if not self.pos_filled(m)]

    def children(self) -> "list[TTT]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append(self.clone())
            self.unplay()

        return cs

    def state_action_pairs(self) -> "list[tuple[TTT, int]]":
        cs = []
        for move in self.legal_moves():
            self.play(move)
            cs.append((self.clone(), move))
            self.unplay()

        return cs

    def random_play(self) -> None:
        self.play(choice(self.legal_moves()))

    # returns a 2x3x3 array, where the first dimension is the two sides, and the second and third are the board
    # this is the format that the neural network expects.
    def vectorise(self) -> np.ndarray:
        return np.reshape(self.node.copy(), (2, 3, 3))

    # returns a 3x3x2 array, where the first two dimensions are the board, and the third is the two sides
    # this is what tensorflow wanted, but we don't use it anymore
    def vectorise_chlast(self) -> np.ndarray:
        out = np.reshape(self.node.copy(), (2, 3, 3))
        return np.swapaxes(out, 0, 2)

    def flatten(self) -> np.ndarray:
        return np.reshape(self.node.copy(), (18))

    def clone(self) -> "TTT":
        return TTT(self.node.copy())

    @classmethod
    def _perft(cls, state: "TTT", ss: "set[TTT]"):
        if state in ss:
            return
        ss.add(state.clone())
        if state.is_terminal():
            return
        for move in state.legal_moves():
            state.push(move)
            cls._perft(state, ss)
            state.pop()

    @classmethod
    def get_every_state(cls) -> "set[TTT]":
        ss = set()
        state = TTT()
        cls._perft(state, ss)
        return ss

    @classmethod
    def state_space(cls) -> int:
        return 5478

    def best_move(self) -> int:
        def negamax(state: "TTT", alpha: int, beta: int) -> "Tuple[int, int]":
            if state.is_terminal():
                return state.evaluate(), -1
            best_score = -2
            best_move = -1
            for move in state.legal_moves():
                state.push(move)
                score, _ = negamax(state, -beta, -alpha)
                state.pop()
                score = -score
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return best_score, best_move
        return negamax(self, -2, 2)[1]


FIRST_9_STATES: "list[TTT]" = [
    TTT().push_ret(0),
    TTT().push_ret(1),
    TTT().push_ret(2),
    TTT().push_ret(3),
    TTT().push_ret(4),
    TTT().push_ret(5),
    TTT().push_ret(6),
    TTT().push_ret(7),
    TTT().push_ret(8)
]
