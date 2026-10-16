"""Rule checks for blackjack.py. Run with:  python test_blackjack.py   (or: pytest)

Each test deals a rigged shoe in a fixed order:
player card 1, dealer upcard, player card 2, dealer hole card, then any hits.
Cards are given by value (2-10, or 'A').
"""
import numpy as np
import blackjack as bj


class RiggedShoe:
    def __init__(self, values):
        ranks = [0 if v == "A" else (9 if v == 10 else v - 1) for v in values]
        self.cards = list(reversed(ranks))
        self.remaining = np.zeros(10)
        self.dealt = 0

    def draw(self):
        return self.cards.pop()


def play(*values):
    return bj.play_round(RiggedShoe(values))


def test_player_blackjack_pays_3_to_2():
    assert play("A", 9, 10, 7) == 1.5

def test_dealer_blackjack_beats_20():
    assert play(10, 10, 10, "A") == -1.0

def test_both_blackjack_push():
    assert play("A", "A", 10, 10) == 0.0

def test_dealer_peek_takes_only_original_bet():
    assert play(6, "A", 5, 10) == -1.0          # player has 11 but never gets to double

def test_push():
    assert play(10, 9, 8, 9) == 0.0

def test_player_bust_loses():
    assert play(10, 10, 2, 7, 10) == -1.0       # 12 vs 10 hits, gets 10, busts

def test_double_down_pays_double():
    assert play(6, 6, 5, 10, 10, 10) == 2.0     # 11 doubles to 21; dealer 16 hits and busts

def test_dealer_hits_soft_17():
    assert play(10, "A", 8, 6, 2) == -1.0       # dealer A,6 hits, makes 19 vs player's 18

def test_split_eights():
    assert play(8, 6, 8, 10, 10, 10, 10) == 2.0

def test_double_after_split():
    assert play(3, 6, 3, 10, 8, 10, 8, 10, 10) == 4.0   # 3,3 split; both hands 11, double, 21

def test_split_aces_get_one_card_and_21_is_not_blackjack():
    assert play("A", 9, "A", 10, 10, 10) == 2.0          # pays 1:1, not 3:2

def test_soft_double():
    assert play("A", 4, 7, 10, 3, 10) == 2.0             # A,7 vs 4 doubles

def test_strategy_spot_checks():
    s = bj.basic_strategy
    assert s([9, 5], 10, True, False) == "H"             # hard 16 vs 10: hit (no surrender)
    assert s([5, 4], 11, True, False) == "D"             # 11 vs ace: double (dealer hits soft 17)
    assert s([0, 6], 2, True, False) == "D"              # soft 18 vs 2: double (H17)
    assert s([0, 6], 9, True, False) == "H"              # soft 18 vs 9: hit
    assert s([8, 8], 7, True, True) == "S"               # 9,9 vs 7: stand
    assert s([7, 7], 11, True, True) == "P"              # 8,8 vs ace: split
    assert s([9, 9], 6, True, True) == "S"               # never split tens


if __name__ == "__main__":
    tests = [f for name, f in dict(globals()).items() if name.startswith("test_")]
    for t in tests:
        t()
    print(f"All {len(tests)} tests passed")
