"""Fast-enough blackjack shoe simulator with basic strategy.

Rules: 6 decks, dealer hits soft 17, blackjack pays 3:2, double on any two cards,
double after split, split once (no re-split), split aces get one card each,
no surrender, dealer peeks for blackjack, reshuffle at 75% penetration.

Card ranks are stored as integers 0-9:
    0 = Ace, 1 = 2, 2 = 3, ..., 8 = 9, 9 = ten-value (10, J, Q, K)
"""
import random
import numpy as np

N_DECKS = 6
PER_DECK = np.array([4] * 9 + [16])          # cards of each rank in one deck
RANK_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
HI_LO = np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1])   # Hi-Lo tag for each rank
VALUE = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


def hand_value(cards):
    """Return (total, is_soft). Aces count 11 when that doesn't bust."""
    total = sum(VALUE[c] for c in cards)
    if 0 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def basic_strategy(cards, up, can_double, can_split):
    """Basic strategy for 6 decks, H17, DAS, no surrender.
    up = dealer upcard value 2-11 (11 = Ace). Returns 'H', 'S', 'D' or 'P'."""
    # Pairs
    if can_split and len(cards) == 2 and cards[0] == cards[1]:
        r = cards[0]
        if r == 0 or r == 7:                       # A,A and 8,8: always split
            return "P"
        if r in (1, 2) and up <= 7: return "P"    # 2,2 / 3,3
        if r == 3 and up in (5, 6): return "P"    # 4,4
        if r == 5 and up <= 6: return "P"         # 6,6
        if r == 6 and up <= 7: return "P"         # 7,7
        if r == 8 and up not in (7, 10, 11): return "P"   # 9,9
        # 5,5 and 10,10 fall through to totals
    total, soft = hand_value(cards)
    if soft:
        if total >= 20: return "S"
        if total == 19:                            # A,8
            return "D" if (up == 6 and can_double) else "S"
        if total == 18:                            # A,7
            if up <= 6: return "D" if can_double else "S"
            return "S" if up in (7, 8) else "H"
        if total == 17: return "D" if (3 <= up <= 6 and can_double) else "H"
        if total in (15, 16): return "D" if (4 <= up <= 6 and can_double) else "H"
        if total in (13, 14): return "D" if (5 <= up <= 6 and can_double) else "H"
        return "H"
    if total >= 17: return "S"
    if total >= 13: return "S" if up <= 6 else "H"
    if total == 12: return "S" if 4 <= up <= 6 else "H"
    if total == 11: return "D" if can_double else "H"
    if total == 10: return "D" if (up <= 9 and can_double) else "H"
    if total == 9: return "D" if (3 <= up <= 6 and can_double) else "H"
    return "H"


class Shoe:
    def __init__(self, n_decks=N_DECKS, penetration=0.75, rng=None):
        self.rng = rng or random.Random()
        self.n_decks = n_decks
        self.cut = int(52 * n_decks * penetration)
        self.shuffle()

    def shuffle(self):
        self.cards = [r for r in range(10) for _ in range(PER_DECK[r] * self.n_decks)]
        self.rng.shuffle(self.cards)
        self.remaining = (PER_DECK * self.n_decks).copy()
        self.dealt = 0

    def draw(self):
        c = self.cards.pop()
        self.remaining[c] -= 1
        self.dealt += 1
        return c

    def needs_shuffle(self):
        return self.dealt >= self.cut


def play_round(shoe, strategy=basic_strategy):
    """Play one round with a 1-unit bet. Returns the player's profit in units."""
    draw = shoe.draw
    p = [draw()]; d = [draw()]; p.append(draw()); d.append(draw())
    up = 11 if d[0] == 0 else VALUE[d[0]]
    player_bj = hand_value(p)[0] == 21
    dealer_bj = hand_value(d)[0] == 21
    if dealer_bj:                                  # dealer peeks
        return 0.0 if player_bj else -1.0
    if player_bj:
        return 1.5

    hands, bets, split_done = [p], [1.0], False
    i = 0
    while i < len(hands):
        h = hands[i]
        split_aces = split_done and h[0] == 0
        if split_done and len(h) == 1:             # deal second card to a split hand
            h.append(draw())
            if split_aces:                          # split aces get one card only
                i += 1
                continue
        while True:
            total, _ = hand_value(h)
            if total >= 21:
                break
            first_two = len(h) == 2
            act = strategy(h, up, can_double=first_two, can_split=first_two and not split_done)
            if act == "S":
                break
            if act == "H":
                h.append(draw())
            elif act == "D":
                bets[i] *= 2
                h.append(draw())
                break
            elif act == "P":
                split_done = True
                hands = [[h[0]], [h[1]]]
                bets = [1.0, 1.0]
                h = hands[0]
                h.append(draw())
                if h[0] == 0:                       # split aces: one card, move on
                    break
        i += 1

    totals = [hand_value(h)[0] for h in hands]
    profit = 0.0
    if any(t <= 21 for t in totals):
        while True:                                 # dealer hits soft 17
            dt, ds = hand_value(d)
            if dt < 17 or (dt == 17 and ds):
                d.append(draw())
            else:
                break
    dt = hand_value(d)[0]
    for t, b in zip(totals, bets):
        if t > 21: profit -= b
        elif dt > 21 or t > dt: profit += b
        elif t < dt: profit -= b
    return profit


def simulate(n_rounds, seed=0, n_decks=N_DECKS, penetration=0.75):
    """Play n_rounds through continuously shuffled shoes.
    Returns (remaining-card counts before each round [n, 10], profit per round [n])."""
    rng = random.Random(seed)
    shoe = Shoe(n_decks, penetration, rng)
    comp = np.empty((n_rounds, 10), dtype=np.int16)
    profit = np.empty(n_rounds, dtype=np.float32)
    for k in range(n_rounds):
        if shoe.needs_shuffle():
            shoe.shuffle()
        comp[k] = shoe.remaining
        profit[k] = play_round(shoe)
    return comp, profit


class _FixedShoe:
    """A shuffled copy of a given set of remaining cards (used for repeated hands)."""
    def __init__(self, cards, rng):
        self.cards = cards[:]
        rng.shuffle(self.cards)
        self.remaining = np.zeros(10, dtype=np.int64)   # not tracked here
        self.dealt = 0

    def draw(self):
        return self.cards.pop()


def simulate_states(n_states, hands_per_state=16, seed=0, n_decks=N_DECKS, penetration=0.75):
    """Variance-reduced data: walk through real shoes, and at each point play
    `hands_per_state` independent hands from a fresh shuffle of the cards that are left.
    Returns (composition [n, 10], average profit per hand [n])."""
    rng = random.Random(seed)
    shoe = Shoe(n_decks, penetration, rng)
    comp = np.empty((n_states, 10), dtype=np.int16)
    avg = np.empty(n_states, dtype=np.float32)
    for k in range(n_states):
        if shoe.needs_shuffle():
            shoe.shuffle()
        comp[k] = shoe.remaining
        left = shoe.cards
        total = 0.0
        for _ in range(hands_per_state):
            total += play_round(_FixedShoe(left, rng))
        avg[k] = total / hands_per_state
        play_round(shoe)                       # advance the real shoe by one round
    return comp, avg


# ---------- helpers used by the notebook ----------

def composition_features(comp):
    """Turn remaining-card counts into 'extra cards of each rank already played, per deck left'.
    0 everywhere = a perfectly neutral shoe. Positive for a rank = more of it is gone than average."""
    comp = np.asarray(comp, dtype=np.float32)
    left = comp.sum(1, keepdims=True)
    expected_left = PER_DECK / 52 * left
    return ((expected_left - comp) / (left / 52)).astype(np.float32)


def hilo_true_count(features):
    """Hi-Lo true count = running count / decks remaining (equivalent formula)."""
    return (features * HI_LO).sum(1)


def _hands_job(args):
    n, seed = args
    return simulate(n, seed=seed)


def _states_job(args):
    n, k, seed = args
    return simulate_states(n, k, seed=seed)


def parallel(job, arg_list, processes=None):
    """Run simulation chunks on several CPU cores and stack the results."""
    from multiprocessing import Pool
    with Pool(processes) as pool:
        parts = pool.map(job, arg_list)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
