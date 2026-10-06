"""
CuiSync - Shapley values over the four entity-type similarity scores

Implements the formula in Chapter 3:

    phi_i = sum over S subset of N\\{i} of  |S|! (|N|-|S|-1)! / |N|!  *  [ v(S U {i}) - v(S) ]

where
    N = the set of entity types {ingredients, actions, cookware, utensils}
    s_i = the cosine similarity of entity type i for the two dishes
    w_i = the weight of entity type i (0.93, 0.04, 0.02, 0.01)
    v(S) = sum of w_i * s_i for i in S      (an entity type left out contributes 0)

With these weights v(N) is exactly the overall similarity score shown on the page, and
the values satisfy the Shapley "efficiency" property: sum(phi) = v(N) - v(empty) = score.

All 2^3 = 8 coalitions per player are enumerated, so this is the exact Shapley value,
not an approximation. (Because v is a weighted sum, the result happens to equal w_i * s_i;
the full computation is kept so the code follows the formula in the paper.)
"""

from itertools import combinations
from math import factorial


def compute_shapley(scores, weights):
    """
    scores:  {"ingredients": 0.86, "actions": 0.69, ...}   raw cosine similarity per entity type
    weights: {"ingredients": 0.93, "actions": 0.04, ...}
    returns: {"ingredients": phi, ...}  in the same units as the overall score (0-1 scale)
    """
    players = list(weights)
    n = len(players)

    def v(coalition):
        return sum(weights[p] * scores[p] for p in coalition)

    phi = {}
    for i in players:
        others = [p for p in players if p != i]
        total = 0.0
        for size in range(n):                       # |S| = 0 .. n-1
            coef = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(others, size):
                total += coef * (v(S + (i,)) - v(S))
        phi[i] = total
    return phi


def influence_shares(phi):
    """
    Each entity type's share of the overall similarity, in percent (sums to 100).
    Negative values (possible only if a cosine score is negative) are treated as 0.
    """
    positive = {k: max(v, 0.0) for k, v in phi.items()}
    total = sum(positive.values())
    if total <= 0:
        return {k: 0.0 for k in phi}
    shares = {k: round(100.0 * v / total, 1) for k, v in positive.items()}
    # Decimal rounding can produce 99.9 or 100.1. Adjust the largest share so the
    # displayed contributions always sum exactly to 100.0%.
    if shares:
        largest = max(shares, key=shares.get)
        shares[largest] = round(shares[largest] + (100.0 - sum(shares.values())), 1)
    return shares
