# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.scenarios.utils import social_metrics


def isoelastic_coin_minus_labor(
    coin_comps, total_labor, isoelastic_etas, labor_coefficient
):
    """Agent utility, concave increasing in coin and linearly decreasing in labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        isoelastic_eta (float): Constant describing the shape of the utility profile
            with respect to coin endowment. Must be between 0 and 1. 0 yields utility
            that increases linearly with coin. 1 yields utility that increases with
            log(coin). Utility from coin uses:
                https://en.wikipedia.org/wiki/Isoelastic_utility
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    coin_comps = np.array(coin_comps)
    isoelastic_etas = np.array(isoelastic_etas)
    assert np.all(coin_comps >= 0)
    assert np.all((isoelastic_etas >= 0)&(isoelastic_etas <= 1))
    # assert 0 <= isoelastic_eta <= 1.0

    # Utility from coin endowment
    eta_sum = 0
    util_c = 1
    for coin_comp, isoelastic_eta in zip(coin_comps, isoelastic_etas):
        if isoelastic_eta == 1.0:  # dangerous
            util_c *= np.log(np.max(1, coin_comp))
        else:  # isoelastic_eta >= 0
            util_c *= ((coin_comp ** (1 - isoelastic_eta) - 1) / (1 - isoelastic_eta))
        eta_sum += isoelastic_eta

    # disutility from labor
    util_l = (total_labor**(2-eta_sum)) * labor_coefficient
    

    # Net utility
    util = util_c - util_l

    return util


def coin_minus_labor_cost(
    coin_comps, total_labor, labor_exponent, labor_coefficient
):
    """Agent utility, linearly increasing in coin and decreasing as a power of labor.

    Args:
        coin_endowment (float, ndarray): The amount of coin owned by the agent(s).
        total_labor (float, ndarray): The amount of labor performed by the agent(s).
        labor_exponent (float): Constant describing the shape of the utility profile
            with respect to total labor. Must be between >1.
        labor_coefficient (float): Constant describing the disutility experienced per
            unit of labor performed. Disutility from labor equals:
                labor_coefficient * total_labor.

    Returns:
        Agent utility (float) or utilities (ndarray).
    """
    # https://en.wikipedia.org/wiki/Isoelastic_utility
    coin_comps = np.array(coin_comps)
    assert np.all(coin_comps >= 0)
    assert labor_exponent > 1

    # Utility from coin endowment
    util_c = np.sum(coin_comps)

    # Disutility from labor
    util_l = (total_labor ** labor_exponent) * labor_coefficient

    # Net utility
    util = util_c - util_l

    return util


def coin_eq_times_productivity(coin_endowments, equality_weight):
    """Social welfare, measured as productivity scaled by the degree of coin equality.

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        equality_weight (float): Constant that determines how productivity is scaled
            by coin equality. Must be between 0 (SW = prod) and 1 (SW = prod * eq).

    Returns:
        Product of coin equality and productivity (float).
    """
    n_agents = len(coin_endowments)
    prod = social_metrics.get_productivity(coin_endowments) / n_agents
    equality = equality_weight * social_metrics.get_equality(coin_endowments) + (
        1 - equality_weight
    )
    return equality * prod


def inv_income_weighted_coin_endowments(coin_endowments):
    """Social welfare, as weighted average endowment (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.

    Returns:
        Weighted average coin endowment (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(coin_endowments * pareto_weights)


def inv_income_weighted_utility(coin_endowments, utilities):
    """Social welfare, as weighted average utility (weighted by inverse endowment).

    Args:
        coin_endowments (ndarray): The array of coin endowments for each of the
            agents in the simulated economy.
        utilities (ndarray): The array of utilities for each of the agents in the
            simulated economy.

    Returns:
        Weighted average utility (float).
    """
    pareto_weights = 1 / np.maximum(coin_endowments, 1)
    pareto_weights = pareto_weights / np.sum(pareto_weights)
    return np.sum(utilities * pareto_weights)
