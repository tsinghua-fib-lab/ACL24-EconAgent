# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np


def annealed_tax_limit(completions, warmup_period, slope, final_max_tax_value=1.0):
    """
    Compute the maximum tax rate available at this stage of tax annealing.

    This function uses the number of episode completions and the annealing schedule
    (warmup_period, slope, & final_max_tax_value) to determine what the maximum tax
    rate can be.
    This type of annealing allows for a tax curriculum where earlier episodes are
    restricted to lower tax rates. As more episodes are played, higher tax values are
    allowed.

    Args:
        completions (int): Number of times the environment has completed an episode.
            Expected to be >= 0.
        warmup_period (int): Until warmup_period completions, only allow 0 tax. Using
            a negative value will enable non-0 taxes at 0 environment completions.
        slope (float): After warmup_period completions, percentage of full tax value
            unmasked with each new completion.
        final_max_tax_value (float): The maximum tax value at the end of annealing.

    Returns:
        A scalar value indicating the maximum tax at this stage of annealing.

    Example:
        >> WARMUP = 100
        >> SLOPE = 0.01
        >> annealed_tax_limit(0, WARMUP, SLOPE)
        0.0
        >> annealed_tax_limit(100, WARMUP, SLOPE)
        0.0
        >> annealed_tax_limit(150, WARMUP, SLOPE)
        0.5
        >> annealed_tax_limit(200, WARMUP, SLOPE)
        1.0
        >> annealed_tax_limit(1000, WARMUP, SLOPE)
        1.0
    """
    # What percentage of the full range is currently visible
    # (between 0 [only 0 tax] and 1 [all taxes visible])
    percentage_visible = np.maximum(
        0.0, np.minimum(1.0, slope * (completions - warmup_period))
    )

    # Determine the highest allowable tax,
    # given the current position in the annealing schedule
    current_max_tax = percentage_visible * final_max_tax_value

    return current_max_tax


def annealed_tax_mask(completions, warmup_period, slope, tax_values):
    """
    Generate a mask applied to a set of tax values for the purpose of tax annealing.

    This function uses the number of episode completions and the annealing schedule
    to determine which of the tax values are considered valid. The most extreme
    tax/subsidy values are unmasked last. Zero tax is always unmasked (i.e. always
    valid).
    This type of annealing allows for a tax curriculum where earlier episodes are
    restricted to lower tax rates. As more episodes are played, higher tax values are
    allowed.

    Args:
        completions (int): Number of times the environment has completed an episode.
            Expected to be >= 0.
        warmup_period (int): Until warmup_period completions, only allow 0 tax. Using
            a negative value will enable non-0 taxes at 0 environment completions.
        slope (float): After warmup_period completions, percentage of full tax value
            unmasked with each new completion.
        tax_values (list): The list of tax values associated with each action to
            which this mask will apply.

    Returns:
        A binary mask with same shape as tax_values, indicating which tax values are
            currently valid.

    Example:
        >> WARMUP = 100
        >> SLOPE = 0.01
        >> TAX_VALUES = [0.0, 0.25, 0.50, 0.75, 1.0]
        >> annealed_tax_limit(0, WARMUP, SLOPE, TAX_VALUES)
        [0, 0, 0, 0, 0]
        >> annealed_tax_limit(100, WARMUP, SLOPE, TAX_VALUES)
        [0, 0, 0, 0, 0]
        >> annealed_tax_limit(150, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 0, 0]
        >> annealed_tax_limit(200, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 1, 1]
        >> annealed_tax_limit(1000, WARMUP, SLOPE, TAX_VALUES)
        [1, 1, 1, 1, 1]
    """
    # Infer the most extreme tax level from the supplied tax values.
    abs_tax = np.abs(tax_values)
    full_tax_amount = np.max(abs_tax)

    # Determine the highest allowable tax, given the current position
    # in the annealing schedule
    max_absolute_visible_tax = annealed_tax_limit(
        completions, warmup_period, slope, full_tax_amount
    )

    # Return a binary mask to allow for taxes
    # at or below the highest absolute visible tax
    return np.less_equal(np.abs(tax_values), max_absolute_visible_tax).astype(
        np.float32
    )
