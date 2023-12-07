# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from copy import deepcopy

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)
from ai_economist.foundation.components.utils import (
    annealed_tax_limit,
    annealed_tax_mask,
)


@component_registry.add
class WealthRedistribution(BaseComponent):
    """Redistributes the total coin of the mobile agents as evenly as possible.

    Note:
        If this component is used, it should always be the last component in the order!
    """

    name = "WealthRedistribution"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    """
    Required methods for implementing components
    --------------------------------------------
    """

    def get_n_actions(self, agent_cls_name):
        """This component is passive: it does not add any actions."""
        return

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Redistributes inventory coins so that all agents have equal coin endowment.
        """
        world = self.world

        # Divide coins evenly
        ic = np.array([agent.state["inventory"]["Coin"] for agent in world.agents])
        ec = np.array([agent.state["escrow"]["Coin"] for agent in world.agents])
        tc = np.sum(ic + ec)
        target_share = tc / self.n_agents
        for agent in world.agents:
            agent.state["inventory"]["Coin"] = float(target_share - ec[agent.idx])

        ic = np.array([agent.state["inventory"]["Coin"] for agent in world.agents])
        ec = np.array([agent.state["escrow"]["Coin"] for agent in world.agents])
        tc_next = np.sum(ic + ec)
        assert np.abs(tc - tc_next) < 1

    def generate_observations(self):
        """This component does not add any observations."""
        obs = {}
        return obs

    def generate_masks(self, completions=0):
        """Passive component. Masks are empty."""
        masks = {}
        return masks


@component_registry.add
class PeriodicBracketTax(BaseComponent):
    """Periodically collect income taxes from agents and do lump-sum redistribution.

    Note:
        If this component is used, it should always be the last component in the order!

    Args:
        disable_taxes (bool): Whether to disable any tax collection, effectively
            enforcing that tax rates are always 0. Useful for removing taxes without
            changing the observation space. Default is False (taxes enabled).
        tax_model (str): Which tax model to use for setting taxes.
            "model_wrapper" (default) uses the actions of the planner agent;
            "saez" uses an adaptation of the theoretical optimal taxation formula
            derived in https://www.nber.org/papers/w7628.
            "us-federal-single-filer-2018-scaled" uses US federal tax rates from 2018;
            "fixed-bracket-rates" uses the rates supplied in fixed_bracket_rates.
        period (int): Length of a tax period in environment timesteps. Taxes are
            updated at the start of each period and collected/redistributed at the
            end of each period. Must be > 0. Default is 100 timesteps.
        rate_min (float): Minimum tax rate within a bracket. Must be >= 0 (default).
        rate_max (float): Maximum tax rate within a bracket. Must be <= 1 (default).
        rate_disc (float): (Only applies for "model_wrapper") the interval separating
            discrete tax rates that the planner can select. Default of 0.05 means,
            for example, the planner can select among [0.0, 0.05, 0.10, ... 1.0].
            Must be > 0 and < 1.
        n_brackets (int): How many tax brackets to use. Must be >=2. Default is 5.
        top_bracket_cutoff (float): The income at the left end of the last tax
            bracket. Must be >= 10. Default is 100 coin.
        usd_scaling (float): Scale by which to divide the US Federal bracket cutoffs
            when using bracket_spacing = "us-federal". Must be > 0. Default is 1000.
        bracket_spacing (str): How bracket cutoffs should be spaced.
            "us-federal" (default) uses scaled cutoffs from the 2018 US Federal
                taxes, with scaling set by usd_scaling (ignores n_brackets and
                top_bracket_cutoff);
            "linear" linearly spaces the n_bracket cutoffs between 0 and
                top_bracket_cutoff;
            "log" is similar to "linear" but with logarithmic spacing.
        fixed_bracket_rates (list): Required if tax_model=="fixed-bracket-rates". A
            list of fixed marginal rates to use for each bracket. Length must be
            equal to the number of brackets (7 for "us-federal" spacing, n_brackets
            otherwise).
        pareto_weight_type (str): Type of Pareto weights to use when computing tax
            rates using the Saez formula. "inverse_income" (default) uses 1/z;
            "uniform" uses 1.
        saez_fixed_elas (float, optional): If supplied, this value will be used as
            the elasticity estimate when computing tax rates using the Saez formula.
            If not given (default), elasticity will be estimated empirically.
        tax_annealing_schedule (list, optional): A length-2 list of
            [tax_annealing_warmup, tax_annealing_slope] describing the tax annealing
            schedule. See annealed_tax_mask function for details. Default behavior is
            no tax annealing.
    """

    name = "PeriodicBracketTax"
    component_type = "PeriodicTax"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent", "BasicPlanner"]

    def __init__(
        self,
        *base_component_args,
        disable_taxes=False,
        tax_model="model_wrapper",
        period=100,
        rate_min=0.0,
        rate_max=1.0,
        rate_disc=0.05,
        n_brackets=5,
        top_bracket_cutoff=100,
        usd_scaling=1000.0,
        bracket_spacing="us-federal",
        fixed_bracket_rates=None,
        pareto_weight_type="inverse_income",
        saez_fixed_elas=None,
        tax_annealing_schedule=None,
        scale_obs=True,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Whether to turn off taxes. Disabling taxes will prevent any taxes from
        # being collected but the observation space will be the same as if taxes were
        # enabled, which can be useful for controlled tax/no-tax comparisons.
        self.disable_taxes = bool(disable_taxes)

        # How to set taxes.
        self.tax_model = tax_model
        assert self.tax_model in [
            "model_wrapper",
            "us-federal-single-filer-2018-scaled",
            "saez",
            "fixed-bracket-rates",
        ]

        # How many timesteps a tax period lasts.
        self.period = int(period)
        assert self.period > 0

        # Minimum marginal bracket rate
        self.rate_min = 0.0 if self.disable_taxes else float(rate_min)
        # Maximum marginal bracket rate
        self.rate_max = 0.0 if self.disable_taxes else float(rate_max)
        assert 0 <= self.rate_min <= self.rate_max <= 1.0

        # Interval for discretizing tax rate options
        # (only applies if tax_model == "model_wrapper").
        self.rate_disc = float(rate_disc)

        self.use_discretized_rates = self.tax_model == "model_wrapper"

        if self.use_discretized_rates:
            self.disc_rates = np.arange(
                self.rate_min, self.rate_max + self.rate_disc, self.rate_disc
            )
            self.disc_rates = self.disc_rates[self.disc_rates <= self.rate_max]
            assert len(self.disc_rates) > 1 or self.disable_taxes
            self.n_disc_rates = len(self.disc_rates)
        else:
            self.disc_rates = None
            self.n_disc_rates = 0

        # === income bracket definitions ===
        self.n_brackets = int(n_brackets)
        assert self.n_brackets >= 2

        self.top_bracket_cutoff = float(top_bracket_cutoff)
        assert self.top_bracket_cutoff >= 10

        self.usd_scale = float(usd_scaling)
        assert self.usd_scale > 0

        self.bracket_spacing = bracket_spacing.lower()
        assert self.bracket_spacing in ["linear", "log", "us-federal"]

        if self.bracket_spacing == "linear":
            self.bracket_cutoffs = np.linspace(
                0, self.top_bracket_cutoff, self.n_brackets
            )

        elif self.bracket_spacing == "log":
            b0_max = self.top_bracket_cutoff / (2 ** (self.n_brackets - 2))
            self.bracket_cutoffs = np.concatenate(
                [
                    [0],
                    2
                    ** np.linspace(
                        np.log2(b0_max),
                        np.log2(self.top_bracket_cutoff),
                        n_brackets - 1,
                    ),
                ]
            )
        elif self.bracket_spacing == "us-federal":
            self.bracket_cutoffs = (
                np.array([0, 9700, 39475, 84200, 160725, 204100, 510300])
                / self.usd_scale
            )
            self.n_brackets = len(self.bracket_cutoffs)
            self.top_bracket_cutoff = float(self.bracket_cutoffs[-1])
        else:
            raise NotImplementedError

        self.bracket_edges = np.concatenate([self.bracket_cutoffs, [np.inf]])
        self.bracket_sizes = self.bracket_edges[1:] - self.bracket_edges[:-1]

        assert self.bracket_cutoffs[0] == 0

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            assert self.bracket_spacing == "us-federal"

        if self.tax_model == "fixed-bracket-rates":
            assert isinstance(fixed_bracket_rates, (tuple, list))
            assert np.min(fixed_bracket_rates) >= 0
            assert np.max(fixed_bracket_rates) <= 1
            assert len(fixed_bracket_rates) == self.n_brackets
            self._fixed_bracket_rates = np.array(fixed_bracket_rates)
        else:
            self._fixed_bracket_rates = None

        # === bracket tax rates ===
        self.curr_bracket_tax_rates = np.zeros_like(self.bracket_cutoffs)
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        # === Pareto weights, elasticity ===
        self.pareto_weight_type = pareto_weight_type
        self.elas_tm1 = 0.5
        self.elas_t = 0.5
        self.log_z0_tm1 = 0
        self.log_z0_t = 0

        self._saez_fixed_elas = saez_fixed_elas
        if self._saez_fixed_elas is not None:
            self._saez_fixed_elas = float(self._saez_fixed_elas)
            assert self._saez_fixed_elas >= 0

        # Size of the local buffer. In a distributed context, the global buffer size
        # will be capped at n_replicas * _buffer_size.
        # NOTE: Saez will use random taxes until it has self._buffer_size samples.
        self._buffer_size = 500
        self._reached_min_samples = False
        self._additions_this_episode = 0
        # Local buffer maintained by this replica.
        self._local_saez_buffer = []
        # "Global" buffer obtained by combining local buffers of individual replicas.
        self._global_saez_buffer = []

        self._saez_n_estimation_bins = 100
        self._saez_top_rate_cutoff = self.bracket_cutoffs[-1]
        self._saez_income_bin_edges = np.linspace(
            0, self._saez_top_rate_cutoff, self._saez_n_estimation_bins + 1
        )
        self._saez_income_bin_sizes = np.concatenate(
            [
                self._saez_income_bin_edges[1:] - self._saez_income_bin_edges[:-1],
                [np.inf],
            ]
        )
        self.running_avg_tax_rates = np.zeros_like(self.curr_bracket_tax_rates)

        # === tax cycle definitions ===
        self.tax_cycle_pos = 1
        self.last_coin = [0 for _ in range(self.n_agents)]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        # === trackers ===
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [0] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self.taxes = []

        # === tax annealing ===
        # for annealing of non-planner max taxes.
        self._annealed_rate_max = float(self.rate_max)
        self._last_completions = 0

        # for annealing of planner actions.
        self.tax_annealing_schedule = tax_annealing_schedule
        if tax_annealing_schedule is not None:
            assert isinstance(self.tax_annealing_schedule, (tuple, list))
            self._annealing_warmup = self.tax_annealing_schedule[0]
            self._annealing_slope = self.tax_annealing_schedule[1]
            self._annealed_rate_max = annealed_tax_limit(
                self._last_completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )
        else:
            self._annealing_warmup = None
            self._annealing_slope = None

        if self.tax_model == "model_wrapper" and not self.disable_taxes:
            planner_action_tuples = self.get_n_actions("BasicPlanner")
            self._planner_tax_val_dict = {
                k: self.disc_rates for k, v in planner_action_tuples
            }
        else:
            self._planner_tax_val_dict = {}
        self._planner_masks = None

        # === placeholders ===
        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]
        
        self.scale_obs = scale_obs

    # Methods for getting/setting marginal tax rates
    # ----------------------------------------------

    # ------- US Federal taxes
    @property
    def us_federal_single_filer_2018_scaled(self):
        """
        https://turbotax.intuit.com/tax-tips/irs-tax-return/current-federal-tax-rate-schedules/L7Bjs1EAD
        If taxable income is over—
        but not over—
        the tax is:
        $0
        $9,700
        10% of the amount over $0
        $9,700
        $39,475
        $970 plus 12% of the amount over $9,700
        $39,475
        $84,200
        $4,543 plus 22% of the amount over $39,475
        $84,200
        $160,725
        $14,382 plus 24% of the amount over $84,200
        $160,725
        $204,100
        $32,748 plus 32% of the amount over $160,725
        $204,100
        $510,300
        $46,628 plus 35% of the amount over $204,100
        $510,300
        no limit
        $153,798 plus 37% of the amount over $510,300
        """
        return [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]

    @us_federal_single_filer_2018_scaled.setter
    def us_federal_single_filer_2018_scaled(self, bracket_rates):
        return bracket_rates

    # ------- fixed-bracket-rates
    @property
    def fixed_bracket_rates(self):
        """Return whatever fixed bracket rates were set during initialization."""
        return self._fixed_bracket_rates

    @property
    def curr_rate_max(self):
        """Maximum allowable tax rate, given current progress of any tax annealing."""
        if self.tax_annealing_schedule is None:
            return self.rate_max
        return self._annealed_rate_max

    @property
    def curr_marginal_rates(self):
        """The current set of marginal tax bracket rates."""
        if self.use_discretized_rates:
            return self.disc_rates[self.curr_rate_indices]

        if self.tax_model == "us-federal-single-filer-2018-scaled":
            marginal_tax_bracket_rates = np.minimum(
                np.array(self.us_federal_single_filer_2018_scaled), self.curr_rate_max
            )
        elif self.tax_model == "saez":
            marginal_tax_bracket_rates = np.minimum(
                self.curr_bracket_tax_rates, self.curr_rate_max
            )
        elif self.tax_model == "fixed-bracket-rates":
            marginal_tax_bracket_rates = np.minimum(
                np.array(self.fixed_bracket_rates), self.curr_rate_max
            )
        else:
            raise NotImplementedError

        return marginal_tax_bracket_rates

    def set_new_period_rates_model(self):
        """Update taxes using actions from the tax model."""
        if self.disable_taxes:
            return

        # AI version
        for i, bracket in enumerate(self.bracket_cutoffs):
            planner_action = self.world.planner.get_component_action(
                self.name, "TaxIndexBracket_{:03d}".format(int(bracket))
            )
            if planner_action == 0:
                pass
            elif planner_action <= self.n_disc_rates:
                self.curr_rate_indices[i] = int(planner_action - 1)
            else:
                raise ValueError

    # ------- Saez formula
    def compute_and_set_new_period_rates_from_saez_formula(
        self, update_elas_tm1=True, update_log_z0_tm1=True
    ):
        """Estimates/sets optimal rates using adaptation of Saez formula

        See: https://www.nber.org/papers/w7628
        """
        # Until we reach the min sample number, keep checking if we have reached it.
        if not self._reached_min_samples:
            # Note: self.saez_buffer includes the global buffer (if applicable).
            if len(self.saez_buffer) >= self._buffer_size:
                self._reached_min_samples = True

        # If no enough samples, use random taxes.
        if not self._reached_min_samples:
            self.curr_bracket_tax_rates = np.random.uniform(
                low=self.rate_min,
                high=self.curr_rate_max,
                size=self.curr_bracket_tax_rates.shape,
            )
            return

        incomes_and_marginal_rates = np.array(self.saez_buffer)

        # Elasticity assumed constant for all incomes.
        # (Run this for the sake of tracking the estimate; will not actually use the
        # estimate if using fixed elasticity).
        if update_elas_tm1:
            self.elas_tm1 = float(self.elas_t)
        if update_log_z0_tm1:
            self.log_z0_tm1 = float(self.log_z0_t)

        elas_t, log_z0_t = self.estimate_uniform_income_elasticity(
            incomes_and_marginal_rates,
            elas_df=0.98,
            elas_tm1=self.elas_tm1,
            log_z0_tm1=self.log_z0_tm1,
            verbose=False,
        )

        if update_elas_tm1:
            self.elas_t = float(elas_t)
        if update_log_z0_tm1:
            self.log_z0_t = float(log_z0_t)

        # If a fixed estimate has been specified, use it in the formulas below.
        if self._saez_fixed_elas is not None:
            elas_t = float(self._saez_fixed_elas)

        # Get Saez parameters at each income bin
        # to compute a marginal tax rate schedule.
        binned_gzs, binned_azs = self.get_binned_saez_welfare_weight_and_pareto_params(
            population_incomes=incomes_and_marginal_rates[:, 0]
        )

        # Use the elasticity to compute this binned schedule using the Saez formula.
        binned_marginal_tax_rates = self.get_saez_marginal_rates(
            binned_gzs, binned_azs, elas_t
        )

        # Adapt the saez tax schedule to the tax brackets.
        self.curr_bracket_tax_rates = np.clip(
            self.bracketize_schedule(
                bin_marginal_rates=binned_marginal_tax_rates,
                bin_edges=self._saez_income_bin_edges,
                bin_sizes=self._saez_income_bin_sizes,
            ),
            self.rate_min,
            self.curr_rate_max,
        )

        self.running_avg_tax_rates = (self.running_avg_tax_rates * 0.99) + (
            self.curr_bracket_tax_rates * 0.01
        )

    # Implementation of the Saez formula in this periodic, bracketed setting
    # ----------------------------------------------------------------------
    @property
    def saez_buffer(self):
        if not self._global_saez_buffer:
            saez_buffer = self._local_saez_buffer
        elif self._additions_this_episode == 0:
            saez_buffer = self._global_saez_buffer
        else:
            saez_buffer = (
                self._global_saez_buffer
                + self._local_saez_buffer[-self._additions_this_episode :]
            )
        return saez_buffer

    def get_local_saez_buffer(self):
        return self._local_saez_buffer

    def set_global_saez_buffer(self, global_saez_buffer):
        assert isinstance(global_saez_buffer, list)
        assert len(global_saez_buffer) >= len(self._local_saez_buffer)
        self._global_saez_buffer = global_saez_buffer

    def _update_saez_buffer(self, tax_info_t):
        # Update the buffer.
        for a_idx in range(self.n_agents):
            z_t = tax_info_t[str(a_idx)]["income"]
            tau_t = tax_info_t[str(a_idx)]["marginal_rate"]
            self._local_saez_buffer.append([z_t, tau_t])
            self._additions_this_episode += 1

        while len(self._local_saez_buffer) > self._buffer_size:
            _ = self._local_saez_buffer.pop(0)

    def reset_saez_buffers(self):
        self._local_saez_buffer = []
        self._global_saez_buffer = []
        self._additions_this_episode = 0
        self._reached_min_samples = False

    def estimate_uniform_income_elasticity(
        self,
        observed_incomes_and_marginal_rates,
        elas_df=0.98,
        elas_tm1=0.5,
        log_z0_tm1=0.5,
        verbose=False,
    ):
        """Estimate elasticity using Ordinary Least Squares regression.
        OLS: https://en.wikipedia.org/wiki/Ordinary_least_squares
        Estimating elasticity: https://www.nber.org/papers/w7512
        """
        zs = []
        taus = []

        for z_t, tau_t in observed_incomes_and_marginal_rates:
            # If z_t is <=0 or tau_t is >=1, the operations below will give us nans
            if z_t > 0 and tau_t < 1:
                zs.append(z_t)
                taus.append(tau_t)

        if len(zs) < 10:
            return float(elas_tm1), float(log_z0_tm1)
        if np.std(taus) < 1e-6:
            return float(elas_tm1), float(log_z0_tm1)

        # Regressing log income against log 1-marginal_rate.
        x = np.log(np.maximum(1 - np.array(taus), 1e-9))
        # (bias term)
        b = np.ones_like(x)
        # Perform OLS.
        X = np.stack([x, b]).T  # Stack linear & bias terms
        Y = np.log(np.maximum(np.array(zs), 1e-9))  # Regression targets
        XXi = np.linalg.inv(X.T.dot(X))
        XY = X.T.dot(Y)
        elas, log_z0 = XXi.T.dot(XY)

        warn_less_than_0 = elas < 0
        instant_elas_t = np.maximum(elas, 0.0)

        elas_t = ((1 - elas_df) * instant_elas_t) + (elas_df * elas_tm1)

        if verbose:
            if warn_less_than_0:
                print("\nWARNING: Recent elasticity estimate is < 0.")
                print("Running elasticity estimate: {:.2f}\n".format(elas_t))
            else:
                print("\nRunning elasticity estimate: {:.2f}\n".format(elas_t))

        return elas_t, log_z0

    def get_binned_saez_welfare_weight_and_pareto_params(self, population_incomes):
        def clip(x, lo=None, hi=None):
            if lo is not None:
                x = max(lo, x)
            if hi is not None:
                x = min(x, hi)
            return x

        def bin_z(left, right):
            return 0.5 * (left + right)

        def get_cumul(counts, incomes_below, incomes_above):
            n_below = len(incomes_below)
            n_above = len(incomes_above)
            n_total = np.sum(counts) + n_below + n_above

            def p(i, counts):
                return counts[i] / n_total

            # Probability that an income is below the taxable threshold.
            p_below = n_below / n_total

            # pz = p(z' = z): probability that [binned] income z' occurs in bin z.
            pz = [p(i, counts) for i in range(len(counts))] + [n_above / n_total]

            # Pz = p(z' <= z): Probability z' is less-than or equal to z.
            cum_pz = [pz[0] + p_below]
            for p in pz[1:]:
                cum_pz.append(clip(cum_pz[-1] + p, 0, 1.0))

            return np.array(pz), np.array(cum_pz)

        def compute_binned_g_distribution(counts, lefts, incomes):
            def pareto(z):
                if self.pareto_weight_type == "uniform":
                    pareto_weights = np.ones_like(z)
                elif self.pareto_weight_type == "inverse_income":
                    pareto_weights = 1.0 / np.maximum(1, z)
                else:
                    raise NotImplementedError
                return pareto_weights

            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # The total (unnormalized) Pareto weight of untaxable incomes.
            if len(incomes_below) > 0:
                pareto_weight_below = np.sum(pareto(np.maximum(incomes_below, 0)))
            else:
                pareto_weight_below = 0

            # The total (unnormalized) Pareto weight within each bin.
            if len(incomes_above) > 0:
                pareto_weight_above = np.sum(pareto(incomes_above))
            else:
                pareto_weight_above = 0

            # The total (unnormalized) Pareto weight within each bin.
            pareto_weight_per_bin = counts * pareto(bin_z(lefts[:-1], lefts[1:]))

            # The aggregate (unnormalized) Pareto weight of all incomes.
            cumulative_pareto_weights = pareto_weight_per_bin.sum()
            cumulative_pareto_weights += pareto_weight_below
            cumulative_pareto_weights += pareto_weight_above

            # Normalize so that the Pareto density sums to 1.
            pareto_norm = cumulative_pareto_weights + 1e-9
            unnormalized_pareto_density = np.concatenate(
                [pareto_weight_per_bin, [pareto_weight_above]]
            )
            normalized_pareto_density = unnormalized_pareto_density / pareto_norm

            # Aggregate Pareto weight of earners with income greater-than or equal to z.
            cumulative_pareto_density_geq_z = np.cumsum(
                normalized_pareto_density[::-1]
            )[::-1]

            # Probability that [binned] income z' is greather-than or equal to z.
            pz, _ = get_cumul(counts, incomes_below, incomes_above)
            cumulative_prob_geq_z = np.cumsum(pz[::-1])[::-1]

            # Average (normalized) Pareto weight of earners with income >= z.
            geq_z_norm = cumulative_prob_geq_z + 1e-9
            avg_pareto_weight_geq_z = cumulative_pareto_density_geq_z / geq_z_norm

            def interpolate_gzs(gz):
                # Assume incomes within a bin are evenly distributed within that bin
                # and re-compute accordingly.
                gz_at_left_edge = gz[:-1]
                gz_at_right_edge = gz[1:]

                avg_bin_gz = 0.5 * (gz_at_left_edge + gz_at_right_edge)
                # Re-attach the gz of the top tax rate (does not need to be
                # interpolated).
                gzs = np.concatenate([avg_bin_gz, [gz[-1]]])
                return gzs

            return interpolate_gzs(avg_pareto_weight_geq_z)

        def compute_binned_a_distribution(counts, lefts, incomes):
            incomes_below = incomes[incomes < lefts[0]]
            incomes_above = incomes[incomes > lefts[-1]]

            # z is defined as the MIDDLE point in a bin.
            # So for a bin [left, right] -> z = (left + right) / 2.
            Az = []

            # cum_pz = p(z' <= z): Probability z' is less-than or equal to z
            pz, cum_pz = get_cumul(counts, incomes_below, incomes_above)

            # Probability z' is greater-than or equal to z
            # Note: The "0.5" coefficient gives results more consistent with theory; it
            # accounts for the assumption that incomes within a particular bin are
            # uniformly spread between the left & right edges of that bin.
            p_geq_z = 1 - cum_pz + (0.5 * pz)

            T = len(lefts[:-1])

            for i in range(T):
                if pz[i] == 0:
                    Az.append(np.nan)
                else:
                    z = bin_z(lefts[i], lefts[i + 1])
                    # paz = z * pz[i] / (clip(1 - Pz[i], 0, 1) + 1e-9)
                    paz = z * pz[i] / (clip(p_geq_z[i], 0, 1) + 1e-9)  # defn of A(z)
                    paz = paz / (lefts[i + 1] - lefts[i])  # norm by bin width
                    Az.append(paz)

            # Az for the incomes past the top cutoff,
            # the bin is [left, infinity]: there is no "middle".
            # Hence, use the mean value in the last bin.
            if len(incomes_above) > 0:
                cutoff = lefts[-1]
                avg_income_above_cutoff = np.mean(incomes_above)
                # use a special formula to compute A(z)
                Az_above = avg_income_above_cutoff / (
                    avg_income_above_cutoff - cutoff + 1e-9
                )
            else:
                Az_above = 0.0

            return np.concatenate([Az, [Az_above]])

        counts, lefts = np.histogram(
            population_incomes, bins=self._saez_income_bin_edges
        )
        population_gz = compute_binned_g_distribution(counts, lefts, population_incomes)
        population_az = compute_binned_a_distribution(counts, lefts, population_incomes)

        # Return the binned stats used to create a schedule of marginal rates.
        return population_gz, population_az

    @staticmethod
    def get_saez_marginal_rates(binned_gz, binned_az, elas, interpolate=True):
        # Marginal rates within each income bin (last tau is the top tax rate).
        taus = (1.0 - binned_gz) / (1.0 - binned_gz + binned_az * elas + 1e-9)

        if interpolate:
            # In bins where there were no incomes found, tau is nan.
            # Interpolate to fill the gaps.
            last_real_rate = 0.0
            last_real_tidx = -1
            for i, tau in enumerate(taus):
                # The current tax rate is a real number.
                if not np.isnan(tau):
                    # This is the end of a gap. Interpolate.
                    if (i - last_real_tidx) > 1:
                        assert (
                            i != 0
                        )  # This should never trigger for the first tax bin.
                        gap_indices = list(range(last_real_tidx + 1, i))
                        intermediate_rates = np.linspace(
                            last_real_rate, tau, len(gap_indices) + 2
                        )[1:-1]
                        assert len(gap_indices) == len(intermediate_rates)
                        for gap_index, intermediate_rate in zip(
                            gap_indices, intermediate_rates
                        ):
                            taus[gap_index] = intermediate_rate
                    # Update the tracker.
                    last_real_rate = float(tau)
                    last_real_tidx = int(i)

                # The current tax rate is a nan. Continue without updating
                # the tracker (indicating the presence of a gap).
                else:
                    pass

        return taus

    def bracketize_schedule(self, bin_marginal_rates, bin_edges, bin_sizes):
        # Compute the amount of tax each bracket would collect
        # if income was >= the right edge.
        # Divide by the bracket size to get
        # the average marginal rate within that bracket.
        last_bracket_total = 0
        bracket_avg_marginal_rates = []
        for b_idx, income in enumerate(self.bracket_cutoffs[1:]):
            # How much income occurs within each bin
            # (including the open-ended, top "bin").
            past_cutoff = np.maximum(0, income - bin_edges)
            bin_income = np.minimum(bin_sizes, past_cutoff)

            # To get the total taxes due,
            # multiply the income within each bin by that bin's marginal rate.
            bin_taxes = bin_marginal_rates * bin_income
            taxes_due = np.maximum(0, np.sum(bin_taxes))

            bracket_tax_burden = taxes_due - last_bracket_total
            bracket_size = self.bracket_sizes[b_idx]

            bracket_avg_marginal_rates.append(bracket_tax_burden / bracket_size)
            last_bracket_total = taxes_due

        # The top bracket tax rate is computed directly already.
        bracket_avg_marginal_rates.append(bin_marginal_rates[-1])

        bracket_rates = np.array(bracket_avg_marginal_rates)
        assert len(bracket_rates) == self.n_brackets

        return bracket_rates

    # Methods for collecting and redistributing taxes
    # -----------------------------------------------

    def income_bin(self, income):
        """Return index of tax bin in which income falls."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.bracket_cutoffs[np.argmax(bracket_bool)]

    def marginal_rate(self, income):
        """Return the marginal tax rate applied at this income level."""
        if income < 0:
            return 0.0
        meets_min = income >= self.bracket_edges[:-1]
        under_max = income < self.bracket_edges[1:]
        bracket_bool = meets_min * under_max
        return self.curr_marginal_rates[np.argmax(bracket_bool)]

    def taxes_due(self, income):
        """Return the total amount of taxes due at this income level."""
        past_cutoff = np.maximum(0, income - self.bracket_cutoffs)
        bin_income = np.minimum(self.bracket_sizes, past_cutoff)
        bin_taxes = self.curr_marginal_rates * bin_income
        return np.sum(bin_taxes)

    def enact_taxes(self):
        """Calculate period income & tax burden. Collect taxes and redistribute."""
        net_tax_revenue = 0
        tax_dict = dict(
            schedule=np.array(self.curr_marginal_rates),
            cutoffs=np.array(self.bracket_cutoffs),
        )

        for curr_rate, bracket_cutoff in zip(
            self.curr_marginal_rates, self.bracket_cutoffs
        ):
            self._schedules["{:03d}".format(int(bracket_cutoff))].append(
                float(curr_rate)
            )

        self.last_income = []
        self.last_effective_tax_rate = []
        self.last_marginal_rate = []
        for agent, last_coin in zip(self.world.agents, self.last_coin):
            income = agent.state["production"] - last_coin
            tax_due = self.taxes_due(income)
            effective_taxes = np.minimum(
                agent.state["inventory"]["Coin"], tax_due
            )  # Don't take from escrow.
            marginal_rate = self.marginal_rate(income)
            effective_tax_rate = float(effective_taxes / np.maximum(0.000001, income))
            tax_dict[str(agent.idx)] = dict(
                income=float(income),
                tax_paid=float(effective_taxes),
                marginal_rate=marginal_rate,
                effective_rate=effective_tax_rate,
            )

            # Actually collect the taxes.
            agent.state["inventory"]["Coin"] -= effective_taxes
            net_tax_revenue += effective_taxes

            self.last_income.append(float(income))
            self.last_marginal_rate.append(float(marginal_rate))
            self.last_effective_tax_rate.append(effective_tax_rate)

            self.all_effective_tax_rates.append(effective_tax_rate)
            self._occupancy["{:03d}".format(int(self.income_bin(income)))] += 1

        self.total_collected_taxes += float(net_tax_revenue)

        lump_sum = net_tax_revenue / self.n_agents
        for agent in self.world.agents:
            agent.state["inventory"]["Coin"] += lump_sum
            tax_dict[str(agent.idx)]["lump_sum"] = float(lump_sum)
            self.last_coin[agent.idx] = float(agent.state["production"])

        self.taxes.append(tax_dict)

        # Pre-compute some things that will be useful for generating observations.
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        # Fold this period's tax data into the saez buffer.
        if self.tax_model == "saez":
            self._update_saez_buffer(tax_dict)

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        If using the "model_wrapper" tax model and taxes are enabled, the planner's
        action space includes an action subspace for each of the tax brackets. Each
        such action space has as many actions as there are discretized tax rates.
        """
        # Only the planner takes actions through this component.
        if agent_cls_name == "BasicPlanner":
            if self.tax_model == "model_wrapper" and not self.disable_taxes:
                # For every bracket, the planner can select one of the discretized
                # tax rates.
                return [
                    ("TaxIndexBracket_{:03d}".format(int(r)), self.n_disc_rates)
                    for r in self.bracket_cutoffs
                ]

        # Return 0 (no added actions) if the other conditions aren't met.
        return 0

    def get_additional_state_fields(self, agent_cls_name):
        """This component does not add any agent state fields."""
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        On the first day of each tax period, update taxes. On the last day, enact them.
        """

        # 1. On the first day of a new tax period: Set up the taxes for this period.
        if self.tax_cycle_pos == 1:
            if self.tax_model == "model_wrapper":
                self.set_new_period_rates_model()

            if self.tax_model == "saez":
                self.compute_and_set_new_period_rates_from_saez_formula()

            # (cache this for faster obs generation)
            self._curr_rates_obs = np.array(self.curr_marginal_rates)

        # 2. On the last day of the tax period: Get $-taxes AND update agent endowments.
        if self.tax_cycle_pos >= self.period:
            self.enact_taxes()
            self.tax_cycle_pos = 0

        else:
            self.taxes.append([])

        # increment timestep.
        self.tax_cycle_pos += 1

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Agents observe where in the tax period cycle they are, information about the
        last period's incomes, and the current marginal tax rates, including the
        marginal rate that will apply to their next unit of income.

        The planner observes the same type of information, but for all of the agents. It
        also sees, for each agent, their marginal tax rate and reported income from
        the previous tax period.
        """
        is_tax_day = float(self.tax_cycle_pos >= self.period)
        is_first_day = float(self.tax_cycle_pos == 1)
        tax_phase = self.tax_cycle_pos / self.period
        consumption_obs = np.sort([agent.consumption['Coin'] for agent in self.world.agents]).astype(np.float64)/(self.world.timestep+1)
        investment_obs = np.sort([agent.investment['Coin'] for agent in self.world.agents]).astype(np.float64)
        saving_obs = np.sort([agent.saving['Coin'] for agent in self.world.agents]).astype(np.float64)
        _last_income_obs_sorted = self._last_income_obs_sorted.copy().astype(np.float64)
        _last_income_obs = self._last_income_obs.copy().astype(np.float64)
        try:
            curr_tax = self.taxes[self.world.timestep-1]
        except:
            curr_tax = {}
        all_tax_paid = np.array([curr_tax.get(str(agent.idx), {}).get('tax_paid', 0) for agent in self.world.agents]).astype(np.float64)
        all_lump_sum = np.array([curr_tax.get(str(agent.idx), {}).get('lump_sum', 0) for agent in self.world.agents]).astype(np.float64)
        if self.scale_obs:
            _last_income_obs_sorted /= (_last_income_obs_sorted.max()+1e-8)
            consumption_obs /= (consumption_obs.max()+1e-8)
            investment_obs /= (investment_obs.max()+1e-8)
            saving_obs /= (saving_obs.max()+1e-8)
            _last_income_obs /= (_last_income_obs.max()+1e-8)
            all_tax_paid /= (all_tax_paid.max()+1e-8)
            all_lump_sum /= (all_lump_sum.max()+1e-8)
            

        obs = dict()
        obs[self.world.planner.idx] = dict(
            is_tax_day=is_tax_day,
            is_first_day=is_first_day,
            tax_phase=tax_phase,
            last_incomes=_last_income_obs_sorted,
            last_consumption=consumption_obs,
            last_investment=investment_obs,
            last_saving=saving_obs,
            curr_rates=self._curr_rates_obs,
        )

        for agent in self.world.agents:
            i = agent.idx
            k = str(i)

            curr_marginal_rate = self.marginal_rate(
                agent.state["production"] - self.last_coin[i]
            )

            obs[k] = dict(
                is_tax_day=is_tax_day,
                is_first_day=is_first_day,
                tax_phase=tax_phase,
                last_incomes=_last_income_obs_sorted,
                curr_rates=self._curr_rates_obs,
                marginal_rate=curr_marginal_rate,
            )

            obs["p" + k] = dict(
                last_income=_last_income_obs[i],
                tax_paid=all_tax_paid[i],
                lump_sum=all_lump_sum[i],
                last_marginal_rate=self.last_marginal_rate[i],
                curr_marginal_rate=curr_marginal_rate,
            )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Masks only apply to the planner and if tax_model == "model_wrapper" and taxes
        are enabled.
        All tax actions are masked (so, only NO-OPs can be sampled) on all timesteps
        except when self.tax_cycle_pos==1 (meaning a new tax period is starting).
        When self.tax_cycle_pos==1, tax actions are masked in order to enforce any
        tax annealing.
        """
        if (
            completions != self._last_completions
            and self.tax_annealing_schedule is not None
        ):
            self._last_completions = int(completions)
            self._annealed_rate_max = annealed_tax_limit(
                completions,
                self._annealing_warmup,
                self._annealing_slope,
                self.rate_max,
            )

        if self.disable_taxes:
            return {}

        if self.tax_model == "model_wrapper":
            # No annealing. Generate masks using default method.
            if self.tax_annealing_schedule is None:
                if self._planner_masks is None:
                    masks = super().generate_masks(completions=completions)
                    self._planner_masks = dict(
                        new_taxes=deepcopy(masks[self.world.planner.idx]),
                        zeros={
                            k: np.zeros_like(v)
                            for k, v in masks[self.world.planner.idx].items()
                        },
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

            # Doing annealing.
            else:
                # Figure out what the masks should be this episode.
                if self._planner_masks is None:
                    planner_masks = {
                        k: annealed_tax_mask(
                            completions,
                            self._annealing_warmup,
                            self._annealing_slope,
                            tax_values,
                        )
                        for k, tax_values in self._planner_tax_val_dict.items()
                    }
                    self._planner_masks = dict(
                        new_taxes=deepcopy(planner_masks),
                        zeros={k: np.zeros_like(v) for k, v in planner_masks.items()},
                    )

                # No need to recompute. Use the cached masks.
                masks = dict()
                if self.tax_cycle_pos != 1 or self.disable_taxes:
                    # Apply zero masks for any timestep where taxes
                    # are not going to be updated.
                    masks[self.world.planner.idx] = self._planner_masks["zeros"]
                else:
                    masks[self.world.planner.idx] = self._planner_masks["new_taxes"]

        # We are not using a learned planner. Generate masks by the default method.
        else:
            masks = super().generate_masks(completions=completions)

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset trackers.
        """
        self.curr_rate_indices = [0 for _ in range(self.n_brackets)]

        self.tax_cycle_pos = 1
        self.last_coin = [
            float(agent.state["production"]) for agent in self.world.agents
        ]
        self.last_income = [0 for _ in range(self.n_agents)]
        self.last_marginal_rate = [0 for _ in range(self.n_agents)]
        self.last_effective_tax_rate = [0 for _ in range(self.n_agents)]

        self._curr_rates_obs = np.array(self.curr_marginal_rates)
        self._last_income_obs = np.array(self.last_income) / self.period
        self._last_income_obs_sorted = self._last_income_obs[
            np.argsort(self._last_income_obs)
        ]

        self.taxes = []
        self.total_collected_taxes = 0
        self.all_effective_tax_rates = []
        self._schedules = {"{:03d}".format(int(r)): [] for r in self.bracket_cutoffs}
        self._occupancy = {"{:03d}".format(int(r)): 0 for r in self.bracket_cutoffs}
        self._planner_masks = None

        if self.tax_model == "saez":
            self.curr_bracket_tax_rates = np.array(self.running_avg_tax_rates)

    def get_metrics(self):
        """
        See base_component.py for detailed description.

        Return metrics related to bracket rates, bracket occupancy, and tax collection.
        """
        out = dict()

        n_observed_incomes = np.maximum(1, np.sum(list(self._occupancy.values())))
        for c in self.bracket_cutoffs:
            k = "{:03d}".format(int(c))
            out["avg_bracket_rate/{}".format(k)] = np.mean(self._schedules[k])
            out["bracket_occupancy/{}".format(k)] = (
                self._occupancy[k] / n_observed_incomes
            )

        if not self.disable_taxes:
            out["avg_effective_tax_rate"] = np.mean(self.all_effective_tax_rates)
            out["total_collected_taxes"] = float(self.total_collected_taxes)

            # Indices of richest and poorest agents.
            agent_coin_endows = np.array(
                [agent.total_endowment("Coin") for agent in self.world.agents]
            )
            idx_poor = np.argmin(agent_coin_endows)
            idx_rich = np.argmax(agent_coin_endows)

            tax_days = self.taxes[(self.period - 1) :: self.period]
            for i, tag in zip([idx_poor, idx_rich], ["poorest", "richest"]):
                total_income = np.maximum(
                    0, [tax_day[str(i)]["income"] for tax_day in tax_days]
                ).sum()
                total_tax_paid = np.sum(
                    [tax_day[str(i)]["tax_paid"] for tax_day in tax_days]
                )
                # Report the overall tax rate over the episode
                # for the richest and poorest agents.
                out["avg_tax_rate/{}".format(tag)] = total_tax_paid / np.maximum(
                    0.001, total_income
                )

            if self.tax_model == "saez":
                # Include the running estimate of elasticity.
                out["saez/estimated_elasticity"] = self.elas_tm1

        return out

    def get_dense_log(self):
        """
        Log taxes.

        Returns:
            taxes (list): A list of tax collections. Each entry corresponds to a single
                timestep. Entries are empty except for timesteps where a tax period
                ended and taxes were collected. For those timesteps, each entry
                contains the tax schedule, each agent's reported income, tax paid,
                and redistribution received.
                Returns None if taxes are disabled.
        """
        if self.disable_taxes:
            return None
        return self.taxes
