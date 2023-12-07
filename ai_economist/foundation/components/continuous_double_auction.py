# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)
from ai_economist.foundation.entities import resource_registry


@component_registry.add
class ContinuousDoubleAuction(BaseComponent):
    """Allows mobile agents to buy/sell collectible resources with one another.

    Implements a commodity-exchange-style market where agents may sell a unit of
        resource by submitting an ask (saying the minimum it will accept in payment)
        or may buy a resource by submitting a bid (saying the maximum it will pay in
        exchange for a unit of a given resource).

    Args:
        max_bid_ask (int): Maximum amount of coin that an agent can bid or ask for.
            Must be >= 1. Default is 10 coin.
        order_labor (float): Amount of labor incurred when an agent creates an order.
            Must be >= 0. Default is 0.25.
        order_duration (int): Number of environment timesteps before an unfilled
            bid/ask expires. Must be >= 1. Default is 50 timesteps.
        max_num_orders (int, optional): Maximum number of bids + asks that an agent can
            have open for a given resource. Must be >= 1. Default is no limit to
            number of orders.
    """

    name = "ContinuousDoubleAuction"
    component_type = "Trade"
    required_entities = ["Coin", "Labor"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *args,
        max_bid_ask=10,
        order_labor=0.25,
        order_duration=50,
        max_num_orders=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # The max amount (in coin) that an agent can bid/ask for 1 unit of a commodity
        self.max_bid_ask = int(max_bid_ask)
        assert self.max_bid_ask >= 1
        self.price_floor = 0
        self.price_ceiling = int(max_bid_ask)

        # The amount of time (in timesteps) that an order stays in the books
        # before it expires
        self.order_duration = int(order_duration)
        assert self.order_duration >= 1

        # The maximum number of bid+ask orders an agent can have open
        # for each type of commodity
        self.max_num_orders = int(max_num_orders or self.order_duration)
        assert self.max_num_orders >= 1

        # The labor cost associated with creating a bid or ask order

        self.order_labor = float(order_labor)
        self.order_labor = max(self.order_labor, 0.0)

        # Each collectible resource in the world can be traded via this component
        self.commodities = [
            r for r in self.world.resources if resource_registry.get(r).collectible
        ]

        # These get reset at the start of an episode:
        self.asks = {c: [] for c in self.commodities}
        self.bids = {c: [] for c in self.commodities}
        self.n_orders = {
            c: {i: 0 for i in range(self.n_agents)} for c in self.commodities
        }
        self.executed_trades = []
        self.price_history = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.bid_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.ask_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }

    # Convenience methods
    # -------------------

    def _price_zeros(self):
        if 1 + self.price_ceiling - self.price_floor <= 0:
            print("ERROR!", self.price_ceiling, self.price_floor)

        return np.zeros(1 + self.price_ceiling - self.price_floor)

    def available_asks(self, resource, agent):
        """
        Get a histogram of asks for resource to which agent could bid against.

        Args:
            resource (str): Name of the resource
            agent (BasicMobileAgent or None): Object of agent for which available
                asks are being queried. If None, all asks are considered available.

        Returns:
            ask_hist (ndarray): For each possible price level, the number of
                available asks.
        """
        if agent is None:
            a_idx = -1
        else:
            a_idx = agent.idx
        ask_hist = self._price_zeros()
        for i, h in self.ask_hists[resource].items():
            if a_idx != i:
                ask_hist += h
        return ask_hist

    def available_bids(self, resource, agent):
        """
        Get a histogram of bids for resource to which agent could ask against.

        Args:
            resource (str): Name of the resource
            agent (BasicMobileAgent or None): Object of agent for which available
                bids are being queried. If None, all bids are considered available.

        Returns:
            bid_hist (ndarray): For each possible price level, the number of
                available bids.
        """
        if agent is None:
            a_idx = -1
        else:
            a_idx = agent.idx
        bid_hist = self._price_zeros()
        for i, h in self.bid_hists[resource].items():
            if a_idx != i:
                bid_hist += h
        return bid_hist

    def can_bid(self, resource, agent):
        """If agent can submit a bid for resource."""
        return self.n_orders[resource][agent.idx] < self.max_num_orders

    def can_ask(self, resource, agent):
        """If agent can submit an ask for resource."""
        return (
            self.n_orders[resource][agent.idx] < self.max_num_orders
            and agent.state["inventory"][resource] > 0
        )

    # Core components for this market
    # -------------------------------

    def create_bid(self, resource, agent, max_payment):
        """Create a new bid for resource, with agent offering max_payment.

        On a successful trade, payment will be at most max_payment, possibly less.

        The agent places the bid coin into escrow so that it may not be spent on
        something else while the order exists.
        """

        # The agent is past the max number of orders
        # or doesn't have enough money, do nothing
        if (not self.can_bid(resource, agent)) or agent.state["inventory"][
            "Coin"
        ] < max_payment:
            return

        assert self.price_floor <= max_payment <= self.price_ceiling

        bid = {
            "buyer": agent.idx,
            "bid": int(max_payment),
            "bid_lifetime": 0,
        }

        # Add this to the bid book
        self.bids[resource].append(bid)
        self.bid_hists[resource][bid["buyer"]][bid["bid"] - self.price_floor] += 1
        self.n_orders[resource][agent.idx] += 1

        # Set aside whatever money the agent is willing to pay
        # (will get excess back if price ends up being less)
        _ = agent.inventory_to_escrow("Coin", int(max_payment))

        # Incur the labor cost of creating an order
        agent.state["endogenous"]["Labor"] += self.order_labor

    def create_ask(self, resource, agent, min_income):
        """
        Create a new ask for resource, with agent asking for min_income.

        On a successful trade, income will be at least min_income, possibly more.

        The agent places one unit of resource into escrow so that it may not be used
        for something else while the order exists.
        """
        # The agent is past the max number of orders
        # or doesn't the resource it's trying to sell, do nothing
        if not self.can_ask(resource, agent):
            return

        # is there an upper limit?
        assert self.price_floor <= min_income <= self.price_ceiling

        ask = {
            "seller": agent.idx,
            "ask": int(min_income),
            "ask_lifetime": 0,
        }

        # Add this to the ask book
        self.asks[resource].append(ask)
        self.ask_hists[resource][ask["seller"]][ask["ask"] - self.price_floor] += 1
        self.n_orders[resource][agent.idx] += 1

        # Set aside the resource the agent is willing to sell
        amount = agent.inventory_to_escrow(resource, 1)
        assert amount == 1

        # Incur the labor cost of creating an order
        agent.state["endogenous"]["Labor"] += self.order_labor

    def match_orders(self):
        """
        This implements the continuous double auction by identifying valid bid/ask
        pairs and executing trades accordingly.

        Higher (lower) bids (asks) are given priority over lower (higher) bids (asks).
        Trades are executed using the price of whichever bid/ask order was placed
        first: bid price if bid was placed first, ask price otherwise.

        Trading removes the payment and resource from bidder's and asker's escrow,
        respectively, and puts them in the other's inventory.
        """
        self.executed_trades.append([])

        for resource in self.commodities:
            possible_match = [True for _ in range(self.n_agents)]
            keep_checking = True

            bids = sorted(
                self.bids[resource],
                key=lambda b: (b["bid"], b["bid_lifetime"]),
                reverse=True,
            )
            asks = sorted(
                self.asks[resource], key=lambda a: (a["ask"], -a["ask_lifetime"])
            )

            while any(possible_match) and keep_checking:
                idx_bid, idx_ask = 0, 0
                while True:
                    # Out of bids to check. Exit both loops.
                    if idx_bid >= len(bids):
                        keep_checking = False
                        break

                    # Already know this buyer is no good for this round.
                    # Skip to next bid.
                    if not possible_match[bids[idx_bid]["buyer"]]:
                        idx_bid += 1

                    # Out of asks to check. This buyer won't find a match on this round.
                    # (maybe) Restart inner loop.
                    elif idx_ask >= len(asks):
                        possible_match[bids[idx_bid]["buyer"]] = False
                        break

                    # Skip to next ask if this ask comes from the buyer
                    # of the current bid.
                    elif asks[idx_ask]["seller"] == bids[idx_bid]["buyer"]:
                        idx_ask += 1

                    # If this bid/ask pair can't be matched, this buyer
                    # can't be matched. (maybe) Restart inner loop.
                    elif bids[idx_bid]["bid"] < asks[idx_ask]["ask"]:
                        possible_match[bids[idx_bid]["buyer"]] = False
                        break

                    # TRADE! (then restart inner loop)
                    else:
                        bid = bids.pop(idx_bid)
                        ask = asks.pop(idx_ask)

                        trade = {"commodity": resource}
                        trade.update(bid)
                        trade.update(ask)

                        if (
                            bid["bid_lifetime"] <= ask["ask_lifetime"]
                        ):  # Ask came earlier. (in other words,
                            # trade triggered by new bid)
                            trade["price"] = int(trade["ask"])
                        else:  # Bid came earlier. (in other words,
                            # trade triggered by new ask)
                            trade["price"] = int(trade["bid"])
                        trade["cost"] = trade["price"]  # What the buyer pays in total
                        trade["income"] = trade[
                            "price"
                        ]  # What the seller receives in total

                        buyer = self.world.agents[trade["buyer"]]
                        seller = self.world.agents[trade["seller"]]

                        # Bookkeeping
                        self.bid_hists[resource][bid["buyer"]][
                            bid["bid"] - self.price_floor
                        ] -= 1
                        self.ask_hists[resource][ask["seller"]][
                            ask["ask"] - self.price_floor
                        ] -= 1
                        self.n_orders[trade["commodity"]][seller.idx] -= 1
                        self.n_orders[trade["commodity"]][buyer.idx] -= 1
                        self.executed_trades[-1].append(trade)
                        self.price_history[resource][trade["seller"]][
                            trade["price"]
                        ] += 1

                        # The resource goes from the seller's escrow
                        # to the buyer's inventory
                        seller.state["escrow"][resource] -= 1
                        buyer.state["inventory"][resource] += 1

                        # Buyer's money (already set aside) leaves escrow
                        pre_payment = int(trade["bid"])
                        buyer.state["escrow"]["Coin"] -= pre_payment
                        assert buyer.state["escrow"]["Coin"] >= 0

                        # Payment is removed from the pre_payment
                        # and given to the seller. Excess returned to buyer.
                        payment_to_seller = int(trade["price"])
                        excess_payment_from_buyer = pre_payment - payment_to_seller
                        assert excess_payment_from_buyer >= 0
                        seller.state["inventory"]["Coin"] += payment_to_seller
                        buyer.state["inventory"]["Coin"] += excess_payment_from_buyer

                        # Restart the inner loop
                        break

            # Keep the unfilled bids/asks
            self.bids[resource] = bids
            self.asks[resource] = asks

    def remove_expired_orders(self):
        """
        Increment the time counter for any unfilled bids/asks and remove expired
        orders from the market.

        When orders expire, the payment or resource is removed from escrow and
        returned to the inventory and the associated order is removed from the order
        books.
        """
        world = self.world

        for resource in self.commodities:

            bids_ = []
            for bid in self.bids[resource]:
                bid["bid_lifetime"] += 1
                # If the bid is not expired, keep it in the bids
                if bid["bid_lifetime"] <= self.order_duration:
                    bids_.append(bid)
                # Otherwise, remove it and do the associated bookkeeping
                else:
                    # Return the set aside money to the buyer
                    amount = world.agents[bid["buyer"]].escrow_to_inventory(
                        "Coin", bid["bid"]
                    )
                    assert amount == bid["bid"]
                    # Adjust the bid histogram to reflect the removal of the bid
                    self.bid_hists[resource][bid["buyer"]][
                        bid["bid"] - self.price_floor
                    ] -= 1
                    # Adjust the order counter
                    self.n_orders[resource][bid["buyer"]] -= 1

            asks_ = []
            for ask in self.asks[resource]:
                ask["ask_lifetime"] += 1
                # If the ask is not expired, keep it in the asks
                if ask["ask_lifetime"] <= self.order_duration:
                    asks_.append(ask)
                # Otherwise, remove it and do the associated bookkeeping
                else:
                    # Return the set aside resource to the seller
                    resource_unit = world.agents[ask["seller"]].escrow_to_inventory(
                        resource, 1
                    )
                    assert resource_unit == 1
                    # Adjust the ask histogram to reflect the removal of the ask
                    self.ask_hists[resource][ask["seller"]][
                        ask["ask"] - self.price_floor
                    ] -= 1
                    # Adjust the order counter
                    self.n_orders[resource][ask["seller"]] -= 1

            self.bids[resource] = bids_
            self.asks[resource] = asks_

    # Required methods for implementing components
    # --------------------------------------------

    def get_n_actions(self, agent_cls_name):
        """
        See base_component.py for detailed description.

        Adds 2*C action spaces [ (bid+ask) * n_commodities ], each with 1 + max_bid_ask
        actions corresponding to price levels 0 to max_bid_ask.
        """
        # This component adds 2*(1+max_bid_ask)*n_resources possible actions:
        # buy/sell x each-price x each-resource
        if agent_cls_name == "BasicMobileAgent":
            trades = []
            for c in self.commodities:
                trades.append(
                    ("Buy_{}".format(c), 1 + self.max_bid_ask)
                )  # How much willing to pay for c
                trades.append(
                    ("Sell_{}".format(c), 1 + self.max_bid_ask)
                )  # How much need to receive to sell c
            return trades

        return None

    def get_additional_state_fields(self, agent_cls_name):
        """
        See base_component.py for detailed description.
        """
        # This component doesn't add any state fields
        return {}

    def component_step(self):
        """
        See base_component.py for detailed description.

        Create new bids and asks, match and execute valid order pairs, and manage
        order expiration.
        """
        world = self.world

        for resource in self.commodities:
            for agent in world.agents:
                self.price_history[resource][agent.idx] *= 0.995

                # Create bid action
                # -----------------
                resource_action = agent.get_component_action(
                    self.name, "Buy_{}".format(resource)
                )

                # No-op
                if resource_action == 0:
                    pass

                # Create a bid
                elif resource_action <= self.max_bid_ask + 1:
                    self.create_bid(resource, agent, max_payment=resource_action - 1)

                else:
                    raise ValueError

                # Create ask action
                # -----------------
                resource_action = agent.get_component_action(
                    self.name, "Sell_{}".format(resource)
                )

                # No-op
                if resource_action == 0:
                    pass

                # Create an ask
                elif resource_action <= self.max_bid_ask + 1:
                    self.create_ask(resource, agent, min_income=resource_action - 1)

                else:
                    raise ValueError

        # Here's where the magic happens:
        self.match_orders()  # Pair bids and asks
        self.remove_expired_orders()  # Get rid of orders that have expired

    def generate_observations(self):
        """
        See base_component.py for detailed description.

        Here, agents and the planner both observe historical market behavior and
        outstanding bids/asks for each tradable commodity. Agents only see the
        outstanding bids/asks to which they could respond (that is, that they did not
        submit). Agents also see their own outstanding bids/asks.
        """
        world = self.world

        obs = {a.idx: {} for a in world.agents + [world.planner]}

        prices = np.arange(self.price_floor, self.price_ceiling + 1)
        for c in self.commodities:
            net_price_history = np.sum(
                np.stack([self.price_history[c][i] for i in range(self.n_agents)]),
                axis=0,
            )
            market_rate = prices.dot(net_price_history) / np.maximum(
                0.001, np.sum(net_price_history)
            )
            scaled_price_history = net_price_history * self.inv_scale

            full_asks = self.available_asks(c, agent=None)
            full_bids = self.available_bids(c, agent=None)

            obs[world.planner.idx].update(
                {
                    "market_rate-{}".format(c): market_rate,
                    "price_history-{}".format(c): scaled_price_history,
                    "full_asks-{}".format(c): full_asks,
                    "full_bids-{}".format(c): full_bids,
                }
            )

            for _, agent in enumerate(world.agents):
                # Private to the agent
                obs[agent.idx].update(
                    {
                        "market_rate-{}".format(c): market_rate,
                        "price_history-{}".format(c): scaled_price_history,
                        "available_asks-{}".format(c): full_asks
                        - self.ask_hists[c][agent.idx],
                        "available_bids-{}".format(c): full_bids
                        - self.bid_hists[c][agent.idx],
                        "my_asks-{}".format(c): self.ask_hists[c][agent.idx],
                        "my_bids-{}".format(c): self.bid_hists[c][agent.idx],
                    }
                )

        return obs

    def generate_masks(self, completions=0):
        """
        See base_component.py for detailed description.

        Agents cannot submit bids/asks for resources where they are at the order
        limit. In addition, they may only submit asks for resources they possess and
        bids for which they can pay.
        """
        world = self.world

        masks = dict()

        for agent in world.agents:
            masks[agent.idx] = {}

            can_pay = np.arange(self.max_bid_ask + 1) <= agent.inventory["Coin"]

            for resource in self.commodities:
                if not self.can_ask(resource, agent):  # asks_maxed:
                    masks[agent.idx]["Sell_{}".format(resource)] = np.zeros(
                        1 + self.max_bid_ask
                    )
                else:
                    masks[agent.idx]["Sell_{}".format(resource)] = np.ones(
                        1 + self.max_bid_ask
                    )

                if not self.can_bid(resource, agent):
                    masks[agent.idx]["Buy_{}".format(resource)] = np.zeros(
                        1 + self.max_bid_ask
                    )
                else:
                    masks[agent.idx]["Buy_{}".format(resource)] = can_pay.astype(
                        np.int32
                    )

        return masks

    # For non-required customization
    # ------------------------------

    def get_metrics(self):
        """
        Metrics that capture what happened through this component.

        Returns:
            metrics (dict): A dictionary of {"metric_name": metric_value},
                where metric_value is a scalar.
        """
        world = self.world

        trade_keys = ["price", "cost", "income"]

        selling_stats = {
            a.idx: {
                c: {k: 0 for k in trade_keys + ["n_sales"]} for c in self.commodities
            }
            for a in world.agents
        }
        buying_stats = {
            a.idx: {
                c: {k: 0 for k in trade_keys + ["n_sales"]} for c in self.commodities
            }
            for a in world.agents
        }

        n_trades = 0

        for trades in self.executed_trades:
            for trade in trades:
                n_trades += 1
                i_s, i_b, c = trade["seller"], trade["buyer"], trade["commodity"]
                selling_stats[i_s][c]["n_sales"] += 1
                buying_stats[i_b][c]["n_sales"] += 1
                for k in trade_keys:
                    selling_stats[i_s][c][k] += trade[k]
                    buying_stats[i_b][c][k] += trade[k]

        out_dict = {}
        for a in world.agents:
            for c in self.commodities:
                for stats, prefix in zip(
                    [selling_stats, buying_stats], ["Sell", "Buy"]
                ):
                    n = stats[a.idx][c]["n_sales"]
                    if n == 0:
                        for k in trade_keys:
                            stats[a.idx][c][k] = np.nan
                    else:
                        for k in trade_keys:
                            stats[a.idx][c][k] /= n

                    for k, v in stats[a.idx][c].items():
                        out_dict["{}/{}{}/{}".format(a.idx, prefix, c, k)] = v

        out_dict["n_trades"] = n_trades

        return out_dict

    def additional_reset_steps(self):
        """
        See base_component.py for detailed description.

        Reset the order books.
        """
        self.bids = {c: [] for c in self.commodities}
        self.asks = {c: [] for c in self.commodities}
        self.n_orders = {
            c: {i: 0 for i in range(self.n_agents)} for c in self.commodities
        }

        self.price_history = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.bid_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }
        self.ask_hists = {
            c: {i: self._price_zeros() for i in range(self.n_agents)}
            for c in self.commodities
        }

        self.executed_trades = []

    def get_dense_log(self):
        """
        Log executed trades.

        Returns:
            trades (list): A list of trade events. Each entry corresponds to a single
                timestep and contains a description of any trades that occurred on
                that timestep.
        """
        return self.executed_trades
