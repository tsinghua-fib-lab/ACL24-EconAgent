# Large Language Model-Empowered Agents for Simulating Macroeconomic Activities
The implementation of macroeconomic simulation is based on [Foundation](https://github.com/MaciejMacko/ai-economist), An Economic Simulation Framework, which is announced by this paper: 

Zheng, Stephan, et al. "The ai economist: Improving equality and productivity with ai-driven tax policies." arXiv preprint arXiv:2004.13332 (2020).

# Run
Simulate with GPT-3.5, 100 agents, and 240 months (fill openai.api_key in simulate_utils.py): 

`python simulate.py --policy_model gpt --num_agents 100 --episode_length 240`

Simulate with Composite, 100 agents, and 240 months:

`python simulate.py --policy_model complex --num_agents 100 --episode_length 240`

For RL approaches, *i.e.*, **The ai economist**, we just follow their training codes and use the trained models for simulations. See appendix in the paper for details.
