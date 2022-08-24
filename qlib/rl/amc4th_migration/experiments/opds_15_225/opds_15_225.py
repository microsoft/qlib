_base_ = ["./twap.yml"]

strategies = {
    "_delete_": True,
    "5min": {"type": "qlib.contrib.strategy.rule_strategy.TWAPStrategy"},
    "30min": {"type": "qlib.rl.order_execution.strategy.MultiplexStrategyOnTradeStep", "strategies": []},
    "1day": {
        "type": "qlib.rl.order_execution.strategy.SAOEIntStrategy",
        "state_interpreter": {
            "type": "qlib.rl.order_execution.interpreter.FullHistoryStateInterpreter",
            "max_step": 8,
            "data_ticks": 240,
            "data_dim": 16,
        },
        "action_interpreter": {
            "type": "qlib.rl.order_execution.interpreter.CategoricalActionInterpreter",
            "values": 4,
            "max_step": 8,
        },
        "network": {
            "type": "qlib.rl.order_execution.network.DualAttentionRNN",
        },
        "policy": {
            "type": "qlib.rl.order_execution.policy.PPO",
            "lr": 1.0e-4,
            "weight_file": "data/amc-checkpoints/opds_15_225/opds_15_225_30r_4_80",
        },
    },
}

import copy

# for mypy
assert isinstance(strategies["1day"], dict)
assert isinstance(strategies["30min"], dict)

for step in range(8):
    step_start, step_end = max(15, step * 30), min(225, step * 30 + 30)
    num_inner_steps = (step_end - step_start + 5 - 1) // 5
    strategy: dict = copy.deepcopy(strategies["1day"])
    strategy["state_interpreter"]["max_step"] = num_inner_steps
    action_values = [3, 6, 6, 6, 6, 6, 6, 3]

    strategy["network"] = {"type": "qlib.rl.order_execution.network.DualAttentionRNN"}
    strategy["action_interpreter"]["values"] = action_values[step]
    strategy["action_interpreter"]["max_step"] = num_inner_steps
    strategy["policy"]["weight_file"] = f"data/amc-checkpoints/opds_15_225/opds_{step_start}_{step_end}"

    strategies["30min"]["strategies"].append(strategy)


del copy, step, step_start, step_end, num_inner_steps, strategy, action_values
