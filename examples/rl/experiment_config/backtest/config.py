_base_ = ["./twap.yml"]

strategies = {
    "_delete_": True,
    "30min": {
        "class": "TWAPStrategy",
        "module_path": "qlib.contrib.strategy.rule_strategy",
        "kwargs": {},
    },
    "1day": {
        "class": "SAOEIntStrategy",
        "module_path": "qlib.rl.order_execution.strategy",
        "kwargs": {
            "state_interpreter": {
                "class": "FullHistoryStateInterpreter",
                "module_path": "qlib.rl.order_execution.interpreter",
                "kwargs": {
                    "max_step": 8,
                    "data_ticks": 240,
                    "data_dim": 6,
                    "processed_data_provider": {
                        "class": "PickleProcessedDataProvider",
                        "module_path": "qlib.rl.data.pickle_styled",
                        "kwargs": {
                            "data_dir": "./data/pickle_dataframe/feature",
                        },
                    },
                },
            },
            "action_interpreter": {
                "class": "CategoricalActionInterpreter",
                "module_path": "qlib.rl.order_execution.interpreter",
                "kwargs": {
                    "values": 14,
                    "max_step": 8,
                },
            },
            "network": {
                "class": "Recurrent",
                "module_path": "qlib.rl.order_execution.network",
                "kwargs": {},
            },
            "policy": {
                "class": "PPO",
                "module_path": "qlib.rl.order_execution.policy",
                "kwargs": {
                    "lr": 1.0e-4,
                    "weight_file": "./checkpoints/latest.pth",
                },
            },
        },
    },
}
