


class AuxiliaryInfoCollector:

    def __init__(self, logger):

    def __call__(self):
        info = {"category": ep_state.flow_dir.value, "reward": rew_info}
        if ep_state.done:
            info["index"] = {"stock_id": sample.stock_id, "date": sample.date}
            info["history"] = {"action": self.action_history}
            info.update(ep_state.logs())

            try:
                # done but loop is not exhausted
                # exhaust the loop manually
                while True:
                    self.collect_data_loop.send(0.)
            except StopIteration:
                pass

            info["qlib"] = {}
            for key, val in list(
                self.executor.trade_account.get_trade_indicator().order_indicator_his.values()
            )[0].to_series().items():
                info["qlib"][key] = val.item()