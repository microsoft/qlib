import env
from vecenv import *
import sampler
import logger
import json
import os
import agent
import network
import policy
import random
import tianshou as ts
import tqdm
from tianshou.utils import tqdm_config, MovAvg
from torch.utils.tensorboard import SummaryWriter
from collector import *
import numpy as np


from util import merge_dicts


def get_best_gpu(force=None):
    if force is not None:
        return force
    s = os.popen("nvidia-smi --query-gpu=memory.free --format=csv")
    a = []
    ss = s.read().replace("MiB", "").replace("memory.free", "").split("\n")
    s.close()
    for i in range(1, len(ss) - 1):
        a.append(int(ss[i]))
    best = int(np.argmax(a))
    print("the best GPU is ", best, " with free memories of ", ss[best + 1])
    return best


def setup_seed(seed):
    """

    :param seed:

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class BaseExecutor(object):
    def __init__(
        self,
        log_dir,
        resources,
        env_conf,
        optim=None,
        policy_conf=None,
        network_conf=None,
        policy_path=None,
        seed=None,
    ):
        """A base class for executor

        :param log_dir: The directory to write all the logs.
        :type log_dir: string
        :param resources: A dict which describes available computational resources.
        :type resources: dict
        :param env_conf: Configurations for the envionments.
        :type env_conf: dict
        :param optim: Optimization configuration, defaults to None
        :type optim: dict, optional
        :param policy_conf: Configurations for the RL algorithm, defaults to None
        :type policy_conf: dict, optional
        :param network_conf: Configurations for policy network_conf, defaults to None
        :type network_conf: dict, optional
        :param policy_path: If is not None, would load the policy from this path, defaults to None
        :type policy_path: string, optional
        :param seed: Random seed, defaults to None
        :type seed: int, optional
        """
        # self.config = config
        self.log_dir = log_dir
        print(self.log_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if resources["device"] == "cuda":
            resources["device"] = "cuda:" + str(get_best_gpu())
        self.device = torch.device(resources["device"])
        if seed:
            setup_seed(seed)

        assert not policy_path is None or not policy_conf is None, "Policy must be defined"
        if policy_path:
            self.policy = torch.load(policy_path, map_location=self.device)
            self.policy.actor.extractor.device = self.device
            # policy.eval()
        elif hasattr(agent, policy_conf["name"]):
            policy_conf["config"] = merge_dicts(policy_conf["config"], resources)
            self.policy = getattr(agent, policy_conf["name"])(policy_conf["config"])
            # print(self.policy)
        else:
            assert not network_conf is None
            if "extractor" in network_conf.keys():
                net = getattr(network, network_conf["extractor"]["name"] + "_Extractor")(
                    device=self.device, **network_conf["config"]
                )
            else:
                net = getattr(network, network_conf["name"] + "_Extractor")(
                    device=self.device, **network_conf["config"]
                )
            net.to(self.device)
            actor = getattr(network, network_conf["name"] + "_Actor")(
                extractor=net, device=self.device, **network_conf["config"]
            )
            actor.to(self.device)
            critic = getattr(network, network_conf["name"] + "_Critic")(
                extractor=net, device=self.device, **network_conf["config"]
            )
            critic.to(self.device)
            self.optim = torch.optim.Adam(
                list(actor.parameters()) + list(critic.parameters()),
                lr=optim["lr"],
                weight_decay=optim["weight_decay"] if "weight_decay" in optim else 0.0,
            )
            self.dist = torch.distributions.Categorical
            try:
                self.policy = getattr(ts.policy, policy_conf["name"])(
                    actor, critic, self.optim, self.dist, **policy_conf["config"]
                )
            except:
                self.policy = getattr(policy, policy_conf["name"])(
                    actor, critic, self.optim, self.dist, **policy_conf["config"]
                )
        self.writer = SummaryWriter(self.log_dir)

    def train(
        self,
        max_epoch,
        step_per_epoch,
        repeat_per_collect,
        collect_per_step,
        batch_size,
        iteration=0,
        global_step=0,
        early_stopping=5,
        *args,
        **kargs,
    ):
        """Run the whole training process.

        :param max_epoch: The total number of epoch.
        :param step_per_epoch: The times of bp in one epoch.
        :param collect_per_step: Number of episodes to collect before one bp.
        :param repeat_per_collect: Times of bps after every rould of experience collecting.
        :param batch_size: Batch size when bp.
        :param iteration: The iteration when starting the training, used when fine tuning. (Default value = 0)
        :param global_step: The number of steps when starting the training, used when fine tuning. (Default value = 0)
        :param early_stopping: If the test reward does not reach a new high in `early_stopping` iterations, the training would stop. (Default value = 5)
        :returns: The result on test set.

        """
        raise NotImplementedError

    def train_round(self, repeat_per_collect, collect_per_step, batch_size, *args, **kargs):
        """Do an round of training

        :param collect_per_step: Number of episodes to collect before one bp.
        :param repeat_per_collect: Times of bps after every rould of experience collecting.
        :param batch_size: Batch size when bp.

        """
        raise NotImplementedError

    def eval(self, order_dir, save_res=False, logdir=None, *args, **kargs):
        """Evaluate the policy on orders in order_dir

        :param order_dir: the orders to be evaluated on.
        :param save_res: whether the result of evaluation be saved to self.logdir/res.json (Default value = False)
        :param logdir: the place to save the .log and .pkl log files to. If None, don't save logfiles. (Default value = None)
        :returns: The result of evaluation.

        """
        raise NotImplementedError


class Executor(BaseExecutor):
    def __init__(
        self,
        log_dir,
        resources,
        env_conf,
        train_paths,
        valid_paths,
        test_paths,
        io_conf,
        optim=None,
        policy_conf=None,
        network_conf=None,
        policy_path=None,
        seed=None,
        share_memory=False,
        buffer_size=200000,
        q_learning=False,
        *args,
        **kargs,
    ):
        """[summary]

        :param log_dir: The directory to write all the logs.
        :type log_dir: string
        :param resources: A dict which describes available computational resources.
        :type resources: dict
        :param env_conf: Configurations for the envionments.
        :type env_conf: dict
        :param train_paths: The paths of training datasets including orders, backtest files and features.
        :type train_paths: string
        :param valid_paths: The paths of validation datasets including orders, backtest files and features.
        :type valid_paths: string
        :param test_paths: The paths of test datasets including orders, backtest files and features.
        :type test_paths: string
        :param io_conf: Configuration for sampler and loggers.
        :type io_conf: dict
        :param share_memory: Whether to use shared memory vecnev, defaults to False
        :type share_memory: bool, optional
        :param buffer_size: The size of replay buffer, defaults to 200000
        :type buffer_size: int, optional
        """
        super().__init__(log_dir, resources, env_conf, optim, policy_conf, network_conf, policy_path, seed)
        single_env = getattr(env, env_conf["name"])
        env_conf = merge_dicts(env_conf, train_paths)
        env_conf["log"] = True
        print("CPU_COUNT:", resources["num_cpus"])
        if share_memory:
            self.env = ShmemVectorEnv([lambda: single_env(env_conf) for _ in range(resources["num_cpus"])])
        else:
            self.env = SubprocVectorEnv([lambda: single_env(env_conf) for _ in range(resources["num_cpus"])])
        self.test_collector = Collector(policy=self.policy, env=self.env, testing=True, reward_metric=np.sum)
        self.train_collector = Collector(
            self.policy, self.env, buffer=ts.data.ReplayBuffer(buffer_size), reward_metric=np.sum,
        )
        self.train_paths = train_paths
        self.test_paths = test_paths
        self.valid_paths = valid_paths
        train_sampler_conf = train_paths
        train_sampler_conf["features"] = env_conf["features"]
        test_sampler_conf = test_paths
        test_sampler_conf["features"] = env_conf["features"]
        self.train_sampler = getattr(sampler, io_conf["train_sampler"])(train_sampler_conf)
        self.test_sampler = getattr(sampler, io_conf["test_sampler"])(test_sampler_conf)
        self.train_logger = logger.InfoLogger()
        self.test_logger = getattr(logger, io_conf["test_logger"])

        self.q_learning = q_learning

    def train(
        self,
        max_epoch,
        step_per_epoch,
        repeat_per_collect,
        collect_per_step,
        batch_size,
        iteration=0,
        global_step=0,
        early_stopping=5,
        train_step_min=0,
        log_valid=True,
        *args,
        **kargs,
    ):
        best_epoch, best_reward = -1, -1
        stat = {}
        for epoch in range(1, 1 + max_epoch):
            with tqdm.tqdm(total=step_per_epoch, desc=f"Epoch #{epoch}", **tqdm_config) as t:
                while t.n < t.total:
                    result, losses = self.train_round(repeat_per_collect, collect_per_step, batch_size, iteration)
                    global_step += result["n/st"]
                    iteration += 1
                    for k in result.keys():
                        self.writer.add_scalar("Train/" + k, result[k], global_step=global_step)
                    for k in losses.keys():
                        if stat.get(k) is None:
                            stat[k] = MovAvg()
                        stat[k].add(losses[k])
                        self.writer.add_scalar("Train/" + k, stat[k].get(), global_step=global_step)
                    t.update(1)
            if t.n <= t.total:
                t.update()
            result = self.eval(
                self.valid_paths["order_dir"], logdir=f"{self.log_dir}/valid/{iteration}/" if log_valid else None,
            )
            for k in result.keys():
                self.writer.add_scalar("Valid/" + k, result[k], global_step=global_step)
            if best_epoch == -1 or best_reward < result["rew"]:
                best_reward = result["rew"]
                best_epoch = epoch
                best_state = self.policy.state_dict()
                early_stop_round = 0
                torch.save(self.policy, f"{self.log_dir}/policy_best")
            elif global_step >= train_step_min:
                early_stop_round += 1
            torch.save(self.policy, f"{self.log_dir}/policy_{epoch}")
            print(
                f'Epoch #{epoch}: test_reward: {result["rew"]:.4f}, '  # train_reward: {result_train["rew"]:.4f}, '
                f"best_reward: {best_reward:.4f} in #{best_epoch}"
            )
            if early_stop_round >= early_stopping:
                print("Early stopped")
                break
        print("Testing...")
        self.policy.load_state_dict(best_state)
        result = self.eval(self.test_paths["order_dir"], logdir=f"{self.log_dir}/test/", save_res=True)
        for k in result.keys():
            self.writer.add_scalar("Test/" + k, result[k], global_step=global_step)
        return result

    def train_round(self, repeat_per_collect, collect_per_step, batch_size, *args, **kargs):
        self.policy.train()
        self.env.toggle_log(False)
        self.env.sampler = self.train_sampler
        if not self.q_learning:
            self.train_collector.reset()
        result = self.train_collector.collect(n_episode=collect_per_step, log_fn=self.train_logger)
        result = merge_dicts(result, self.train_logger.summary())
        if not self.q_learning:
            losses = self.policy.update(
                0, self.train_collector.buffer, batch_size=batch_size, repeat=repeat_per_collect,
            )
        else:
            losses = self.policy.update(batch_size, self.train_collector.buffer,)
        return result, losses

    def eval(self, order_dir, save_res=False, logdir=None, *args, **kargs):
        print(f"start evaluating on {order_dir}")
        self.policy.eval()
        self.env.toggle_log(True)
        self.test_sampler.reset(order_dir)
        self.env.sampler = self.test_sampler
        self.test_collector.reset()
        if not logdir is None:
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            eval_logger = self.test_logger(logdir, order_dir)
            eval_logger.reset()
        else:
            eval_logger = self.train_logger
        result = self.test_collector.collect(log_fn=eval_logger)
        result = merge_dicts(result, eval_logger.summary())
        if save_res:
            with open(self.log_dir + "/res.json", "w") as f:
                json.dump(result, f, sort_keys=True, indent=4)
        print(f"finish evaluating on {order_dir}")
        return result
