# Requirements

Here is the minimal hardware requirements to run the `workflow_by_code` example.
- Memory: 16G
- Free Disk: 5G


# NOTE
The results will slightly vary on different OSs(the variance of annualized return will be less than 2%).
The evaluation results in the `README.md` page are from Linux OS.

# Recorder artifact trust

Recorder loading is restricted by default. Prediction and label data normally need
no opt-in, but resuming a workflow can require executable model, dataset or task
objects. Only enable `trusted=True` after verifying **both the writer and
the artifact store**, including who can replace files in a shared experiment.
Unrestricted pickle loading can execute code. Do not automatically retry a refused
load with trust enabled.

See the [artifact loading migration guide](https://qlib.readthedocs.io/en/latest/start/artifact_migration.html)
for supported NumPy/pandas data, version limitations, custom recorder/loader changes
and low-level `load_object(..., trusted=True)` usage. Some notebooks and direct-load
examples already opt in explicitly; their model/task loads assume your own trusted
runs, not arbitrary downloaded or shared artifacts.

## Online workflows

The three `online_srv` CLIs expose `--trusted`, defaulting to `False`.
For **your own artifacts in an access-controlled store**, run from `examples/`:

```bash
python online_srv/update_online_pred.py --trusted=True main
# Subsequent daily updates use the same explicit consent:
python online_srv/update_online_pred.py --trusted=True update_online_pred
```

The simulation and rolling-management examples also accept the flag. **Their
`main`/`first_run` methods reset experiment data (and rolling task pools); use
dedicated experiment stores and save anything important before running them.**
With Qlib data and, for rolling management, your MongoDB service configured:

```bash
python online_srv/online_management_simulate.py --trusted=True --exp_name=my_own_rolling_exp main
python online_srv/rolling_online_management.py --trusted=True --task_url=mongodb://localhost:27017/ first_run
python online_srv/rolling_online_management.py --trusted=True --task_url=mongodb://localhost:27017/ add_strategy
python online_srv/rolling_online_management.py --task_url=mongodb://localhost:27017/ routine
```

Consent is passed to each strategy, including newly added strategies, and through
its online tool/updater to executable artifact loads. Prediction, label and
numerical-report reads stay restricted. `OnlineManager` has no trust constructor
or global permission.

`RollingOnlineExample` creates a `DelayTrainerRM` with the selected consent only
when no trainer is supplied. In Python, configure a caller-supplied delayed trainer
yourself; the example does not change its policy:

```python
from qlib.model.trainer import DelayTrainerR
from online_srv.rolling_online_management import RollingOnlineExample

example = RollingOnlineExample(
    trainer=DelayTrainerR(trusted=True),
    trusted=True,
)
```

Ordinary `TrainerR`/`TrainerRM` constructors do not accept `trusted`.
The simulation example leaves those trainers unchanged and configures its strategy.

Rolling management saves a local `.RollingOnlineExample` pickle. Only restore a
file you independently trust. Restoring it retains its saved strategy, tool and trainer
settings; legacy components without a saved flag default to restricted loading.
An example constructor/CLI flag does not override a subsequently loaded manager.
After reviewing artifact sources, explicitly reconfigure or recreate each
strategy, its `strategy.tool`, and any delayed trainer; changing a strategy's flag
alone does not update its existing tool. `add_strategy` uses the current CLI flag
for **new** strategies only. Remember that this example's `first_run` is destructive.
Ordinary trusted artifacts do not require deleting experiments or full retraining
to migrate; select consent on the actual components that reload them.

## Other migrations

- [DDG-DA](benchmarks_dynamic/DDG-DA/README.md#recorder-artifacts-and-local-working-files):
  the workflow's opt-in covers necessary recorder and handler/internal-data cache
  reads. Verify local `working_dir`, configuration directory and `h_path` contents
  as well as MLflow storage.
- [HIST](benchmarks/HIST/README.md#stock-index-mapping-migration): update the
  stock-index mapping path to JSON, including on restored models; legacy
  object-pickled `.npy` mappings are not accepted.

These settings cover scoped artifact loads, not every Qlib deserialization API.
Other local model files, handler caches, YAML configurations and task stores
retain their own trust requirements.
