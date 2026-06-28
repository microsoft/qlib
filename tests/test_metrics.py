import pandas as pd

from qlib.config import C
from qlib.data.cache import H, MemoryCalendarCache, SimpleDatasetCache
from qlib.metrics import configure_metrics, get_metrics_recorder


def teardown_function():
    configure_metrics({"enabled": False})
    H.clear()
    C.reset()


class FakeDatasetProvider:
    def __init__(self):
        self.calls = 0

    def dataset(self, instruments, fields, start_time=None, end_time=None, freq="day", inst_processors=[]):
        self.calls += 1
        index = pd.MultiIndex.from_product(
            [["SH600000"], [pd.Timestamp("2020-01-01")]], names=["instrument", "datetime"]
        )
        return pd.DataFrame([[1.0]], index=index, columns=fields)


class FakeCalendarProvider:
    def __init__(self):
        self.calls = 0

    @staticmethod
    def _uri(start_time=None, end_time=None, freq="day", future=False):
        return f"{start_time}:{end_time}:{freq}:{future}"

    def calendar(self, start_time=None, end_time=None, freq="day", future=False):
        self.calls += 1
        return [pd.Timestamp("2020-01-01")]


def test_metrics_are_disabled_by_default():
    configure_metrics({"enabled": False})

    metrics = get_metrics_recorder()
    metrics.increment("qlib.cache.hit")
    metrics.gauge("qlib.memory.usage_mb", 128)
    metrics.timing("qlib.data.load_seconds", 0.25)

    assert metrics.enabled is False
    assert metrics.snapshot() == {"counters": {}, "gauges": {}, "timings": {}}


def test_enabled_metrics_record_counters_gauges_and_timings():
    configure_metrics({"enabled": True})

    metrics = get_metrics_recorder()
    metrics.increment("qlib.cache.hit")
    metrics.increment("qlib.cache.hit", 2)
    metrics.gauge("qlib.memory.usage_mb", 128)
    metrics.timing("qlib.data.load_seconds", 0.25)
    metrics.timing("qlib.data.load_seconds", 0.75)

    snapshot = metrics.snapshot()
    assert snapshot["counters"]["qlib.cache.hit"] == 3
    assert snapshot["gauges"]["qlib.memory.usage_mb"] == 128
    assert snapshot["timings"]["qlib.data.load_seconds"] == {
        "count": 2,
        "total": 1.0,
        "last": 0.75,
        "min": 0.25,
        "max": 0.75,
    }


def test_metrics_support_tags():
    configure_metrics({"enabled": True})

    metrics = get_metrics_recorder()
    metrics.increment("qlib.cache.hit", tags={"provider": "local", "freq": "day"})

    assert metrics.snapshot()["counters"]["qlib.cache.hit{freq=day,provider=local}"] == 1


def test_metrics_timer_context_records_elapsed_time():
    configure_metrics({"enabled": True})

    metrics = get_metrics_recorder()
    with metrics.timer("qlib.workflow.step_seconds"):
        pass

    timing = metrics.snapshot()["timings"]["qlib.workflow.step_seconds"]
    assert timing["count"] == 1
    assert timing["total"] >= 0
    assert timing["last"] >= 0


def test_metrics_can_be_enabled_from_qlib_config():
    C.set(metrics_config={"enabled": True})

    metrics = get_metrics_recorder()
    metrics.increment("qlib.config.enabled")

    assert metrics.enabled is True
    assert metrics.snapshot()["counters"]["qlib.config.enabled"] == 1


def test_simple_dataset_cache_emits_hit_miss_and_timing_metrics(tmp_path):
    C["local_cache_path"] = str(tmp_path)
    configure_metrics({"enabled": True})
    provider = FakeDatasetProvider()
    cache = SimpleDatasetCache(provider)

    cache.dataset(["SH600000"], ["close"], freq="day")
    cache.dataset(["SH600000"], ["close"], freq="day")

    snapshot = get_metrics_recorder().snapshot()
    assert provider.calls == 1
    assert snapshot["counters"]["qlib.cache.dataset.miss{cache=simple,freq=day}"] == 1
    assert snapshot["counters"]["qlib.cache.dataset.hit{cache=simple,freq=day}"] == 1
    assert snapshot["timings"]["qlib.cache.dataset.load_seconds{cache=simple,freq=day}"]["count"] == 2


def test_memory_calendar_cache_emits_hit_miss_and_timing_metrics():
    configure_metrics({"enabled": True})
    provider = FakeCalendarProvider()
    cache = MemoryCalendarCache(provider)

    cache.calendar(freq="day")
    cache.calendar(freq="day")

    snapshot = get_metrics_recorder().snapshot()
    assert provider.calls == 1
    assert snapshot["counters"]["qlib.cache.calendar.miss{cache=memory,freq=day}"] == 1
    assert snapshot["counters"]["qlib.cache.calendar.hit{cache=memory,freq=day}"] == 1
    assert snapshot["timings"]["qlib.cache.calendar.load_seconds{cache=memory,freq=day}"]["count"] == 2
