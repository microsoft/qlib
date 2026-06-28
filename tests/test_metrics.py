from qlib.config import C
from qlib.metrics import configure_metrics, get_metrics_recorder


def teardown_function():
    configure_metrics({"enabled": False})
    C.reset()


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
