from qlib.config import C
from qlib.trace import configure_tracing, get_current_span, get_current_trace_context, is_tracing_enabled, trace_span


def teardown_function():
    configure_tracing({"enabled": False})
    C.reset()


def test_trace_span_is_noop_when_tracing_disabled():
    configure_tracing({"enabled": False})

    with trace_span("dataset.prepare") as span:
        assert span is None
        assert get_current_span() is None
        assert get_current_trace_context() == {}


def test_trace_span_creates_context_when_enabled():
    configure_tracing({"enabled": True})

    with trace_span("dataset.prepare", trace_id="trace-1") as span:
        assert span is not None
        assert span.name == "dataset.prepare"
        assert span.trace_id == "trace-1"
        assert get_current_span() == span
        assert get_current_trace_context() == {
            "trace_id": span.trace_id,
            "span_id": span.span_id,
            "span_name": "dataset.prepare",
        }

    assert get_current_span() is None


def test_nested_trace_spans_share_trace_id_and_set_parent():
    configure_tracing({"enabled": True})

    with trace_span("workflow") as parent:
        with trace_span("dataset.prepare") as child:
            assert child.trace_id == parent.trace_id
            assert child.parent_span_id == parent.span_id
            assert get_current_trace_context()["parent_span_id"] == parent.span_id

        assert get_current_span() == parent


def test_tracing_can_be_enabled_from_qlib_config():
    C.set(tracing_config={"enabled": True})

    assert is_tracing_enabled() is True
    with trace_span("model.fit") as span:
        assert span is not None
