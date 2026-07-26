from qlib.contrib.model import gbdt, highfreq_gdbt_model


def test_lgb_model_skips_none_early_stopping(monkeypatch):
    early_stopping_calls = []
    train_kwargs = {}
    model = gbdt.LGBModel(early_stopping_rounds=None)

    monkeypatch.setattr(
        model,
        "_prepare_data",
        lambda dataset, reweighter=None: [("train-dataset", "train")],
    )
    monkeypatch.setattr(
        gbdt.lgb,
        "early_stopping",
        lambda rounds: early_stopping_calls.append(rounds) or "early",
    )
    monkeypatch.setattr(gbdt.lgb, "log_evaluation", lambda period: "log")
    monkeypatch.setattr(gbdt.lgb, "record_evaluation", lambda evals_result: "record")

    def train(*args, **kwargs):
        train_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(gbdt.lgb, "train", train)

    model.fit(object(), evals_result={"train": {}})

    assert early_stopping_calls == []
    assert train_kwargs["callbacks"] == ["log", "record"]


def test_hflgb_model_skips_none_early_stopping(monkeypatch):
    early_stopping_calls = []
    train_kwargs = {}
    model = highfreq_gdbt_model.HFLGBModel()

    monkeypatch.setattr(
        model, "_prepare_data", lambda dataset: ("train-dataset", "valid-dataset")
    )
    monkeypatch.setattr(
        highfreq_gdbt_model.lgb,
        "early_stopping",
        lambda rounds: early_stopping_calls.append(rounds) or "early",
    )
    monkeypatch.setattr(highfreq_gdbt_model.lgb, "log_evaluation", lambda period: "log")
    monkeypatch.setattr(
        highfreq_gdbt_model.lgb, "record_evaluation", lambda evals_result: "record"
    )

    def train(*args, **kwargs):
        train_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(highfreq_gdbt_model.lgb, "train", train)

    model.fit(
        object(),
        early_stopping_rounds=None,
        evals_result={"train": {"loss": [1.0]}, "valid": {"loss": [1.1]}},
    )

    assert early_stopping_calls == []
    assert train_kwargs["callbacks"] == ["log", "record"]
