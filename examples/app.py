# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Instruction:
1. Install streamlit by conda or pip.
2. qrun the ymal file in the examples folder at least once in order to make a mlruns directory inside examples.
3. run "streamlit run app.py" and enjoy.
"""

import itertools

import pandas as pd
import streamlit as st

import qlib
from qlib.contrib.report import analysis_model, analysis_position
from qlib.utils.exceptions import LoadObjectError
from qlib.workflow import R


def _max_width_():
    # set the app width with a big number in order to adjust wide screens
    max_width_str = f"max-width: 4000px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )


@st.cache
def _init():
    # cache the initiation in order to only initiate once
    qlib.init()


_init()
_max_width_()

experiments = R.list_experiments()
exp_names = list(experiments.keys())
selected_exp_name = st.sidebar.selectbox("Please select the experiment", exp_names)

recorders = R.list_recorders(experiment_name=selected_exp_name)
recorder_names = list(recorders.keys())
selected_recoder_name = st.sidebar.selectbox(
    "Please select the recoder", recorder_names
)

selected_recoder = recorders[selected_recoder_name]


@st.cache(max_entries=5, allow_output_mutation=True)
def get_recorder_artifacts(recoder):

    artifacts = dict()
    artifacts["params"] = recoder.list_params()
    artifacts["metrics"] = recoder.list_metrics()
    artifacts["tags"] = recoder.list_tags()
    try:
        artifacts["report_normal_df"] = recoder.load_object(
            "portfolio_analysis/report_normal.pkl"
        )
    except LoadObjectError:
        pass
    try:
        artifacts["analysis_df"] = recoder.load_object(
            "portfolio_analysis/port_analysis.pkl"
        )
    except LoadObjectError:
        pass
    try:
        pred_df = recoder.load_object("pred.pkl")
        label_df = recoder.load_object("label.pkl")
        label_df.columns = ["label"]
        artifacts["pred_label"] = pd.concat(
            [label_df, pred_df], axis=1, sort=True
        ).reindex(label_df.index)
    except LoadObjectError:
        pass
    try:
        artifacts["positions"] = selected_recoder.load_object(
            "portfolio_analysis/positions_normal.pkl"
        )
    except LoadObjectError:
        pass

    return artifacts


loaded_artifacts = get_recorder_artifacts(selected_recoder)

params = loaded_artifacts.get("params")
metrics = loaded_artifacts.get("metrics")
tags = loaded_artifacts.get("tags")

st.write(params)
st.write(metrics)
st.write(tags)

positions = loaded_artifacts.get("positions")

if positions is not None:
    dates = list(positions.keys())
    col1, col2 = st.beta_columns(2)
    with col1:
        try:
            date = st.date_input("Please select a date", value=dates[0])
            date = pd.Timestamp(date)
            position_at_date = pd.DataFrame(positions[date])

            st.dataframe(
                position_at_date.loc[
                    :, ~position_at_date.columns.isin(["cash", "today_account_value"])
                ].T.style.background_gradient("Reds")
            )
        except (ValueError, KeyError):
            st.write("This is not a valid date")

    generator = (value.keys() for key, value in positions.items())
    stocks = list(set(itertools.chain.from_iterable(generator)))
    stocks.sort()

    with col2:
        selected_stock = st.selectbox("Please select a stock", stocks)
        try:
            weight = pd.Series(
                (
                    positions[d].get(selected_stock, {"weight": 0.0})["weight"]
                    for d in dates
                ),
                index=pd.DatetimeIndex(dates),
                name="weight",
                dtype="float",
            )

            st.line_chart(weight)
        except TypeError:
            st.write("This is not a valid stock code")


report_normal_df = loaded_artifacts.get("report_normal_df")
analysis_df = loaded_artifacts.get("analysis_df")

if report_normal_df is not None:
    for fig in analysis_position.report_graph(report_normal_df, show_notebook=False):
        st.plotly_chart(fig, use_container_width=True)
    if analysis_df is not None:
        for fig in analysis_position.risk_analysis_graph(
            analysis_df, report_normal_df, show_notebook=False
        ):
            st.plotly_chart(fig, use_container_width=True)

pred_label = loaded_artifacts.get("pred_label")

if pred_label is not None:
    for fig in analysis_position.score_ic_graph(pred_label, show_notebook=False):
        st.plotly_chart(fig, use_container_width=True)
    for fig in analysis_model.model_performance_graph(pred_label, show_notebook=False):
        st.plotly_chart(fig, use_container_width=True)
