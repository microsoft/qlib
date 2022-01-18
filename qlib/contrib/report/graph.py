# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import importlib
from typing import Iterable

import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go

from plotly.subplots import make_subplots
from plotly.figure_factory import create_distplot


class BaseGraph:

    _name = None

    def __init__(
        self, df: pd.DataFrame = None, layout: dict = None, graph_kwargs: dict = None, name_dict: dict = None, **kwargs
    ):
        """

        :param df:
        :param layout:
        :param graph_kwargs:
        :param name_dict:
        :param kwargs:
            layout: dict
                go.Layout parameters
            graph_kwargs: dict
                Graph parameters, eg: go.Bar(**graph_kwargs)
        """
        self._df = df

        self._layout = dict() if layout is None else layout
        self._graph_kwargs = dict() if graph_kwargs is None else graph_kwargs
        self._name_dict = name_dict

        self.data = None

        self._init_parameters(**kwargs)
        self._init_data()

    def _init_data(self):
        """

        :return:
        """
        if self._df.empty:
            raise ValueError("df is empty.")

        self.data = self._get_data()

    def _init_parameters(self, **kwargs):
        """

        :param kwargs
        """

        # Instantiate graphics parameters
        self._graph_type = self._name.lower().capitalize()

        # Displayed column name
        if self._name_dict is None:
            self._name_dict = {_item: _item for _item in self._df.columns}

    @staticmethod
    def get_instance_with_graph_parameters(graph_type: str = None, **kwargs):
        """

        :param graph_type:
        :param kwargs:
        :return:
        """
        try:
            _graph_module = importlib.import_module("plotly.graph_objs")
            _graph_class = getattr(_graph_module, graph_type)
        except AttributeError:
            _graph_module = importlib.import_module("qlib.contrib.report.graph")
            _graph_class = getattr(_graph_module, graph_type)
        return _graph_class(**kwargs)

    @staticmethod
    def show_graph_in_notebook(figure_list: Iterable[go.Figure] = None):
        """

        :param figure_list:
        :return:
        """
        py.init_notebook_mode()
        for _fig in figure_list:
            # NOTE: displays figures: https://plotly.com/python/renderers/
            # default: plotly_mimetype+notebook
            # support renderers: import plotly.io as pio; print(pio.renderers)
            renderer = None
            try:
                # in notebook
                _ipykernel = str(type(get_ipython()))
                if "google.colab" in _ipykernel:
                    renderer = "colab"
            except NameError:
                pass

            _fig.show(renderer=renderer)

    def _get_layout(self) -> go.Layout:
        """

        :return:
        """
        return go.Layout(**self._layout)

    def _get_data(self) -> list:
        """

        :return:
        """

        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df.index, y=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data

    @property
    def figure(self) -> go.Figure:
        """

        :return:
        """
        _figure = go.Figure(data=self.data, layout=self._get_layout())
        # NOTE: Use the default theme from plotly version 3.x, template=None
        _figure["layout"].update(template=None)
        return _figure


class ScatterGraph(BaseGraph):
    _name = "scatter"


class BarGraph(BaseGraph):
    _name = "bar"


class DistplotGraph(BaseGraph):
    _name = "distplot"

    def _get_data(self):
        """

        :return:
        """
        _t_df = self._df.dropna()
        _data_list = [_t_df[_col] for _col in self._name_dict]
        _label_list = list(self._name_dict.values())
        _fig = create_distplot(_data_list, _label_list, show_rug=False, **self._graph_kwargs)

        return _fig["data"]


class HeatmapGraph(BaseGraph):
    _name = "heatmap"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type,
                x=self._df.columns,
                y=self._df.index,
                z=self._df.values.tolist(),
                **self._graph_kwargs
            )
        ]
        return _data


class HistogramGraph(BaseGraph):
    _name = "histogram"

    def _get_data(self):
        """

        :return:
        """
        _data = [
            self.get_instance_with_graph_parameters(
                graph_type=self._graph_type, x=self._df[_col], name=_name, **self._graph_kwargs
            )
            for _col, _name in self._name_dict.items()
        ]
        return _data


class SubplotsGraph:
    """Create subplots same as df.plot(subplots=True)

    Simple package for `plotly.tools.subplots`
    """

    def __init__(
        self,
        df: pd.DataFrame = None,
        kind_map: dict = None,
        layout: dict = None,
        sub_graph_layout: dict = None,
        sub_graph_data: list = None,
        subplots_kwargs: dict = None,
        **kwargs
    ):
        """

        :param df: pd.DataFrame

        :param kind_map: dict, subplots graph kind and kwargs
            eg: dict(kind='ScatterGraph', kwargs=dict())

        :param layout: `go.Layout` parameters

        :param sub_graph_layout: Layout of each graphic, similar to 'layout'

        :param sub_graph_data: Instantiation parameters for each sub-graphic
            eg: [(column_name, instance_parameters), ]

            column_name: str or go.Figure

            Instance_parameters:

                - row: int, the row where the graph is located

                - col: int, the col where the graph is located

                - name: str, show name, default column_name in 'df'

                - kind: str, graph kind, default `kind` param, eg: bar, scatter, ...

                - graph_kwargs: dict, graph kwargs, default {}, used in `go.Bar(**graph_kwargs)`

        :param subplots_kwargs: `plotly.tools.make_subplots` original parameters

                - shared_xaxes: bool, default False

                - shared_yaxes: bool, default False

                - vertical_spacing: float, default 0.3 / rows

                - subplot_titles: list, default []
                    If `sub_graph_data` is None, will generate 'subplot_titles' according to `df.columns`,
                    this field will be discarded


                - specs: list, see `make_subplots` docs

                - rows: int, Number of rows in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'rows' according to `df`, this field will be discarded

                - cols: int, Number of cols in the subplot grid, default 1
                    If `sub_graph_data` is None, will generate 'cols' according to `df`, this field will be discarded


        :param kwargs:

        """

        self._df = df
        self._layout = layout
        self._sub_graph_layout = sub_graph_layout

        self._kind_map = kind_map
        if self._kind_map is None:
            self._kind_map = dict(kind="ScatterGraph", kwargs=dict())

        self._subplots_kwargs = subplots_kwargs
        if self._subplots_kwargs is None:
            self._init_subplots_kwargs()

        self.__cols = self._subplots_kwargs.get("cols", 2)
        self.__rows = self._subplots_kwargs.get("rows", math.ceil(len(self._df.columns) / self.__cols))

        self._sub_graph_data = sub_graph_data
        if self._sub_graph_data is None:
            self._init_sub_graph_data()

        self._init_figure()

    def _init_sub_graph_data(self):
        """

        :return:
        """
        self._sub_graph_data = []
        self._subplot_titles = []

        for i, column_name in enumerate(self._df.columns):
            row = math.ceil((i + 1) / self.__cols)
            _temp = (i + 1) % self.__cols
            col = _temp if _temp else self.__cols
            res_name = column_name.replace("_", " ")
            _temp_row_data = (
                column_name,
                dict(
                    row=row,
                    col=col,
                    name=res_name,
                    kind=self._kind_map["kind"],
                    graph_kwargs=self._kind_map["kwargs"],
                ),
            )
            self._sub_graph_data.append(_temp_row_data)
            self._subplot_titles.append(res_name)

    def _init_subplots_kwargs(self):
        """

        :return:
        """
        # Default cols, rows
        _cols = 2
        _rows = math.ceil(len(self._df.columns) / 2)
        self._subplots_kwargs = dict()
        self._subplots_kwargs["rows"] = _rows
        self._subplots_kwargs["cols"] = _cols
        self._subplots_kwargs["shared_xaxes"] = False
        self._subplots_kwargs["shared_yaxes"] = False
        self._subplots_kwargs["vertical_spacing"] = 0.3 / _rows
        self._subplots_kwargs["print_grid"] = False
        self._subplots_kwargs["subplot_titles"] = self._df.columns.tolist()

    def _init_figure(self):
        """

        :return:
        """
        self._figure = make_subplots(**self._subplots_kwargs)

        for column_name, column_map in self._sub_graph_data:
            if isinstance(column_name, go.Figure):
                _graph_obj = column_name
            elif isinstance(column_name, str):
                temp_name = column_map.get("name", column_name.replace("_", " "))
                kind = column_map.get("kind", self._kind_map.get("kind", "ScatterGraph"))
                _graph_kwargs = column_map.get("graph_kwargs", self._kind_map.get("kwargs", {}))
                _graph_obj = BaseGraph.get_instance_with_graph_parameters(
                    kind,
                    **dict(
                        df=self._df.loc[:, [column_name]],
                        name_dict={column_name: temp_name},
                        graph_kwargs=_graph_kwargs,
                    )
                )
            else:
                raise TypeError()

            row = column_map["row"]
            col = column_map["col"]

            _graph_data = getattr(_graph_obj, "data")
            # for _item in _graph_data:
            #     _item.pop('xaxis', None)
            #     _item.pop('yaxis', None)

            for _g_obj in _graph_data:
                self._figure.add_trace(_g_obj, row=row, col=col)

        if self._sub_graph_layout is not None:
            for k, v in self._sub_graph_layout.items():
                self._figure["layout"][k].update(v)

        # NOTE: Use the default theme from plotly version 3.x: template=None
        self._figure["layout"].update(template=None)
        self._figure["layout"].update(self._layout)

    @property
    def figure(self):
        return self._figure
