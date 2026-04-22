
/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/


/**
 * AUTO-GENERATED FILE. DO NOT MODIFY.
 */

/*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
import { __extends } from "tslib";
import SeriesData from '../../data/SeriesData.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import { defaultEmphasis } from '../../util/model.js';
import Model from '../../model/Model.js';
import createGraphFromNodeEdge from '../helper/createGraphFromNodeEdge.js';
import LegendVisualProvider from '../../visual/LegendVisualProvider.js';
import SeriesModel from '../../model/Series.js';
import { createTooltipMarkup } from '../../component/tooltip/tooltipMarkup.js';
import { defaultSeriesFormatTooltip } from '../../component/tooltip/seriesFormatTooltip.js';
import { initCurvenessList, createEdgeMapForCurveness } from '../helper/multipleGraphEdgeHelper.js';
var GraphSeriesModel = /** @class */function (_super) {
  __extends(GraphSeriesModel, _super);
  function GraphSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = GraphSeriesModel.type;
    _this.hasSymbolVisual = true;
    return _this;
  }
  GraphSeriesModel.prototype.init = function (option) {
    _super.prototype.init.apply(this, arguments);
    var self = this;
    function getCategoriesData() {
      return self._categoriesData;
    }
    // Provide data for legend select
    this.legendVisualProvider = new LegendVisualProvider(getCategoriesData, getCategoriesData);
    this.fillDataTextStyle(option.edges || option.links);
    this._updateCategoriesData();
  };
  GraphSeriesModel.prototype.mergeOption = function (option) {
    _super.prototype.mergeOption.apply(this, arguments);
    this.fillDataTextStyle(option.edges || option.links);
    this._updateCategoriesData();
  };
  GraphSeriesModel.prototype.mergeDefaultAndTheme = function (option) {
    _super.prototype.mergeDefaultAndTheme.apply(this, arguments);
    defaultEmphasis(option, 'edgeLabel', ['show']);
  };
  GraphSeriesModel.prototype.getInitialData = function (option, ecModel) {
    var edges = option.edges || option.links || [];
    var nodes = option.data || option.nodes || [];
    var self = this;
    if (nodes && edges) {
      // auto curveness
      initCurvenessList(this);
      var graph = createGraphFromNodeEdge(nodes, edges, this, true, beforeLink);
      zrUtil.each(graph.edges, function (edge) {
        createEdgeMapForCurveness(edge.node1, edge.node2, this, edge.dataIndex);
      }, this);
      return graph.data;
    }
    function beforeLink(nodeData, edgeData) {
      // Overwrite nodeData.getItemModel to
      nodeData.wrapMethod('getItemModel', function (model) {
        var categoriesModels = self._categoriesModels;
        var categoryIdx = model.getShallow('category');
        var categoryModel = categoriesModels[categoryIdx];
        if (categoryModel) {
          categoryModel.parentModel = model.parentModel;
          model.parentModel = categoryModel;
        }
        return model;
      });
      // TODO Inherit resolveParentPath by default in Model#getModel?
      var oldGetModel = Model.prototype.getModel;
      function newGetModel(path, parentModel) {
        var model = oldGetModel.call(this, path, parentModel);
        model.resolveParentPath = resolveParentPath;
        return model;
      }
      edgeData.wrapMethod('getItemModel', function (model) {
        model.resolveParentPath = resolveParentPath;
        model.getModel = newGetModel;
        return model;
      });
      function resolveParentPath(pathArr) {
        if (pathArr && (pathArr[0] === 'label' || pathArr[1] === 'label')) {
          var newPathArr = pathArr.slice();
          if (pathArr[0] === 'label') {
            newPathArr[0] = 'edgeLabel';
          } else if (pathArr[1] === 'label') {
            newPathArr[1] = 'edgeLabel';
          }
          return newPathArr;
        }
        return pathArr;
      }
    }
  };
  GraphSeriesModel.prototype.getGraph = function () {
    return this.getData().graph;
  };
  GraphSeriesModel.prototype.getEdgeData = function () {
    return this.getGraph().edgeData;
  };
  GraphSeriesModel.prototype.getCategoriesData = function () {
    return this._categoriesData;
  };
  GraphSeriesModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    if (dataType === 'edge') {
      var nodeData = this.getData();
      var params = this.getDataParams(dataIndex, dataType);
      var edge = nodeData.graph.getEdgeByIndex(dataIndex);
      var sourceName = nodeData.getName(edge.node1.dataIndex);
      var targetName = nodeData.getName(edge.node2.dataIndex);
      var nameArr = [];
      sourceName != null && nameArr.push(sourceName);
      targetName != null && nameArr.push(targetName);
      return createTooltipMarkup('nameValue', {
        name: nameArr.join(' > '),
        value: params.value,
        noValue: params.value == null
      });
    }
    // dataType === 'node' or empty
    var nodeMarkup = defaultSeriesFormatTooltip({
      series: this,
      dataIndex: dataIndex,
      multipleSeries: multipleSeries
    });
    return nodeMarkup;
  };
  GraphSeriesModel.prototype._updateCategoriesData = function () {
    var categories = zrUtil.map(this.option.categories || [], function (category) {
      // Data must has value
      return category.value != null ? category : zrUtil.extend({
        value: 0
      }, category);
    });
    var categoriesData = new SeriesData(['value'], this);
    categoriesData.initData(categories);
    this._categoriesData = categoriesData;
    this._categoriesModels = categoriesData.mapArray(function (idx) {
      return categoriesData.getItemModel(idx);
    });
  };
  GraphSeriesModel.prototype.setZoom = function (zoom) {
    this.option.zoom = zoom;
  };
  GraphSeriesModel.prototype.setCenter = function (center) {
    this.option.center = center;
  };
  GraphSeriesModel.prototype.isAnimationEnabled = function () {
    return _super.prototype.isAnimationEnabled.call(this)
    // Not enable animation when do force layout
    && !(this.get('layout') === 'force' && this.get(['force', 'layoutAnimation']));
  };
  GraphSeriesModel.type = 'series.graph';
  GraphSeriesModel.dependencies = ['grid', 'polar', 'geo', 'singleAxis', 'calendar'];
  GraphSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    coordinateSystem: 'view',
    // Default option for all coordinate systems
    // xAxisIndex: 0,
    // yAxisIndex: 0,
    // polarIndex: 0,
    // geoIndex: 0,
    legendHoverLink: true,
    layout: null,
    // Configuration of circular layout
    circular: {
      rotateLabel: false
    },
    // Configuration of force directed layout
    force: {
      initLayout: null,
      // Node repulsion. Can be an array to represent range.
      repulsion: [0, 50],
      gravity: 0.1,
      // Initial friction
      friction: 0.6,
      // Edge length. Can be an array to represent range.
      edgeLength: 30,
      layoutAnimation: true
    },
    left: 'center',
    top: 'center',
    // right: null,
    // bottom: null,
    // width: '80%',
    // height: '80%',
    symbol: 'circle',
    symbolSize: 10,
    edgeSymbol: ['none', 'none'],
    edgeSymbolSize: 10,
    edgeLabel: {
      position: 'middle',
      distance: 5
    },
    draggable: false,
    roam: false,
    // Default on center of graph
    center: null,
    zoom: 1,
    // Symbol size scale ratio in roam
    nodeScaleRatio: 0.6,
    // cursor: null,
    // categories: [],
    // data: []
    // Or
    // nodes: []
    //
    // links: []
    // Or
    // edges: []
    label: {
      show: false,
      formatter: '{b}'
    },
    itemStyle: {},
    lineStyle: {
      color: '#aaa',
      width: 1,
      opacity: 0.5
    },
    emphasis: {
      scale: true,
      label: {
        show: true
      }
    },
    select: {
      itemStyle: {
        borderColor: '#212121'
      }
    }
  };
  return GraphSeriesModel;
}(SeriesModel);
export default GraphSeriesModel;