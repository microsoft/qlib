
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
import SeriesModel from '../../model/Series.js';
import createGraphFromNodeEdge from '../helper/createGraphFromNodeEdge.js';
import Model from '../../model/Model.js';
import { createTooltipMarkup } from '../../component/tooltip/tooltipMarkup.js';
var SankeySeriesModel = /** @class */function (_super) {
  __extends(SankeySeriesModel, _super);
  function SankeySeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SankeySeriesModel.type;
    return _this;
  }
  /**
   * Init a graph data structure from data in option series
   */
  SankeySeriesModel.prototype.getInitialData = function (option, ecModel) {
    var links = option.edges || option.links || [];
    var nodes = option.data || option.nodes || [];
    var levels = option.levels || [];
    this.levelModels = [];
    var levelModels = this.levelModels;
    for (var i = 0; i < levels.length; i++) {
      if (levels[i].depth != null && levels[i].depth >= 0) {
        levelModels[levels[i].depth] = new Model(levels[i], this, ecModel);
      } else {
        if (process.env.NODE_ENV !== 'production') {
          throw new Error('levels[i].depth is mandatory and should be natural number');
        }
      }
    }
    var graph = createGraphFromNodeEdge(nodes, links, this, true, beforeLink);
    return graph.data;
    function beforeLink(nodeData, edgeData) {
      nodeData.wrapMethod('getItemModel', function (model, idx) {
        var seriesModel = model.parentModel;
        var layout = seriesModel.getData().getItemLayout(idx);
        if (layout) {
          var nodeDepth = layout.depth;
          var levelModel = seriesModel.levelModels[nodeDepth];
          if (levelModel) {
            model.parentModel = levelModel;
          }
        }
        return model;
      });
      edgeData.wrapMethod('getItemModel', function (model, idx) {
        var seriesModel = model.parentModel;
        var edge = seriesModel.getGraph().getEdgeByIndex(idx);
        var layout = edge.node1.getLayout();
        if (layout) {
          var depth = layout.depth;
          var levelModel = seriesModel.levelModels[depth];
          if (levelModel) {
            model.parentModel = levelModel;
          }
        }
        return model;
      });
    }
  };
  SankeySeriesModel.prototype.setNodePosition = function (dataIndex, localPosition) {
    var nodes = this.option.data || this.option.nodes;
    var dataItem = nodes[dataIndex];
    dataItem.localX = localPosition[0];
    dataItem.localY = localPosition[1];
  };
  /**
   * Return the graphic data structure
   *
   * @return graphic data structure
   */
  SankeySeriesModel.prototype.getGraph = function () {
    return this.getData().graph;
  };
  /**
   * Get edge data of graphic data structure
   *
   * @return data structure of list
   */
  SankeySeriesModel.prototype.getEdgeData = function () {
    return this.getGraph().edgeData;
  };
  SankeySeriesModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    function noValue(val) {
      return isNaN(val) || val == null;
    }
    // dataType === 'node' or empty do not show tooltip by default
    if (dataType === 'edge') {
      var params = this.getDataParams(dataIndex, dataType);
      var rawDataOpt = params.data;
      var edgeValue = params.value;
      var edgeName = rawDataOpt.source + ' -- ' + rawDataOpt.target;
      return createTooltipMarkup('nameValue', {
        name: edgeName,
        value: edgeValue,
        noValue: noValue(edgeValue)
      });
    }
    // dataType === 'node'
    else {
      var node = this.getGraph().getNodeByIndex(dataIndex);
      var value = node.getLayout().value;
      var name_1 = this.getDataParams(dataIndex, dataType).data.name;
      return createTooltipMarkup('nameValue', {
        name: name_1 != null ? name_1 + '' : null,
        value: value,
        noValue: noValue(value)
      });
    }
  };
  SankeySeriesModel.prototype.optionUpdated = function () {};
  // Override Series.getDataParams()
  SankeySeriesModel.prototype.getDataParams = function (dataIndex, dataType) {
    var params = _super.prototype.getDataParams.call(this, dataIndex, dataType);
    if (params.value == null && dataType === 'node') {
      var node = this.getGraph().getNodeByIndex(dataIndex);
      var nodeValue = node.getLayout().value;
      params.value = nodeValue;
    }
    return params;
  };
  SankeySeriesModel.type = 'series.sankey';
  SankeySeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    coordinateSystem: 'view',
    left: '5%',
    top: '5%',
    right: '20%',
    bottom: '5%',
    orient: 'horizontal',
    nodeWidth: 20,
    nodeGap: 8,
    draggable: true,
    layoutIterations: 32,
    label: {
      show: true,
      position: 'right',
      fontSize: 12
    },
    edgeLabel: {
      show: false,
      fontSize: 12
    },
    levels: [],
    nodeAlign: 'justify',
    lineStyle: {
      color: '#314656',
      opacity: 0.2,
      curveness: 0.5
    },
    emphasis: {
      label: {
        show: true
      },
      lineStyle: {
        opacity: 0.5
      }
    },
    select: {
      itemStyle: {
        borderColor: '#212121'
      }
    },
    animationEasing: 'linear',
    animationDuration: 1000
  };
  return SankeySeriesModel;
}(SeriesModel);
export default SankeySeriesModel;