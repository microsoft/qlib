
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
import Tree from '../../data/Tree.js';
import Model from '../../model/Model.js';
import { createTooltipMarkup } from '../../component/tooltip/tooltipMarkup.js';
import { wrapTreePathInfo } from '../helper/treeHelper.js';
var TreeSeriesModel = /** @class */function (_super) {
  __extends(TreeSeriesModel, _super);
  function TreeSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.hasSymbolVisual = true;
    // Do it self.
    _this.ignoreStyleOnData = true;
    return _this;
  }
  /**
   * Init a tree data structure from data in option series
   */
  TreeSeriesModel.prototype.getInitialData = function (option) {
    // create a virtual root
    var root = {
      name: option.name,
      children: option.data
    };
    var leaves = option.leaves || {};
    var leavesModel = new Model(leaves, this, this.ecModel);
    var tree = Tree.createTree(root, this, beforeLink);
    function beforeLink(nodeData) {
      nodeData.wrapMethod('getItemModel', function (model, idx) {
        var node = tree.getNodeByDataIndex(idx);
        if (!(node && node.children.length && node.isExpand)) {
          model.parentModel = leavesModel;
        }
        return model;
      });
    }
    var treeDepth = 0;
    tree.eachNode('preorder', function (node) {
      if (node.depth > treeDepth) {
        treeDepth = node.depth;
      }
    });
    var expandAndCollapse = option.expandAndCollapse;
    var expandTreeDepth = expandAndCollapse && option.initialTreeDepth >= 0 ? option.initialTreeDepth : treeDepth;
    tree.root.eachNode('preorder', function (node) {
      var item = node.hostTree.data.getRawDataItem(node.dataIndex);
      // Add item.collapsed != null, because users can collapse node original in the series.data.
      node.isExpand = item && item.collapsed != null ? !item.collapsed : node.depth <= expandTreeDepth;
    });
    return tree.data;
  };
  /**
   * Make the configuration 'orient' backward compatibly, with 'horizontal = LR', 'vertical = TB'.
   * @returns {string} orient
   */
  TreeSeriesModel.prototype.getOrient = function () {
    var orient = this.get('orient');
    if (orient === 'horizontal') {
      orient = 'LR';
    } else if (orient === 'vertical') {
      orient = 'TB';
    }
    return orient;
  };
  TreeSeriesModel.prototype.setZoom = function (zoom) {
    this.option.zoom = zoom;
  };
  TreeSeriesModel.prototype.setCenter = function (center) {
    this.option.center = center;
  };
  TreeSeriesModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    var tree = this.getData().tree;
    var realRoot = tree.root.children[0];
    var node = tree.getNodeByDataIndex(dataIndex);
    var value = node.getValue();
    var name = node.name;
    while (node && node !== realRoot) {
      name = node.parentNode.name + '.' + name;
      node = node.parentNode;
    }
    return createTooltipMarkup('nameValue', {
      name: name,
      value: value,
      noValue: isNaN(value) || value == null
    });
  };
  // Add tree path to tooltip param
  TreeSeriesModel.prototype.getDataParams = function (dataIndex) {
    var params = _super.prototype.getDataParams.apply(this, arguments);
    var node = this.getData().tree.getNodeByDataIndex(dataIndex);
    params.treeAncestors = wrapTreePathInfo(node, this);
    params.collapsed = !node.isExpand;
    return params;
  };
  TreeSeriesModel.type = 'series.tree';
  // can support the position parameters 'left', 'top','right','bottom', 'width',
  // 'height' in the setOption() with 'merge' mode normal.
  TreeSeriesModel.layoutMode = 'box';
  TreeSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    coordinateSystem: 'view',
    // the position of the whole view
    left: '12%',
    top: '12%',
    right: '12%',
    bottom: '12%',
    // the layout of the tree, two value can be selected, 'orthogonal' or 'radial'
    layout: 'orthogonal',
    // value can be 'polyline'
    edgeShape: 'curve',
    edgeForkPosition: '50%',
    // true | false | 'move' | 'scale', see module:component/helper/RoamController.
    roam: false,
    // Symbol size scale ratio in roam
    nodeScaleRatio: 0.4,
    // Default on center of graph
    center: null,
    zoom: 1,
    orient: 'LR',
    symbol: 'emptyCircle',
    symbolSize: 7,
    expandAndCollapse: true,
    initialTreeDepth: 2,
    lineStyle: {
      color: '#ccc',
      width: 1.5,
      curveness: 0.5
    },
    itemStyle: {
      color: 'lightsteelblue',
      // borderColor: '#c23531',
      borderWidth: 1.5
    },
    label: {
      show: true
    },
    animationEasing: 'linear',
    animationDuration: 700,
    animationDurationUpdate: 500
  };
  return TreeSeriesModel;
}(SeriesModel);
export default TreeSeriesModel;