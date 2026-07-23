
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
import SymbolDraw from '../helper/SymbolDraw.js';
import LineDraw from '../helper/LineDraw.js';
import RoamController from '../../component/helper/RoamController.js';
import * as roamHelper from '../../component/helper/roamHelper.js';
import { onIrrelevantElement } from '../../component/helper/cursorHelper.js';
import * as graphic from '../../util/graphic.js';
import adjustEdge from './adjustEdge.js';
import { getNodeGlobalScale } from './graphHelper.js';
import ChartView from '../../view/Chart.js';
import { getECData } from '../../util/innerStore.js';
import { simpleLayoutEdge } from './simpleLayoutHelper.js';
import { circularLayout, rotateNodeLabel } from './circularLayoutHelper.js';
function isViewCoordSys(coordSys) {
  return coordSys.type === 'view';
}
var GraphView = /** @class */function (_super) {
  __extends(GraphView, _super);
  function GraphView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = GraphView.type;
    return _this;
  }
  GraphView.prototype.init = function (ecModel, api) {
    var symbolDraw = new SymbolDraw();
    var lineDraw = new LineDraw();
    var group = this.group;
    this._controller = new RoamController(api.getZr());
    this._controllerHost = {
      target: group
    };
    group.add(symbolDraw.group);
    group.add(lineDraw.group);
    this._symbolDraw = symbolDraw;
    this._lineDraw = lineDraw;
    this._firstRender = true;
  };
  GraphView.prototype.render = function (seriesModel, ecModel, api) {
    var _this = this;
    var coordSys = seriesModel.coordinateSystem;
    this._model = seriesModel;
    var symbolDraw = this._symbolDraw;
    var lineDraw = this._lineDraw;
    var group = this.group;
    if (isViewCoordSys(coordSys)) {
      var groupNewProp = {
        x: coordSys.x,
        y: coordSys.y,
        scaleX: coordSys.scaleX,
        scaleY: coordSys.scaleY
      };
      if (this._firstRender) {
        group.attr(groupNewProp);
      } else {
        graphic.updateProps(group, groupNewProp, seriesModel);
      }
    }
    // Fix edge contact point with node
    adjustEdge(seriesModel.getGraph(), getNodeGlobalScale(seriesModel));
    var data = seriesModel.getData();
    symbolDraw.updateData(data);
    var edgeData = seriesModel.getEdgeData();
    // TODO: TYPE
    lineDraw.updateData(edgeData);
    this._updateNodeAndLinkScale();
    this._updateController(seriesModel, ecModel, api);
    clearTimeout(this._layoutTimeout);
    var forceLayout = seriesModel.forceLayout;
    var layoutAnimation = seriesModel.get(['force', 'layoutAnimation']);
    if (forceLayout) {
      this._startForceLayoutIteration(forceLayout, layoutAnimation);
    }
    var layout = seriesModel.get('layout');
    data.graph.eachNode(function (node) {
      var idx = node.dataIndex;
      var el = node.getGraphicEl();
      var itemModel = node.getModel();
      if (!el) {
        return;
      }
      // Update draggable
      el.off('drag').off('dragend');
      var draggable = itemModel.get('draggable');
      if (draggable) {
        el.on('drag', function (e) {
          switch (layout) {
            case 'force':
              forceLayout.warmUp();
              !_this._layouting && _this._startForceLayoutIteration(forceLayout, layoutAnimation);
              forceLayout.setFixed(idx);
              // Write position back to layout
              data.setItemLayout(idx, [el.x, el.y]);
              break;
            case 'circular':
              data.setItemLayout(idx, [el.x, el.y]);
              // mark node fixed
              node.setLayout({
                fixed: true
              }, true);
              // recalculate circular layout
              circularLayout(seriesModel, 'symbolSize', node, [e.offsetX, e.offsetY]);
              _this.updateLayout(seriesModel);
              break;
            case 'none':
            default:
              data.setItemLayout(idx, [el.x, el.y]);
              // update edge
              simpleLayoutEdge(seriesModel.getGraph(), seriesModel);
              _this.updateLayout(seriesModel);
              break;
          }
        }).on('dragend', function () {
          if (forceLayout) {
            forceLayout.setUnfixed(idx);
          }
        });
      }
      el.setDraggable(draggable, !!itemModel.get('cursor'));
      var focus = itemModel.get(['emphasis', 'focus']);
      if (focus === 'adjacency') {
        getECData(el).focus = node.getAdjacentDataIndices();
      }
    });
    data.graph.eachEdge(function (edge) {
      var el = edge.getGraphicEl();
      var focus = edge.getModel().get(['emphasis', 'focus']);
      if (!el) {
        return;
      }
      if (focus === 'adjacency') {
        getECData(el).focus = {
          edge: [edge.dataIndex],
          node: [edge.node1.dataIndex, edge.node2.dataIndex]
        };
      }
    });
    var circularRotateLabel = seriesModel.get('layout') === 'circular' && seriesModel.get(['circular', 'rotateLabel']);
    var cx = data.getLayout('cx');
    var cy = data.getLayout('cy');
    data.graph.eachNode(function (node) {
      rotateNodeLabel(node, circularRotateLabel, cx, cy);
    });
    this._firstRender = false;
  };
  GraphView.prototype.dispose = function () {
    this.remove();
    this._controller && this._controller.dispose();
    this._controllerHost = null;
  };
  GraphView.prototype._startForceLayoutIteration = function (forceLayout, layoutAnimation) {
    var self = this;
    (function step() {
      forceLayout.step(function (stopped) {
        self.updateLayout(self._model);
        (self._layouting = !stopped) && (layoutAnimation ? self._layoutTimeout = setTimeout(step, 16) : step());
      });
    })();
  };
  GraphView.prototype._updateController = function (seriesModel, ecModel, api) {
    var _this = this;
    var controller = this._controller;
    var controllerHost = this._controllerHost;
    var group = this.group;
    controller.setPointerChecker(function (e, x, y) {
      var rect = group.getBoundingRect();
      rect.applyTransform(group.transform);
      return rect.contain(x, y) && !onIrrelevantElement(e, api, seriesModel);
    });
    if (!isViewCoordSys(seriesModel.coordinateSystem)) {
      controller.disable();
      return;
    }
    controller.enable(seriesModel.get('roam'));
    controllerHost.zoomLimit = seriesModel.get('scaleLimit');
    controllerHost.zoom = seriesModel.coordinateSystem.getZoom();
    controller.off('pan').off('zoom').on('pan', function (e) {
      roamHelper.updateViewOnPan(controllerHost, e.dx, e.dy);
      api.dispatchAction({
        seriesId: seriesModel.id,
        type: 'graphRoam',
        dx: e.dx,
        dy: e.dy
      });
    }).on('zoom', function (e) {
      roamHelper.updateViewOnZoom(controllerHost, e.scale, e.originX, e.originY);
      api.dispatchAction({
        seriesId: seriesModel.id,
        type: 'graphRoam',
        zoom: e.scale,
        originX: e.originX,
        originY: e.originY
      });
      _this._updateNodeAndLinkScale();
      adjustEdge(seriesModel.getGraph(), getNodeGlobalScale(seriesModel));
      _this._lineDraw.updateLayout();
      // Only update label layout on zoom
      api.updateLabelLayout();
    });
  };
  GraphView.prototype._updateNodeAndLinkScale = function () {
    var seriesModel = this._model;
    var data = seriesModel.getData();
    var nodeScale = getNodeGlobalScale(seriesModel);
    data.eachItemGraphicEl(function (el, idx) {
      el && el.setSymbolScale(nodeScale);
    });
  };
  GraphView.prototype.updateLayout = function (seriesModel) {
    adjustEdge(seriesModel.getGraph(), getNodeGlobalScale(seriesModel));
    this._symbolDraw.updateLayout();
    this._lineDraw.updateLayout();
  };
  GraphView.prototype.remove = function () {
    clearTimeout(this._layoutTimeout);
    this._layouting = false;
    this._layoutTimeout = null;
    this._symbolDraw && this._symbolDraw.remove();
    this._lineDraw && this._lineDraw.remove();
  };
  GraphView.type = 'graph';
  return GraphView;
}(ChartView);
export default GraphView;