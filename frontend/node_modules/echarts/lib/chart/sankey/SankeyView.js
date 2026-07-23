
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
import * as graphic from '../../util/graphic.js';
import { enterEmphasis, leaveEmphasis, toggleHoverEmphasis, setStatesStylesFromModel } from '../../util/states.js';
import ChartView from '../../view/Chart.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import { getECData } from '../../util/innerStore.js';
import { isString, retrieve3 } from 'zrender/lib/core/util.js';
var SankeyPathShape = /** @class */function () {
  function SankeyPathShape() {
    this.x1 = 0;
    this.y1 = 0;
    this.x2 = 0;
    this.y2 = 0;
    this.cpx1 = 0;
    this.cpy1 = 0;
    this.cpx2 = 0;
    this.cpy2 = 0;
    this.extent = 0;
  }
  return SankeyPathShape;
}();
var SankeyPath = /** @class */function (_super) {
  __extends(SankeyPath, _super);
  function SankeyPath(opts) {
    return _super.call(this, opts) || this;
  }
  SankeyPath.prototype.getDefaultShape = function () {
    return new SankeyPathShape();
  };
  SankeyPath.prototype.buildPath = function (ctx, shape) {
    var extent = shape.extent;
    ctx.moveTo(shape.x1, shape.y1);
    ctx.bezierCurveTo(shape.cpx1, shape.cpy1, shape.cpx2, shape.cpy2, shape.x2, shape.y2);
    if (shape.orient === 'vertical') {
      ctx.lineTo(shape.x2 + extent, shape.y2);
      ctx.bezierCurveTo(shape.cpx2 + extent, shape.cpy2, shape.cpx1 + extent, shape.cpy1, shape.x1 + extent, shape.y1);
    } else {
      ctx.lineTo(shape.x2, shape.y2 + extent);
      ctx.bezierCurveTo(shape.cpx2, shape.cpy2 + extent, shape.cpx1, shape.cpy1 + extent, shape.x1, shape.y1 + extent);
    }
    ctx.closePath();
  };
  SankeyPath.prototype.highlight = function () {
    enterEmphasis(this);
  };
  SankeyPath.prototype.downplay = function () {
    leaveEmphasis(this);
  };
  return SankeyPath;
}(graphic.Path);
var SankeyView = /** @class */function (_super) {
  __extends(SankeyView, _super);
  function SankeyView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SankeyView.type;
    _this._focusAdjacencyDisabled = false;
    return _this;
  }
  SankeyView.prototype.render = function (seriesModel, ecModel, api) {
    var sankeyView = this;
    var graph = seriesModel.getGraph();
    var group = this.group;
    var layoutInfo = seriesModel.layoutInfo;
    // view width
    var width = layoutInfo.width;
    // view height
    var height = layoutInfo.height;
    var nodeData = seriesModel.getData();
    var edgeData = seriesModel.getData('edge');
    var orient = seriesModel.get('orient');
    this._model = seriesModel;
    group.removeAll();
    group.x = layoutInfo.x;
    group.y = layoutInfo.y;
    // generate a bezire Curve for each edge
    graph.eachEdge(function (edge) {
      var curve = new SankeyPath();
      var ecData = getECData(curve);
      ecData.dataIndex = edge.dataIndex;
      ecData.seriesIndex = seriesModel.seriesIndex;
      ecData.dataType = 'edge';
      var edgeModel = edge.getModel();
      var lineStyleModel = edgeModel.getModel('lineStyle');
      var curvature = lineStyleModel.get('curveness');
      var n1Layout = edge.node1.getLayout();
      var node1Model = edge.node1.getModel();
      var dragX1 = node1Model.get('localX');
      var dragY1 = node1Model.get('localY');
      var n2Layout = edge.node2.getLayout();
      var node2Model = edge.node2.getModel();
      var dragX2 = node2Model.get('localX');
      var dragY2 = node2Model.get('localY');
      var edgeLayout = edge.getLayout();
      var x1;
      var y1;
      var x2;
      var y2;
      var cpx1;
      var cpy1;
      var cpx2;
      var cpy2;
      curve.shape.extent = Math.max(1, edgeLayout.dy);
      curve.shape.orient = orient;
      if (orient === 'vertical') {
        x1 = (dragX1 != null ? dragX1 * width : n1Layout.x) + edgeLayout.sy;
        y1 = (dragY1 != null ? dragY1 * height : n1Layout.y) + n1Layout.dy;
        x2 = (dragX2 != null ? dragX2 * width : n2Layout.x) + edgeLayout.ty;
        y2 = dragY2 != null ? dragY2 * height : n2Layout.y;
        cpx1 = x1;
        cpy1 = y1 * (1 - curvature) + y2 * curvature;
        cpx2 = x2;
        cpy2 = y1 * curvature + y2 * (1 - curvature);
      } else {
        x1 = (dragX1 != null ? dragX1 * width : n1Layout.x) + n1Layout.dx;
        y1 = (dragY1 != null ? dragY1 * height : n1Layout.y) + edgeLayout.sy;
        x2 = dragX2 != null ? dragX2 * width : n2Layout.x;
        y2 = (dragY2 != null ? dragY2 * height : n2Layout.y) + edgeLayout.ty;
        cpx1 = x1 * (1 - curvature) + x2 * curvature;
        cpy1 = y1;
        cpx2 = x1 * curvature + x2 * (1 - curvature);
        cpy2 = y2;
      }
      curve.setShape({
        x1: x1,
        y1: y1,
        x2: x2,
        y2: y2,
        cpx1: cpx1,
        cpy1: cpy1,
        cpx2: cpx2,
        cpy2: cpy2
      });
      curve.useStyle(lineStyleModel.getItemStyle());
      // Special color, use source node color or target node color
      applyCurveStyle(curve.style, orient, edge);
      var defaultEdgeLabelText = "" + edgeModel.get('value');
      var edgeLabelStateModels = getLabelStatesModels(edgeModel, 'edgeLabel');
      setLabelStyle(curve, edgeLabelStateModels, {
        labelFetcher: {
          getFormattedLabel: function (dataIndex, stateName, dataType, labelDimIndex, formatter, extendParams) {
            return seriesModel.getFormattedLabel(dataIndex, stateName, 'edge', labelDimIndex,
            // ensure edgeLabel formatter is provided
            // to prevent the inheritance from `label.formatter` of the series
            retrieve3(formatter, edgeLabelStateModels.normal && edgeLabelStateModels.normal.get('formatter'), defaultEdgeLabelText), extendParams);
          }
        },
        labelDataIndex: edge.dataIndex,
        defaultText: defaultEdgeLabelText
      });
      curve.setTextConfig({
        position: 'inside'
      });
      var emphasisModel = edgeModel.getModel('emphasis');
      setStatesStylesFromModel(curve, edgeModel, 'lineStyle', function (model) {
        var style = model.getItemStyle();
        applyCurveStyle(style, orient, edge);
        return style;
      });
      group.add(curve);
      edgeData.setItemGraphicEl(edge.dataIndex, curve);
      var focus = emphasisModel.get('focus');
      toggleHoverEmphasis(curve, focus === 'adjacency' ? edge.getAdjacentDataIndices() : focus === 'trajectory' ? edge.getTrajectoryDataIndices() : focus, emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
    });
    // Generate a rect for each node
    graph.eachNode(function (node) {
      var layout = node.getLayout();
      var itemModel = node.getModel();
      var dragX = itemModel.get('localX');
      var dragY = itemModel.get('localY');
      var emphasisModel = itemModel.getModel('emphasis');
      var borderRadius = itemModel.get(['itemStyle', 'borderRadius']) || 0;
      var rect = new graphic.Rect({
        shape: {
          x: dragX != null ? dragX * width : layout.x,
          y: dragY != null ? dragY * height : layout.y,
          width: layout.dx,
          height: layout.dy,
          r: borderRadius
        },
        style: itemModel.getModel('itemStyle').getItemStyle(),
        z2: 10
      });
      setLabelStyle(rect, getLabelStatesModels(itemModel), {
        labelFetcher: {
          getFormattedLabel: function (dataIndex, stateName) {
            return seriesModel.getFormattedLabel(dataIndex, stateName, 'node');
          }
        },
        labelDataIndex: node.dataIndex,
        defaultText: node.id
      });
      rect.disableLabelAnimation = true;
      rect.setStyle('fill', node.getVisual('color'));
      rect.setStyle('decal', node.getVisual('style').decal);
      setStatesStylesFromModel(rect, itemModel);
      group.add(rect);
      nodeData.setItemGraphicEl(node.dataIndex, rect);
      getECData(rect).dataType = 'node';
      var focus = emphasisModel.get('focus');
      toggleHoverEmphasis(rect, focus === 'adjacency' ? node.getAdjacentDataIndices() : focus === 'trajectory' ? node.getTrajectoryDataIndices() : focus, emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
    });
    nodeData.eachItemGraphicEl(function (el, dataIndex) {
      var itemModel = nodeData.getItemModel(dataIndex);
      if (itemModel.get('draggable')) {
        el.drift = function (dx, dy) {
          sankeyView._focusAdjacencyDisabled = true;
          this.shape.x += dx;
          this.shape.y += dy;
          this.dirty();
          api.dispatchAction({
            type: 'dragNode',
            seriesId: seriesModel.id,
            dataIndex: nodeData.getRawIndex(dataIndex),
            localX: this.shape.x / width,
            localY: this.shape.y / height
          });
        };
        el.ondragend = function () {
          sankeyView._focusAdjacencyDisabled = false;
        };
        el.draggable = true;
        el.cursor = 'move';
      }
    });
    if (!this._data && seriesModel.isAnimationEnabled()) {
      group.setClipPath(createGridClipShape(group.getBoundingRect(), seriesModel, function () {
        group.removeClipPath();
      }));
    }
    this._data = seriesModel.getData();
  };
  SankeyView.prototype.dispose = function () {};
  SankeyView.type = 'sankey';
  return SankeyView;
}(ChartView);
/**
 * Special color, use source node color or target node color
 * @param curveProps curve's style to parse
 * @param orient direction
 * @param edge current curve data
 */
function applyCurveStyle(curveProps, orient, edge) {
  switch (curveProps.fill) {
    case 'source':
      curveProps.fill = edge.node1.getVisual('color');
      curveProps.decal = edge.node1.getVisual('style').decal;
      break;
    case 'target':
      curveProps.fill = edge.node2.getVisual('color');
      curveProps.decal = edge.node2.getVisual('style').decal;
      break;
    case 'gradient':
      var sourceColor = edge.node1.getVisual('color');
      var targetColor = edge.node2.getVisual('color');
      if (isString(sourceColor) && isString(targetColor)) {
        curveProps.fill = new graphic.LinearGradient(0, 0, +(orient === 'horizontal'), +(orient === 'vertical'), [{
          color: sourceColor,
          offset: 0
        }, {
          color: targetColor,
          offset: 1
        }]);
      }
  }
}
// Add animation to the view
function createGridClipShape(rect, seriesModel, cb) {
  var rectEl = new graphic.Rect({
    shape: {
      x: rect.x - 10,
      y: rect.y - 10,
      width: 0,
      height: rect.height + 20
    }
  });
  graphic.initProps(rectEl, {
    shape: {
      width: rect.width + 20
    }
  }, seriesModel, cb);
  return rectEl;
}
export default SankeyView;