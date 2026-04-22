
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
import * as zrUtil from 'zrender/lib/core/util.js';
import * as graphic from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import SymbolClz from '../helper/Symbol.js';
import { radialCoordinate } from './layoutHelper.js';
import * as bbox from 'zrender/lib/core/bbox.js';
import View from '../../coord/View.js';
import * as roamHelper from '../../component/helper/roamHelper.js';
import RoamController from '../../component/helper/RoamController.js';
import { onIrrelevantElement } from '../../component/helper/cursorHelper.js';
import { parsePercent } from '../../util/number.js';
import ChartView from '../../view/Chart.js';
import Path from 'zrender/lib/graphic/Path.js';
import { setStatesStylesFromModel, setStatesFlag, setDefaultStateProxy, HOVER_STATE_BLUR } from '../../util/states.js';
var TreeEdgeShape = /** @class */function () {
  function TreeEdgeShape() {
    this.parentPoint = [];
    this.childPoints = [];
  }
  return TreeEdgeShape;
}();
var TreePath = /** @class */function (_super) {
  __extends(TreePath, _super);
  function TreePath(opts) {
    return _super.call(this, opts) || this;
  }
  TreePath.prototype.getDefaultStyle = function () {
    return {
      stroke: '#000',
      fill: null
    };
  };
  TreePath.prototype.getDefaultShape = function () {
    return new TreeEdgeShape();
  };
  TreePath.prototype.buildPath = function (ctx, shape) {
    var childPoints = shape.childPoints;
    var childLen = childPoints.length;
    var parentPoint = shape.parentPoint;
    var firstChildPos = childPoints[0];
    var lastChildPos = childPoints[childLen - 1];
    if (childLen === 1) {
      ctx.moveTo(parentPoint[0], parentPoint[1]);
      ctx.lineTo(firstChildPos[0], firstChildPos[1]);
      return;
    }
    var orient = shape.orient;
    var forkDim = orient === 'TB' || orient === 'BT' ? 0 : 1;
    var otherDim = 1 - forkDim;
    var forkPosition = parsePercent(shape.forkPosition, 1);
    var tmpPoint = [];
    tmpPoint[forkDim] = parentPoint[forkDim];
    tmpPoint[otherDim] = parentPoint[otherDim] + (lastChildPos[otherDim] - parentPoint[otherDim]) * forkPosition;
    ctx.moveTo(parentPoint[0], parentPoint[1]);
    ctx.lineTo(tmpPoint[0], tmpPoint[1]);
    ctx.moveTo(firstChildPos[0], firstChildPos[1]);
    tmpPoint[forkDim] = firstChildPos[forkDim];
    ctx.lineTo(tmpPoint[0], tmpPoint[1]);
    tmpPoint[forkDim] = lastChildPos[forkDim];
    ctx.lineTo(tmpPoint[0], tmpPoint[1]);
    ctx.lineTo(lastChildPos[0], lastChildPos[1]);
    for (var i = 1; i < childLen - 1; i++) {
      var point = childPoints[i];
      ctx.moveTo(point[0], point[1]);
      tmpPoint[forkDim] = point[forkDim];
      ctx.lineTo(tmpPoint[0], tmpPoint[1]);
    }
  };
  return TreePath;
}(Path);
var TreeView = /** @class */function (_super) {
  __extends(TreeView, _super);
  function TreeView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TreeView.type;
    _this._mainGroup = new graphic.Group();
    return _this;
  }
  TreeView.prototype.init = function (ecModel, api) {
    this._controller = new RoamController(api.getZr());
    this._controllerHost = {
      target: this.group
    };
    this.group.add(this._mainGroup);
  };
  TreeView.prototype.render = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    var layoutInfo = seriesModel.layoutInfo;
    var group = this._mainGroup;
    var layout = seriesModel.get('layout');
    if (layout === 'radial') {
      group.x = layoutInfo.x + layoutInfo.width / 2;
      group.y = layoutInfo.y + layoutInfo.height / 2;
    } else {
      group.x = layoutInfo.x;
      group.y = layoutInfo.y;
    }
    this._updateViewCoordSys(seriesModel, api);
    this._updateController(seriesModel, ecModel, api);
    var oldData = this._data;
    data.diff(oldData).add(function (newIdx) {
      if (symbolNeedsDraw(data, newIdx)) {
        // Create node and edge
        updateNode(data, newIdx, null, group, seriesModel);
      }
    }).update(function (newIdx, oldIdx) {
      var symbolEl = oldData.getItemGraphicEl(oldIdx);
      if (!symbolNeedsDraw(data, newIdx)) {
        symbolEl && removeNode(oldData, oldIdx, symbolEl, group, seriesModel);
        return;
      }
      // Update node and edge
      updateNode(data, newIdx, symbolEl, group, seriesModel);
    }).remove(function (oldIdx) {
      var symbolEl = oldData.getItemGraphicEl(oldIdx);
      // When remove a collapsed node of subtree, since the collapsed
      // node haven't been initialized with a symbol element,
      // you can't found it's symbol element through index.
      // so if we want to remove the symbol element we should insure
      // that the symbol element is not null.
      if (symbolEl) {
        removeNode(oldData, oldIdx, symbolEl, group, seriesModel);
      }
    }).execute();
    this._nodeScaleRatio = seriesModel.get('nodeScaleRatio');
    this._updateNodeAndLinkScale(seriesModel);
    if (seriesModel.get('expandAndCollapse') === true) {
      data.eachItemGraphicEl(function (el, dataIndex) {
        el.off('click').on('click', function () {
          api.dispatchAction({
            type: 'treeExpandAndCollapse',
            seriesId: seriesModel.id,
            dataIndex: dataIndex
          });
        });
      });
    }
    this._data = data;
  };
  TreeView.prototype._updateViewCoordSys = function (seriesModel, api) {
    var data = seriesModel.getData();
    var points = [];
    data.each(function (idx) {
      var layout = data.getItemLayout(idx);
      if (layout && !isNaN(layout.x) && !isNaN(layout.y)) {
        points.push([+layout.x, +layout.y]);
      }
    });
    var min = [];
    var max = [];
    bbox.fromPoints(points, min, max);
    // If don't Store min max when collapse the root node after roam,
    // the root node will disappear.
    var oldMin = this._min;
    var oldMax = this._max;
    // If width or height is 0
    if (max[0] - min[0] === 0) {
      min[0] = oldMin ? oldMin[0] : min[0] - 1;
      max[0] = oldMax ? oldMax[0] : max[0] + 1;
    }
    if (max[1] - min[1] === 0) {
      min[1] = oldMin ? oldMin[1] : min[1] - 1;
      max[1] = oldMax ? oldMax[1] : max[1] + 1;
    }
    var viewCoordSys = seriesModel.coordinateSystem = new View();
    viewCoordSys.zoomLimit = seriesModel.get('scaleLimit');
    viewCoordSys.setBoundingRect(min[0], min[1], max[0] - min[0], max[1] - min[1]);
    viewCoordSys.setCenter(seriesModel.get('center'), api);
    viewCoordSys.setZoom(seriesModel.get('zoom'));
    // Here we use viewCoordSys just for computing the 'position' and 'scale' of the group
    this.group.attr({
      x: viewCoordSys.x,
      y: viewCoordSys.y,
      scaleX: viewCoordSys.scaleX,
      scaleY: viewCoordSys.scaleY
    });
    this._min = min;
    this._max = max;
  };
  TreeView.prototype._updateController = function (seriesModel, ecModel, api) {
    var _this = this;
    var controller = this._controller;
    var controllerHost = this._controllerHost;
    var group = this.group;
    controller.setPointerChecker(function (e, x, y) {
      var rect = group.getBoundingRect();
      rect.applyTransform(group.transform);
      return rect.contain(x, y) && !onIrrelevantElement(e, api, seriesModel);
    });
    controller.enable(seriesModel.get('roam'));
    controllerHost.zoomLimit = seriesModel.get('scaleLimit');
    controllerHost.zoom = seriesModel.coordinateSystem.getZoom();
    controller.off('pan').off('zoom').on('pan', function (e) {
      roamHelper.updateViewOnPan(controllerHost, e.dx, e.dy);
      api.dispatchAction({
        seriesId: seriesModel.id,
        type: 'treeRoam',
        dx: e.dx,
        dy: e.dy
      });
    }).on('zoom', function (e) {
      roamHelper.updateViewOnZoom(controllerHost, e.scale, e.originX, e.originY);
      api.dispatchAction({
        seriesId: seriesModel.id,
        type: 'treeRoam',
        zoom: e.scale,
        originX: e.originX,
        originY: e.originY
      });
      _this._updateNodeAndLinkScale(seriesModel);
      // Only update label layout on zoom
      api.updateLabelLayout();
    });
  };
  TreeView.prototype._updateNodeAndLinkScale = function (seriesModel) {
    var data = seriesModel.getData();
    var nodeScale = this._getNodeGlobalScale(seriesModel);
    data.eachItemGraphicEl(function (el, idx) {
      el.setSymbolScale(nodeScale);
    });
  };
  TreeView.prototype._getNodeGlobalScale = function (seriesModel) {
    var coordSys = seriesModel.coordinateSystem;
    if (coordSys.type !== 'view') {
      return 1;
    }
    var nodeScaleRatio = this._nodeScaleRatio;
    var groupZoom = coordSys.scaleX || 1;
    // Scale node when zoom changes
    var roamZoom = coordSys.getZoom();
    var nodeScale = (roamZoom - 1) * nodeScaleRatio + 1;
    return nodeScale / groupZoom;
  };
  TreeView.prototype.dispose = function () {
    this._controller && this._controller.dispose();
    this._controllerHost = null;
  };
  TreeView.prototype.remove = function () {
    this._mainGroup.removeAll();
    this._data = null;
  };
  TreeView.type = 'tree';
  return TreeView;
}(ChartView);
function symbolNeedsDraw(data, dataIndex) {
  var layout = data.getItemLayout(dataIndex);
  return layout && !isNaN(layout.x) && !isNaN(layout.y);
}
function updateNode(data, dataIndex, symbolEl, group, seriesModel) {
  var isInit = !symbolEl;
  var node = data.tree.getNodeByDataIndex(dataIndex);
  var itemModel = node.getModel();
  var visualColor = node.getVisual('style').fill;
  var symbolInnerColor = node.isExpand === false && node.children.length !== 0 ? visualColor : '#fff';
  var virtualRoot = data.tree.root;
  var source = node.parentNode === virtualRoot ? node : node.parentNode || node;
  var sourceSymbolEl = data.getItemGraphicEl(source.dataIndex);
  var sourceLayout = source.getLayout();
  var sourceOldLayout = sourceSymbolEl ? {
    x: sourceSymbolEl.__oldX,
    y: sourceSymbolEl.__oldY,
    rawX: sourceSymbolEl.__radialOldRawX,
    rawY: sourceSymbolEl.__radialOldRawY
  } : sourceLayout;
  var targetLayout = node.getLayout();
  if (isInit) {
    symbolEl = new SymbolClz(data, dataIndex, null, {
      symbolInnerColor: symbolInnerColor,
      useNameLabel: true
    });
    symbolEl.x = sourceOldLayout.x;
    symbolEl.y = sourceOldLayout.y;
  } else {
    symbolEl.updateData(data, dataIndex, null, {
      symbolInnerColor: symbolInnerColor,
      useNameLabel: true
    });
  }
  symbolEl.__radialOldRawX = symbolEl.__radialRawX;
  symbolEl.__radialOldRawY = symbolEl.__radialRawY;
  symbolEl.__radialRawX = targetLayout.rawX;
  symbolEl.__radialRawY = targetLayout.rawY;
  group.add(symbolEl);
  data.setItemGraphicEl(dataIndex, symbolEl);
  symbolEl.__oldX = symbolEl.x;
  symbolEl.__oldY = symbolEl.y;
  graphic.updateProps(symbolEl, {
    x: targetLayout.x,
    y: targetLayout.y
  }, seriesModel);
  var symbolPath = symbolEl.getSymbolPath();
  if (seriesModel.get('layout') === 'radial') {
    var realRoot = virtualRoot.children[0];
    var rootLayout = realRoot.getLayout();
    var length_1 = realRoot.children.length;
    var rad = void 0;
    var isLeft = void 0;
    if (targetLayout.x === rootLayout.x && node.isExpand === true && realRoot.children.length) {
      var center = {
        x: (realRoot.children[0].getLayout().x + realRoot.children[length_1 - 1].getLayout().x) / 2,
        y: (realRoot.children[0].getLayout().y + realRoot.children[length_1 - 1].getLayout().y) / 2
      };
      rad = Math.atan2(center.y - rootLayout.y, center.x - rootLayout.x);
      if (rad < 0) {
        rad = Math.PI * 2 + rad;
      }
      isLeft = center.x < rootLayout.x;
      if (isLeft) {
        rad = rad - Math.PI;
      }
    } else {
      rad = Math.atan2(targetLayout.y - rootLayout.y, targetLayout.x - rootLayout.x);
      if (rad < 0) {
        rad = Math.PI * 2 + rad;
      }
      if (node.children.length === 0 || node.children.length !== 0 && node.isExpand === false) {
        isLeft = targetLayout.x < rootLayout.x;
        if (isLeft) {
          rad = rad - Math.PI;
        }
      } else {
        isLeft = targetLayout.x > rootLayout.x;
        if (!isLeft) {
          rad = rad - Math.PI;
        }
      }
    }
    var textPosition = isLeft ? 'left' : 'right';
    var normalLabelModel = itemModel.getModel('label');
    var rotate = normalLabelModel.get('rotate');
    var labelRotateRadian = rotate * (Math.PI / 180);
    var textContent = symbolPath.getTextContent();
    if (textContent) {
      symbolPath.setTextConfig({
        position: normalLabelModel.get('position') || textPosition,
        rotation: rotate == null ? -rad : labelRotateRadian,
        origin: 'center'
      });
      textContent.setStyle('verticalAlign', 'middle');
    }
  }
  // Handle status
  var focus = itemModel.get(['emphasis', 'focus']);
  var focusDataIndices = focus === 'relative' ? zrUtil.concatArray(node.getAncestorsIndices(), node.getDescendantIndices()) : focus === 'ancestor' ? node.getAncestorsIndices() : focus === 'descendant' ? node.getDescendantIndices() : null;
  if (focusDataIndices) {
    // Modify the focus to data indices.
    getECData(symbolEl).focus = focusDataIndices;
  }
  drawEdge(seriesModel, node, virtualRoot, symbolEl, sourceOldLayout, sourceLayout, targetLayout, group);
  if (symbolEl.__edge) {
    symbolEl.onHoverStateChange = function (toState) {
      if (toState !== 'blur') {
        // NOTE: Ensure the parent elements will been blurred firstly.
        // According to the return of getAncestorsIndices and getDescendantIndices
        // TODO: A bit tricky.
        var parentEl = node.parentNode && data.getItemGraphicEl(node.parentNode.dataIndex);
        if (!(parentEl && parentEl.hoverState === HOVER_STATE_BLUR)) {
          setStatesFlag(symbolEl.__edge, toState);
        }
      }
    };
  }
}
function drawEdge(seriesModel, node, virtualRoot, symbolEl, sourceOldLayout, sourceLayout, targetLayout, group) {
  var itemModel = node.getModel();
  var edgeShape = seriesModel.get('edgeShape');
  var layout = seriesModel.get('layout');
  var orient = seriesModel.getOrient();
  var curvature = seriesModel.get(['lineStyle', 'curveness']);
  var edgeForkPosition = seriesModel.get('edgeForkPosition');
  var lineStyle = itemModel.getModel('lineStyle').getLineStyle();
  var edge = symbolEl.__edge;
  // curve edge from node -> parent
  // polyline edge from node -> children
  if (edgeShape === 'curve') {
    if (node.parentNode && node.parentNode !== virtualRoot) {
      if (!edge) {
        edge = symbolEl.__edge = new graphic.BezierCurve({
          shape: getEdgeShape(layout, orient, curvature, sourceOldLayout, sourceOldLayout)
        });
      }
      graphic.updateProps(edge, {
        shape: getEdgeShape(layout, orient, curvature, sourceLayout, targetLayout)
      }, seriesModel);
    }
  } else if (edgeShape === 'polyline') {
    if (layout === 'orthogonal') {
      if (node !== virtualRoot && node.children && node.children.length !== 0 && node.isExpand === true) {
        var children = node.children;
        var childPoints = [];
        for (var i = 0; i < children.length; i++) {
          var childLayout = children[i].getLayout();
          childPoints.push([childLayout.x, childLayout.y]);
        }
        if (!edge) {
          edge = symbolEl.__edge = new TreePath({
            shape: {
              parentPoint: [targetLayout.x, targetLayout.y],
              childPoints: [[targetLayout.x, targetLayout.y]],
              orient: orient,
              forkPosition: edgeForkPosition
            }
          });
        }
        graphic.updateProps(edge, {
          shape: {
            parentPoint: [targetLayout.x, targetLayout.y],
            childPoints: childPoints
          }
        }, seriesModel);
      }
    } else {
      if (process.env.NODE_ENV !== 'production') {
        throw new Error('The polyline edgeShape can only be used in orthogonal layout');
      }
    }
  }
  // show all edge when edgeShape is 'curve', filter node `isExpand` is false when edgeShape is 'polyline'
  if (edge && !(edgeShape === 'polyline' && !node.isExpand)) {
    edge.useStyle(zrUtil.defaults({
      strokeNoScale: true,
      fill: null
    }, lineStyle));
    setStatesStylesFromModel(edge, itemModel, 'lineStyle');
    setDefaultStateProxy(edge);
    group.add(edge);
  }
}
function removeNodeEdge(node, data, group, seriesModel, removeAnimationOpt) {
  var virtualRoot = data.tree.root;
  var _a = getSourceNode(virtualRoot, node),
    source = _a.source,
    sourceLayout = _a.sourceLayout;
  var symbolEl = data.getItemGraphicEl(node.dataIndex);
  if (!symbolEl) {
    return;
  }
  var sourceSymbolEl = data.getItemGraphicEl(source.dataIndex);
  var sourceEdge = sourceSymbolEl.__edge;
  // 1. when expand the sub tree, delete the children node should delete the edge of
  // the source at the same time. because the polyline edge shape is only owned by the source.
  // 2.when the node is the only children of the source, delete the node should delete the edge of
  // the source at the same time. the same reason as above.
  var edge = symbolEl.__edge || (source.isExpand === false || source.children.length === 1 ? sourceEdge : undefined);
  var edgeShape = seriesModel.get('edgeShape');
  var layoutOpt = seriesModel.get('layout');
  var orient = seriesModel.get('orient');
  var curvature = seriesModel.get(['lineStyle', 'curveness']);
  if (edge) {
    if (edgeShape === 'curve') {
      graphic.removeElement(edge, {
        shape: getEdgeShape(layoutOpt, orient, curvature, sourceLayout, sourceLayout),
        style: {
          opacity: 0
        }
      }, seriesModel, {
        cb: function () {
          group.remove(edge);
        },
        removeOpt: removeAnimationOpt
      });
    } else if (edgeShape === 'polyline' && seriesModel.get('layout') === 'orthogonal') {
      graphic.removeElement(edge, {
        shape: {
          parentPoint: [sourceLayout.x, sourceLayout.y],
          childPoints: [[sourceLayout.x, sourceLayout.y]]
        },
        style: {
          opacity: 0
        }
      }, seriesModel, {
        cb: function () {
          group.remove(edge);
        },
        removeOpt: removeAnimationOpt
      });
    }
  }
}
function getSourceNode(virtualRoot, node) {
  var source = node.parentNode === virtualRoot ? node : node.parentNode || node;
  var sourceLayout;
  while (sourceLayout = source.getLayout(), sourceLayout == null) {
    source = source.parentNode === virtualRoot ? source : source.parentNode || source;
  }
  return {
    source: source,
    sourceLayout: sourceLayout
  };
}
function removeNode(data, dataIndex, symbolEl, group, seriesModel) {
  var node = data.tree.getNodeByDataIndex(dataIndex);
  var virtualRoot = data.tree.root;
  var sourceLayout = getSourceNode(virtualRoot, node).sourceLayout;
  // Use same duration and easing with update to have more consistent animation.
  var removeAnimationOpt = {
    duration: seriesModel.get('animationDurationUpdate'),
    easing: seriesModel.get('animationEasingUpdate')
  };
  graphic.removeElement(symbolEl, {
    x: sourceLayout.x + 1,
    y: sourceLayout.y + 1
  }, seriesModel, {
    cb: function () {
      group.remove(symbolEl);
      data.setItemGraphicEl(dataIndex, null);
    },
    removeOpt: removeAnimationOpt
  });
  symbolEl.fadeOut(null, data.hostModel, {
    fadeLabel: true,
    animation: removeAnimationOpt
  });
  // remove edge as parent node
  node.children.forEach(function (childNode) {
    removeNodeEdge(childNode, data, group, seriesModel, removeAnimationOpt);
  });
  // remove edge as child node
  removeNodeEdge(node, data, group, seriesModel, removeAnimationOpt);
}
function getEdgeShape(layoutOpt, orient, curvature, sourceLayout, targetLayout) {
  var cpx1;
  var cpy1;
  var cpx2;
  var cpy2;
  var x1;
  var x2;
  var y1;
  var y2;
  if (layoutOpt === 'radial') {
    x1 = sourceLayout.rawX;
    y1 = sourceLayout.rawY;
    x2 = targetLayout.rawX;
    y2 = targetLayout.rawY;
    var radialCoor1 = radialCoordinate(x1, y1);
    var radialCoor2 = radialCoordinate(x1, y1 + (y2 - y1) * curvature);
    var radialCoor3 = radialCoordinate(x2, y2 + (y1 - y2) * curvature);
    var radialCoor4 = radialCoordinate(x2, y2);
    return {
      x1: radialCoor1.x || 0,
      y1: radialCoor1.y || 0,
      x2: radialCoor4.x || 0,
      y2: radialCoor4.y || 0,
      cpx1: radialCoor2.x || 0,
      cpy1: radialCoor2.y || 0,
      cpx2: radialCoor3.x || 0,
      cpy2: radialCoor3.y || 0
    };
  } else {
    x1 = sourceLayout.x;
    y1 = sourceLayout.y;
    x2 = targetLayout.x;
    y2 = targetLayout.y;
    if (orient === 'LR' || orient === 'RL') {
      cpx1 = x1 + (x2 - x1) * curvature;
      cpy1 = y1;
      cpx2 = x2 + (x1 - x2) * curvature;
      cpy2 = y2;
    }
    if (orient === 'TB' || orient === 'BT') {
      cpx1 = x1;
      cpy1 = y1 + (y2 - y1) * curvature;
      cpx2 = x2;
      cpy2 = y2 + (y1 - y2) * curvature;
    }
  }
  return {
    x1: x1,
    y1: y1,
    x2: x2,
    y2: y2,
    cpx1: cpx1,
    cpy1: cpy1,
    cpx2: cpx2,
    cpy2: cpy2
  };
}
export default TreeView;