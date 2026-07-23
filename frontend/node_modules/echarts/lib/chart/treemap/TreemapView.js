
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
import { bind, each, indexOf, curry, extend, normalizeCssArray, isFunction } from 'zrender/lib/core/util.js';
import * as graphic from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import { isHighDownDispatcher, setAsHighDownDispatcher, setDefaultStateProxy, enableHoverFocus, Z2_EMPHASIS_LIFT } from '../../util/states.js';
import DataDiffer from '../../data/DataDiffer.js';
import * as helper from '../helper/treeHelper.js';
import Breadcrumb from './Breadcrumb.js';
import RoamController from '../../component/helper/RoamController.js';
import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import * as matrix from 'zrender/lib/core/matrix.js';
import * as animationUtil from '../../util/animation.js';
import makeStyleMapper from '../../model/mixin/makeStyleMapper.js';
import ChartView from '../../view/Chart.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
import { makeInner, convertOptionIdName } from '../../util/model.js';
import { windowOpen } from '../../util/format.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
var Group = graphic.Group;
var Rect = graphic.Rect;
var DRAG_THRESHOLD = 3;
var PATH_LABEL_NOAMAL = 'label';
var PATH_UPPERLABEL_NORMAL = 'upperLabel';
// Should larger than emphasis states lift z
var Z2_BASE = Z2_EMPHASIS_LIFT * 10; // Should bigger than every z2.
var Z2_BG = Z2_EMPHASIS_LIFT * 2;
var Z2_CONTENT = Z2_EMPHASIS_LIFT * 3;
var getStateItemStyle = makeStyleMapper([['fill', 'color'],
// `borderColor` and `borderWidth` has been occupied,
// so use `stroke` to indicate the stroke of the rect.
['stroke', 'strokeColor'], ['lineWidth', 'strokeWidth'], ['shadowBlur'], ['shadowOffsetX'], ['shadowOffsetY'], ['shadowColor']
// Option decal is in `DecalObject` but style.decal is in `PatternObject`.
// So do not transfer decal directly.
]);
var getItemStyleNormal = function (model) {
  // Normal style props should include emphasis style props.
  var itemStyle = getStateItemStyle(model);
  // Clear styles set by emphasis.
  itemStyle.stroke = itemStyle.fill = itemStyle.lineWidth = null;
  return itemStyle;
};
var inner = makeInner();
var TreemapView = /** @class */function (_super) {
  __extends(TreemapView, _super);
  function TreemapView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TreemapView.type;
    _this._state = 'ready';
    _this._storage = createStorage();
    return _this;
  }
  /**
   * @override
   */
  TreemapView.prototype.render = function (seriesModel, ecModel, api, payload) {
    var models = ecModel.findComponents({
      mainType: 'series',
      subType: 'treemap',
      query: payload
    });
    if (indexOf(models, seriesModel) < 0) {
      return;
    }
    this.seriesModel = seriesModel;
    this.api = api;
    this.ecModel = ecModel;
    var types = ['treemapZoomToNode', 'treemapRootToNode'];
    var targetInfo = helper.retrieveTargetInfo(payload, types, seriesModel);
    var payloadType = payload && payload.type;
    var layoutInfo = seriesModel.layoutInfo;
    var isInit = !this._oldTree;
    var thisStorage = this._storage;
    // Mark new root when action is treemapRootToNode.
    var reRoot = payloadType === 'treemapRootToNode' && targetInfo && thisStorage ? {
      rootNodeGroup: thisStorage.nodeGroup[targetInfo.node.getRawIndex()],
      direction: payload.direction
    } : null;
    var containerGroup = this._giveContainerGroup(layoutInfo);
    var hasAnimation = seriesModel.get('animation');
    var renderResult = this._doRender(containerGroup, seriesModel, reRoot);
    hasAnimation && !isInit && (!payloadType || payloadType === 'treemapZoomToNode' || payloadType === 'treemapRootToNode') ? this._doAnimation(containerGroup, renderResult, seriesModel, reRoot) : renderResult.renderFinally();
    this._resetController(api);
    this._renderBreadcrumb(seriesModel, api, targetInfo);
  };
  TreemapView.prototype._giveContainerGroup = function (layoutInfo) {
    var containerGroup = this._containerGroup;
    if (!containerGroup) {
      // FIXME
      // 加一层containerGroup是为了clip，但是现在clip功能并没有实现。
      containerGroup = this._containerGroup = new Group();
      this._initEvents(containerGroup);
      this.group.add(containerGroup);
    }
    containerGroup.x = layoutInfo.x;
    containerGroup.y = layoutInfo.y;
    return containerGroup;
  };
  TreemapView.prototype._doRender = function (containerGroup, seriesModel, reRoot) {
    var thisTree = seriesModel.getData().tree;
    var oldTree = this._oldTree;
    // Clear last shape records.
    var lastsForAnimation = createStorage();
    var thisStorage = createStorage();
    var oldStorage = this._storage;
    var willInvisibleEls = [];
    function doRenderNode(thisNode, oldNode, parentGroup, depth) {
      return renderNode(seriesModel, thisStorage, oldStorage, reRoot, lastsForAnimation, willInvisibleEls, thisNode, oldNode, parentGroup, depth);
    }
    // Notice: When thisTree and oldTree are the same tree (see list.cloneShallow),
    // the oldTree is actually losted, so we cannot find all of the old graphic
    // elements from tree. So we use this strategy: make element storage, move
    // from old storage to new storage, clear old storage.
    dualTravel(thisTree.root ? [thisTree.root] : [], oldTree && oldTree.root ? [oldTree.root] : [], containerGroup, thisTree === oldTree || !oldTree, 0);
    // Process all removing.
    var willDeleteEls = clearStorage(oldStorage);
    this._oldTree = thisTree;
    this._storage = thisStorage;
    if (this._controllerHost) {
      var _oldRootLayout = this.seriesModel.layoutInfo;
      var rootLayout = thisTree.root.getLayout();
      if (rootLayout.width === _oldRootLayout.width && rootLayout.height === _oldRootLayout.height) {
        this._controllerHost.zoom = 1;
      }
    }
    return {
      lastsForAnimation: lastsForAnimation,
      willDeleteEls: willDeleteEls,
      renderFinally: renderFinally
    };
    function dualTravel(thisViewChildren, oldViewChildren, parentGroup, sameTree, depth) {
      // When 'render' is triggered by action,
      // 'this' and 'old' may be the same tree,
      // we use rawIndex in that case.
      if (sameTree) {
        oldViewChildren = thisViewChildren;
        each(thisViewChildren, function (child, index) {
          !child.isRemoved() && processNode(index, index);
        });
      }
      // Diff hierarchically (diff only in each subtree, but not whole).
      // because, consistency of view is important.
      else {
        new DataDiffer(oldViewChildren, thisViewChildren, getKey, getKey).add(processNode).update(processNode).remove(curry(processNode, null)).execute();
      }
      function getKey(node) {
        // Identify by name or raw index.
        return node.getId();
      }
      function processNode(newIndex, oldIndex) {
        var thisNode = newIndex != null ? thisViewChildren[newIndex] : null;
        var oldNode = oldIndex != null ? oldViewChildren[oldIndex] : null;
        var group = doRenderNode(thisNode, oldNode, parentGroup, depth);
        group && dualTravel(thisNode && thisNode.viewChildren || [], oldNode && oldNode.viewChildren || [], group, sameTree, depth + 1);
      }
    }
    function clearStorage(storage) {
      var willDeleteEls = createStorage();
      storage && each(storage, function (store, storageName) {
        var delEls = willDeleteEls[storageName];
        each(store, function (el) {
          el && (delEls.push(el), inner(el).willDelete = true);
        });
      });
      return willDeleteEls;
    }
    function renderFinally() {
      each(willDeleteEls, function (els) {
        each(els, function (el) {
          el.parent && el.parent.remove(el);
        });
      });
      each(willInvisibleEls, function (el) {
        el.invisible = true;
        // Setting invisible is for optimizing, so no need to set dirty,
        // just mark as invisible.
        el.dirty();
      });
    }
  };
  TreemapView.prototype._doAnimation = function (containerGroup, renderResult, seriesModel, reRoot) {
    var durationOption = seriesModel.get('animationDurationUpdate');
    var easingOption = seriesModel.get('animationEasing');
    // TODO: do not support function until necessary.
    var duration = (isFunction(durationOption) ? 0 : durationOption) || 0;
    var easing = (isFunction(easingOption) ? null : easingOption) || 'cubicOut';
    var animationWrap = animationUtil.createWrap();
    // Make delete animations.
    each(renderResult.willDeleteEls, function (store, storageName) {
      each(store, function (el, rawIndex) {
        if (el.invisible) {
          return;
        }
        var parent = el.parent; // Always has parent, and parent is nodeGroup.
        var target;
        var innerStore = inner(parent);
        if (reRoot && reRoot.direction === 'drillDown') {
          target = parent === reRoot.rootNodeGroup
          // This is the content element of view root.
          // Only `content` will enter this branch, because
          // `background` and `nodeGroup` will not be deleted.
          ? {
            shape: {
              x: 0,
              y: 0,
              width: innerStore.nodeWidth,
              height: innerStore.nodeHeight
            },
            style: {
              opacity: 0
            }
          }
          // Others.
          : {
            style: {
              opacity: 0
            }
          };
        } else {
          var targetX = 0;
          var targetY = 0;
          if (!innerStore.willDelete) {
            // Let node animate to right-bottom corner, cooperating with fadeout,
            // which is appropriate for user understanding.
            // Divided by 2 for reRoot rolling up effect.
            targetX = innerStore.nodeWidth / 2;
            targetY = innerStore.nodeHeight / 2;
          }
          target = storageName === 'nodeGroup' ? {
            x: targetX,
            y: targetY,
            style: {
              opacity: 0
            }
          } : {
            shape: {
              x: targetX,
              y: targetY,
              width: 0,
              height: 0
            },
            style: {
              opacity: 0
            }
          };
        }
        // TODO: do not support delay until necessary.
        target && animationWrap.add(el, target, duration, 0, easing);
      });
    });
    // Make other animations
    each(this._storage, function (store, storageName) {
      each(store, function (el, rawIndex) {
        var last = renderResult.lastsForAnimation[storageName][rawIndex];
        var target = {};
        if (!last) {
          return;
        }
        if (el instanceof graphic.Group) {
          if (last.oldX != null) {
            target.x = el.x;
            target.y = el.y;
            el.x = last.oldX;
            el.y = last.oldY;
          }
        } else {
          if (last.oldShape) {
            target.shape = extend({}, el.shape);
            el.setShape(last.oldShape);
          }
          if (last.fadein) {
            el.setStyle('opacity', 0);
            target.style = {
              opacity: 1
            };
          }
          // When animation is stopped for succedent animation starting,
          // el.style.opacity might not be 1
          else if (el.style.opacity !== 1) {
            target.style = {
              opacity: 1
            };
          }
        }
        animationWrap.add(el, target, duration, 0, easing);
      });
    }, this);
    this._state = 'animating';
    animationWrap.finished(bind(function () {
      this._state = 'ready';
      renderResult.renderFinally();
    }, this)).start();
  };
  TreemapView.prototype._resetController = function (api) {
    var controller = this._controller;
    var controllerHost = this._controllerHost;
    if (!controllerHost) {
      this._controllerHost = {
        target: this.group
      };
      controllerHost = this._controllerHost;
    }
    // Init controller.
    if (!controller) {
      controller = this._controller = new RoamController(api.getZr());
      controller.enable(this.seriesModel.get('roam'));
      controllerHost.zoomLimit = this.seriesModel.get('scaleLimit');
      controllerHost.zoom = this.seriesModel.get('zoom');
      controller.on('pan', bind(this._onPan, this));
      controller.on('zoom', bind(this._onZoom, this));
    }
    var rect = new BoundingRect(0, 0, api.getWidth(), api.getHeight());
    controller.setPointerChecker(function (e, x, y) {
      return rect.contain(x, y);
    });
  };
  TreemapView.prototype._clearController = function () {
    var controller = this._controller;
    this._controllerHost = null;
    if (controller) {
      controller.dispose();
      controller = null;
    }
  };
  TreemapView.prototype._onPan = function (e) {
    if (this._state !== 'animating' && (Math.abs(e.dx) > DRAG_THRESHOLD || Math.abs(e.dy) > DRAG_THRESHOLD)) {
      // These param must not be cached.
      var root = this.seriesModel.getData().tree.root;
      if (!root) {
        return;
      }
      var rootLayout = root.getLayout();
      if (!rootLayout) {
        return;
      }
      this.api.dispatchAction({
        type: 'treemapMove',
        from: this.uid,
        seriesId: this.seriesModel.id,
        rootRect: {
          x: rootLayout.x + e.dx,
          y: rootLayout.y + e.dy,
          width: rootLayout.width,
          height: rootLayout.height
        }
      });
    }
  };
  TreemapView.prototype._onZoom = function (e) {
    var mouseX = e.originX;
    var mouseY = e.originY;
    var zoomDelta = e.scale;
    if (this._state !== 'animating') {
      // These param must not be cached.
      var root = this.seriesModel.getData().tree.root;
      if (!root) {
        return;
      }
      var rootLayout = root.getLayout();
      if (!rootLayout) {
        return;
      }
      var rect = new BoundingRect(rootLayout.x, rootLayout.y, rootLayout.width, rootLayout.height);
      // scaleLimit
      var zoomLimit = null;
      var _controllerHost = this._controllerHost;
      zoomLimit = _controllerHost.zoomLimit;
      var newZoom = _controllerHost.zoom = _controllerHost.zoom || 1;
      newZoom *= zoomDelta;
      if (zoomLimit) {
        var zoomMin = zoomLimit.min || 0;
        var zoomMax = zoomLimit.max || Infinity;
        newZoom = Math.max(Math.min(zoomMax, newZoom), zoomMin);
      }
      var zoomScale = newZoom / _controllerHost.zoom;
      _controllerHost.zoom = newZoom;
      var layoutInfo = this.seriesModel.layoutInfo;
      // Transform mouse coord from global to containerGroup.
      mouseX -= layoutInfo.x;
      mouseY -= layoutInfo.y;
      // Scale root bounding rect.
      var m = matrix.create();
      matrix.translate(m, m, [-mouseX, -mouseY]);
      matrix.scale(m, m, [zoomScale, zoomScale]);
      matrix.translate(m, m, [mouseX, mouseY]);
      rect.applyTransform(m);
      this.api.dispatchAction({
        type: 'treemapRender',
        from: this.uid,
        seriesId: this.seriesModel.id,
        rootRect: {
          x: rect.x,
          y: rect.y,
          width: rect.width,
          height: rect.height
        }
      });
    }
  };
  TreemapView.prototype._initEvents = function (containerGroup) {
    var _this = this;
    containerGroup.on('click', function (e) {
      if (_this._state !== 'ready') {
        return;
      }
      var nodeClick = _this.seriesModel.get('nodeClick', true);
      if (!nodeClick) {
        return;
      }
      var targetInfo = _this.findTarget(e.offsetX, e.offsetY);
      if (!targetInfo) {
        return;
      }
      var node = targetInfo.node;
      if (node.getLayout().isLeafRoot) {
        _this._rootToNode(targetInfo);
      } else {
        if (nodeClick === 'zoomToNode') {
          _this._zoomToNode(targetInfo);
        } else if (nodeClick === 'link') {
          var itemModel = node.hostTree.data.getItemModel(node.dataIndex);
          var link = itemModel.get('link', true);
          var linkTarget = itemModel.get('target', true) || 'blank';
          link && windowOpen(link, linkTarget);
        }
      }
    }, this);
  };
  TreemapView.prototype._renderBreadcrumb = function (seriesModel, api, targetInfo) {
    var _this = this;
    if (!targetInfo) {
      targetInfo = seriesModel.get('leafDepth', true) != null ? {
        node: seriesModel.getViewRoot()
      }
      // FIXME
      // better way?
      // Find breadcrumb tail on center of containerGroup.
      : this.findTarget(api.getWidth() / 2, api.getHeight() / 2);
      if (!targetInfo) {
        targetInfo = {
          node: seriesModel.getData().tree.root
        };
      }
    }
    (this._breadcrumb || (this._breadcrumb = new Breadcrumb(this.group))).render(seriesModel, api, targetInfo.node, function (node) {
      if (_this._state !== 'animating') {
        helper.aboveViewRoot(seriesModel.getViewRoot(), node) ? _this._rootToNode({
          node: node
        }) : _this._zoomToNode({
          node: node
        });
      }
    });
  };
  /**
   * @override
   */
  TreemapView.prototype.remove = function () {
    this._clearController();
    this._containerGroup && this._containerGroup.removeAll();
    this._storage = createStorage();
    this._state = 'ready';
    this._breadcrumb && this._breadcrumb.remove();
  };
  TreemapView.prototype.dispose = function () {
    this._clearController();
  };
  TreemapView.prototype._zoomToNode = function (targetInfo) {
    this.api.dispatchAction({
      type: 'treemapZoomToNode',
      from: this.uid,
      seriesId: this.seriesModel.id,
      targetNode: targetInfo.node
    });
  };
  TreemapView.prototype._rootToNode = function (targetInfo) {
    this.api.dispatchAction({
      type: 'treemapRootToNode',
      from: this.uid,
      seriesId: this.seriesModel.id,
      targetNode: targetInfo.node
    });
  };
  /**
   * @public
   * @param {number} x Global coord x.
   * @param {number} y Global coord y.
   * @return {Object} info If not found, return undefined;
   * @return {number} info.node Target node.
   * @return {number} info.offsetX x refer to target node.
   * @return {number} info.offsetY y refer to target node.
   */
  TreemapView.prototype.findTarget = function (x, y) {
    var targetInfo;
    var viewRoot = this.seriesModel.getViewRoot();
    viewRoot.eachNode({
      attr: 'viewChildren',
      order: 'preorder'
    }, function (node) {
      var bgEl = this._storage.background[node.getRawIndex()];
      // If invisible, there might be no element.
      if (bgEl) {
        var point = bgEl.transformCoordToLocal(x, y);
        var shape = bgEl.shape;
        // For performance consideration, don't use 'getBoundingRect'.
        if (shape.x <= point[0] && point[0] <= shape.x + shape.width && shape.y <= point[1] && point[1] <= shape.y + shape.height) {
          targetInfo = {
            node: node,
            offsetX: point[0],
            offsetY: point[1]
          };
        } else {
          return false; // Suppress visit subtree.
        }
      }
    }, this);
    return targetInfo;
  };
  TreemapView.type = 'treemap';
  return TreemapView;
}(ChartView);
/**
 * @inner
 */
function createStorage() {
  return {
    nodeGroup: [],
    background: [],
    content: []
  };
}
/**
 * @inner
 * @return Return undefined means do not travel further.
 */
function renderNode(seriesModel, thisStorage, oldStorage, reRoot, lastsForAnimation, willInvisibleEls, thisNode, oldNode, parentGroup, depth) {
  // Whether under viewRoot.
  if (!thisNode) {
    // Deleting nodes will be performed finally. This method just find
    // element from old storage, or create new element, set them to new
    // storage, and set styles.
    return;
  }
  // -------------------------------------------------------------------
  // Start of closure variables available in "Procedures in renderNode".
  var thisLayout = thisNode.getLayout();
  var data = seriesModel.getData();
  var nodeModel = thisNode.getModel();
  // Only for enabling highlight/downplay. Clear firstly.
  // Because some node will not be rendered.
  data.setItemGraphicEl(thisNode.dataIndex, null);
  if (!thisLayout || !thisLayout.isInView) {
    return;
  }
  var thisWidth = thisLayout.width;
  var thisHeight = thisLayout.height;
  var borderWidth = thisLayout.borderWidth;
  var thisInvisible = thisLayout.invisible;
  var thisRawIndex = thisNode.getRawIndex();
  var oldRawIndex = oldNode && oldNode.getRawIndex();
  var thisViewChildren = thisNode.viewChildren;
  var upperHeight = thisLayout.upperHeight;
  var isParent = thisViewChildren && thisViewChildren.length;
  var itemStyleNormalModel = nodeModel.getModel('itemStyle');
  var itemStyleEmphasisModel = nodeModel.getModel(['emphasis', 'itemStyle']);
  var itemStyleBlurModel = nodeModel.getModel(['blur', 'itemStyle']);
  var itemStyleSelectModel = nodeModel.getModel(['select', 'itemStyle']);
  var borderRadius = itemStyleNormalModel.get('borderRadius') || 0;
  // End of closure ariables available in "Procedures in renderNode".
  // -----------------------------------------------------------------
  // Node group
  var group = giveGraphic('nodeGroup', Group);
  if (!group) {
    return;
  }
  parentGroup.add(group);
  // x,y are not set when el is above view root.
  group.x = thisLayout.x || 0;
  group.y = thisLayout.y || 0;
  group.markRedraw();
  inner(group).nodeWidth = thisWidth;
  inner(group).nodeHeight = thisHeight;
  if (thisLayout.isAboveViewRoot) {
    return group;
  }
  // Background
  var bg = giveGraphic('background', Rect, depth, Z2_BG);
  bg && renderBackground(group, bg, isParent && thisLayout.upperLabelHeight);
  var emphasisModel = nodeModel.getModel('emphasis');
  var focus = emphasisModel.get('focus');
  var blurScope = emphasisModel.get('blurScope');
  var isDisabled = emphasisModel.get('disabled');
  var focusOrIndices = focus === 'ancestor' ? thisNode.getAncestorsIndices() : focus === 'descendant' ? thisNode.getDescendantIndices() : focus;
  // No children, render content.
  if (isParent) {
    // Because of the implementation about "traverse" in graphic hover style, we
    // can not set hover listener on the "group" of non-leaf node. Otherwise the
    // hover event from the descendents will be listenered.
    if (isHighDownDispatcher(group)) {
      setAsHighDownDispatcher(group, false);
    }
    if (bg) {
      setAsHighDownDispatcher(bg, !isDisabled);
      // Only for enabling highlight/downplay.
      data.setItemGraphicEl(thisNode.dataIndex, bg);
      enableHoverFocus(bg, focusOrIndices, blurScope);
    }
  } else {
    var content = giveGraphic('content', Rect, depth, Z2_CONTENT);
    content && renderContent(group, content);
    bg.disableMorphing = true;
    if (bg && isHighDownDispatcher(bg)) {
      setAsHighDownDispatcher(bg, false);
    }
    setAsHighDownDispatcher(group, !isDisabled);
    // Only for enabling highlight/downplay.
    data.setItemGraphicEl(thisNode.dataIndex, group);
    var cursorStyle = nodeModel.getShallow('cursor');
    cursorStyle && content.attr('cursor', cursorStyle);
    enableHoverFocus(group, focusOrIndices, blurScope);
  }
  return group;
  // ----------------------------
  // | Procedures in renderNode |
  // ----------------------------
  function renderBackground(group, bg, useUpperLabel) {
    var ecData = getECData(bg);
    // For tooltip.
    ecData.dataIndex = thisNode.dataIndex;
    ecData.seriesIndex = seriesModel.seriesIndex;
    bg.setShape({
      x: 0,
      y: 0,
      width: thisWidth,
      height: thisHeight,
      r: borderRadius
    });
    if (thisInvisible) {
      // If invisible, do not set visual, otherwise the element will
      // change immediately before animation. We think it is OK to
      // remain its origin color when moving out of the view window.
      processInvisible(bg);
    } else {
      bg.invisible = false;
      var style = thisNode.getVisual('style');
      var visualBorderColor = style.stroke;
      var normalStyle = getItemStyleNormal(itemStyleNormalModel);
      normalStyle.fill = visualBorderColor;
      var emphasisStyle = getStateItemStyle(itemStyleEmphasisModel);
      emphasisStyle.fill = itemStyleEmphasisModel.get('borderColor');
      var blurStyle = getStateItemStyle(itemStyleBlurModel);
      blurStyle.fill = itemStyleBlurModel.get('borderColor');
      var selectStyle = getStateItemStyle(itemStyleSelectModel);
      selectStyle.fill = itemStyleSelectModel.get('borderColor');
      if (useUpperLabel) {
        var upperLabelWidth = thisWidth - 2 * borderWidth;
        prepareText(
        // PENDING: convert ZRColor to ColorString for text.
        bg, visualBorderColor, style.opacity, {
          x: borderWidth,
          y: 0,
          width: upperLabelWidth,
          height: upperHeight
        });
      }
      // For old bg.
      else {
        bg.removeTextContent();
      }
      bg.setStyle(normalStyle);
      bg.ensureState('emphasis').style = emphasisStyle;
      bg.ensureState('blur').style = blurStyle;
      bg.ensureState('select').style = selectStyle;
      setDefaultStateProxy(bg);
    }
    group.add(bg);
  }
  function renderContent(group, content) {
    var ecData = getECData(content);
    // For tooltip.
    ecData.dataIndex = thisNode.dataIndex;
    ecData.seriesIndex = seriesModel.seriesIndex;
    var contentWidth = Math.max(thisWidth - 2 * borderWidth, 0);
    var contentHeight = Math.max(thisHeight - 2 * borderWidth, 0);
    content.culling = true;
    content.setShape({
      x: borderWidth,
      y: borderWidth,
      width: contentWidth,
      height: contentHeight,
      r: borderRadius
    });
    if (thisInvisible) {
      // If invisible, do not set visual, otherwise the element will
      // change immediately before animation. We think it is OK to
      // remain its origin color when moving out of the view window.
      processInvisible(content);
    } else {
      content.invisible = false;
      var nodeStyle = thisNode.getVisual('style');
      var visualColor = nodeStyle.fill;
      var normalStyle = getItemStyleNormal(itemStyleNormalModel);
      normalStyle.fill = visualColor;
      normalStyle.decal = nodeStyle.decal;
      var emphasisStyle = getStateItemStyle(itemStyleEmphasisModel);
      var blurStyle = getStateItemStyle(itemStyleBlurModel);
      var selectStyle = getStateItemStyle(itemStyleSelectModel);
      // PENDING: convert ZRColor to ColorString for text.
      prepareText(content, visualColor, nodeStyle.opacity, null);
      content.setStyle(normalStyle);
      content.ensureState('emphasis').style = emphasisStyle;
      content.ensureState('blur').style = blurStyle;
      content.ensureState('select').style = selectStyle;
      setDefaultStateProxy(content);
    }
    group.add(content);
  }
  function processInvisible(element) {
    // Delay invisible setting utill animation finished,
    // avoid element vanish suddenly before animation.
    !element.invisible && willInvisibleEls.push(element);
  }
  function prepareText(rectEl, visualColor, visualOpacity,
  // Can be null/undefined
  upperLabelRect) {
    var normalLabelModel = nodeModel.getModel(upperLabelRect ? PATH_UPPERLABEL_NORMAL : PATH_LABEL_NOAMAL);
    var defaultText = convertOptionIdName(nodeModel.get('name'), null);
    var isShow = normalLabelModel.getShallow('show');
    setLabelStyle(rectEl, getLabelStatesModels(nodeModel, upperLabelRect ? PATH_UPPERLABEL_NORMAL : PATH_LABEL_NOAMAL), {
      defaultText: isShow ? defaultText : null,
      inheritColor: visualColor,
      defaultOpacity: visualOpacity,
      labelFetcher: seriesModel,
      labelDataIndex: thisNode.dataIndex
    });
    var textEl = rectEl.getTextContent();
    if (!textEl) {
      return;
    }
    var textStyle = textEl.style;
    var textPadding = normalizeCssArray(textStyle.padding || 0);
    if (upperLabelRect) {
      rectEl.setTextConfig({
        layoutRect: upperLabelRect
      });
      textEl.disableLabelLayout = true;
    }
    textEl.beforeUpdate = function () {
      var width = Math.max((upperLabelRect ? upperLabelRect.width : rectEl.shape.width) - textPadding[1] - textPadding[3], 0);
      var height = Math.max((upperLabelRect ? upperLabelRect.height : rectEl.shape.height) - textPadding[0] - textPadding[2], 0);
      if (textStyle.width !== width || textStyle.height !== height) {
        textEl.setStyle({
          width: width,
          height: height
        });
      }
    };
    textStyle.truncateMinChar = 2;
    textStyle.lineOverflow = 'truncate';
    addDrillDownIcon(textStyle, upperLabelRect, thisLayout);
    var textEmphasisState = textEl.getState('emphasis');
    addDrillDownIcon(textEmphasisState ? textEmphasisState.style : null, upperLabelRect, thisLayout);
  }
  function addDrillDownIcon(style, upperLabelRect, thisLayout) {
    var text = style ? style.text : null;
    if (!upperLabelRect && thisLayout.isLeafRoot && text != null) {
      var iconChar = seriesModel.get('drillDownIcon', true);
      style.text = iconChar ? iconChar + ' ' + text : text;
    }
  }
  function giveGraphic(storageName, Ctor, depth, z) {
    var element = oldRawIndex != null && oldStorage[storageName][oldRawIndex];
    var lasts = lastsForAnimation[storageName];
    if (element) {
      // Remove from oldStorage
      oldStorage[storageName][oldRawIndex] = null;
      prepareAnimationWhenHasOld(lasts, element);
    }
    // If invisible and no old element, do not create new element (for optimizing).
    else if (!thisInvisible) {
      element = new Ctor();
      if (element instanceof Displayable) {
        element.z2 = calculateZ2(depth, z);
      }
      prepareAnimationWhenNoOld(lasts, element);
    }
    // Set to thisStorage
    return thisStorage[storageName][thisRawIndex] = element;
  }
  function prepareAnimationWhenHasOld(lasts, element) {
    var lastCfg = lasts[thisRawIndex] = {};
    if (element instanceof Group) {
      lastCfg.oldX = element.x;
      lastCfg.oldY = element.y;
    } else {
      lastCfg.oldShape = extend({}, element.shape);
    }
  }
  // If a element is new, we need to find the animation start point carefully,
  // otherwise it will looks strange when 'zoomToNode'.
  function prepareAnimationWhenNoOld(lasts, element) {
    var lastCfg = lasts[thisRawIndex] = {};
    var parentNode = thisNode.parentNode;
    var isGroup = element instanceof graphic.Group;
    if (parentNode && (!reRoot || reRoot.direction === 'drillDown')) {
      var parentOldX = 0;
      var parentOldY = 0;
      // New nodes appear from right-bottom corner in 'zoomToNode' animation.
      // For convenience, get old bounding rect from background.
      var parentOldBg = lastsForAnimation.background[parentNode.getRawIndex()];
      if (!reRoot && parentOldBg && parentOldBg.oldShape) {
        parentOldX = parentOldBg.oldShape.width;
        parentOldY = parentOldBg.oldShape.height;
      }
      // When no parent old shape found, its parent is new too,
      // so we can just use {x:0, y:0}.
      if (isGroup) {
        lastCfg.oldX = 0;
        lastCfg.oldY = parentOldY;
      } else {
        lastCfg.oldShape = {
          x: parentOldX,
          y: parentOldY,
          width: 0,
          height: 0
        };
      }
    }
    // Fade in, user can be aware that these nodes are new.
    lastCfg.fadein = !isGroup;
  }
}
// We cannot set all background with the same z, because the behaviour of
// drill down and roll up differ background creation sequence from tree
// hierarchy sequence, which cause lower background elements to overlap
// upper ones. So we calculate z based on depth.
// Moreover, we try to shrink down z interval to [0, 1] to avoid that
// treemap with large z overlaps other components.
function calculateZ2(depth, z2InLevel) {
  return depth * Z2_BASE + z2InLevel;
}
export default TreemapView;