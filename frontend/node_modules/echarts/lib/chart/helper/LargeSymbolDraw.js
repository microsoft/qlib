
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
/* global Float32Array */
// TODO Batch by color
import * as graphic from '../../util/graphic.js';
import { createSymbol } from '../../util/symbol.js';
import { getECData } from '../../util/innerStore.js';
var BOOST_SIZE_THRESHOLD = 4;
var LargeSymbolPathShape = /** @class */function () {
  function LargeSymbolPathShape() {}
  return LargeSymbolPathShape;
}();
var LargeSymbolPath = /** @class */function (_super) {
  __extends(LargeSymbolPath, _super);
  function LargeSymbolPath(opts) {
    var _this = _super.call(this, opts) || this;
    _this._off = 0;
    _this.hoverDataIdx = -1;
    return _this;
  }
  LargeSymbolPath.prototype.getDefaultShape = function () {
    return new LargeSymbolPathShape();
  };
  LargeSymbolPath.prototype.reset = function () {
    this.notClear = false;
    this._off = 0;
  };
  LargeSymbolPath.prototype.buildPath = function (path, shape) {
    var points = shape.points;
    var size = shape.size;
    var symbolProxy = this.symbolProxy;
    var symbolProxyShape = symbolProxy.shape;
    var ctx = path.getContext ? path.getContext() : path;
    var canBoost = ctx && size[0] < BOOST_SIZE_THRESHOLD;
    var softClipShape = this.softClipShape;
    var i;
    // Do draw in afterBrush.
    if (canBoost) {
      this._ctx = ctx;
      return;
    }
    this._ctx = null;
    for (i = this._off; i < points.length;) {
      var x = points[i++];
      var y = points[i++];
      if (isNaN(x) || isNaN(y)) {
        continue;
      }
      if (softClipShape && !softClipShape.contain(x, y)) {
        continue;
      }
      symbolProxyShape.x = x - size[0] / 2;
      symbolProxyShape.y = y - size[1] / 2;
      symbolProxyShape.width = size[0];
      symbolProxyShape.height = size[1];
      symbolProxy.buildPath(path, symbolProxyShape, true);
    }
    if (this.incremental) {
      this._off = i;
      this.notClear = true;
    }
  };
  LargeSymbolPath.prototype.afterBrush = function () {
    var shape = this.shape;
    var points = shape.points;
    var size = shape.size;
    var ctx = this._ctx;
    var softClipShape = this.softClipShape;
    var i;
    if (!ctx) {
      return;
    }
    // PENDING If style or other canvas status changed?
    for (i = this._off; i < points.length;) {
      var x = points[i++];
      var y = points[i++];
      if (isNaN(x) || isNaN(y)) {
        continue;
      }
      if (softClipShape && !softClipShape.contain(x, y)) {
        continue;
      }
      // fillRect is faster than building a rect path and draw.
      // And it support light globalCompositeOperation.
      ctx.fillRect(x - size[0] / 2, y - size[1] / 2, size[0], size[1]);
    }
    if (this.incremental) {
      this._off = i;
      this.notClear = true;
    }
  };
  LargeSymbolPath.prototype.findDataIndex = function (x, y) {
    // TODO ???
    // Consider transform
    var shape = this.shape;
    var points = shape.points;
    var size = shape.size;
    var w = Math.max(size[0], 4);
    var h = Math.max(size[1], 4);
    // Not consider transform
    // Treat each element as a rect
    // top down traverse
    for (var idx = points.length / 2 - 1; idx >= 0; idx--) {
      var i = idx * 2;
      var x0 = points[i] - w / 2;
      var y0 = points[i + 1] - h / 2;
      if (x >= x0 && y >= y0 && x <= x0 + w && y <= y0 + h) {
        return idx;
      }
    }
    return -1;
  };
  LargeSymbolPath.prototype.contain = function (x, y) {
    var localPos = this.transformCoordToLocal(x, y);
    var rect = this.getBoundingRect();
    x = localPos[0];
    y = localPos[1];
    if (rect.contain(x, y)) {
      // Cache found data index.
      var dataIdx = this.hoverDataIdx = this.findDataIndex(x, y);
      return dataIdx >= 0;
    }
    this.hoverDataIdx = -1;
    return false;
  };
  LargeSymbolPath.prototype.getBoundingRect = function () {
    // Ignore stroke for large symbol draw.
    var rect = this._rect;
    if (!rect) {
      var shape = this.shape;
      var points = shape.points;
      var size = shape.size;
      var w = size[0];
      var h = size[1];
      var minX = Infinity;
      var minY = Infinity;
      var maxX = -Infinity;
      var maxY = -Infinity;
      for (var i = 0; i < points.length;) {
        var x = points[i++];
        var y = points[i++];
        minX = Math.min(x, minX);
        maxX = Math.max(x, maxX);
        minY = Math.min(y, minY);
        maxY = Math.max(y, maxY);
      }
      rect = this._rect = new graphic.BoundingRect(minX - w / 2, minY - h / 2, maxX - minX + w, maxY - minY + h);
    }
    return rect;
  };
  return LargeSymbolPath;
}(graphic.Path);
var LargeSymbolDraw = /** @class */function () {
  function LargeSymbolDraw() {
    this.group = new graphic.Group();
  }
  /**
   * Update symbols draw by new data
   */
  LargeSymbolDraw.prototype.updateData = function (data, opt) {
    this._clear();
    var symbolEl = this._create();
    symbolEl.setShape({
      points: data.getLayout('points')
    });
    this._setCommon(symbolEl, data, opt);
  };
  LargeSymbolDraw.prototype.updateLayout = function (data) {
    var points = data.getLayout('points');
    this.group.eachChild(function (child) {
      if (child.startIndex != null) {
        var len = (child.endIndex - child.startIndex) * 2;
        var byteOffset = child.startIndex * 4 * 2;
        points = new Float32Array(points.buffer, byteOffset, len);
      }
      child.setShape('points', points);
      // Reset draw cursor.
      child.reset();
    });
  };
  LargeSymbolDraw.prototype.incrementalPrepareUpdate = function (data) {
    this._clear();
  };
  LargeSymbolDraw.prototype.incrementalUpdate = function (taskParams, data, opt) {
    var lastAdded = this._newAdded[0];
    var points = data.getLayout('points');
    var oldPoints = lastAdded && lastAdded.shape.points;
    // Merging the exists. Each element has 1e4 points.
    // Consider the performance balance between too much elements and too much points in one shape(may affect hover optimization)
    if (oldPoints && oldPoints.length < 2e4) {
      var oldLen = oldPoints.length;
      var newPoints = new Float32Array(oldLen + points.length);
      // Concat two array
      newPoints.set(oldPoints);
      newPoints.set(points, oldLen);
      // Update endIndex
      lastAdded.endIndex = taskParams.end;
      lastAdded.setShape({
        points: newPoints
      });
    } else {
      // Clear
      this._newAdded = [];
      var symbolEl = this._create();
      symbolEl.startIndex = taskParams.start;
      symbolEl.endIndex = taskParams.end;
      symbolEl.incremental = true;
      symbolEl.setShape({
        points: points
      });
      this._setCommon(symbolEl, data, opt);
    }
  };
  LargeSymbolDraw.prototype.eachRendered = function (cb) {
    this._newAdded[0] && cb(this._newAdded[0]);
  };
  LargeSymbolDraw.prototype._create = function () {
    var symbolEl = new LargeSymbolPath({
      cursor: 'default'
    });
    symbolEl.ignoreCoarsePointer = true;
    this.group.add(symbolEl);
    this._newAdded.push(symbolEl);
    return symbolEl;
  };
  LargeSymbolDraw.prototype._setCommon = function (symbolEl, data, opt) {
    var hostModel = data.hostModel;
    opt = opt || {};
    var size = data.getVisual('symbolSize');
    symbolEl.setShape('size', size instanceof Array ? size : [size, size]);
    symbolEl.softClipShape = opt.clipShape || null;
    // Create symbolProxy to build path for each data
    symbolEl.symbolProxy = createSymbol(data.getVisual('symbol'), 0, 0, 0, 0);
    // Use symbolProxy setColor method
    symbolEl.setColor = symbolEl.symbolProxy.setColor;
    var extrudeShadow = symbolEl.shape.size[0] < BOOST_SIZE_THRESHOLD;
    symbolEl.useStyle(
    // Draw shadow when doing fillRect is extremely slow.
    hostModel.getModel('itemStyle').getItemStyle(extrudeShadow ? ['color', 'shadowBlur', 'shadowColor'] : ['color']));
    var globalStyle = data.getVisual('style');
    var visualColor = globalStyle && globalStyle.fill;
    if (visualColor) {
      symbolEl.setColor(visualColor);
    }
    var ecData = getECData(symbolEl);
    // Enable tooltip
    // PENDING May have performance issue when path is extremely large
    ecData.seriesIndex = hostModel.seriesIndex;
    symbolEl.on('mousemove', function (e) {
      ecData.dataIndex = null;
      var dataIndex = symbolEl.hoverDataIdx;
      if (dataIndex >= 0) {
        // Provide dataIndex for tooltip
        ecData.dataIndex = dataIndex + (symbolEl.startIndex || 0);
      }
    });
  };
  LargeSymbolDraw.prototype.remove = function () {
    this._clear();
  };
  LargeSymbolDraw.prototype._clear = function () {
    this._newAdded = [];
    this.group.removeAll();
  };
  return LargeSymbolDraw;
}();
export default LargeSymbolDraw;