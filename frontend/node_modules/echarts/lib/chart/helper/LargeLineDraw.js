
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
// TODO Batch by color
import * as graphic from '../../util/graphic.js';
import * as lineContain from 'zrender/lib/contain/line.js';
import * as quadraticContain from 'zrender/lib/contain/quadratic.js';
import { getECData } from '../../util/innerStore.js';
var LargeLinesPathShape = /** @class */function () {
  function LargeLinesPathShape() {
    this.polyline = false;
    this.curveness = 0;
    this.segs = [];
  }
  return LargeLinesPathShape;
}();
var LargeLinesPath = /** @class */function (_super) {
  __extends(LargeLinesPath, _super);
  function LargeLinesPath(opts) {
    var _this = _super.call(this, opts) || this;
    _this._off = 0;
    _this.hoverDataIdx = -1;
    return _this;
  }
  LargeLinesPath.prototype.reset = function () {
    this.notClear = false;
    this._off = 0;
  };
  LargeLinesPath.prototype.getDefaultStyle = function () {
    return {
      stroke: '#000',
      fill: null
    };
  };
  LargeLinesPath.prototype.getDefaultShape = function () {
    return new LargeLinesPathShape();
  };
  LargeLinesPath.prototype.buildPath = function (ctx, shape) {
    var segs = shape.segs;
    var curveness = shape.curveness;
    var i;
    if (shape.polyline) {
      for (i = this._off; i < segs.length;) {
        var count = segs[i++];
        if (count > 0) {
          ctx.moveTo(segs[i++], segs[i++]);
          for (var k = 1; k < count; k++) {
            ctx.lineTo(segs[i++], segs[i++]);
          }
        }
      }
    } else {
      for (i = this._off; i < segs.length;) {
        var x0 = segs[i++];
        var y0 = segs[i++];
        var x1 = segs[i++];
        var y1 = segs[i++];
        ctx.moveTo(x0, y0);
        if (curveness > 0) {
          var x2 = (x0 + x1) / 2 - (y0 - y1) * curveness;
          var y2 = (y0 + y1) / 2 - (x1 - x0) * curveness;
          ctx.quadraticCurveTo(x2, y2, x1, y1);
        } else {
          ctx.lineTo(x1, y1);
        }
      }
    }
    if (this.incremental) {
      this._off = i;
      this.notClear = true;
    }
  };
  LargeLinesPath.prototype.findDataIndex = function (x, y) {
    var shape = this.shape;
    var segs = shape.segs;
    var curveness = shape.curveness;
    var lineWidth = this.style.lineWidth;
    if (shape.polyline) {
      var dataIndex = 0;
      for (var i = 0; i < segs.length;) {
        var count = segs[i++];
        if (count > 0) {
          var x0 = segs[i++];
          var y0 = segs[i++];
          for (var k = 1; k < count; k++) {
            var x1 = segs[i++];
            var y1 = segs[i++];
            if (lineContain.containStroke(x0, y0, x1, y1, lineWidth, x, y)) {
              return dataIndex;
            }
          }
        }
        dataIndex++;
      }
    } else {
      var dataIndex = 0;
      for (var i = 0; i < segs.length;) {
        var x0 = segs[i++];
        var y0 = segs[i++];
        var x1 = segs[i++];
        var y1 = segs[i++];
        if (curveness > 0) {
          var x2 = (x0 + x1) / 2 - (y0 - y1) * curveness;
          var y2 = (y0 + y1) / 2 - (x1 - x0) * curveness;
          if (quadraticContain.containStroke(x0, y0, x2, y2, x1, y1, lineWidth, x, y)) {
            return dataIndex;
          }
        } else {
          if (lineContain.containStroke(x0, y0, x1, y1, lineWidth, x, y)) {
            return dataIndex;
          }
        }
        dataIndex++;
      }
    }
    return -1;
  };
  LargeLinesPath.prototype.contain = function (x, y) {
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
  LargeLinesPath.prototype.getBoundingRect = function () {
    // Ignore stroke for large symbol draw.
    var rect = this._rect;
    if (!rect) {
      var shape = this.shape;
      var points = shape.segs;
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
      rect = this._rect = new graphic.BoundingRect(minX, minY, maxX, maxY);
    }
    return rect;
  };
  return LargeLinesPath;
}(graphic.Path);
var LargeLineDraw = /** @class */function () {
  function LargeLineDraw() {
    this.group = new graphic.Group();
  }
  /**
   * Update symbols draw by new data
   */
  LargeLineDraw.prototype.updateData = function (data) {
    this._clear();
    var lineEl = this._create();
    lineEl.setShape({
      segs: data.getLayout('linesPoints')
    });
    this._setCommon(lineEl, data);
  };
  ;
  /**
   * @override
   */
  LargeLineDraw.prototype.incrementalPrepareUpdate = function (data) {
    this.group.removeAll();
    this._clear();
  };
  ;
  /**
   * @override
   */
  LargeLineDraw.prototype.incrementalUpdate = function (taskParams, data) {
    var lastAdded = this._newAdded[0];
    var linePoints = data.getLayout('linesPoints');
    var oldSegs = lastAdded && lastAdded.shape.segs;
    // Merging the exists. Each element has 1e4 points.
    // Consider the performance balance between too much elements and too much points in one shape(may affect hover optimization)
    if (oldSegs && oldSegs.length < 2e4) {
      var oldLen = oldSegs.length;
      var newSegs = new Float32Array(oldLen + linePoints.length);
      // Concat two array
      newSegs.set(oldSegs);
      newSegs.set(linePoints, oldLen);
      lastAdded.setShape({
        segs: newSegs
      });
    } else {
      // Clear
      this._newAdded = [];
      var lineEl = this._create();
      lineEl.incremental = true;
      lineEl.setShape({
        segs: linePoints
      });
      this._setCommon(lineEl, data);
      lineEl.__startIndex = taskParams.start;
    }
  };
  /**
   * @override
   */
  LargeLineDraw.prototype.remove = function () {
    this._clear();
  };
  LargeLineDraw.prototype.eachRendered = function (cb) {
    this._newAdded[0] && cb(this._newAdded[0]);
  };
  LargeLineDraw.prototype._create = function () {
    var lineEl = new LargeLinesPath({
      cursor: 'default',
      ignoreCoarsePointer: true
    });
    this._newAdded.push(lineEl);
    this.group.add(lineEl);
    return lineEl;
  };
  LargeLineDraw.prototype._setCommon = function (lineEl, data, isIncremental) {
    var hostModel = data.hostModel;
    lineEl.setShape({
      polyline: hostModel.get('polyline'),
      curveness: hostModel.get(['lineStyle', 'curveness'])
    });
    lineEl.useStyle(hostModel.getModel('lineStyle').getLineStyle());
    lineEl.style.strokeNoScale = true;
    var style = data.getVisual('style');
    if (style && style.stroke) {
      lineEl.setStyle('stroke', style.stroke);
    }
    lineEl.setStyle('fill', null);
    var ecData = getECData(lineEl);
    // Enable tooltip
    // PENDING May have performance issue when path is extremely large
    ecData.seriesIndex = hostModel.seriesIndex;
    lineEl.on('mousemove', function (e) {
      ecData.dataIndex = null;
      var dataIndex = lineEl.hoverDataIdx;
      if (dataIndex > 0) {
        // Provide dataIndex for tooltip
        ecData.dataIndex = dataIndex + lineEl.__startIndex;
      }
    });
  };
  ;
  LargeLineDraw.prototype._clear = function () {
    this._newAdded = [];
    this.group.removeAll();
  };
  ;
  return LargeLineDraw;
}();
export default LargeLineDraw;