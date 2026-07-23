
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
import DataZoomView from './DataZoomView.js';
import sliderMove from '../helper/sliderMove.js';
import * as roams from './roams.js';
import { bind } from 'zrender/lib/core/util.js';
var InsideZoomView = /** @class */function (_super) {
  __extends(InsideZoomView, _super);
  function InsideZoomView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = 'dataZoom.inside';
    return _this;
  }
  InsideZoomView.prototype.render = function (dataZoomModel, ecModel, api) {
    _super.prototype.render.apply(this, arguments);
    if (dataZoomModel.noTarget()) {
      this._clear();
      return;
    }
    // Hence the `throttle` util ensures to preserve command order,
    // here simply updating range all the time will not cause missing
    // any of the the roam change.
    this.range = dataZoomModel.getPercentRange();
    // Reset controllers.
    roams.setViewInfoToCoordSysRecord(api, dataZoomModel, {
      pan: bind(getRangeHandlers.pan, this),
      zoom: bind(getRangeHandlers.zoom, this),
      scrollMove: bind(getRangeHandlers.scrollMove, this)
    });
  };
  InsideZoomView.prototype.dispose = function () {
    this._clear();
    _super.prototype.dispose.apply(this, arguments);
  };
  InsideZoomView.prototype._clear = function () {
    roams.disposeCoordSysRecordIfNeeded(this.api, this.dataZoomModel);
    this.range = null;
  };
  InsideZoomView.type = 'dataZoom.inside';
  return InsideZoomView;
}(DataZoomView);
var getRangeHandlers = {
  zoom: function (coordSysInfo, coordSysMainType, controller, e) {
    var lastRange = this.range;
    var range = lastRange.slice();
    // Calculate transform by the first axis.
    var axisModel = coordSysInfo.axisModels[0];
    if (!axisModel) {
      return;
    }
    var directionInfo = getDirectionInfo[coordSysMainType](null, [e.originX, e.originY], axisModel, controller, coordSysInfo);
    var percentPoint = (directionInfo.signal > 0 ? directionInfo.pixelStart + directionInfo.pixelLength - directionInfo.pixel : directionInfo.pixel - directionInfo.pixelStart) / directionInfo.pixelLength * (range[1] - range[0]) + range[0];
    var scale = Math.max(1 / e.scale, 0);
    range[0] = (range[0] - percentPoint) * scale + percentPoint;
    range[1] = (range[1] - percentPoint) * scale + percentPoint;
    // Restrict range.
    var minMaxSpan = this.dataZoomModel.findRepresentativeAxisProxy().getMinMaxSpan();
    sliderMove(0, range, [0, 100], 0, minMaxSpan.minSpan, minMaxSpan.maxSpan);
    this.range = range;
    if (lastRange[0] !== range[0] || lastRange[1] !== range[1]) {
      return range;
    }
  },
  pan: makeMover(function (range, axisModel, coordSysInfo, coordSysMainType, controller, e) {
    var directionInfo = getDirectionInfo[coordSysMainType]([e.oldX, e.oldY], [e.newX, e.newY], axisModel, controller, coordSysInfo);
    return directionInfo.signal * (range[1] - range[0]) * directionInfo.pixel / directionInfo.pixelLength;
  }),
  scrollMove: makeMover(function (range, axisModel, coordSysInfo, coordSysMainType, controller, e) {
    var directionInfo = getDirectionInfo[coordSysMainType]([0, 0], [e.scrollDelta, e.scrollDelta], axisModel, controller, coordSysInfo);
    return directionInfo.signal * (range[1] - range[0]) * e.scrollDelta;
  })
};
function makeMover(getPercentDelta) {
  return function (coordSysInfo, coordSysMainType, controller, e) {
    var lastRange = this.range;
    var range = lastRange.slice();
    // Calculate transform by the first axis.
    var axisModel = coordSysInfo.axisModels[0];
    if (!axisModel) {
      return;
    }
    var percentDelta = getPercentDelta(range, axisModel, coordSysInfo, coordSysMainType, controller, e);
    sliderMove(percentDelta, range, [0, 100], 'all');
    this.range = range;
    if (lastRange[0] !== range[0] || lastRange[1] !== range[1]) {
      return range;
    }
  };
}
var getDirectionInfo = {
  grid: function (oldPoint, newPoint, axisModel, controller, coordSysInfo) {
    var axis = axisModel.axis;
    var ret = {};
    var rect = coordSysInfo.model.coordinateSystem.getRect();
    oldPoint = oldPoint || [0, 0];
    if (axis.dim === 'x') {
      ret.pixel = newPoint[0] - oldPoint[0];
      ret.pixelLength = rect.width;
      ret.pixelStart = rect.x;
      ret.signal = axis.inverse ? 1 : -1;
    } else {
      // axis.dim === 'y'
      ret.pixel = newPoint[1] - oldPoint[1];
      ret.pixelLength = rect.height;
      ret.pixelStart = rect.y;
      ret.signal = axis.inverse ? -1 : 1;
    }
    return ret;
  },
  polar: function (oldPoint, newPoint, axisModel, controller, coordSysInfo) {
    var axis = axisModel.axis;
    var ret = {};
    var polar = coordSysInfo.model.coordinateSystem;
    var radiusExtent = polar.getRadiusAxis().getExtent();
    var angleExtent = polar.getAngleAxis().getExtent();
    oldPoint = oldPoint ? polar.pointToCoord(oldPoint) : [0, 0];
    newPoint = polar.pointToCoord(newPoint);
    if (axisModel.mainType === 'radiusAxis') {
      ret.pixel = newPoint[0] - oldPoint[0];
      // ret.pixelLength = Math.abs(radiusExtent[1] - radiusExtent[0]);
      // ret.pixelStart = Math.min(radiusExtent[0], radiusExtent[1]);
      ret.pixelLength = radiusExtent[1] - radiusExtent[0];
      ret.pixelStart = radiusExtent[0];
      ret.signal = axis.inverse ? 1 : -1;
    } else {
      // 'angleAxis'
      ret.pixel = newPoint[1] - oldPoint[1];
      // ret.pixelLength = Math.abs(angleExtent[1] - angleExtent[0]);
      // ret.pixelStart = Math.min(angleExtent[0], angleExtent[1]);
      ret.pixelLength = angleExtent[1] - angleExtent[0];
      ret.pixelStart = angleExtent[0];
      ret.signal = axis.inverse ? -1 : 1;
    }
    return ret;
  },
  singleAxis: function (oldPoint, newPoint, axisModel, controller, coordSysInfo) {
    var axis = axisModel.axis;
    var rect = coordSysInfo.model.coordinateSystem.getRect();
    var ret = {};
    oldPoint = oldPoint || [0, 0];
    if (axis.orient === 'horizontal') {
      ret.pixel = newPoint[0] - oldPoint[0];
      ret.pixelLength = rect.width;
      ret.pixelStart = rect.x;
      ret.signal = axis.inverse ? 1 : -1;
    } else {
      // 'vertical'
      ret.pixel = newPoint[1] - oldPoint[1];
      ret.pixelLength = rect.height;
      ret.pixelStart = rect.y;
      ret.signal = axis.inverse ? -1 : 1;
    }
    return ret;
  }
};
export default InsideZoomView;