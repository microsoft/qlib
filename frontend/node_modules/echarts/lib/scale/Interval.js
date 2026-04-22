
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
import * as numberUtil from '../util/number.js';
import * as formatUtil from '../util/format.js';
import Scale from './Scale.js';
import * as helper from './helper.js';
var roundNumber = numberUtil.round;
var IntervalScale = /** @class */function (_super) {
  __extends(IntervalScale, _super);
  function IntervalScale() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = 'interval';
    // Step is calculated in adjustExtent.
    _this._interval = 0;
    _this._intervalPrecision = 2;
    return _this;
  }
  IntervalScale.prototype.parse = function (val) {
    return val;
  };
  IntervalScale.prototype.contain = function (val) {
    return helper.contain(val, this._extent);
  };
  IntervalScale.prototype.normalize = function (val) {
    return helper.normalize(val, this._extent);
  };
  IntervalScale.prototype.scale = function (val) {
    return helper.scale(val, this._extent);
  };
  IntervalScale.prototype.setExtent = function (start, end) {
    var thisExtent = this._extent;
    // start,end may be a Number like '25',so...
    if (!isNaN(start)) {
      thisExtent[0] = parseFloat(start);
    }
    if (!isNaN(end)) {
      thisExtent[1] = parseFloat(end);
    }
  };
  IntervalScale.prototype.unionExtent = function (other) {
    var extent = this._extent;
    other[0] < extent[0] && (extent[0] = other[0]);
    other[1] > extent[1] && (extent[1] = other[1]);
    // unionExtent may called by it's sub classes
    this.setExtent(extent[0], extent[1]);
  };
  IntervalScale.prototype.getInterval = function () {
    return this._interval;
  };
  IntervalScale.prototype.setInterval = function (interval) {
    this._interval = interval;
    // Dropped auto calculated niceExtent and use user-set extent.
    // We assume user wants to set both interval, min, max to get a better result.
    this._niceExtent = this._extent.slice();
    this._intervalPrecision = helper.getIntervalPrecision(interval);
  };
  /**
   * @param expandToNicedExtent Whether expand the ticks to niced extent.
   */
  IntervalScale.prototype.getTicks = function (expandToNicedExtent) {
    var interval = this._interval;
    var extent = this._extent;
    var niceTickExtent = this._niceExtent;
    var intervalPrecision = this._intervalPrecision;
    var ticks = [];
    // If interval is 0, return [];
    if (!interval) {
      return ticks;
    }
    // Consider this case: using dataZoom toolbox, zoom and zoom.
    var safeLimit = 10000;
    if (extent[0] < niceTickExtent[0]) {
      if (expandToNicedExtent) {
        ticks.push({
          value: roundNumber(niceTickExtent[0] - interval, intervalPrecision)
        });
      } else {
        ticks.push({
          value: extent[0]
        });
      }
    }
    var tick = niceTickExtent[0];
    while (tick <= niceTickExtent[1]) {
      ticks.push({
        value: tick
      });
      // Avoid rounding error
      tick = roundNumber(tick + interval, intervalPrecision);
      if (tick === ticks[ticks.length - 1].value) {
        // Consider out of safe float point, e.g.,
        // -3711126.9907707 + 2e-10 === -3711126.9907707
        break;
      }
      if (ticks.length > safeLimit) {
        return [];
      }
    }
    // Consider this case: the last item of ticks is smaller
    // than niceTickExtent[1] and niceTickExtent[1] === extent[1].
    var lastNiceTick = ticks.length ? ticks[ticks.length - 1].value : niceTickExtent[1];
    if (extent[1] > lastNiceTick) {
      if (expandToNicedExtent) {
        ticks.push({
          value: roundNumber(lastNiceTick + interval, intervalPrecision)
        });
      } else {
        ticks.push({
          value: extent[1]
        });
      }
    }
    return ticks;
  };
  IntervalScale.prototype.getMinorTicks = function (splitNumber) {
    var ticks = this.getTicks(true);
    var minorTicks = [];
    var extent = this.getExtent();
    for (var i = 1; i < ticks.length; i++) {
      var nextTick = ticks[i];
      var prevTick = ticks[i - 1];
      var count = 0;
      var minorTicksGroup = [];
      var interval = nextTick.value - prevTick.value;
      var minorInterval = interval / splitNumber;
      while (count < splitNumber - 1) {
        var minorTick = roundNumber(prevTick.value + (count + 1) * minorInterval);
        // For the first and last interval. The count may be less than splitNumber.
        if (minorTick > extent[0] && minorTick < extent[1]) {
          minorTicksGroup.push(minorTick);
        }
        count++;
      }
      minorTicks.push(minorTicksGroup);
    }
    return minorTicks;
  };
  /**
   * @param opt.precision If 'auto', use nice presision.
   * @param opt.pad returns 1.50 but not 1.5 if precision is 2.
   */
  IntervalScale.prototype.getLabel = function (data, opt) {
    if (data == null) {
      return '';
    }
    var precision = opt && opt.precision;
    if (precision == null) {
      precision = numberUtil.getPrecision(data.value) || 0;
    } else if (precision === 'auto') {
      // Should be more precise then tick.
      precision = this._intervalPrecision;
    }
    // (1) If `precision` is set, 12.005 should be display as '12.00500'.
    // (2) Use roundNumber (toFixed) to avoid scientific notation like '3.5e-7'.
    var dataNum = roundNumber(data.value, precision, true);
    return formatUtil.addCommas(dataNum);
  };
  /**
   * @param splitNumber By default `5`.
   */
  IntervalScale.prototype.calcNiceTicks = function (splitNumber, minInterval, maxInterval) {
    splitNumber = splitNumber || 5;
    var extent = this._extent;
    var span = extent[1] - extent[0];
    if (!isFinite(span)) {
      return;
    }
    // User may set axis min 0 and data are all negative
    // FIXME If it needs to reverse ?
    if (span < 0) {
      span = -span;
      extent.reverse();
    }
    var result = helper.intervalScaleNiceTicks(extent, splitNumber, minInterval, maxInterval);
    this._intervalPrecision = result.intervalPrecision;
    this._interval = result.interval;
    this._niceExtent = result.niceTickExtent;
  };
  IntervalScale.prototype.calcNiceExtent = function (opt) {
    var extent = this._extent;
    // If extent start and end are same, expand them
    if (extent[0] === extent[1]) {
      if (extent[0] !== 0) {
        // Expand extent
        // Note that extents can be both negative. See #13154
        var expandSize = Math.abs(extent[0]);
        // In the fowllowing case
        //      Axis has been fixed max 100
        //      Plus data are all 100 and axis extent are [100, 100].
        // Extend to the both side will cause expanded max is larger than fixed max.
        // So only expand to the smaller side.
        if (!opt.fixMax) {
          extent[1] += expandSize / 2;
          extent[0] -= expandSize / 2;
        } else {
          extent[0] -= expandSize / 2;
        }
      } else {
        extent[1] = 1;
      }
    }
    var span = extent[1] - extent[0];
    // If there are no data and extent are [Infinity, -Infinity]
    if (!isFinite(span)) {
      extent[0] = 0;
      extent[1] = 1;
    }
    this.calcNiceTicks(opt.splitNumber, opt.minInterval, opt.maxInterval);
    // let extent = this._extent;
    var interval = this._interval;
    if (!opt.fixMin) {
      extent[0] = roundNumber(Math.floor(extent[0] / interval) * interval);
    }
    if (!opt.fixMax) {
      extent[1] = roundNumber(Math.ceil(extent[1] / interval) * interval);
    }
  };
  IntervalScale.prototype.setNiceExtent = function (min, max) {
    this._niceExtent = [min, max];
  };
  IntervalScale.type = 'interval';
  return IntervalScale;
}(Scale);
Scale.registerClass(IntervalScale);
export default IntervalScale;