
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
import { getPrecisionSafe, round } from '../util/number.js';
import IntervalScale from '../scale/Interval.js';
import { getScaleExtent } from './axisHelper.js';
import { warn } from '../util/log.js';
import { increaseInterval, isValueNice } from '../scale/helper.js';
var mathLog = Math.log;
export function alignScaleTicks(scale, axisModel, alignToScale) {
  var intervalScaleProto = IntervalScale.prototype;
  // NOTE: There is a precondition for log scale  here:
  // In log scale we store _interval and _extent of exponent value.
  // So if we use the method of InternalScale to set/get these data.
  // It process the exponent value, which is linear and what we want here.
  var alignToTicks = intervalScaleProto.getTicks.call(alignToScale);
  var alignToNicedTicks = intervalScaleProto.getTicks.call(alignToScale, true);
  var alignToSplitNumber = alignToTicks.length - 1;
  var alignToInterval = intervalScaleProto.getInterval.call(alignToScale);
  var scaleExtent = getScaleExtent(scale, axisModel);
  var rawExtent = scaleExtent.extent;
  var isMinFixed = scaleExtent.fixMin;
  var isMaxFixed = scaleExtent.fixMax;
  if (scale.type === 'log') {
    var logBase = mathLog(scale.base);
    rawExtent = [mathLog(rawExtent[0]) / logBase, mathLog(rawExtent[1]) / logBase];
  }
  scale.setExtent(rawExtent[0], rawExtent[1]);
  scale.calcNiceExtent({
    splitNumber: alignToSplitNumber,
    fixMin: isMinFixed,
    fixMax: isMaxFixed
  });
  var extent = intervalScaleProto.getExtent.call(scale);
  // Need to update the rawExtent.
  // Because value in rawExtent may be not parsed. e.g. 'dataMin', 'dataMax'
  if (isMinFixed) {
    rawExtent[0] = extent[0];
  }
  if (isMaxFixed) {
    rawExtent[1] = extent[1];
  }
  var interval = intervalScaleProto.getInterval.call(scale);
  var min = rawExtent[0];
  var max = rawExtent[1];
  if (isMinFixed && isMaxFixed) {
    // User set min, max, divide to get new interval
    interval = (max - min) / alignToSplitNumber;
  } else if (isMinFixed) {
    max = rawExtent[0] + interval * alignToSplitNumber;
    // User set min, expand extent on the other side
    while (max < rawExtent[1] && isFinite(max) && isFinite(rawExtent[1])) {
      interval = increaseInterval(interval);
      max = rawExtent[0] + interval * alignToSplitNumber;
    }
  } else if (isMaxFixed) {
    // User set max, expand extent on the other side
    min = rawExtent[1] - interval * alignToSplitNumber;
    while (min > rawExtent[0] && isFinite(min) && isFinite(rawExtent[0])) {
      interval = increaseInterval(interval);
      min = rawExtent[1] - interval * alignToSplitNumber;
    }
  } else {
    var nicedSplitNumber = scale.getTicks().length - 1;
    if (nicedSplitNumber > alignToSplitNumber) {
      interval = increaseInterval(interval);
    }
    var range = interval * alignToSplitNumber;
    max = Math.ceil(rawExtent[1] / interval) * interval;
    min = round(max - range);
    // Not change the result that crossing zero.
    if (min < 0 && rawExtent[0] >= 0) {
      min = 0;
      max = round(range);
    } else if (max > 0 && rawExtent[1] <= 0) {
      max = 0;
      min = -round(range);
    }
  }
  // Adjust min, max based on the extent of alignTo. When min or max is set in alignTo scale
  var t0 = (alignToTicks[0].value - alignToNicedTicks[0].value) / alignToInterval;
  var t1 = (alignToTicks[alignToSplitNumber].value - alignToNicedTicks[alignToSplitNumber].value) / alignToInterval;
  // NOTE: Must in setExtent -> setInterval -> setNiceExtent order.
  intervalScaleProto.setExtent.call(scale, min + interval * t0, max + interval * t1);
  intervalScaleProto.setInterval.call(scale, interval);
  if (t0 || t1) {
    intervalScaleProto.setNiceExtent.call(scale, min + interval, max - interval);
  }
  if (process.env.NODE_ENV !== 'production') {
    var ticks = intervalScaleProto.getTicks.call(scale);
    if (ticks[1] && (!isValueNice(interval) || getPrecisionSafe(ticks[1].value) > getPrecisionSafe(interval))) {
      warn(
      // eslint-disable-next-line
      "The ticks may be not readable when set min: " + axisModel.get('min') + ", max: " + axisModel.get('max') + " and alignTicks: true");
    }
  }
}