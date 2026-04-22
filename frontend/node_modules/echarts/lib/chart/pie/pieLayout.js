
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
import { parsePercent, linearMap } from '../../util/number.js';
import * as layout from '../../util/layout.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import { normalizeArcAngles } from 'zrender/lib/core/PathProxy.js';
import { makeInner } from '../../util/model.js';
var PI2 = Math.PI * 2;
var RADIAN = Math.PI / 180;
function getViewRect(seriesModel, api) {
  return layout.getLayoutRect(seriesModel.getBoxLayoutParams(), {
    width: api.getWidth(),
    height: api.getHeight()
  });
}
export function getBasicPieLayout(seriesModel, api) {
  var viewRect = getViewRect(seriesModel, api);
  // center can be string or number when coordinateSystem is specified
  var center = seriesModel.get('center');
  var radius = seriesModel.get('radius');
  if (!zrUtil.isArray(radius)) {
    radius = [0, radius];
  }
  var width = parsePercent(viewRect.width, api.getWidth());
  var height = parsePercent(viewRect.height, api.getHeight());
  var size = Math.min(width, height);
  var r0 = parsePercent(radius[0], size / 2);
  var r = parsePercent(radius[1], size / 2);
  var cx;
  var cy;
  var coordSys = seriesModel.coordinateSystem;
  if (coordSys) {
    // percentage is not allowed when coordinate system is specified
    var point = coordSys.dataToPoint(center);
    cx = point[0] || 0;
    cy = point[1] || 0;
  } else {
    if (!zrUtil.isArray(center)) {
      center = [center, center];
    }
    cx = parsePercent(center[0], width) + viewRect.x;
    cy = parsePercent(center[1], height) + viewRect.y;
  }
  return {
    cx: cx,
    cy: cy,
    r0: r0,
    r: r
  };
}
export default function pieLayout(seriesType, ecModel, api) {
  ecModel.eachSeriesByType(seriesType, function (seriesModel) {
    var data = seriesModel.getData();
    var valueDim = data.mapDimension('value');
    var viewRect = getViewRect(seriesModel, api);
    var _a = getBasicPieLayout(seriesModel, api),
      cx = _a.cx,
      cy = _a.cy,
      r = _a.r,
      r0 = _a.r0;
    var startAngle = -seriesModel.get('startAngle') * RADIAN;
    var endAngle = seriesModel.get('endAngle');
    var padAngle = seriesModel.get('padAngle') * RADIAN;
    endAngle = endAngle === 'auto' ? startAngle - PI2 : -endAngle * RADIAN;
    var minAngle = seriesModel.get('minAngle') * RADIAN;
    var minAndPadAngle = minAngle + padAngle;
    var validDataCount = 0;
    data.each(valueDim, function (value) {
      !isNaN(value) && validDataCount++;
    });
    var sum = data.getSum(valueDim);
    // Sum may be 0
    var unitRadian = Math.PI / (sum || validDataCount) * 2;
    var clockwise = seriesModel.get('clockwise');
    var roseType = seriesModel.get('roseType');
    var stillShowZeroSum = seriesModel.get('stillShowZeroSum');
    // [0...max]
    var extent = data.getDataExtent(valueDim);
    extent[0] = 0;
    var dir = clockwise ? 1 : -1;
    var angles = [startAngle, endAngle];
    var halfPadAngle = dir * padAngle / 2;
    normalizeArcAngles(angles, !clockwise);
    startAngle = angles[0], endAngle = angles[1];
    var layoutData = getSeriesLayoutData(seriesModel);
    layoutData.startAngle = startAngle;
    layoutData.endAngle = endAngle;
    layoutData.clockwise = clockwise;
    var angleRange = Math.abs(endAngle - startAngle);
    // In the case some sector angle is smaller than minAngle
    var restAngle = angleRange;
    var valueSumLargerThanMinAngle = 0;
    var currentAngle = startAngle;
    data.setLayout({
      viewRect: viewRect,
      r: r
    });
    data.each(valueDim, function (value, idx) {
      var angle;
      if (isNaN(value)) {
        data.setItemLayout(idx, {
          angle: NaN,
          startAngle: NaN,
          endAngle: NaN,
          clockwise: clockwise,
          cx: cx,
          cy: cy,
          r0: r0,
          r: roseType ? NaN : r
        });
        return;
      }
      // FIXME 兼容 2.0 但是 roseType 是 area 的时候才是这样？
      if (roseType !== 'area') {
        angle = sum === 0 && stillShowZeroSum ? unitRadian : value * unitRadian;
      } else {
        angle = angleRange / validDataCount;
      }
      if (angle < minAndPadAngle) {
        angle = minAndPadAngle;
        restAngle -= minAndPadAngle;
      } else {
        valueSumLargerThanMinAngle += value;
      }
      var endAngle = currentAngle + dir * angle;
      // calculate display angle
      var actualStartAngle = 0;
      var actualEndAngle = 0;
      if (padAngle > angle) {
        actualStartAngle = currentAngle + dir * angle / 2;
        actualEndAngle = actualStartAngle;
      } else {
        actualStartAngle = currentAngle + halfPadAngle;
        actualEndAngle = endAngle - halfPadAngle;
      }
      data.setItemLayout(idx, {
        angle: angle,
        startAngle: actualStartAngle,
        endAngle: actualEndAngle,
        clockwise: clockwise,
        cx: cx,
        cy: cy,
        r0: r0,
        r: roseType ? linearMap(value, extent, [r0, r]) : r
      });
      currentAngle = endAngle;
    });
    // Some sector is constrained by minAngle and padAngle
    // Rest sectors needs recalculate angle
    if (restAngle < PI2 && validDataCount) {
      // Average the angle if rest angle is not enough after all angles is
      // Constrained by minAngle and padAngle
      if (restAngle <= 1e-3) {
        var angle_1 = angleRange / validDataCount;
        data.each(valueDim, function (value, idx) {
          if (!isNaN(value)) {
            var layout_1 = data.getItemLayout(idx);
            layout_1.angle = angle_1;
            var actualStartAngle = 0;
            var actualEndAngle = 0;
            if (angle_1 < padAngle) {
              actualStartAngle = startAngle + dir * (idx + 1 / 2) * angle_1;
              actualEndAngle = actualStartAngle;
            } else {
              actualStartAngle = startAngle + dir * idx * angle_1 + halfPadAngle;
              actualEndAngle = startAngle + dir * (idx + 1) * angle_1 - halfPadAngle;
            }
            layout_1.startAngle = actualStartAngle;
            layout_1.endAngle = actualEndAngle;
          }
        });
      } else {
        unitRadian = restAngle / valueSumLargerThanMinAngle;
        currentAngle = startAngle;
        data.each(valueDim, function (value, idx) {
          if (!isNaN(value)) {
            var layout_2 = data.getItemLayout(idx);
            var angle = layout_2.angle === minAndPadAngle ? minAndPadAngle : value * unitRadian;
            var actualStartAngle = 0;
            var actualEndAngle = 0;
            if (angle < padAngle) {
              actualStartAngle = currentAngle + dir * angle / 2;
              actualEndAngle = actualStartAngle;
            } else {
              actualStartAngle = currentAngle + halfPadAngle;
              actualEndAngle = currentAngle + dir * angle - halfPadAngle;
            }
            layout_2.startAngle = actualStartAngle;
            layout_2.endAngle = actualEndAngle;
            currentAngle += dir * angle;
          }
        });
      }
    }
  });
}
export var getSeriesLayoutData = makeInner();