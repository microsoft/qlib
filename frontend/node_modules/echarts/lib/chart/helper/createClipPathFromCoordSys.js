
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
import * as graphic from '../../util/graphic.js';
import { round } from '../../util/number.js';
import { isFunction } from 'zrender/lib/core/util.js';
function createGridClipPath(cartesian, hasAnimation, seriesModel, done, during) {
  var rect = cartesian.getArea();
  var x = rect.x;
  var y = rect.y;
  var width = rect.width;
  var height = rect.height;
  var lineWidth = seriesModel.get(['lineStyle', 'width']) || 0;
  // Expand the clip path a bit to avoid the border is clipped and looks thinner
  x -= lineWidth / 2;
  y -= lineWidth / 2;
  width += lineWidth;
  height += lineWidth;
  // fix: https://github.com/apache/incubator-echarts/issues/11369
  width = Math.ceil(width);
  if (x !== Math.floor(x)) {
    x = Math.floor(x);
    // if no extra 1px on `width`, it will still be clipped since `x` is floored
    width++;
  }
  var clipPath = new graphic.Rect({
    shape: {
      x: x,
      y: y,
      width: width,
      height: height
    }
  });
  if (hasAnimation) {
    var baseAxis = cartesian.getBaseAxis();
    var isHorizontal = baseAxis.isHorizontal();
    var isAxisInversed = baseAxis.inverse;
    if (isHorizontal) {
      if (isAxisInversed) {
        clipPath.shape.x += width;
      }
      clipPath.shape.width = 0;
    } else {
      if (!isAxisInversed) {
        clipPath.shape.y += height;
      }
      clipPath.shape.height = 0;
    }
    var duringCb = isFunction(during) ? function (percent) {
      during(percent, clipPath);
    } : null;
    graphic.initProps(clipPath, {
      shape: {
        width: width,
        height: height,
        x: x,
        y: y
      }
    }, seriesModel, null, done, duringCb);
  }
  return clipPath;
}
function createPolarClipPath(polar, hasAnimation, seriesModel) {
  var sectorArea = polar.getArea();
  // Avoid float number rounding error for symbol on the edge of axis extent.
  var r0 = round(sectorArea.r0, 1);
  var r = round(sectorArea.r, 1);
  var clipPath = new graphic.Sector({
    shape: {
      cx: round(polar.cx, 1),
      cy: round(polar.cy, 1),
      r0: r0,
      r: r,
      startAngle: sectorArea.startAngle,
      endAngle: sectorArea.endAngle,
      clockwise: sectorArea.clockwise
    }
  });
  if (hasAnimation) {
    var isRadial = polar.getBaseAxis().dim === 'angle';
    if (isRadial) {
      clipPath.shape.endAngle = sectorArea.startAngle;
    } else {
      clipPath.shape.r = r0;
    }
    graphic.initProps(clipPath, {
      shape: {
        endAngle: sectorArea.endAngle,
        r: r
      }
    }, seriesModel);
  }
  return clipPath;
}
function createClipPath(coordSys, hasAnimation, seriesModel, done, during) {
  if (!coordSys) {
    return null;
  } else if (coordSys.type === 'polar') {
    return createPolarClipPath(coordSys, hasAnimation, seriesModel);
  } else if (coordSys.type === 'cartesian2d') {
    return createGridClipPath(coordSys, hasAnimation, seriesModel, done, during);
  }
  return null;
}
export { createGridClipPath, createPolarClipPath, createClipPath };