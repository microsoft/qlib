
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
import BaseAxisPointer from './BaseAxisPointer.js';
import * as viewHelper from './viewHelper.js';
import * as singleAxisHelper from '../../coord/single/singleAxisHelper.js';
var XY = ['x', 'y'];
var WH = ['width', 'height'];
var SingleAxisPointer = /** @class */function (_super) {
  __extends(SingleAxisPointer, _super);
  function SingleAxisPointer() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  /**
   * @override
   */
  SingleAxisPointer.prototype.makeElOption = function (elOption, value, axisModel, axisPointerModel, api) {
    var axis = axisModel.axis;
    var coordSys = axis.coordinateSystem;
    var otherExtent = getGlobalExtent(coordSys, 1 - getPointDimIndex(axis));
    var pixelValue = coordSys.dataToPoint(value)[0];
    var axisPointerType = axisPointerModel.get('type');
    if (axisPointerType && axisPointerType !== 'none') {
      var elStyle = viewHelper.buildElStyle(axisPointerModel);
      var pointerOption = pointerShapeBuilder[axisPointerType](axis, pixelValue, otherExtent);
      pointerOption.style = elStyle;
      elOption.graphicKey = pointerOption.type;
      elOption.pointer = pointerOption;
    }
    var layoutInfo = singleAxisHelper.layout(axisModel);
    viewHelper.buildCartesianSingleLabelElOption(
    // @ts-ignore
    value, elOption, layoutInfo, axisModel, axisPointerModel, api);
  };
  /**
   * @override
   */
  SingleAxisPointer.prototype.getHandleTransform = function (value, axisModel, axisPointerModel) {
    var layoutInfo = singleAxisHelper.layout(axisModel, {
      labelInside: false
    });
    // @ts-ignore
    layoutInfo.labelMargin = axisPointerModel.get(['handle', 'margin']);
    var position = viewHelper.getTransformedPosition(axisModel.axis, value, layoutInfo);
    return {
      x: position[0],
      y: position[1],
      rotation: layoutInfo.rotation + (layoutInfo.labelDirection < 0 ? Math.PI : 0)
    };
  };
  /**
   * @override
   */
  SingleAxisPointer.prototype.updateHandleTransform = function (transform, delta, axisModel, axisPointerModel) {
    var axis = axisModel.axis;
    var coordSys = axis.coordinateSystem;
    var dimIndex = getPointDimIndex(axis);
    var axisExtent = getGlobalExtent(coordSys, dimIndex);
    var currPosition = [transform.x, transform.y];
    currPosition[dimIndex] += delta[dimIndex];
    currPosition[dimIndex] = Math.min(axisExtent[1], currPosition[dimIndex]);
    currPosition[dimIndex] = Math.max(axisExtent[0], currPosition[dimIndex]);
    var otherExtent = getGlobalExtent(coordSys, 1 - dimIndex);
    var cursorOtherValue = (otherExtent[1] + otherExtent[0]) / 2;
    var cursorPoint = [cursorOtherValue, cursorOtherValue];
    cursorPoint[dimIndex] = currPosition[dimIndex];
    return {
      x: currPosition[0],
      y: currPosition[1],
      rotation: transform.rotation,
      cursorPoint: cursorPoint,
      tooltipOption: {
        verticalAlign: 'middle'
      }
    };
  };
  return SingleAxisPointer;
}(BaseAxisPointer);
var pointerShapeBuilder = {
  line: function (axis, pixelValue, otherExtent) {
    var targetShape = viewHelper.makeLineShape([pixelValue, otherExtent[0]], [pixelValue, otherExtent[1]], getPointDimIndex(axis));
    return {
      type: 'Line',
      subPixelOptimize: true,
      shape: targetShape
    };
  },
  shadow: function (axis, pixelValue, otherExtent) {
    var bandWidth = axis.getBandWidth();
    var span = otherExtent[1] - otherExtent[0];
    return {
      type: 'Rect',
      shape: viewHelper.makeRectShape([pixelValue - bandWidth / 2, otherExtent[0]], [bandWidth, span], getPointDimIndex(axis))
    };
  }
};
function getPointDimIndex(axis) {
  return axis.isHorizontal() ? 0 : 1;
}
function getGlobalExtent(coordSys, dimIndex) {
  var rect = coordSys.getRect();
  return [rect[XY[dimIndex]], rect[XY[dimIndex]] + rect[WH[dimIndex]]];
}
export default SingleAxisPointer;