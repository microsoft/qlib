
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
import AxisBuilder from './AxisBuilder.js';
import AxisView from './AxisView.js';
var axisBuilderAttrs = ['axisLine', 'axisTickLabel', 'axisName'];
var selfBuilderAttrs = ['splitLine', 'splitArea', 'minorSplitLine'];
var RadiusAxisView = /** @class */function (_super) {
  __extends(RadiusAxisView, _super);
  function RadiusAxisView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = RadiusAxisView.type;
    _this.axisPointerClass = 'PolarAxisPointer';
    return _this;
  }
  RadiusAxisView.prototype.render = function (radiusAxisModel, ecModel) {
    this.group.removeAll();
    if (!radiusAxisModel.get('show')) {
      return;
    }
    var oldAxisGroup = this._axisGroup;
    var newAxisGroup = this._axisGroup = new graphic.Group();
    this.group.add(newAxisGroup);
    var radiusAxis = radiusAxisModel.axis;
    var polar = radiusAxis.polar;
    var angleAxis = polar.getAngleAxis();
    var ticksCoords = radiusAxis.getTicksCoords();
    var minorTicksCoords = radiusAxis.getMinorTicksCoords();
    var axisAngle = angleAxis.getExtent()[0];
    var radiusExtent = radiusAxis.getExtent();
    var layout = layoutAxis(polar, radiusAxisModel, axisAngle);
    var axisBuilder = new AxisBuilder(radiusAxisModel, layout);
    zrUtil.each(axisBuilderAttrs, axisBuilder.add, axisBuilder);
    newAxisGroup.add(axisBuilder.getGroup());
    graphic.groupTransition(oldAxisGroup, newAxisGroup, radiusAxisModel);
    zrUtil.each(selfBuilderAttrs, function (name) {
      if (radiusAxisModel.get([name, 'show']) && !radiusAxis.scale.isBlank()) {
        axisElementBuilders[name](this.group, radiusAxisModel, polar, axisAngle, radiusExtent, ticksCoords, minorTicksCoords);
      }
    }, this);
  };
  RadiusAxisView.type = 'radiusAxis';
  return RadiusAxisView;
}(AxisView);
var axisElementBuilders = {
  splitLine: function (group, radiusAxisModel, polar, axisAngle, radiusExtent, ticksCoords) {
    var splitLineModel = radiusAxisModel.getModel('splitLine');
    var lineStyleModel = splitLineModel.getModel('lineStyle');
    var lineColors = lineStyleModel.get('color');
    var lineCount = 0;
    var angleAxis = polar.getAngleAxis();
    var RADIAN = Math.PI / 180;
    var angleExtent = angleAxis.getExtent();
    var shapeType = Math.abs(angleExtent[1] - angleExtent[0]) === 360 ? 'Circle' : 'Arc';
    lineColors = lineColors instanceof Array ? lineColors : [lineColors];
    var splitLines = [];
    for (var i = 0; i < ticksCoords.length; i++) {
      var colorIndex = lineCount++ % lineColors.length;
      splitLines[colorIndex] = splitLines[colorIndex] || [];
      splitLines[colorIndex].push(new graphic[shapeType]({
        shape: {
          cx: polar.cx,
          cy: polar.cy,
          // ensure circle radius >= 0
          r: Math.max(ticksCoords[i].coord, 0),
          startAngle: -angleExtent[0] * RADIAN,
          endAngle: -angleExtent[1] * RADIAN,
          clockwise: angleAxis.inverse
        }
      }));
    }
    // Simple optimization
    // Batching the lines if color are the same
    for (var i = 0; i < splitLines.length; i++) {
      group.add(graphic.mergePath(splitLines[i], {
        style: zrUtil.defaults({
          stroke: lineColors[i % lineColors.length],
          fill: null
        }, lineStyleModel.getLineStyle()),
        silent: true
      }));
    }
  },
  minorSplitLine: function (group, radiusAxisModel, polar, axisAngle, radiusExtent, ticksCoords, minorTicksCoords) {
    if (!minorTicksCoords.length) {
      return;
    }
    var minorSplitLineModel = radiusAxisModel.getModel('minorSplitLine');
    var lineStyleModel = minorSplitLineModel.getModel('lineStyle');
    var lines = [];
    for (var i = 0; i < minorTicksCoords.length; i++) {
      for (var k = 0; k < minorTicksCoords[i].length; k++) {
        lines.push(new graphic.Circle({
          shape: {
            cx: polar.cx,
            cy: polar.cy,
            r: minorTicksCoords[i][k].coord
          }
        }));
      }
    }
    group.add(graphic.mergePath(lines, {
      style: zrUtil.defaults({
        fill: null
      }, lineStyleModel.getLineStyle()),
      silent: true
    }));
  },
  splitArea: function (group, radiusAxisModel, polar, axisAngle, radiusExtent, ticksCoords) {
    if (!ticksCoords.length) {
      return;
    }
    var splitAreaModel = radiusAxisModel.getModel('splitArea');
    var areaStyleModel = splitAreaModel.getModel('areaStyle');
    var areaColors = areaStyleModel.get('color');
    var lineCount = 0;
    areaColors = areaColors instanceof Array ? areaColors : [areaColors];
    var splitAreas = [];
    var prevRadius = ticksCoords[0].coord;
    for (var i = 1; i < ticksCoords.length; i++) {
      var colorIndex = lineCount++ % areaColors.length;
      splitAreas[colorIndex] = splitAreas[colorIndex] || [];
      splitAreas[colorIndex].push(new graphic.Sector({
        shape: {
          cx: polar.cx,
          cy: polar.cy,
          r0: prevRadius,
          r: ticksCoords[i].coord,
          startAngle: 0,
          endAngle: Math.PI * 2
        },
        silent: true
      }));
      prevRadius = ticksCoords[i].coord;
    }
    // Simple optimization
    // Batching the lines if color are the same
    for (var i = 0; i < splitAreas.length; i++) {
      group.add(graphic.mergePath(splitAreas[i], {
        style: zrUtil.defaults({
          fill: areaColors[i % areaColors.length]
        }, areaStyleModel.getAreaStyle()),
        silent: true
      }));
    }
  }
};
/**
 * @inner
 */
function layoutAxis(polar, radiusAxisModel, axisAngle) {
  return {
    position: [polar.cx, polar.cy],
    rotation: axisAngle / 180 * Math.PI,
    labelDirection: -1,
    tickDirection: -1,
    nameDirection: 1,
    labelRotate: radiusAxisModel.getModel('axisLabel').get('rotate'),
    // Over splitLine and splitArea
    z2: 1
  };
}
export default RadiusAxisView;