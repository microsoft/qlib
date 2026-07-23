
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
import { createTextStyle } from '../../label/labelStyle.js';
import Model from '../../model/Model.js';
import AxisView from './AxisView.js';
import AxisBuilder from './AxisBuilder.js';
import { getECData } from '../../util/innerStore.js';
var elementList = ['axisLine', 'axisLabel', 'axisTick', 'minorTick', 'splitLine', 'minorSplitLine', 'splitArea'];
function getAxisLineShape(polar, rExtent, angle) {
  rExtent[1] > rExtent[0] && (rExtent = rExtent.slice().reverse());
  var start = polar.coordToPoint([rExtent[0], angle]);
  var end = polar.coordToPoint([rExtent[1], angle]);
  return {
    x1: start[0],
    y1: start[1],
    x2: end[0],
    y2: end[1]
  };
}
function getRadiusIdx(polar) {
  var radiusAxis = polar.getRadiusAxis();
  return radiusAxis.inverse ? 0 : 1;
}
// Remove the last tick which will overlap the first tick
function fixAngleOverlap(list) {
  var firstItem = list[0];
  var lastItem = list[list.length - 1];
  if (firstItem && lastItem && Math.abs(Math.abs(firstItem.coord - lastItem.coord) - 360) < 1e-4) {
    list.pop();
  }
}
var AngleAxisView = /** @class */function (_super) {
  __extends(AngleAxisView, _super);
  function AngleAxisView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = AngleAxisView.type;
    _this.axisPointerClass = 'PolarAxisPointer';
    return _this;
  }
  AngleAxisView.prototype.render = function (angleAxisModel, ecModel) {
    this.group.removeAll();
    if (!angleAxisModel.get('show')) {
      return;
    }
    var angleAxis = angleAxisModel.axis;
    var polar = angleAxis.polar;
    var radiusExtent = polar.getRadiusAxis().getExtent();
    var ticksAngles = angleAxis.getTicksCoords();
    var minorTickAngles = angleAxis.getMinorTicksCoords();
    var labels = zrUtil.map(angleAxis.getViewLabels(), function (labelItem) {
      labelItem = zrUtil.clone(labelItem);
      var scale = angleAxis.scale;
      var tickValue = scale.type === 'ordinal' ? scale.getRawOrdinalNumber(labelItem.tickValue) : labelItem.tickValue;
      labelItem.coord = angleAxis.dataToCoord(tickValue);
      return labelItem;
    });
    fixAngleOverlap(labels);
    fixAngleOverlap(ticksAngles);
    zrUtil.each(elementList, function (name) {
      if (angleAxisModel.get([name, 'show']) && (!angleAxis.scale.isBlank() || name === 'axisLine')) {
        angelAxisElementsBuilders[name](this.group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent, labels);
      }
    }, this);
  };
  AngleAxisView.type = 'angleAxis';
  return AngleAxisView;
}(AxisView);
var angelAxisElementsBuilders = {
  axisLine: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent) {
    var lineStyleModel = angleAxisModel.getModel(['axisLine', 'lineStyle']);
    var angleAxis = polar.getAngleAxis();
    var RADIAN = Math.PI / 180;
    var angleExtent = angleAxis.getExtent();
    // extent id of the axis radius (r0 and r)
    var rId = getRadiusIdx(polar);
    var r0Id = rId ? 0 : 1;
    var shape;
    var shapeType = Math.abs(angleExtent[1] - angleExtent[0]) === 360 ? 'Circle' : 'Arc';
    if (radiusExtent[r0Id] === 0) {
      shape = new graphic[shapeType]({
        shape: {
          cx: polar.cx,
          cy: polar.cy,
          r: radiusExtent[rId],
          startAngle: -angleExtent[0] * RADIAN,
          endAngle: -angleExtent[1] * RADIAN,
          clockwise: angleAxis.inverse
        },
        style: lineStyleModel.getLineStyle(),
        z2: 1,
        silent: true
      });
    } else {
      shape = new graphic.Ring({
        shape: {
          cx: polar.cx,
          cy: polar.cy,
          r: radiusExtent[rId],
          r0: radiusExtent[r0Id]
        },
        style: lineStyleModel.getLineStyle(),
        z2: 1,
        silent: true
      });
    }
    shape.style.fill = null;
    group.add(shape);
  },
  axisTick: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent) {
    var tickModel = angleAxisModel.getModel('axisTick');
    var tickLen = (tickModel.get('inside') ? -1 : 1) * tickModel.get('length');
    var radius = radiusExtent[getRadiusIdx(polar)];
    var lines = zrUtil.map(ticksAngles, function (tickAngleItem) {
      return new graphic.Line({
        shape: getAxisLineShape(polar, [radius, radius + tickLen], tickAngleItem.coord)
      });
    });
    group.add(graphic.mergePath(lines, {
      style: zrUtil.defaults(tickModel.getModel('lineStyle').getLineStyle(), {
        stroke: angleAxisModel.get(['axisLine', 'lineStyle', 'color'])
      })
    }));
  },
  minorTick: function (group, angleAxisModel, polar, tickAngles, minorTickAngles, radiusExtent) {
    if (!minorTickAngles.length) {
      return;
    }
    var tickModel = angleAxisModel.getModel('axisTick');
    var minorTickModel = angleAxisModel.getModel('minorTick');
    var tickLen = (tickModel.get('inside') ? -1 : 1) * minorTickModel.get('length');
    var radius = radiusExtent[getRadiusIdx(polar)];
    var lines = [];
    for (var i = 0; i < minorTickAngles.length; i++) {
      for (var k = 0; k < minorTickAngles[i].length; k++) {
        lines.push(new graphic.Line({
          shape: getAxisLineShape(polar, [radius, radius + tickLen], minorTickAngles[i][k].coord)
        }));
      }
    }
    group.add(graphic.mergePath(lines, {
      style: zrUtil.defaults(minorTickModel.getModel('lineStyle').getLineStyle(), zrUtil.defaults(tickModel.getLineStyle(), {
        stroke: angleAxisModel.get(['axisLine', 'lineStyle', 'color'])
      }))
    }));
  },
  axisLabel: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent, labels) {
    var rawCategoryData = angleAxisModel.getCategories(true);
    var commonLabelModel = angleAxisModel.getModel('axisLabel');
    var labelMargin = commonLabelModel.get('margin');
    var triggerEvent = angleAxisModel.get('triggerEvent');
    // Use length of ticksAngles because it may remove the last tick to avoid overlapping
    zrUtil.each(labels, function (labelItem, idx) {
      var labelModel = commonLabelModel;
      var tickValue = labelItem.tickValue;
      var r = radiusExtent[getRadiusIdx(polar)];
      var p = polar.coordToPoint([r + labelMargin, labelItem.coord]);
      var cx = polar.cx;
      var cy = polar.cy;
      var labelTextAlign = Math.abs(p[0] - cx) / r < 0.3 ? 'center' : p[0] > cx ? 'left' : 'right';
      var labelTextVerticalAlign = Math.abs(p[1] - cy) / r < 0.3 ? 'middle' : p[1] > cy ? 'top' : 'bottom';
      if (rawCategoryData && rawCategoryData[tickValue]) {
        var rawCategoryItem = rawCategoryData[tickValue];
        if (zrUtil.isObject(rawCategoryItem) && rawCategoryItem.textStyle) {
          labelModel = new Model(rawCategoryItem.textStyle, commonLabelModel, commonLabelModel.ecModel);
        }
      }
      var textEl = new graphic.Text({
        silent: AxisBuilder.isLabelSilent(angleAxisModel),
        style: createTextStyle(labelModel, {
          x: p[0],
          y: p[1],
          fill: labelModel.getTextColor() || angleAxisModel.get(['axisLine', 'lineStyle', 'color']),
          text: labelItem.formattedLabel,
          align: labelTextAlign,
          verticalAlign: labelTextVerticalAlign
        })
      });
      group.add(textEl);
      // Pack data for mouse event
      if (triggerEvent) {
        var eventData = AxisBuilder.makeAxisEventDataBase(angleAxisModel);
        eventData.targetType = 'axisLabel';
        eventData.value = labelItem.rawLabel;
        getECData(textEl).eventData = eventData;
      }
    }, this);
  },
  splitLine: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent) {
    var splitLineModel = angleAxisModel.getModel('splitLine');
    var lineStyleModel = splitLineModel.getModel('lineStyle');
    var lineColors = lineStyleModel.get('color');
    var lineCount = 0;
    lineColors = lineColors instanceof Array ? lineColors : [lineColors];
    var splitLines = [];
    for (var i = 0; i < ticksAngles.length; i++) {
      var colorIndex = lineCount++ % lineColors.length;
      splitLines[colorIndex] = splitLines[colorIndex] || [];
      splitLines[colorIndex].push(new graphic.Line({
        shape: getAxisLineShape(polar, radiusExtent, ticksAngles[i].coord)
      }));
    }
    // Simple optimization
    // Batching the lines if color are the same
    for (var i = 0; i < splitLines.length; i++) {
      group.add(graphic.mergePath(splitLines[i], {
        style: zrUtil.defaults({
          stroke: lineColors[i % lineColors.length]
        }, lineStyleModel.getLineStyle()),
        silent: true,
        z: angleAxisModel.get('z')
      }));
    }
  },
  minorSplitLine: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent) {
    if (!minorTickAngles.length) {
      return;
    }
    var minorSplitLineModel = angleAxisModel.getModel('minorSplitLine');
    var lineStyleModel = minorSplitLineModel.getModel('lineStyle');
    var lines = [];
    for (var i = 0; i < minorTickAngles.length; i++) {
      for (var k = 0; k < minorTickAngles[i].length; k++) {
        lines.push(new graphic.Line({
          shape: getAxisLineShape(polar, radiusExtent, minorTickAngles[i][k].coord)
        }));
      }
    }
    group.add(graphic.mergePath(lines, {
      style: lineStyleModel.getLineStyle(),
      silent: true,
      z: angleAxisModel.get('z')
    }));
  },
  splitArea: function (group, angleAxisModel, polar, ticksAngles, minorTickAngles, radiusExtent) {
    if (!ticksAngles.length) {
      return;
    }
    var splitAreaModel = angleAxisModel.getModel('splitArea');
    var areaStyleModel = splitAreaModel.getModel('areaStyle');
    var areaColors = areaStyleModel.get('color');
    var lineCount = 0;
    areaColors = areaColors instanceof Array ? areaColors : [areaColors];
    var splitAreas = [];
    var RADIAN = Math.PI / 180;
    var prevAngle = -ticksAngles[0].coord * RADIAN;
    var r0 = Math.min(radiusExtent[0], radiusExtent[1]);
    var r1 = Math.max(radiusExtent[0], radiusExtent[1]);
    var clockwise = angleAxisModel.get('clockwise');
    for (var i = 1, len = ticksAngles.length; i <= len; i++) {
      var coord = i === len ? ticksAngles[0].coord : ticksAngles[i].coord;
      var colorIndex = lineCount++ % areaColors.length;
      splitAreas[colorIndex] = splitAreas[colorIndex] || [];
      splitAreas[colorIndex].push(new graphic.Sector({
        shape: {
          cx: polar.cx,
          cy: polar.cy,
          r0: r0,
          r: r1,
          startAngle: prevAngle,
          endAngle: -coord * RADIAN,
          clockwise: clockwise
        },
        silent: true
      }));
      prevAngle = -coord * RADIAN;
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
export default AngleAxisView;