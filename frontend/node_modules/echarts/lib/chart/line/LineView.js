
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
// FIXME step not support polar
import * as zrUtil from 'zrender/lib/core/util.js';
import SymbolDraw from '../helper/SymbolDraw.js';
import SymbolClz from '../helper/Symbol.js';
import lineAnimationDiff from './lineAnimationDiff.js';
import * as graphic from '../../util/graphic.js';
import * as modelUtil from '../../util/model.js';
import { ECPolyline, ECPolygon } from './poly.js';
import ChartView from '../../view/Chart.js';
import { prepareDataCoordInfo, getStackedOnPoint } from './helper.js';
import { createGridClipPath, createPolarClipPath } from '../helper/createClipPathFromCoordSys.js';
import { isCoordinateSystemType } from '../../coord/CoordinateSystem.js';
import { setStatesStylesFromModel, setStatesFlag, toggleHoverEmphasis, SPECIAL_STATES } from '../../util/states.js';
import { setLabelStyle, getLabelStatesModels, labelInner } from '../../label/labelStyle.js';
import { getDefaultLabel, getDefaultInterpolatedLabel } from '../helper/labelHelper.js';
import { getECData } from '../../util/innerStore.js';
import { createFloat32Array } from '../../util/vendor.js';
import { convertToColorString } from '../../util/format.js';
import { lerp } from 'zrender/lib/tool/color.js';
function isPointsSame(points1, points2) {
  if (points1.length !== points2.length) {
    return;
  }
  for (var i = 0; i < points1.length; i++) {
    if (points1[i] !== points2[i]) {
      return;
    }
  }
  return true;
}
function bboxFromPoints(points) {
  var minX = Infinity;
  var minY = Infinity;
  var maxX = -Infinity;
  var maxY = -Infinity;
  for (var i = 0; i < points.length;) {
    var x = points[i++];
    var y = points[i++];
    if (!isNaN(x)) {
      minX = Math.min(x, minX);
      maxX = Math.max(x, maxX);
    }
    if (!isNaN(y)) {
      minY = Math.min(y, minY);
      maxY = Math.max(y, maxY);
    }
  }
  return [[minX, minY], [maxX, maxY]];
}
function getBoundingDiff(points1, points2) {
  var _a = bboxFromPoints(points1),
    min1 = _a[0],
    max1 = _a[1];
  var _b = bboxFromPoints(points2),
    min2 = _b[0],
    max2 = _b[1];
  // Get a max value from each corner of two boundings.
  return Math.max(Math.abs(min1[0] - min2[0]), Math.abs(min1[1] - min2[1]), Math.abs(max1[0] - max2[0]), Math.abs(max1[1] - max2[1]));
}
function getSmooth(smooth) {
  return zrUtil.isNumber(smooth) ? smooth : smooth ? 0.5 : 0;
}
function getStackedOnPoints(coordSys, data, dataCoordInfo) {
  if (!dataCoordInfo.valueDim) {
    return [];
  }
  var len = data.count();
  var points = createFloat32Array(len * 2);
  for (var idx = 0; idx < len; idx++) {
    var pt = getStackedOnPoint(dataCoordInfo, coordSys, data, idx);
    points[idx * 2] = pt[0];
    points[idx * 2 + 1] = pt[1];
  }
  return points;
}
/**
 * Filter the null data and extend data for step considering `stepTurnAt`
 *
 * @param points data to convert, that may containing null
 * @param basePoints base data to reference, used only for areaStyle points
 * @param coordSys coordinate system
 * @param stepTurnAt 'start' | 'end' | 'middle' | true
 * @param connectNulls whether to connect nulls
 * @returns converted point positions
 */
function turnPointsIntoStep(points, basePoints, coordSys, stepTurnAt, connectNulls) {
  var baseAxis = coordSys.getBaseAxis();
  var baseIndex = baseAxis.dim === 'x' || baseAxis.dim === 'radius' ? 0 : 1;
  var stepPoints = [];
  var i = 0;
  var stepPt = [];
  var pt = [];
  var nextPt = [];
  var filteredPoints = [];
  if (connectNulls) {
    for (i = 0; i < points.length; i += 2) {
      /**
       * For areaStyle of stepped lines, `stackedOnPoints` should be
       * filtered the same as `points` so that the base axis values
       * should stay the same as the lines above. See #20021
       */
      var reference = basePoints || points;
      if (!isNaN(reference[i]) && !isNaN(reference[i + 1])) {
        filteredPoints.push(points[i], points[i + 1]);
      }
    }
    points = filteredPoints;
  }
  for (i = 0; i < points.length - 2; i += 2) {
    nextPt[0] = points[i + 2];
    nextPt[1] = points[i + 3];
    pt[0] = points[i];
    pt[1] = points[i + 1];
    stepPoints.push(pt[0], pt[1]);
    switch (stepTurnAt) {
      case 'end':
        stepPt[baseIndex] = nextPt[baseIndex];
        stepPt[1 - baseIndex] = pt[1 - baseIndex];
        stepPoints.push(stepPt[0], stepPt[1]);
        break;
      case 'middle':
        var middle = (pt[baseIndex] + nextPt[baseIndex]) / 2;
        var stepPt2 = [];
        stepPt[baseIndex] = stepPt2[baseIndex] = middle;
        stepPt[1 - baseIndex] = pt[1 - baseIndex];
        stepPt2[1 - baseIndex] = nextPt[1 - baseIndex];
        stepPoints.push(stepPt[0], stepPt[1]);
        stepPoints.push(stepPt2[0], stepPt2[1]);
        break;
      default:
        // default is start
        stepPt[baseIndex] = pt[baseIndex];
        stepPt[1 - baseIndex] = nextPt[1 - baseIndex];
        stepPoints.push(stepPt[0], stepPt[1]);
    }
  }
  // Last points
  stepPoints.push(points[i++], points[i++]);
  return stepPoints;
}
/**
 * Clip color stops to edge. Avoid creating too large gradients.
 * Which may lead to blurry when GPU acceleration is enabled. See #15680
 *
 * The stops has been sorted from small to large.
 */
function clipColorStops(colorStops, maxSize) {
  var newColorStops = [];
  var len = colorStops.length;
  // coord will always < 0 in prevOutOfRangeColorStop.
  var prevOutOfRangeColorStop;
  var prevInRangeColorStop;
  function lerpStop(stop0, stop1, clippedCoord) {
    var coord0 = stop0.coord;
    var p = (clippedCoord - coord0) / (stop1.coord - coord0);
    var color = lerp(p, [stop0.color, stop1.color]);
    return {
      coord: clippedCoord,
      color: color
    };
  }
  for (var i = 0; i < len; i++) {
    var stop_1 = colorStops[i];
    var coord = stop_1.coord;
    if (coord < 0) {
      prevOutOfRangeColorStop = stop_1;
    } else if (coord > maxSize) {
      if (prevInRangeColorStop) {
        newColorStops.push(lerpStop(prevInRangeColorStop, stop_1, maxSize));
      } else if (prevOutOfRangeColorStop) {
        // If there are two stops and coord range is between these two stops
        newColorStops.push(lerpStop(prevOutOfRangeColorStop, stop_1, 0), lerpStop(prevOutOfRangeColorStop, stop_1, maxSize));
      }
      // All following stop will be out of range. So just ignore them.
      break;
    } else {
      if (prevOutOfRangeColorStop) {
        newColorStops.push(lerpStop(prevOutOfRangeColorStop, stop_1, 0));
        // Reset
        prevOutOfRangeColorStop = null;
      }
      newColorStops.push(stop_1);
      prevInRangeColorStop = stop_1;
    }
  }
  return newColorStops;
}
function getVisualGradient(data, coordSys, api) {
  var visualMetaList = data.getVisual('visualMeta');
  if (!visualMetaList || !visualMetaList.length || !data.count()) {
    // When data.count() is 0, gradient range can not be calculated.
    return;
  }
  if (coordSys.type !== 'cartesian2d') {
    if (process.env.NODE_ENV !== 'production') {
      console.warn('Visual map on line style is only supported on cartesian2d.');
    }
    return;
  }
  var coordDim;
  var visualMeta;
  for (var i = visualMetaList.length - 1; i >= 0; i--) {
    var dimInfo = data.getDimensionInfo(visualMetaList[i].dimension);
    coordDim = dimInfo && dimInfo.coordDim;
    // Can only be x or y
    if (coordDim === 'x' || coordDim === 'y') {
      visualMeta = visualMetaList[i];
      break;
    }
  }
  if (!visualMeta) {
    if (process.env.NODE_ENV !== 'production') {
      console.warn('Visual map on line style only support x or y dimension.');
    }
    return;
  }
  // If the area to be rendered is bigger than area defined by LinearGradient,
  // the canvas spec prescribes that the color of the first stop and the last
  // stop should be used. But if two stops are added at offset 0, in effect
  // browsers use the color of the second stop to render area outside
  // LinearGradient. So we can only infinitesimally extend area defined in
  // LinearGradient to render `outerColors`.
  var axis = coordSys.getAxis(coordDim);
  // dataToCoord mapping may not be linear, but must be monotonic.
  var colorStops = zrUtil.map(visualMeta.stops, function (stop) {
    // offset will be calculated later.
    return {
      coord: axis.toGlobalCoord(axis.dataToCoord(stop.value)),
      color: stop.color
    };
  });
  var stopLen = colorStops.length;
  var outerColors = visualMeta.outerColors.slice();
  if (stopLen && colorStops[0].coord > colorStops[stopLen - 1].coord) {
    colorStops.reverse();
    outerColors.reverse();
  }
  var colorStopsInRange = clipColorStops(colorStops, coordDim === 'x' ? api.getWidth() : api.getHeight());
  var inRangeStopLen = colorStopsInRange.length;
  if (!inRangeStopLen && stopLen) {
    // All stops are out of range. All will be the same color.
    return colorStops[0].coord < 0 ? outerColors[1] ? outerColors[1] : colorStops[stopLen - 1].color : outerColors[0] ? outerColors[0] : colorStops[0].color;
  }
  var tinyExtent = 10; // Arbitrary value: 10px
  var minCoord = colorStopsInRange[0].coord - tinyExtent;
  var maxCoord = colorStopsInRange[inRangeStopLen - 1].coord + tinyExtent;
  var coordSpan = maxCoord - minCoord;
  if (coordSpan < 1e-3) {
    return 'transparent';
  }
  zrUtil.each(colorStopsInRange, function (stop) {
    stop.offset = (stop.coord - minCoord) / coordSpan;
  });
  colorStopsInRange.push({
    // NOTE: inRangeStopLen may still be 0 if stoplen is zero.
    offset: inRangeStopLen ? colorStopsInRange[inRangeStopLen - 1].offset : 0.5,
    color: outerColors[1] || 'transparent'
  });
  colorStopsInRange.unshift({
    offset: inRangeStopLen ? colorStopsInRange[0].offset : 0.5,
    color: outerColors[0] || 'transparent'
  });
  var gradient = new graphic.LinearGradient(0, 0, 0, 0, colorStopsInRange, true);
  gradient[coordDim] = minCoord;
  gradient[coordDim + '2'] = maxCoord;
  return gradient;
}
function getIsIgnoreFunc(seriesModel, data, coordSys) {
  var showAllSymbol = seriesModel.get('showAllSymbol');
  var isAuto = showAllSymbol === 'auto';
  if (showAllSymbol && !isAuto) {
    return;
  }
  var categoryAxis = coordSys.getAxesByScale('ordinal')[0];
  if (!categoryAxis) {
    return;
  }
  // Note that category label interval strategy might bring some weird effect
  // in some scenario: users may wonder why some of the symbols are not
  // displayed. So we show all symbols as possible as we can.
  if (isAuto
  // Simplify the logic, do not determine label overlap here.
  && canShowAllSymbolForCategory(categoryAxis, data)) {
    return;
  }
  // Otherwise follow the label interval strategy on category axis.
  var categoryDataDim = data.mapDimension(categoryAxis.dim);
  var labelMap = {};
  zrUtil.each(categoryAxis.getViewLabels(), function (labelItem) {
    var ordinalNumber = categoryAxis.scale.getRawOrdinalNumber(labelItem.tickValue);
    labelMap[ordinalNumber] = 1;
  });
  return function (dataIndex) {
    return !labelMap.hasOwnProperty(data.get(categoryDataDim, dataIndex));
  };
}
function canShowAllSymbolForCategory(categoryAxis, data) {
  // In most cases, line is monotonous on category axis, and the label size
  // is close with each other. So we check the symbol size and some of the
  // label size alone with the category axis to estimate whether all symbol
  // can be shown without overlap.
  var axisExtent = categoryAxis.getExtent();
  var availSize = Math.abs(axisExtent[1] - axisExtent[0]) / categoryAxis.scale.count();
  isNaN(availSize) && (availSize = 0); // 0/0 is NaN.
  // Sampling some points, max 5.
  var dataLen = data.count();
  var step = Math.max(1, Math.round(dataLen / 5));
  for (var dataIndex = 0; dataIndex < dataLen; dataIndex += step) {
    if (SymbolClz.getSymbolSize(data, dataIndex
    // Only for cartesian, where `isHorizontal` exists.
    )[categoryAxis.isHorizontal() ? 1 : 0]
    // Empirical number
    * 1.5 > availSize) {
      return false;
    }
  }
  return true;
}
function isPointNull(x, y) {
  return isNaN(x) || isNaN(y);
}
function getLastIndexNotNull(points) {
  var len = points.length / 2;
  for (; len > 0; len--) {
    if (!isPointNull(points[len * 2 - 2], points[len * 2 - 1])) {
      break;
    }
  }
  return len - 1;
}
function getPointAtIndex(points, idx) {
  return [points[idx * 2], points[idx * 2 + 1]];
}
function getIndexRange(points, xOrY, dim) {
  var len = points.length / 2;
  var dimIdx = dim === 'x' ? 0 : 1;
  var a;
  var b;
  var prevIndex = 0;
  var nextIndex = -1;
  for (var i = 0; i < len; i++) {
    b = points[i * 2 + dimIdx];
    if (isNaN(b) || isNaN(points[i * 2 + 1 - dimIdx])) {
      continue;
    }
    if (i === 0) {
      a = b;
      continue;
    }
    if (a <= xOrY && b >= xOrY || a >= xOrY && b <= xOrY) {
      nextIndex = i;
      break;
    }
    prevIndex = i;
    a = b;
  }
  return {
    range: [prevIndex, nextIndex],
    t: (xOrY - a) / (b - a)
  };
}
function anyStateShowEndLabel(seriesModel) {
  if (seriesModel.get(['endLabel', 'show'])) {
    return true;
  }
  for (var i = 0; i < SPECIAL_STATES.length; i++) {
    if (seriesModel.get([SPECIAL_STATES[i], 'endLabel', 'show'])) {
      return true;
    }
  }
  return false;
}
function createLineClipPath(lineView, coordSys, hasAnimation, seriesModel) {
  if (isCoordinateSystemType(coordSys, 'cartesian2d')) {
    var endLabelModel_1 = seriesModel.getModel('endLabel');
    var valueAnimation_1 = endLabelModel_1.get('valueAnimation');
    var data_1 = seriesModel.getData();
    var labelAnimationRecord_1 = {
      lastFrameIndex: 0
    };
    var during = anyStateShowEndLabel(seriesModel) ? function (percent, clipRect) {
      lineView._endLabelOnDuring(percent, clipRect, data_1, labelAnimationRecord_1, valueAnimation_1, endLabelModel_1, coordSys);
    } : null;
    var isHorizontal = coordSys.getBaseAxis().isHorizontal();
    var clipPath = createGridClipPath(coordSys, hasAnimation, seriesModel, function () {
      var endLabel = lineView._endLabel;
      if (endLabel && hasAnimation) {
        if (labelAnimationRecord_1.originalX != null) {
          endLabel.attr({
            x: labelAnimationRecord_1.originalX,
            y: labelAnimationRecord_1.originalY
          });
        }
      }
    }, during);
    // Expand clip shape to avoid clipping when line value exceeds axis
    if (!seriesModel.get('clip', true)) {
      var rectShape = clipPath.shape;
      var expandSize = Math.max(rectShape.width, rectShape.height);
      if (isHorizontal) {
        rectShape.y -= expandSize;
        rectShape.height += expandSize * 2;
      } else {
        rectShape.x -= expandSize;
        rectShape.width += expandSize * 2;
      }
    }
    // Set to the final frame. To make sure label layout is right.
    if (during) {
      during(1, clipPath);
    }
    return clipPath;
  } else {
    if (process.env.NODE_ENV !== 'production') {
      if (seriesModel.get(['endLabel', 'show'])) {
        console.warn('endLabel is not supported for lines in polar systems.');
      }
    }
    return createPolarClipPath(coordSys, hasAnimation, seriesModel);
  }
}
function getEndLabelStateSpecified(endLabelModel, coordSys) {
  var baseAxis = coordSys.getBaseAxis();
  var isHorizontal = baseAxis.isHorizontal();
  var isBaseInversed = baseAxis.inverse;
  var align = isHorizontal ? isBaseInversed ? 'right' : 'left' : 'center';
  var verticalAlign = isHorizontal ? 'middle' : isBaseInversed ? 'top' : 'bottom';
  return {
    normal: {
      align: endLabelModel.get('align') || align,
      verticalAlign: endLabelModel.get('verticalAlign') || verticalAlign
    }
  };
}
var LineView = /** @class */function (_super) {
  __extends(LineView, _super);
  function LineView() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  LineView.prototype.init = function () {
    var lineGroup = new graphic.Group();
    var symbolDraw = new SymbolDraw();
    this.group.add(symbolDraw.group);
    this._symbolDraw = symbolDraw;
    this._lineGroup = lineGroup;
    this._changePolyState = zrUtil.bind(this._changePolyState, this);
  };
  LineView.prototype.render = function (seriesModel, ecModel, api) {
    var coordSys = seriesModel.coordinateSystem;
    var group = this.group;
    var data = seriesModel.getData();
    var lineStyleModel = seriesModel.getModel('lineStyle');
    var areaStyleModel = seriesModel.getModel('areaStyle');
    var points = data.getLayout('points') || [];
    var isCoordSysPolar = coordSys.type === 'polar';
    var prevCoordSys = this._coordSys;
    var symbolDraw = this._symbolDraw;
    var polyline = this._polyline;
    var polygon = this._polygon;
    var lineGroup = this._lineGroup;
    var hasAnimation = !ecModel.ssr && seriesModel.get('animation');
    var isAreaChart = !areaStyleModel.isEmpty();
    var valueOrigin = areaStyleModel.get('origin');
    var dataCoordInfo = prepareDataCoordInfo(coordSys, data, valueOrigin);
    var stackedOnPoints = isAreaChart && getStackedOnPoints(coordSys, data, dataCoordInfo);
    var showSymbol = seriesModel.get('showSymbol');
    var connectNulls = seriesModel.get('connectNulls');
    var isIgnoreFunc = showSymbol && !isCoordSysPolar && getIsIgnoreFunc(seriesModel, data, coordSys);
    // Remove temporary symbols
    var oldData = this._data;
    oldData && oldData.eachItemGraphicEl(function (el, idx) {
      if (el.__temp) {
        group.remove(el);
        oldData.setItemGraphicEl(idx, null);
      }
    });
    // Remove previous created symbols if showSymbol changed to false
    if (!showSymbol) {
      symbolDraw.remove();
    }
    group.add(lineGroup);
    // FIXME step not support polar
    var step = !isCoordSysPolar ? seriesModel.get('step') : false;
    var clipShapeForSymbol;
    if (coordSys && coordSys.getArea && seriesModel.get('clip', true)) {
      clipShapeForSymbol = coordSys.getArea();
      // Avoid float number rounding error for symbol on the edge of axis extent.
      // See #7913 and `test/dataZoom-clip.html`.
      if (clipShapeForSymbol.width != null) {
        clipShapeForSymbol.x -= 0.1;
        clipShapeForSymbol.y -= 0.1;
        clipShapeForSymbol.width += 0.2;
        clipShapeForSymbol.height += 0.2;
      } else if (clipShapeForSymbol.r0) {
        clipShapeForSymbol.r0 -= 0.5;
        clipShapeForSymbol.r += 0.5;
      }
    }
    this._clipShapeForSymbol = clipShapeForSymbol;
    var visualColor = getVisualGradient(data, coordSys, api) || data.getVisual('style')[data.getVisual('drawType')];
    // Initialization animation or coordinate system changed
    if (!(polyline && prevCoordSys.type === coordSys.type && step === this._step)) {
      showSymbol && symbolDraw.updateData(data, {
        isIgnore: isIgnoreFunc,
        clipShape: clipShapeForSymbol,
        disableAnimation: true,
        getSymbolPoint: function (idx) {
          return [points[idx * 2], points[idx * 2 + 1]];
        }
      });
      hasAnimation && this._initSymbolLabelAnimation(data, coordSys, clipShapeForSymbol);
      if (step) {
        if (stackedOnPoints) {
          stackedOnPoints = turnPointsIntoStep(stackedOnPoints, points, coordSys, step, connectNulls);
        }
        // TODO If stacked series is not step
        points = turnPointsIntoStep(points, null, coordSys, step, connectNulls);
      }
      polyline = this._newPolyline(points);
      if (isAreaChart) {
        polygon = this._newPolygon(points, stackedOnPoints);
      } // If areaStyle is removed
      else if (polygon) {
        lineGroup.remove(polygon);
        polygon = this._polygon = null;
      }
      // NOTE: Must update _endLabel before setClipPath.
      if (!isCoordSysPolar) {
        this._initOrUpdateEndLabel(seriesModel, coordSys, convertToColorString(visualColor));
      }
      lineGroup.setClipPath(createLineClipPath(this, coordSys, true, seriesModel));
    } else {
      if (isAreaChart && !polygon) {
        // If areaStyle is added
        polygon = this._newPolygon(points, stackedOnPoints);
      } else if (polygon && !isAreaChart) {
        // If areaStyle is removed
        lineGroup.remove(polygon);
        polygon = this._polygon = null;
      }
      // NOTE: Must update _endLabel before setClipPath.
      if (!isCoordSysPolar) {
        this._initOrUpdateEndLabel(seriesModel, coordSys, convertToColorString(visualColor));
      }
      // Update clipPath
      var oldClipPath = lineGroup.getClipPath();
      if (oldClipPath) {
        var newClipPath = createLineClipPath(this, coordSys, false, seriesModel);
        graphic.initProps(oldClipPath, {
          shape: newClipPath.shape
        }, seriesModel);
      } else {
        lineGroup.setClipPath(createLineClipPath(this, coordSys, true, seriesModel));
      }
      // Always update, or it is wrong in the case turning on legend
      // because points are not changed.
      showSymbol && symbolDraw.updateData(data, {
        isIgnore: isIgnoreFunc,
        clipShape: clipShapeForSymbol,
        disableAnimation: true,
        getSymbolPoint: function (idx) {
          return [points[idx * 2], points[idx * 2 + 1]];
        }
      });
      // In the case data zoom triggered refreshing frequently
      // Data may not change if line has a category axis. So it should animate nothing.
      if (!isPointsSame(this._stackedOnPoints, stackedOnPoints) || !isPointsSame(this._points, points)) {
        if (hasAnimation) {
          this._doUpdateAnimation(data, stackedOnPoints, coordSys, api, step, valueOrigin, connectNulls);
        } else {
          // Not do it in update with animation
          if (step) {
            if (stackedOnPoints) {
              stackedOnPoints = turnPointsIntoStep(stackedOnPoints, points, coordSys, step, connectNulls);
            }
            // TODO If stacked series is not step
            points = turnPointsIntoStep(points, null, coordSys, step, connectNulls);
          }
          polyline.setShape({
            points: points
          });
          polygon && polygon.setShape({
            points: points,
            stackedOnPoints: stackedOnPoints
          });
        }
      }
    }
    var emphasisModel = seriesModel.getModel('emphasis');
    var focus = emphasisModel.get('focus');
    var blurScope = emphasisModel.get('blurScope');
    var emphasisDisabled = emphasisModel.get('disabled');
    polyline.useStyle(zrUtil.defaults(
    // Use color in lineStyle first
    lineStyleModel.getLineStyle(), {
      fill: 'none',
      stroke: visualColor,
      lineJoin: 'bevel'
    }));
    setStatesStylesFromModel(polyline, seriesModel, 'lineStyle');
    if (polyline.style.lineWidth > 0 && seriesModel.get(['emphasis', 'lineStyle', 'width']) === 'bolder') {
      var emphasisLineStyle = polyline.getState('emphasis').style;
      emphasisLineStyle.lineWidth = +polyline.style.lineWidth + 1;
    }
    // Needs seriesIndex for focus
    getECData(polyline).seriesIndex = seriesModel.seriesIndex;
    toggleHoverEmphasis(polyline, focus, blurScope, emphasisDisabled);
    var smooth = getSmooth(seriesModel.get('smooth'));
    var smoothMonotone = seriesModel.get('smoothMonotone');
    polyline.setShape({
      smooth: smooth,
      smoothMonotone: smoothMonotone,
      connectNulls: connectNulls
    });
    if (polygon) {
      var stackedOnSeries = data.getCalculationInfo('stackedOnSeries');
      var stackedOnSmooth = 0;
      polygon.useStyle(zrUtil.defaults(areaStyleModel.getAreaStyle(), {
        fill: visualColor,
        opacity: 0.7,
        lineJoin: 'bevel',
        decal: data.getVisual('style').decal
      }));
      if (stackedOnSeries) {
        stackedOnSmooth = getSmooth(stackedOnSeries.get('smooth'));
      }
      polygon.setShape({
        smooth: smooth,
        stackedOnSmooth: stackedOnSmooth,
        smoothMonotone: smoothMonotone,
        connectNulls: connectNulls
      });
      setStatesStylesFromModel(polygon, seriesModel, 'areaStyle');
      // Needs seriesIndex for focus
      getECData(polygon).seriesIndex = seriesModel.seriesIndex;
      toggleHoverEmphasis(polygon, focus, blurScope, emphasisDisabled);
    }
    var changePolyState = this._changePolyState;
    data.eachItemGraphicEl(function (el) {
      // Switch polyline / polygon state if element changed its state.
      el && (el.onHoverStateChange = changePolyState);
    });
    this._polyline.onHoverStateChange = changePolyState;
    this._data = data;
    // Save the coordinate system for transition animation when data changed
    this._coordSys = coordSys;
    this._stackedOnPoints = stackedOnPoints;
    this._points = points;
    this._step = step;
    this._valueOrigin = valueOrigin;
    if (seriesModel.get('triggerLineEvent')) {
      this.packEventData(seriesModel, polyline);
      polygon && this.packEventData(seriesModel, polygon);
    }
  };
  LineView.prototype.packEventData = function (seriesModel, el) {
    getECData(el).eventData = {
      componentType: 'series',
      componentSubType: 'line',
      componentIndex: seriesModel.componentIndex,
      seriesIndex: seriesModel.seriesIndex,
      seriesName: seriesModel.name,
      seriesType: 'line'
    };
  };
  LineView.prototype.highlight = function (seriesModel, ecModel, api, payload) {
    var data = seriesModel.getData();
    var dataIndex = modelUtil.queryDataIndex(data, payload);
    this._changePolyState('emphasis');
    if (!(dataIndex instanceof Array) && dataIndex != null && dataIndex >= 0) {
      var points = data.getLayout('points');
      var symbol = data.getItemGraphicEl(dataIndex);
      if (!symbol) {
        // Create a temporary symbol if it is not exists
        var x = points[dataIndex * 2];
        var y = points[dataIndex * 2 + 1];
        if (isNaN(x) || isNaN(y)) {
          // Null data
          return;
        }
        // fix #11360: shouldn't draw symbol outside clipShapeForSymbol
        if (this._clipShapeForSymbol && !this._clipShapeForSymbol.contain(x, y)) {
          return;
        }
        var zlevel = seriesModel.get('zlevel') || 0;
        var z = seriesModel.get('z') || 0;
        symbol = new SymbolClz(data, dataIndex);
        symbol.x = x;
        symbol.y = y;
        symbol.setZ(zlevel, z);
        // ensure label text of the temporary symbol is in front of line and area polygon
        var symbolLabel = symbol.getSymbolPath().getTextContent();
        if (symbolLabel) {
          symbolLabel.zlevel = zlevel;
          symbolLabel.z = z;
          symbolLabel.z2 = this._polyline.z2 + 1;
        }
        symbol.__temp = true;
        data.setItemGraphicEl(dataIndex, symbol);
        // Stop scale animation
        symbol.stopSymbolAnimation(true);
        this.group.add(symbol);
      }
      symbol.highlight();
    } else {
      // Highlight whole series
      ChartView.prototype.highlight.call(this, seriesModel, ecModel, api, payload);
    }
  };
  LineView.prototype.downplay = function (seriesModel, ecModel, api, payload) {
    var data = seriesModel.getData();
    var dataIndex = modelUtil.queryDataIndex(data, payload);
    this._changePolyState('normal');
    if (dataIndex != null && dataIndex >= 0) {
      var symbol = data.getItemGraphicEl(dataIndex);
      if (symbol) {
        if (symbol.__temp) {
          data.setItemGraphicEl(dataIndex, null);
          this.group.remove(symbol);
        } else {
          symbol.downplay();
        }
      }
    } else {
      // FIXME
      // can not downplay completely.
      // Downplay whole series
      ChartView.prototype.downplay.call(this, seriesModel, ecModel, api, payload);
    }
  };
  LineView.prototype._changePolyState = function (toState) {
    var polygon = this._polygon;
    setStatesFlag(this._polyline, toState);
    polygon && setStatesFlag(polygon, toState);
  };
  LineView.prototype._newPolyline = function (points) {
    var polyline = this._polyline;
    // Remove previous created polyline
    if (polyline) {
      this._lineGroup.remove(polyline);
    }
    polyline = new ECPolyline({
      shape: {
        points: points
      },
      segmentIgnoreThreshold: 2,
      z2: 10
    });
    this._lineGroup.add(polyline);
    this._polyline = polyline;
    return polyline;
  };
  LineView.prototype._newPolygon = function (points, stackedOnPoints) {
    var polygon = this._polygon;
    // Remove previous created polygon
    if (polygon) {
      this._lineGroup.remove(polygon);
    }
    polygon = new ECPolygon({
      shape: {
        points: points,
        stackedOnPoints: stackedOnPoints
      },
      segmentIgnoreThreshold: 2
    });
    this._lineGroup.add(polygon);
    this._polygon = polygon;
    return polygon;
  };
  LineView.prototype._initSymbolLabelAnimation = function (data, coordSys, clipShape) {
    var isHorizontalOrRadial;
    var isCoordSysPolar;
    var baseAxis = coordSys.getBaseAxis();
    var isAxisInverse = baseAxis.inverse;
    if (coordSys.type === 'cartesian2d') {
      isHorizontalOrRadial = baseAxis.isHorizontal();
      isCoordSysPolar = false;
    } else if (coordSys.type === 'polar') {
      isHorizontalOrRadial = baseAxis.dim === 'angle';
      isCoordSysPolar = true;
    }
    var seriesModel = data.hostModel;
    var seriesDuration = seriesModel.get('animationDuration');
    if (zrUtil.isFunction(seriesDuration)) {
      seriesDuration = seriesDuration(null);
    }
    var seriesDelay = seriesModel.get('animationDelay') || 0;
    var seriesDelayValue = zrUtil.isFunction(seriesDelay) ? seriesDelay(null) : seriesDelay;
    data.eachItemGraphicEl(function (symbol, idx) {
      var el = symbol;
      if (el) {
        var point = [symbol.x, symbol.y];
        var start = void 0;
        var end = void 0;
        var current = void 0;
        if (clipShape) {
          if (isCoordSysPolar) {
            var polarClip = clipShape;
            var coord = coordSys.pointToCoord(point);
            if (isHorizontalOrRadial) {
              start = polarClip.startAngle;
              end = polarClip.endAngle;
              current = -coord[1] / 180 * Math.PI;
            } else {
              start = polarClip.r0;
              end = polarClip.r;
              current = coord[0];
            }
          } else {
            var gridClip = clipShape;
            if (isHorizontalOrRadial) {
              start = gridClip.x;
              end = gridClip.x + gridClip.width;
              current = symbol.x;
            } else {
              start = gridClip.y + gridClip.height;
              end = gridClip.y;
              current = symbol.y;
            }
          }
        }
        var ratio = end === start ? 0 : (current - start) / (end - start);
        if (isAxisInverse) {
          ratio = 1 - ratio;
        }
        var delay = zrUtil.isFunction(seriesDelay) ? seriesDelay(idx) : seriesDuration * ratio + seriesDelayValue;
        var symbolPath = el.getSymbolPath();
        var text = symbolPath.getTextContent();
        el.attr({
          scaleX: 0,
          scaleY: 0
        });
        el.animateTo({
          scaleX: 1,
          scaleY: 1
        }, {
          duration: 200,
          setToFinal: true,
          delay: delay
        });
        if (text) {
          text.animateFrom({
            style: {
              opacity: 0
            }
          }, {
            duration: 300,
            delay: delay
          });
        }
        symbolPath.disableLabelAnimation = true;
      }
    });
  };
  LineView.prototype._initOrUpdateEndLabel = function (seriesModel, coordSys, inheritColor) {
    var endLabelModel = seriesModel.getModel('endLabel');
    if (anyStateShowEndLabel(seriesModel)) {
      var data_2 = seriesModel.getData();
      var polyline = this._polyline;
      // series may be filtered.
      var points = data_2.getLayout('points');
      if (!points) {
        polyline.removeTextContent();
        this._endLabel = null;
        return;
      }
      var endLabel = this._endLabel;
      if (!endLabel) {
        endLabel = this._endLabel = new graphic.Text({
          z2: 200 // should be higher than item symbol
        });
        endLabel.ignoreClip = true;
        polyline.setTextContent(this._endLabel);
        polyline.disableLabelAnimation = true;
      }
      // Find last non-NaN data to display data
      var dataIndex = getLastIndexNotNull(points);
      if (dataIndex >= 0) {
        setLabelStyle(polyline, getLabelStatesModels(seriesModel, 'endLabel'), {
          inheritColor: inheritColor,
          labelFetcher: seriesModel,
          labelDataIndex: dataIndex,
          defaultText: function (dataIndex, opt, interpolatedValue) {
            return interpolatedValue != null ? getDefaultInterpolatedLabel(data_2, interpolatedValue) : getDefaultLabel(data_2, dataIndex);
          },
          enableTextSetter: true
        }, getEndLabelStateSpecified(endLabelModel, coordSys));
        polyline.textConfig.position = null;
      }
    } else if (this._endLabel) {
      this._polyline.removeTextContent();
      this._endLabel = null;
    }
  };
  LineView.prototype._endLabelOnDuring = function (percent, clipRect, data, animationRecord, valueAnimation, endLabelModel, coordSys) {
    var endLabel = this._endLabel;
    var polyline = this._polyline;
    if (endLabel) {
      // NOTE: Don't remove percent < 1. percent === 1 means the first frame during render.
      // The label is not prepared at this time.
      if (percent < 1 && animationRecord.originalX == null) {
        animationRecord.originalX = endLabel.x;
        animationRecord.originalY = endLabel.y;
      }
      var points = data.getLayout('points');
      var seriesModel = data.hostModel;
      var connectNulls = seriesModel.get('connectNulls');
      var precision = endLabelModel.get('precision');
      var distance = endLabelModel.get('distance') || 0;
      var baseAxis = coordSys.getBaseAxis();
      var isHorizontal = baseAxis.isHorizontal();
      var isBaseInversed = baseAxis.inverse;
      var clipShape = clipRect.shape;
      var xOrY = isBaseInversed ? isHorizontal ? clipShape.x : clipShape.y + clipShape.height : isHorizontal ? clipShape.x + clipShape.width : clipShape.y;
      var distanceX = (isHorizontal ? distance : 0) * (isBaseInversed ? -1 : 1);
      var distanceY = (isHorizontal ? 0 : -distance) * (isBaseInversed ? -1 : 1);
      var dim = isHorizontal ? 'x' : 'y';
      var dataIndexRange = getIndexRange(points, xOrY, dim);
      var indices = dataIndexRange.range;
      var diff = indices[1] - indices[0];
      var value = void 0;
      if (diff >= 1) {
        // diff > 1 && connectNulls, which is on the null data.
        if (diff > 1 && !connectNulls) {
          var pt = getPointAtIndex(points, indices[0]);
          endLabel.attr({
            x: pt[0] + distanceX,
            y: pt[1] + distanceY
          });
          valueAnimation && (value = seriesModel.getRawValue(indices[0]));
        } else {
          var pt = polyline.getPointOn(xOrY, dim);
          pt && endLabel.attr({
            x: pt[0] + distanceX,
            y: pt[1] + distanceY
          });
          var startValue = seriesModel.getRawValue(indices[0]);
          var endValue = seriesModel.getRawValue(indices[1]);
          valueAnimation && (value = modelUtil.interpolateRawValues(data, precision, startValue, endValue, dataIndexRange.t));
        }
        animationRecord.lastFrameIndex = indices[0];
      } else {
        // If diff <= 0, which is the range is not found(Include NaN)
        // Choose the first point or last point.
        var idx = percent === 1 || animationRecord.lastFrameIndex > 0 ? indices[0] : 0;
        var pt = getPointAtIndex(points, idx);
        valueAnimation && (value = seriesModel.getRawValue(idx));
        endLabel.attr({
          x: pt[0] + distanceX,
          y: pt[1] + distanceY
        });
      }
      if (valueAnimation) {
        var inner = labelInner(endLabel);
        if (typeof inner.setLabelText === 'function') {
          inner.setLabelText(value);
        }
      }
    }
  };
  /**
   * @private
   */
  // FIXME Two value axis
  LineView.prototype._doUpdateAnimation = function (data, stackedOnPoints, coordSys, api, step, valueOrigin, connectNulls) {
    var polyline = this._polyline;
    var polygon = this._polygon;
    var seriesModel = data.hostModel;
    var diff = lineAnimationDiff(this._data, data, this._stackedOnPoints, stackedOnPoints, this._coordSys, coordSys, this._valueOrigin, valueOrigin);
    var current = diff.current;
    var stackedOnCurrent = diff.stackedOnCurrent;
    var next = diff.next;
    var stackedOnNext = diff.stackedOnNext;
    if (step) {
      // TODO If stacked series is not step
      stackedOnCurrent = turnPointsIntoStep(diff.stackedOnCurrent, diff.current, coordSys, step, connectNulls);
      current = turnPointsIntoStep(diff.current, null, coordSys, step, connectNulls);
      stackedOnNext = turnPointsIntoStep(diff.stackedOnNext, diff.next, coordSys, step, connectNulls);
      next = turnPointsIntoStep(diff.next, null, coordSys, step, connectNulls);
    }
    // Don't apply animation if diff is large.
    // For better result and avoid memory explosion problems like
    // https://github.com/apache/incubator-echarts/issues/12229
    if (getBoundingDiff(current, next) > 3000 || polygon && getBoundingDiff(stackedOnCurrent, stackedOnNext) > 3000) {
      polyline.stopAnimation();
      polyline.setShape({
        points: next
      });
      if (polygon) {
        polygon.stopAnimation();
        polygon.setShape({
          points: next,
          stackedOnPoints: stackedOnNext
        });
      }
      return;
    }
    polyline.shape.__points = diff.current;
    polyline.shape.points = current;
    var target = {
      shape: {
        points: next
      }
    };
    // Also animate the original points.
    // If points reference is changed when turning into step line.
    if (diff.current !== current) {
      target.shape.__points = diff.next;
    }
    // Stop previous animation.
    polyline.stopAnimation();
    graphic.updateProps(polyline, target, seriesModel);
    if (polygon) {
      polygon.setShape({
        // Reuse the points with polyline.
        points: current,
        stackedOnPoints: stackedOnCurrent
      });
      polygon.stopAnimation();
      graphic.updateProps(polygon, {
        shape: {
          stackedOnPoints: stackedOnNext
        }
      }, seriesModel);
      // If use attr directly in updateProps.
      if (polyline.shape.points !== polygon.shape.points) {
        polygon.shape.points = polyline.shape.points;
      }
    }
    var updatedDataInfo = [];
    var diffStatus = diff.status;
    for (var i = 0; i < diffStatus.length; i++) {
      var cmd = diffStatus[i].cmd;
      if (cmd === '=') {
        var el = data.getItemGraphicEl(diffStatus[i].idx1);
        if (el) {
          updatedDataInfo.push({
            el: el,
            ptIdx: i // Index of points
          });
        }
      }
    }
    if (polyline.animators && polyline.animators.length) {
      polyline.animators[0].during(function () {
        polygon && polygon.dirtyShape();
        var points = polyline.shape.__points;
        for (var i = 0; i < updatedDataInfo.length; i++) {
          var el = updatedDataInfo[i].el;
          var offset = updatedDataInfo[i].ptIdx * 2;
          el.x = points[offset];
          el.y = points[offset + 1];
          el.markRedraw();
        }
      });
    }
  };
  LineView.prototype.remove = function (ecModel) {
    var group = this.group;
    var oldData = this._data;
    this._lineGroup.removeAll();
    this._symbolDraw.remove(true);
    // Remove temporary created elements when highlighting
    oldData && oldData.eachItemGraphicEl(function (el, idx) {
      if (el.__temp) {
        group.remove(el);
        oldData.setItemGraphicEl(idx, null);
      }
    });
    this._polyline = this._polygon = this._coordSys = this._points = this._stackedOnPoints = this._endLabel = this._data = null;
  };
  LineView.type = 'line';
  return LineView;
}(ChartView);
export default LineView;