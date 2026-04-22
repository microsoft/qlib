
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
import PointerPath from './PointerPath.js';
import * as graphic from '../../util/graphic.js';
import { setStatesStylesFromModel, toggleHoverEmphasis } from '../../util/states.js';
import { createTextStyle, setLabelValueAnimation, animateLabelValue } from '../../label/labelStyle.js';
import ChartView from '../../view/Chart.js';
import { parsePercent, round, linearMap } from '../../util/number.js';
import Sausage from '../../util/shape/sausage.js';
import { createSymbol } from '../../util/symbol.js';
import ZRImage from 'zrender/lib/graphic/Image.js';
import { extend, isFunction, isString, isNumber, each } from 'zrender/lib/core/util.js';
import { setCommonECData } from '../../util/innerStore.js';
import { normalizeArcAngles } from 'zrender/lib/core/PathProxy.js';
function parsePosition(seriesModel, api) {
  var center = seriesModel.get('center');
  var width = api.getWidth();
  var height = api.getHeight();
  var size = Math.min(width, height);
  var cx = parsePercent(center[0], api.getWidth());
  var cy = parsePercent(center[1], api.getHeight());
  var r = parsePercent(seriesModel.get('radius'), size / 2);
  return {
    cx: cx,
    cy: cy,
    r: r
  };
}
function formatLabel(value, labelFormatter) {
  var label = value == null ? '' : value + '';
  if (labelFormatter) {
    if (isString(labelFormatter)) {
      label = labelFormatter.replace('{value}', label);
    } else if (isFunction(labelFormatter)) {
      label = labelFormatter(value);
    }
  }
  return label;
}
var GaugeView = /** @class */function (_super) {
  __extends(GaugeView, _super);
  function GaugeView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = GaugeView.type;
    return _this;
  }
  GaugeView.prototype.render = function (seriesModel, ecModel, api) {
    this.group.removeAll();
    var colorList = seriesModel.get(['axisLine', 'lineStyle', 'color']);
    var posInfo = parsePosition(seriesModel, api);
    this._renderMain(seriesModel, ecModel, api, colorList, posInfo);
    this._data = seriesModel.getData();
  };
  GaugeView.prototype.dispose = function () {};
  GaugeView.prototype._renderMain = function (seriesModel, ecModel, api, colorList, posInfo) {
    var group = this.group;
    var clockwise = seriesModel.get('clockwise');
    var startAngle = -seriesModel.get('startAngle') / 180 * Math.PI;
    var endAngle = -seriesModel.get('endAngle') / 180 * Math.PI;
    var axisLineModel = seriesModel.getModel('axisLine');
    var roundCap = axisLineModel.get('roundCap');
    var MainPath = roundCap ? Sausage : graphic.Sector;
    var showAxis = axisLineModel.get('show');
    var lineStyleModel = axisLineModel.getModel('lineStyle');
    var axisLineWidth = lineStyleModel.get('width');
    var angles = [startAngle, endAngle];
    normalizeArcAngles(angles, !clockwise);
    startAngle = angles[0];
    endAngle = angles[1];
    var angleRangeSpan = endAngle - startAngle;
    var prevEndAngle = startAngle;
    var sectors = [];
    for (var i = 0; showAxis && i < colorList.length; i++) {
      // Clamp
      var percent = Math.min(Math.max(colorList[i][0], 0), 1);
      endAngle = startAngle + angleRangeSpan * percent;
      var sector = new MainPath({
        shape: {
          startAngle: prevEndAngle,
          endAngle: endAngle,
          cx: posInfo.cx,
          cy: posInfo.cy,
          clockwise: clockwise,
          r0: posInfo.r - axisLineWidth,
          r: posInfo.r
        },
        silent: true
      });
      sector.setStyle({
        fill: colorList[i][1]
      });
      sector.setStyle(lineStyleModel.getLineStyle(
      // Because we use sector to simulate arc
      // so the properties for stroking are useless
      ['color', 'width']));
      sectors.push(sector);
      prevEndAngle = endAngle;
    }
    sectors.reverse();
    each(sectors, function (sector) {
      return group.add(sector);
    });
    var getColor = function (percent) {
      // Less than 0
      if (percent <= 0) {
        return colorList[0][1];
      }
      var i;
      for (i = 0; i < colorList.length; i++) {
        if (colorList[i][0] >= percent && (i === 0 ? 0 : colorList[i - 1][0]) < percent) {
          return colorList[i][1];
        }
      }
      // More than 1
      return colorList[i - 1][1];
    };
    this._renderTicks(seriesModel, ecModel, api, getColor, posInfo, startAngle, endAngle, clockwise, axisLineWidth);
    this._renderTitleAndDetail(seriesModel, ecModel, api, getColor, posInfo);
    this._renderAnchor(seriesModel, posInfo);
    this._renderPointer(seriesModel, ecModel, api, getColor, posInfo, startAngle, endAngle, clockwise, axisLineWidth);
  };
  GaugeView.prototype._renderTicks = function (seriesModel, ecModel, api, getColor, posInfo, startAngle, endAngle, clockwise, axisLineWidth) {
    var group = this.group;
    var cx = posInfo.cx;
    var cy = posInfo.cy;
    var r = posInfo.r;
    var minVal = +seriesModel.get('min');
    var maxVal = +seriesModel.get('max');
    var splitLineModel = seriesModel.getModel('splitLine');
    var tickModel = seriesModel.getModel('axisTick');
    var labelModel = seriesModel.getModel('axisLabel');
    var splitNumber = seriesModel.get('splitNumber');
    var subSplitNumber = tickModel.get('splitNumber');
    var splitLineLen = parsePercent(splitLineModel.get('length'), r);
    var tickLen = parsePercent(tickModel.get('length'), r);
    var angle = startAngle;
    var step = (endAngle - startAngle) / splitNumber;
    var subStep = step / subSplitNumber;
    var splitLineStyle = splitLineModel.getModel('lineStyle').getLineStyle();
    var tickLineStyle = tickModel.getModel('lineStyle').getLineStyle();
    var splitLineDistance = splitLineModel.get('distance');
    var unitX;
    var unitY;
    for (var i = 0; i <= splitNumber; i++) {
      unitX = Math.cos(angle);
      unitY = Math.sin(angle);
      // Split line
      if (splitLineModel.get('show')) {
        var distance = splitLineDistance ? splitLineDistance + axisLineWidth : axisLineWidth;
        var splitLine = new graphic.Line({
          shape: {
            x1: unitX * (r - distance) + cx,
            y1: unitY * (r - distance) + cy,
            x2: unitX * (r - splitLineLen - distance) + cx,
            y2: unitY * (r - splitLineLen - distance) + cy
          },
          style: splitLineStyle,
          silent: true
        });
        if (splitLineStyle.stroke === 'auto') {
          splitLine.setStyle({
            stroke: getColor(i / splitNumber)
          });
        }
        group.add(splitLine);
      }
      // Label
      if (labelModel.get('show')) {
        var distance = labelModel.get('distance') + splitLineDistance;
        var label = formatLabel(round(i / splitNumber * (maxVal - minVal) + minVal), labelModel.get('formatter'));
        var autoColor = getColor(i / splitNumber);
        var textStyleX = unitX * (r - splitLineLen - distance) + cx;
        var textStyleY = unitY * (r - splitLineLen - distance) + cy;
        var rotateType = labelModel.get('rotate');
        var rotate = 0;
        if (rotateType === 'radial') {
          rotate = -angle + 2 * Math.PI;
          if (rotate > Math.PI / 2) {
            rotate += Math.PI;
          }
        } else if (rotateType === 'tangential') {
          rotate = -angle - Math.PI / 2;
        } else if (isNumber(rotateType)) {
          rotate = rotateType * Math.PI / 180;
        }
        if (rotate === 0) {
          group.add(new graphic.Text({
            style: createTextStyle(labelModel, {
              text: label,
              x: textStyleX,
              y: textStyleY,
              verticalAlign: unitY < -0.8 ? 'top' : unitY > 0.8 ? 'bottom' : 'middle',
              align: unitX < -0.4 ? 'left' : unitX > 0.4 ? 'right' : 'center'
            }, {
              inheritColor: autoColor
            }),
            silent: true
          }));
        } else {
          group.add(new graphic.Text({
            style: createTextStyle(labelModel, {
              text: label,
              x: textStyleX,
              y: textStyleY,
              verticalAlign: 'middle',
              align: 'center'
            }, {
              inheritColor: autoColor
            }),
            silent: true,
            originX: textStyleX,
            originY: textStyleY,
            rotation: rotate
          }));
        }
      }
      // Axis tick
      if (tickModel.get('show') && i !== splitNumber) {
        var distance = tickModel.get('distance');
        distance = distance ? distance + axisLineWidth : axisLineWidth;
        for (var j = 0; j <= subSplitNumber; j++) {
          unitX = Math.cos(angle);
          unitY = Math.sin(angle);
          var tickLine = new graphic.Line({
            shape: {
              x1: unitX * (r - distance) + cx,
              y1: unitY * (r - distance) + cy,
              x2: unitX * (r - tickLen - distance) + cx,
              y2: unitY * (r - tickLen - distance) + cy
            },
            silent: true,
            style: tickLineStyle
          });
          if (tickLineStyle.stroke === 'auto') {
            tickLine.setStyle({
              stroke: getColor((i + j / subSplitNumber) / splitNumber)
            });
          }
          group.add(tickLine);
          angle += subStep;
        }
        angle -= subStep;
      } else {
        angle += step;
      }
    }
  };
  GaugeView.prototype._renderPointer = function (seriesModel, ecModel, api, getColor, posInfo, startAngle, endAngle, clockwise, axisLineWidth) {
    var group = this.group;
    var oldData = this._data;
    var oldProgressData = this._progressEls;
    var progressList = [];
    var showPointer = seriesModel.get(['pointer', 'show']);
    var progressModel = seriesModel.getModel('progress');
    var showProgress = progressModel.get('show');
    var data = seriesModel.getData();
    var valueDim = data.mapDimension('value');
    var minVal = +seriesModel.get('min');
    var maxVal = +seriesModel.get('max');
    var valueExtent = [minVal, maxVal];
    var angleExtent = [startAngle, endAngle];
    function createPointer(idx, angle) {
      var itemModel = data.getItemModel(idx);
      var pointerModel = itemModel.getModel('pointer');
      var pointerWidth = parsePercent(pointerModel.get('width'), posInfo.r);
      var pointerLength = parsePercent(pointerModel.get('length'), posInfo.r);
      var pointerStr = seriesModel.get(['pointer', 'icon']);
      var pointerOffset = pointerModel.get('offsetCenter');
      var pointerOffsetX = parsePercent(pointerOffset[0], posInfo.r);
      var pointerOffsetY = parsePercent(pointerOffset[1], posInfo.r);
      var pointerKeepAspect = pointerModel.get('keepAspect');
      var pointer;
      // not exist icon type will be set 'rect'
      if (pointerStr) {
        pointer = createSymbol(pointerStr, pointerOffsetX - pointerWidth / 2, pointerOffsetY - pointerLength, pointerWidth, pointerLength, null, pointerKeepAspect);
      } else {
        pointer = new PointerPath({
          shape: {
            angle: -Math.PI / 2,
            width: pointerWidth,
            r: pointerLength,
            x: pointerOffsetX,
            y: pointerOffsetY
          }
        });
      }
      pointer.rotation = -(angle + Math.PI / 2);
      pointer.x = posInfo.cx;
      pointer.y = posInfo.cy;
      return pointer;
    }
    function createProgress(idx, endAngle) {
      var roundCap = progressModel.get('roundCap');
      var ProgressPath = roundCap ? Sausage : graphic.Sector;
      var isOverlap = progressModel.get('overlap');
      var progressWidth = isOverlap ? progressModel.get('width') : axisLineWidth / data.count();
      var r0 = isOverlap ? posInfo.r - progressWidth : posInfo.r - (idx + 1) * progressWidth;
      var r = isOverlap ? posInfo.r : posInfo.r - idx * progressWidth;
      var progress = new ProgressPath({
        shape: {
          startAngle: startAngle,
          endAngle: endAngle,
          cx: posInfo.cx,
          cy: posInfo.cy,
          clockwise: clockwise,
          r0: r0,
          r: r
        }
      });
      isOverlap && (progress.z2 = linearMap(data.get(valueDim, idx), [minVal, maxVal], [100, 0], true));
      return progress;
    }
    if (showProgress || showPointer) {
      data.diff(oldData).add(function (idx) {
        var val = data.get(valueDim, idx);
        if (showPointer) {
          var pointer = createPointer(idx, startAngle);
          // TODO hide pointer on NaN value?
          graphic.initProps(pointer, {
            rotation: -((isNaN(+val) ? angleExtent[0] : linearMap(val, valueExtent, angleExtent, true)) + Math.PI / 2)
          }, seriesModel);
          group.add(pointer);
          data.setItemGraphicEl(idx, pointer);
        }
        if (showProgress) {
          var progress = createProgress(idx, startAngle);
          var isClip = progressModel.get('clip');
          graphic.initProps(progress, {
            shape: {
              endAngle: linearMap(val, valueExtent, angleExtent, isClip)
            }
          }, seriesModel);
          group.add(progress);
          // Add data index and series index for indexing the data by element
          // Useful in tooltip
          setCommonECData(seriesModel.seriesIndex, data.dataType, idx, progress);
          progressList[idx] = progress;
        }
      }).update(function (newIdx, oldIdx) {
        var val = data.get(valueDim, newIdx);
        if (showPointer) {
          var previousPointer = oldData.getItemGraphicEl(oldIdx);
          var previousRotate = previousPointer ? previousPointer.rotation : startAngle;
          var pointer = createPointer(newIdx, previousRotate);
          pointer.rotation = previousRotate;
          graphic.updateProps(pointer, {
            rotation: -((isNaN(+val) ? angleExtent[0] : linearMap(val, valueExtent, angleExtent, true)) + Math.PI / 2)
          }, seriesModel);
          group.add(pointer);
          data.setItemGraphicEl(newIdx, pointer);
        }
        if (showProgress) {
          var previousProgress = oldProgressData[oldIdx];
          var previousEndAngle = previousProgress ? previousProgress.shape.endAngle : startAngle;
          var progress = createProgress(newIdx, previousEndAngle);
          var isClip = progressModel.get('clip');
          graphic.updateProps(progress, {
            shape: {
              endAngle: linearMap(val, valueExtent, angleExtent, isClip)
            }
          }, seriesModel);
          group.add(progress);
          // Add data index and series index for indexing the data by element
          // Useful in tooltip
          setCommonECData(seriesModel.seriesIndex, data.dataType, newIdx, progress);
          progressList[newIdx] = progress;
        }
      }).execute();
      data.each(function (idx) {
        var itemModel = data.getItemModel(idx);
        var emphasisModel = itemModel.getModel('emphasis');
        var focus = emphasisModel.get('focus');
        var blurScope = emphasisModel.get('blurScope');
        var emphasisDisabled = emphasisModel.get('disabled');
        if (showPointer) {
          var pointer = data.getItemGraphicEl(idx);
          var symbolStyle = data.getItemVisual(idx, 'style');
          var visualColor = symbolStyle.fill;
          if (pointer instanceof ZRImage) {
            var pathStyle = pointer.style;
            pointer.useStyle(extend({
              image: pathStyle.image,
              x: pathStyle.x,
              y: pathStyle.y,
              width: pathStyle.width,
              height: pathStyle.height
            }, symbolStyle));
          } else {
            pointer.useStyle(symbolStyle);
            pointer.type !== 'pointer' && pointer.setColor(visualColor);
          }
          pointer.setStyle(itemModel.getModel(['pointer', 'itemStyle']).getItemStyle());
          if (pointer.style.fill === 'auto') {
            pointer.setStyle('fill', getColor(linearMap(data.get(valueDim, idx), valueExtent, [0, 1], true)));
          }
          pointer.z2EmphasisLift = 0;
          setStatesStylesFromModel(pointer, itemModel);
          toggleHoverEmphasis(pointer, focus, blurScope, emphasisDisabled);
        }
        if (showProgress) {
          var progress = progressList[idx];
          progress.useStyle(data.getItemVisual(idx, 'style'));
          progress.setStyle(itemModel.getModel(['progress', 'itemStyle']).getItemStyle());
          progress.z2EmphasisLift = 0;
          setStatesStylesFromModel(progress, itemModel);
          toggleHoverEmphasis(progress, focus, blurScope, emphasisDisabled);
        }
      });
      this._progressEls = progressList;
    }
  };
  GaugeView.prototype._renderAnchor = function (seriesModel, posInfo) {
    var anchorModel = seriesModel.getModel('anchor');
    var showAnchor = anchorModel.get('show');
    if (showAnchor) {
      var anchorSize = anchorModel.get('size');
      var anchorType = anchorModel.get('icon');
      var offsetCenter = anchorModel.get('offsetCenter');
      var anchorKeepAspect = anchorModel.get('keepAspect');
      var anchor = createSymbol(anchorType, posInfo.cx - anchorSize / 2 + parsePercent(offsetCenter[0], posInfo.r), posInfo.cy - anchorSize / 2 + parsePercent(offsetCenter[1], posInfo.r), anchorSize, anchorSize, null, anchorKeepAspect);
      anchor.z2 = anchorModel.get('showAbove') ? 1 : 0;
      anchor.setStyle(anchorModel.getModel('itemStyle').getItemStyle());
      this.group.add(anchor);
    }
  };
  GaugeView.prototype._renderTitleAndDetail = function (seriesModel, ecModel, api, getColor, posInfo) {
    var _this = this;
    var data = seriesModel.getData();
    var valueDim = data.mapDimension('value');
    var minVal = +seriesModel.get('min');
    var maxVal = +seriesModel.get('max');
    var contentGroup = new graphic.Group();
    var newTitleEls = [];
    var newDetailEls = [];
    var hasAnimation = seriesModel.isAnimationEnabled();
    var showPointerAbove = seriesModel.get(['pointer', 'showAbove']);
    data.diff(this._data).add(function (idx) {
      newTitleEls[idx] = new graphic.Text({
        silent: true
      });
      newDetailEls[idx] = new graphic.Text({
        silent: true
      });
    }).update(function (idx, oldIdx) {
      newTitleEls[idx] = _this._titleEls[oldIdx];
      newDetailEls[idx] = _this._detailEls[oldIdx];
    }).execute();
    data.each(function (idx) {
      var itemModel = data.getItemModel(idx);
      var value = data.get(valueDim, idx);
      var itemGroup = new graphic.Group();
      var autoColor = getColor(linearMap(value, [minVal, maxVal], [0, 1], true));
      var itemTitleModel = itemModel.getModel('title');
      if (itemTitleModel.get('show')) {
        var titleOffsetCenter = itemTitleModel.get('offsetCenter');
        var titleX = posInfo.cx + parsePercent(titleOffsetCenter[0], posInfo.r);
        var titleY = posInfo.cy + parsePercent(titleOffsetCenter[1], posInfo.r);
        var labelEl = newTitleEls[idx];
        labelEl.attr({
          z2: showPointerAbove ? 0 : 2,
          style: createTextStyle(itemTitleModel, {
            x: titleX,
            y: titleY,
            text: data.getName(idx),
            align: 'center',
            verticalAlign: 'middle'
          }, {
            inheritColor: autoColor
          })
        });
        itemGroup.add(labelEl);
      }
      var itemDetailModel = itemModel.getModel('detail');
      if (itemDetailModel.get('show')) {
        var detailOffsetCenter = itemDetailModel.get('offsetCenter');
        var detailX = posInfo.cx + parsePercent(detailOffsetCenter[0], posInfo.r);
        var detailY = posInfo.cy + parsePercent(detailOffsetCenter[1], posInfo.r);
        var width = parsePercent(itemDetailModel.get('width'), posInfo.r);
        var height = parsePercent(itemDetailModel.get('height'), posInfo.r);
        var detailColor = seriesModel.get(['progress', 'show']) ? data.getItemVisual(idx, 'style').fill : autoColor;
        var labelEl = newDetailEls[idx];
        var formatter_1 = itemDetailModel.get('formatter');
        labelEl.attr({
          z2: showPointerAbove ? 0 : 2,
          style: createTextStyle(itemDetailModel, {
            x: detailX,
            y: detailY,
            text: formatLabel(value, formatter_1),
            width: isNaN(width) ? null : width,
            height: isNaN(height) ? null : height,
            align: 'center',
            verticalAlign: 'middle'
          }, {
            inheritColor: detailColor
          })
        });
        setLabelValueAnimation(labelEl, {
          normal: itemDetailModel
        }, value, function (value) {
          return formatLabel(value, formatter_1);
        });
        hasAnimation && animateLabelValue(labelEl, idx, data, seriesModel, {
          getFormattedLabel: function (labelDataIndex, status, dataType, labelDimIndex, fmt, extendParams) {
            return formatLabel(extendParams ? extendParams.interpolatedValue : value, formatter_1);
          }
        });
        itemGroup.add(labelEl);
      }
      contentGroup.add(itemGroup);
    });
    this.group.add(contentGroup);
    this._titleEls = newTitleEls;
    this._detailEls = newDetailEls;
  };
  GaugeView.type = 'gauge';
  return GaugeView;
}(ChartView);
export default GaugeView;