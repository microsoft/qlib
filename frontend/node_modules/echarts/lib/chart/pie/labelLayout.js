
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
// FIXME emphasis label position is not same with normal label position
import { parsePercent } from '../../util/number.js';
import { Point } from '../../util/graphic.js';
import { each, isNumber } from 'zrender/lib/core/util.js';
import { limitTurnAngle, limitSurfaceAngle } from '../../label/labelGuideHelper.js';
import { shiftLayoutOnY } from '../../label/labelLayoutHelper.js';
var RADIAN = Math.PI / 180;
function adjustSingleSide(list, cx, cy, r, dir, viewWidth, viewHeight, viewLeft, viewTop, farthestX) {
  if (list.length < 2) {
    return;
  }
  ;
  function recalculateXOnSemiToAlignOnEllipseCurve(semi) {
    var rB = semi.rB;
    var rB2 = rB * rB;
    for (var i = 0; i < semi.list.length; i++) {
      var item = semi.list[i];
      var dy = Math.abs(item.label.y - cy);
      // horizontal r is always same with original r because x is not changed.
      var rA = r + item.len;
      var rA2 = rA * rA;
      // Use ellipse implicit function to calculate x
      var dx = Math.sqrt(Math.abs((1 - dy * dy / rB2) * rA2));
      var newX = cx + (dx + item.len2) * dir;
      var deltaX = newX - item.label.x;
      var newTargetWidth = item.targetTextWidth - deltaX * dir;
      // text x is changed, so need to recalculate width.
      constrainTextWidth(item, newTargetWidth, true);
      item.label.x = newX;
    }
  }
  // Adjust X based on the shifted y. Make tight labels aligned on an ellipse curve.
  function recalculateX(items) {
    // Extremes of
    var topSemi = {
      list: [],
      maxY: 0
    };
    var bottomSemi = {
      list: [],
      maxY: 0
    };
    for (var i = 0; i < items.length; i++) {
      if (items[i].labelAlignTo !== 'none') {
        continue;
      }
      var item = items[i];
      var semi = item.label.y > cy ? bottomSemi : topSemi;
      var dy = Math.abs(item.label.y - cy);
      if (dy >= semi.maxY) {
        var dx = item.label.x - cx - item.len2 * dir;
        // horizontal r is always same with original r because x is not changed.
        var rA = r + item.len;
        // Canculate rB based on the topest / bottemest label.
        var rB = Math.abs(dx) < rA ? Math.sqrt(dy * dy / (1 - dx * dx / rA / rA)) : rA;
        semi.rB = rB;
        semi.maxY = dy;
      }
      semi.list.push(item);
    }
    recalculateXOnSemiToAlignOnEllipseCurve(topSemi);
    recalculateXOnSemiToAlignOnEllipseCurve(bottomSemi);
  }
  var len = list.length;
  for (var i = 0; i < len; i++) {
    if (list[i].position === 'outer' && list[i].labelAlignTo === 'labelLine') {
      var dx = list[i].label.x - farthestX;
      list[i].linePoints[1][0] += dx;
      list[i].label.x = farthestX;
    }
  }
  if (shiftLayoutOnY(list, viewTop, viewTop + viewHeight)) {
    recalculateX(list);
  }
}
function avoidOverlap(labelLayoutList, cx, cy, r, viewWidth, viewHeight, viewLeft, viewTop) {
  var leftList = [];
  var rightList = [];
  var leftmostX = Number.MAX_VALUE;
  var rightmostX = -Number.MAX_VALUE;
  for (var i = 0; i < labelLayoutList.length; i++) {
    var label = labelLayoutList[i].label;
    if (isPositionCenter(labelLayoutList[i])) {
      continue;
    }
    if (label.x < cx) {
      leftmostX = Math.min(leftmostX, label.x);
      leftList.push(labelLayoutList[i]);
    } else {
      rightmostX = Math.max(rightmostX, label.x);
      rightList.push(labelLayoutList[i]);
    }
  }
  for (var i = 0; i < labelLayoutList.length; i++) {
    var layout = labelLayoutList[i];
    if (!isPositionCenter(layout) && layout.linePoints) {
      if (layout.labelStyleWidth != null) {
        continue;
      }
      var label = layout.label;
      var linePoints = layout.linePoints;
      var targetTextWidth = void 0;
      if (layout.labelAlignTo === 'edge') {
        if (label.x < cx) {
          targetTextWidth = linePoints[2][0] - layout.labelDistance - viewLeft - layout.edgeDistance;
        } else {
          targetTextWidth = viewLeft + viewWidth - layout.edgeDistance - linePoints[2][0] - layout.labelDistance;
        }
      } else if (layout.labelAlignTo === 'labelLine') {
        if (label.x < cx) {
          targetTextWidth = leftmostX - viewLeft - layout.bleedMargin;
        } else {
          targetTextWidth = viewLeft + viewWidth - rightmostX - layout.bleedMargin;
        }
      } else {
        if (label.x < cx) {
          targetTextWidth = label.x - viewLeft - layout.bleedMargin;
        } else {
          targetTextWidth = viewLeft + viewWidth - label.x - layout.bleedMargin;
        }
      }
      layout.targetTextWidth = targetTextWidth;
      constrainTextWidth(layout, targetTextWidth);
    }
  }
  adjustSingleSide(rightList, cx, cy, r, 1, viewWidth, viewHeight, viewLeft, viewTop, rightmostX);
  adjustSingleSide(leftList, cx, cy, r, -1, viewWidth, viewHeight, viewLeft, viewTop, leftmostX);
  for (var i = 0; i < labelLayoutList.length; i++) {
    var layout = labelLayoutList[i];
    if (!isPositionCenter(layout) && layout.linePoints) {
      var label = layout.label;
      var linePoints = layout.linePoints;
      var isAlignToEdge = layout.labelAlignTo === 'edge';
      var padding = label.style.padding;
      var paddingH = padding ? padding[1] + padding[3] : 0;
      // textRect.width already contains paddingH if bgColor is set
      var extraPaddingH = label.style.backgroundColor ? 0 : paddingH;
      var realTextWidth = layout.rect.width + extraPaddingH;
      var dist = linePoints[1][0] - linePoints[2][0];
      if (isAlignToEdge) {
        if (label.x < cx) {
          linePoints[2][0] = viewLeft + layout.edgeDistance + realTextWidth + layout.labelDistance;
        } else {
          linePoints[2][0] = viewLeft + viewWidth - layout.edgeDistance - realTextWidth - layout.labelDistance;
        }
      } else {
        if (label.x < cx) {
          linePoints[2][0] = label.x + layout.labelDistance;
        } else {
          linePoints[2][0] = label.x - layout.labelDistance;
        }
        linePoints[1][0] = linePoints[2][0] + dist;
      }
      linePoints[1][1] = linePoints[2][1] = label.y;
    }
  }
}
/**
 * Set max width of each label, and then wrap each label to the max width.
 *
 * @param layout label layout
 * @param availableWidth max width for the label to display
 * @param forceRecalculate recaculate the text layout even if the current width
 * is smaller than `availableWidth`. This is useful when the text was previously
 * wrapped by calling `constrainTextWidth` but now `availableWidth` changed, in
 * which case, previous wrapping should be redo.
 */
function constrainTextWidth(layout, availableWidth, forceRecalculate) {
  if (forceRecalculate === void 0) {
    forceRecalculate = false;
  }
  if (layout.labelStyleWidth != null) {
    // User-defined style.width has the highest priority.
    return;
  }
  var label = layout.label;
  var style = label.style;
  var textRect = layout.rect;
  var bgColor = style.backgroundColor;
  var padding = style.padding;
  var paddingH = padding ? padding[1] + padding[3] : 0;
  var overflow = style.overflow;
  // textRect.width already contains paddingH if bgColor is set
  var oldOuterWidth = textRect.width + (bgColor ? 0 : paddingH);
  if (availableWidth < oldOuterWidth || forceRecalculate) {
    var oldHeight = textRect.height;
    if (overflow && overflow.match('break')) {
      // Temporarily set background to be null to calculate
      // the bounding box without background.
      label.setStyle('backgroundColor', null);
      // Set constraining width
      label.setStyle('width', availableWidth - paddingH);
      // This is the real bounding box of the text without padding.
      var innerRect = label.getBoundingRect();
      label.setStyle('width', Math.ceil(innerRect.width));
      label.setStyle('backgroundColor', bgColor);
    } else {
      var availableInnerWidth = availableWidth - paddingH;
      var newWidth = availableWidth < oldOuterWidth
      // Current text is too wide, use `availableWidth` as max width.
      ? availableInnerWidth :
      // Current available width is enough, but the text may have
      // already been wrapped with a smaller available width.
      forceRecalculate ? availableInnerWidth > layout.unconstrainedWidth
      // Current available is larger than text width,
      // so don't constrain width (otherwise it may have
      // empty space in the background).
      ? null
      // Current available is smaller than text width, so
      // use the current available width as constraining
      // width.
      : availableInnerWidth
      // Current available width is enough, so no need to
      // constrain.
      : null;
      label.setStyle('width', newWidth);
    }
    var newRect = label.getBoundingRect();
    textRect.width = newRect.width;
    var margin = (label.style.margin || 0) + 2.1;
    textRect.height = newRect.height + margin;
    textRect.y -= (textRect.height - oldHeight) / 2;
  }
}
function isPositionCenter(sectorShape) {
  // Not change x for center label
  return sectorShape.position === 'center';
}
export default function pieLabelLayout(seriesModel) {
  var data = seriesModel.getData();
  var labelLayoutList = [];
  var cx;
  var cy;
  var hasLabelRotate = false;
  var minShowLabelRadian = (seriesModel.get('minShowLabelAngle') || 0) * RADIAN;
  var viewRect = data.getLayout('viewRect');
  var r = data.getLayout('r');
  var viewWidth = viewRect.width;
  var viewLeft = viewRect.x;
  var viewTop = viewRect.y;
  var viewHeight = viewRect.height;
  function setNotShow(el) {
    el.ignore = true;
  }
  function isLabelShown(label) {
    if (!label.ignore) {
      return true;
    }
    for (var key in label.states) {
      if (label.states[key].ignore === false) {
        return true;
      }
    }
    return false;
  }
  data.each(function (idx) {
    var sector = data.getItemGraphicEl(idx);
    var sectorShape = sector.shape;
    var label = sector.getTextContent();
    var labelLine = sector.getTextGuideLine();
    var itemModel = data.getItemModel(idx);
    var labelModel = itemModel.getModel('label');
    // Use position in normal or emphasis
    var labelPosition = labelModel.get('position') || itemModel.get(['emphasis', 'label', 'position']);
    var labelDistance = labelModel.get('distanceToLabelLine');
    var labelAlignTo = labelModel.get('alignTo');
    var edgeDistance = parsePercent(labelModel.get('edgeDistance'), viewWidth);
    var bleedMargin = labelModel.get('bleedMargin');
    var labelLineModel = itemModel.getModel('labelLine');
    var labelLineLen = labelLineModel.get('length');
    labelLineLen = parsePercent(labelLineLen, viewWidth);
    var labelLineLen2 = labelLineModel.get('length2');
    labelLineLen2 = parsePercent(labelLineLen2, viewWidth);
    if (Math.abs(sectorShape.endAngle - sectorShape.startAngle) < minShowLabelRadian) {
      each(label.states, setNotShow);
      label.ignore = true;
      if (labelLine) {
        each(labelLine.states, setNotShow);
        labelLine.ignore = true;
      }
      return;
    }
    if (!isLabelShown(label)) {
      return;
    }
    var midAngle = (sectorShape.startAngle + sectorShape.endAngle) / 2;
    var nx = Math.cos(midAngle);
    var ny = Math.sin(midAngle);
    var textX;
    var textY;
    var linePoints;
    var textAlign;
    cx = sectorShape.cx;
    cy = sectorShape.cy;
    var isLabelInside = labelPosition === 'inside' || labelPosition === 'inner';
    if (labelPosition === 'center') {
      textX = sectorShape.cx;
      textY = sectorShape.cy;
      textAlign = 'center';
    } else {
      var x1 = (isLabelInside ? (sectorShape.r + sectorShape.r0) / 2 * nx : sectorShape.r * nx) + cx;
      var y1 = (isLabelInside ? (sectorShape.r + sectorShape.r0) / 2 * ny : sectorShape.r * ny) + cy;
      textX = x1 + nx * 3;
      textY = y1 + ny * 3;
      if (!isLabelInside) {
        // For roseType
        var x2 = x1 + nx * (labelLineLen + r - sectorShape.r);
        var y2 = y1 + ny * (labelLineLen + r - sectorShape.r);
        var x3 = x2 + (nx < 0 ? -1 : 1) * labelLineLen2;
        var y3 = y2;
        if (labelAlignTo === 'edge') {
          // Adjust textX because text align of edge is opposite
          textX = nx < 0 ? viewLeft + edgeDistance : viewLeft + viewWidth - edgeDistance;
        } else {
          textX = x3 + (nx < 0 ? -labelDistance : labelDistance);
        }
        textY = y3;
        linePoints = [[x1, y1], [x2, y2], [x3, y3]];
      }
      textAlign = isLabelInside ? 'center' : labelAlignTo === 'edge' ? nx > 0 ? 'right' : 'left' : nx > 0 ? 'left' : 'right';
    }
    var PI = Math.PI;
    var labelRotate = 0;
    var rotate = labelModel.get('rotate');
    if (isNumber(rotate)) {
      labelRotate = rotate * (PI / 180);
    } else if (labelPosition === 'center') {
      labelRotate = 0;
    } else if (rotate === 'radial' || rotate === true) {
      var radialAngle = nx < 0 ? -midAngle + PI : -midAngle;
      labelRotate = radialAngle;
    } else if (rotate === 'tangential' && labelPosition !== 'outside' && labelPosition !== 'outer') {
      var rad = Math.atan2(nx, ny);
      if (rad < 0) {
        rad = PI * 2 + rad;
      }
      var isDown = ny > 0;
      if (isDown) {
        rad = PI + rad;
      }
      labelRotate = rad - PI;
    }
    hasLabelRotate = !!labelRotate;
    label.x = textX;
    label.y = textY;
    label.rotation = labelRotate;
    label.setStyle({
      verticalAlign: 'middle'
    });
    // Not sectorShape the inside label
    if (!isLabelInside) {
      var textRect = label.getBoundingRect().clone();
      textRect.applyTransform(label.getComputedTransform());
      // Text has a default 1px stroke. Exclude this.
      var margin = (label.style.margin || 0) + 2.1;
      textRect.y -= margin / 2;
      textRect.height += margin;
      labelLayoutList.push({
        label: label,
        labelLine: labelLine,
        position: labelPosition,
        len: labelLineLen,
        len2: labelLineLen2,
        minTurnAngle: labelLineModel.get('minTurnAngle'),
        maxSurfaceAngle: labelLineModel.get('maxSurfaceAngle'),
        surfaceNormal: new Point(nx, ny),
        linePoints: linePoints,
        textAlign: textAlign,
        labelDistance: labelDistance,
        labelAlignTo: labelAlignTo,
        edgeDistance: edgeDistance,
        bleedMargin: bleedMargin,
        rect: textRect,
        unconstrainedWidth: textRect.width,
        labelStyleWidth: label.style.width
      });
    } else {
      label.setStyle({
        align: textAlign
      });
      var selectState = label.states.select;
      if (selectState) {
        selectState.x += label.x;
        selectState.y += label.y;
      }
    }
    sector.setTextConfig({
      inside: isLabelInside
    });
  });
  if (!hasLabelRotate && seriesModel.get('avoidLabelOverlap')) {
    avoidOverlap(labelLayoutList, cx, cy, r, viewWidth, viewHeight, viewLeft, viewTop);
  }
  for (var i = 0; i < labelLayoutList.length; i++) {
    var layout = labelLayoutList[i];
    var label = layout.label;
    var labelLine = layout.labelLine;
    var notShowLabel = isNaN(label.x) || isNaN(label.y);
    if (label) {
      label.setStyle({
        align: layout.textAlign
      });
      if (notShowLabel) {
        each(label.states, setNotShow);
        label.ignore = true;
      }
      var selectState = label.states.select;
      if (selectState) {
        selectState.x += label.x;
        selectState.y += label.y;
      }
    }
    if (labelLine) {
      var linePoints = layout.linePoints;
      if (notShowLabel || !linePoints) {
        each(labelLine.states, setNotShow);
        labelLine.ignore = true;
      } else {
        limitTurnAngle(linePoints, layout.minTurnAngle);
        limitSurfaceAngle(linePoints, layout.surfaceNormal, layout.maxSurfaceAngle);
        labelLine.setShape({
          points: linePoints
        });
        // Set the anchor to the midpoint of sector
        label.__hostTarget.textGuideLineConfig = {
          anchor: new Point(linePoints[0][0], linePoints[0][1])
        };
      }
    }
  }
}