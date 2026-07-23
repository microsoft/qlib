
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
import { toggleHoverEmphasis } from '../../util/states.js';
import { createSymbol, normalizeSymbolOffset } from '../../util/symbol.js';
import { parsePercent, isNumeric } from '../../util/number.js';
import ChartView from '../../view/Chart.js';
import { getDefaultLabel } from '../helper/labelHelper.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import ZRImage from 'zrender/lib/graphic/Image.js';
import { getECData } from '../../util/innerStore.js';
import { createClipPath } from '../helper/createClipPathFromCoordSys.js';
var BAR_BORDER_WIDTH_QUERY = ['itemStyle', 'borderWidth'];
// index: +isHorizontal
var LAYOUT_ATTRS = [{
  xy: 'x',
  wh: 'width',
  index: 0,
  posDesc: ['left', 'right']
}, {
  xy: 'y',
  wh: 'height',
  index: 1,
  posDesc: ['top', 'bottom']
}];
var pathForLineWidth = new graphic.Circle();
var PictorialBarView = /** @class */function (_super) {
  __extends(PictorialBarView, _super);
  function PictorialBarView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = PictorialBarView.type;
    return _this;
  }
  PictorialBarView.prototype.render = function (seriesModel, ecModel, api) {
    var group = this.group;
    var data = seriesModel.getData();
    var oldData = this._data;
    var cartesian = seriesModel.coordinateSystem;
    var baseAxis = cartesian.getBaseAxis();
    var isHorizontal = baseAxis.isHorizontal();
    var coordSysRect = cartesian.master.getRect();
    var opt = {
      ecSize: {
        width: api.getWidth(),
        height: api.getHeight()
      },
      seriesModel: seriesModel,
      coordSys: cartesian,
      coordSysExtent: [[coordSysRect.x, coordSysRect.x + coordSysRect.width], [coordSysRect.y, coordSysRect.y + coordSysRect.height]],
      isHorizontal: isHorizontal,
      valueDim: LAYOUT_ATTRS[+isHorizontal],
      categoryDim: LAYOUT_ATTRS[1 - +isHorizontal]
    };
    data.diff(oldData).add(function (dataIndex) {
      if (!data.hasValue(dataIndex)) {
        return;
      }
      var itemModel = getItemModel(data, dataIndex);
      var symbolMeta = getSymbolMeta(data, dataIndex, itemModel, opt);
      var bar = createBar(data, opt, symbolMeta);
      data.setItemGraphicEl(dataIndex, bar);
      group.add(bar);
      updateCommon(bar, opt, symbolMeta);
    }).update(function (newIndex, oldIndex) {
      var bar = oldData.getItemGraphicEl(oldIndex);
      if (!data.hasValue(newIndex)) {
        group.remove(bar);
        return;
      }
      var itemModel = getItemModel(data, newIndex);
      var symbolMeta = getSymbolMeta(data, newIndex, itemModel, opt);
      var pictorialShapeStr = getShapeStr(data, symbolMeta);
      if (bar && pictorialShapeStr !== bar.__pictorialShapeStr) {
        group.remove(bar);
        data.setItemGraphicEl(newIndex, null);
        bar = null;
      }
      if (bar) {
        updateBar(bar, opt, symbolMeta);
      } else {
        bar = createBar(data, opt, symbolMeta, true);
      }
      data.setItemGraphicEl(newIndex, bar);
      bar.__pictorialSymbolMeta = symbolMeta;
      // Add back
      group.add(bar);
      updateCommon(bar, opt, symbolMeta);
    }).remove(function (dataIndex) {
      var bar = oldData.getItemGraphicEl(dataIndex);
      bar && removeBar(oldData, dataIndex, bar.__pictorialSymbolMeta.animationModel, bar);
    }).execute();
    // Do clipping
    var clipPath = seriesModel.get('clip', true) ? createClipPath(seriesModel.coordinateSystem, false, seriesModel) : null;
    if (clipPath) {
      group.setClipPath(clipPath);
    } else {
      group.removeClipPath();
    }
    this._data = data;
    return this.group;
  };
  PictorialBarView.prototype.remove = function (ecModel, api) {
    var group = this.group;
    var data = this._data;
    if (ecModel.get('animation')) {
      if (data) {
        data.eachItemGraphicEl(function (bar) {
          removeBar(data, getECData(bar).dataIndex, ecModel, bar);
        });
      }
    } else {
      group.removeAll();
    }
  };
  PictorialBarView.type = 'pictorialBar';
  return PictorialBarView;
}(ChartView);
// Set or calculate default value about symbol, and calculate layout info.
function getSymbolMeta(data, dataIndex, itemModel, opt) {
  var layout = data.getItemLayout(dataIndex);
  var symbolRepeat = itemModel.get('symbolRepeat');
  var symbolClip = itemModel.get('symbolClip');
  var symbolPosition = itemModel.get('symbolPosition') || 'start';
  var symbolRotate = itemModel.get('symbolRotate');
  var rotation = (symbolRotate || 0) * Math.PI / 180 || 0;
  var symbolPatternSize = itemModel.get('symbolPatternSize') || 2;
  var isAnimationEnabled = itemModel.isAnimationEnabled();
  var symbolMeta = {
    dataIndex: dataIndex,
    layout: layout,
    itemModel: itemModel,
    symbolType: data.getItemVisual(dataIndex, 'symbol') || 'circle',
    style: data.getItemVisual(dataIndex, 'style'),
    symbolClip: symbolClip,
    symbolRepeat: symbolRepeat,
    symbolRepeatDirection: itemModel.get('symbolRepeatDirection'),
    symbolPatternSize: symbolPatternSize,
    rotation: rotation,
    animationModel: isAnimationEnabled ? itemModel : null,
    hoverScale: isAnimationEnabled && itemModel.get(['emphasis', 'scale']),
    z2: itemModel.getShallow('z', true) || 0
  };
  prepareBarLength(itemModel, symbolRepeat, layout, opt, symbolMeta);
  prepareSymbolSize(data, dataIndex, layout, symbolRepeat, symbolClip, symbolMeta.boundingLength, symbolMeta.pxSign, symbolPatternSize, opt, symbolMeta);
  prepareLineWidth(itemModel, symbolMeta.symbolScale, rotation, opt, symbolMeta);
  var symbolSize = symbolMeta.symbolSize;
  var symbolOffset = normalizeSymbolOffset(itemModel.get('symbolOffset'), symbolSize);
  prepareLayoutInfo(itemModel, symbolSize, layout, symbolRepeat, symbolClip, symbolOffset, symbolPosition, symbolMeta.valueLineWidth, symbolMeta.boundingLength, symbolMeta.repeatCutLength, opt, symbolMeta);
  return symbolMeta;
}
// bar length can be negative.
function prepareBarLength(itemModel, symbolRepeat, layout, opt, outputSymbolMeta) {
  var valueDim = opt.valueDim;
  var symbolBoundingData = itemModel.get('symbolBoundingData');
  var valueAxis = opt.coordSys.getOtherAxis(opt.coordSys.getBaseAxis());
  var zeroPx = valueAxis.toGlobalCoord(valueAxis.dataToCoord(0));
  var pxSignIdx = 1 - +(layout[valueDim.wh] <= 0);
  var boundingLength;
  if (zrUtil.isArray(symbolBoundingData)) {
    var symbolBoundingExtent = [convertToCoordOnAxis(valueAxis, symbolBoundingData[0]) - zeroPx, convertToCoordOnAxis(valueAxis, symbolBoundingData[1]) - zeroPx];
    symbolBoundingExtent[1] < symbolBoundingExtent[0] && symbolBoundingExtent.reverse();
    boundingLength = symbolBoundingExtent[pxSignIdx];
  } else if (symbolBoundingData != null) {
    boundingLength = convertToCoordOnAxis(valueAxis, symbolBoundingData) - zeroPx;
  } else if (symbolRepeat) {
    boundingLength = opt.coordSysExtent[valueDim.index][pxSignIdx] - zeroPx;
  } else {
    boundingLength = layout[valueDim.wh];
  }
  outputSymbolMeta.boundingLength = boundingLength;
  if (symbolRepeat) {
    outputSymbolMeta.repeatCutLength = layout[valueDim.wh];
  }
  // if 'pxSign' means sign of pixel,  it can't be zero, or symbolScale will be zero
  // and when borderWidth be settled, the actual linewidth will be NaN
  var isXAxis = valueDim.xy === 'x';
  var isInverse = valueAxis.inverse;
  outputSymbolMeta.pxSign = isXAxis && !isInverse || !isXAxis && isInverse ? boundingLength >= 0 ? 1 : -1 : boundingLength > 0 ? 1 : -1;
}
function convertToCoordOnAxis(axis, value) {
  return axis.toGlobalCoord(axis.dataToCoord(axis.scale.parse(value)));
}
// Support ['100%', '100%']
function prepareSymbolSize(data, dataIndex, layout, symbolRepeat, symbolClip, boundingLength, pxSign, symbolPatternSize, opt, outputSymbolMeta) {
  var valueDim = opt.valueDim;
  var categoryDim = opt.categoryDim;
  var categorySize = Math.abs(layout[categoryDim.wh]);
  var symbolSize = data.getItemVisual(dataIndex, 'symbolSize');
  var parsedSymbolSize;
  if (zrUtil.isArray(symbolSize)) {
    parsedSymbolSize = symbolSize.slice();
  } else {
    if (symbolSize == null) {
      // will parse to number below
      parsedSymbolSize = ['100%', '100%'];
    } else {
      parsedSymbolSize = [symbolSize, symbolSize];
    }
  }
  // Note: percentage symbolSize (like '100%') do not consider lineWidth, because it is
  // to complicated to calculate real percent value if considering scaled lineWidth.
  // So the actual size will bigger than layout size if lineWidth is bigger than zero,
  // which can be tolerated in pictorial chart.
  parsedSymbolSize[categoryDim.index] = parsePercent(parsedSymbolSize[categoryDim.index], categorySize);
  parsedSymbolSize[valueDim.index] = parsePercent(parsedSymbolSize[valueDim.index], symbolRepeat ? categorySize : Math.abs(boundingLength));
  outputSymbolMeta.symbolSize = parsedSymbolSize;
  // If x or y is less than zero, show reversed shape.
  var symbolScale = outputSymbolMeta.symbolScale = [parsedSymbolSize[0] / symbolPatternSize, parsedSymbolSize[1] / symbolPatternSize];
  // Follow convention, 'right' and 'top' is the normal scale.
  symbolScale[valueDim.index] *= (opt.isHorizontal ? -1 : 1) * pxSign;
}
function prepareLineWidth(itemModel, symbolScale, rotation, opt, outputSymbolMeta) {
  // In symbols are drawn with scale, so do not need to care about the case that width
  // or height are too small. But symbol use strokeNoScale, where acture lineWidth should
  // be calculated.
  var valueLineWidth = itemModel.get(BAR_BORDER_WIDTH_QUERY) || 0;
  if (valueLineWidth) {
    pathForLineWidth.attr({
      scaleX: symbolScale[0],
      scaleY: symbolScale[1],
      rotation: rotation
    });
    pathForLineWidth.updateTransform();
    valueLineWidth /= pathForLineWidth.getLineScale();
    valueLineWidth *= symbolScale[opt.valueDim.index];
  }
  outputSymbolMeta.valueLineWidth = valueLineWidth || 0;
}
function prepareLayoutInfo(itemModel, symbolSize, layout, symbolRepeat, symbolClip, symbolOffset, symbolPosition, valueLineWidth, boundingLength, repeatCutLength, opt, outputSymbolMeta) {
  var categoryDim = opt.categoryDim;
  var valueDim = opt.valueDim;
  var pxSign = outputSymbolMeta.pxSign;
  var unitLength = Math.max(symbolSize[valueDim.index] + valueLineWidth, 0);
  var pathLen = unitLength;
  // Note: rotation will not effect the layout of symbols, because user may
  // want symbols to rotate on its center, which should not be translated
  // when rotating.
  if (symbolRepeat) {
    var absBoundingLength = Math.abs(boundingLength);
    var symbolMargin = zrUtil.retrieve(itemModel.get('symbolMargin'), '15%') + '';
    var hasEndGap = false;
    if (symbolMargin.lastIndexOf('!') === symbolMargin.length - 1) {
      hasEndGap = true;
      symbolMargin = symbolMargin.slice(0, symbolMargin.length - 1);
    }
    var symbolMarginNumeric = parsePercent(symbolMargin, symbolSize[valueDim.index]);
    var uLenWithMargin = Math.max(unitLength + symbolMarginNumeric * 2, 0);
    // When symbol margin is less than 0, margin at both ends will be subtracted
    // to ensure that all of the symbols will not be overflow the given area.
    var endFix = hasEndGap ? 0 : symbolMarginNumeric * 2;
    // Both final repeatTimes and final symbolMarginNumeric area calculated based on
    // boundingLength.
    var repeatSpecified = isNumeric(symbolRepeat);
    var repeatTimes = repeatSpecified ? symbolRepeat : toIntTimes((absBoundingLength + endFix) / uLenWithMargin);
    // Adjust calculate margin, to ensure each symbol is displayed
    // entirely in the given layout area.
    var mDiff = absBoundingLength - repeatTimes * unitLength;
    symbolMarginNumeric = mDiff / 2 / (hasEndGap ? repeatTimes : Math.max(repeatTimes - 1, 1));
    uLenWithMargin = unitLength + symbolMarginNumeric * 2;
    endFix = hasEndGap ? 0 : symbolMarginNumeric * 2;
    // Update repeatTimes when not all symbol will be shown.
    if (!repeatSpecified && symbolRepeat !== 'fixed') {
      repeatTimes = repeatCutLength ? toIntTimes((Math.abs(repeatCutLength) + endFix) / uLenWithMargin) : 0;
    }
    pathLen = repeatTimes * uLenWithMargin - endFix;
    outputSymbolMeta.repeatTimes = repeatTimes;
    outputSymbolMeta.symbolMargin = symbolMarginNumeric;
  }
  var sizeFix = pxSign * (pathLen / 2);
  var pathPosition = outputSymbolMeta.pathPosition = [];
  pathPosition[categoryDim.index] = layout[categoryDim.wh] / 2;
  pathPosition[valueDim.index] = symbolPosition === 'start' ? sizeFix : symbolPosition === 'end' ? boundingLength - sizeFix : boundingLength / 2; // 'center'
  if (symbolOffset) {
    pathPosition[0] += symbolOffset[0];
    pathPosition[1] += symbolOffset[1];
  }
  var bundlePosition = outputSymbolMeta.bundlePosition = [];
  bundlePosition[categoryDim.index] = layout[categoryDim.xy];
  bundlePosition[valueDim.index] = layout[valueDim.xy];
  var barRectShape = outputSymbolMeta.barRectShape = zrUtil.extend({}, layout);
  barRectShape[valueDim.wh] = pxSign * Math.max(Math.abs(layout[valueDim.wh]), Math.abs(pathPosition[valueDim.index] + sizeFix));
  barRectShape[categoryDim.wh] = layout[categoryDim.wh];
  var clipShape = outputSymbolMeta.clipShape = {};
  // Consider that symbol may be overflow layout rect.
  clipShape[categoryDim.xy] = -layout[categoryDim.xy];
  clipShape[categoryDim.wh] = opt.ecSize[categoryDim.wh];
  clipShape[valueDim.xy] = 0;
  clipShape[valueDim.wh] = layout[valueDim.wh];
}
function createPath(symbolMeta) {
  var symbolPatternSize = symbolMeta.symbolPatternSize;
  var path = createSymbol(
  // Consider texture img, make a big size.
  symbolMeta.symbolType, -symbolPatternSize / 2, -symbolPatternSize / 2, symbolPatternSize, symbolPatternSize);
  path.attr({
    culling: true
  });
  path.type !== 'image' && path.setStyle({
    strokeNoScale: true
  });
  return path;
}
function createOrUpdateRepeatSymbols(bar, opt, symbolMeta, isUpdate) {
  var bundle = bar.__pictorialBundle;
  var symbolSize = symbolMeta.symbolSize;
  var valueLineWidth = symbolMeta.valueLineWidth;
  var pathPosition = symbolMeta.pathPosition;
  var valueDim = opt.valueDim;
  var repeatTimes = symbolMeta.repeatTimes || 0;
  var index = 0;
  var unit = symbolSize[opt.valueDim.index] + valueLineWidth + symbolMeta.symbolMargin * 2;
  eachPath(bar, function (path) {
    path.__pictorialAnimationIndex = index;
    path.__pictorialRepeatTimes = repeatTimes;
    if (index < repeatTimes) {
      updateAttr(path, null, makeTarget(index), symbolMeta, isUpdate);
    } else {
      updateAttr(path, null, {
        scaleX: 0,
        scaleY: 0
      }, symbolMeta, isUpdate, function () {
        bundle.remove(path);
      });
    }
    // updateHoverAnimation(path, symbolMeta);
    index++;
  });
  for (; index < repeatTimes; index++) {
    var path = createPath(symbolMeta);
    path.__pictorialAnimationIndex = index;
    path.__pictorialRepeatTimes = repeatTimes;
    bundle.add(path);
    var target = makeTarget(index);
    updateAttr(path, {
      x: target.x,
      y: target.y,
      scaleX: 0,
      scaleY: 0
    }, {
      scaleX: target.scaleX,
      scaleY: target.scaleY,
      rotation: target.rotation
    }, symbolMeta, isUpdate);
  }
  function makeTarget(index) {
    var position = pathPosition.slice();
    // (start && pxSign > 0) || (end && pxSign < 0): i = repeatTimes - index
    // Otherwise: i = index;
    var pxSign = symbolMeta.pxSign;
    var i = index;
    if (symbolMeta.symbolRepeatDirection === 'start' ? pxSign > 0 : pxSign < 0) {
      i = repeatTimes - 1 - index;
    }
    position[valueDim.index] = unit * (i - repeatTimes / 2 + 0.5) + pathPosition[valueDim.index];
    return {
      x: position[0],
      y: position[1],
      scaleX: symbolMeta.symbolScale[0],
      scaleY: symbolMeta.symbolScale[1],
      rotation: symbolMeta.rotation
    };
  }
}
function createOrUpdateSingleSymbol(bar, opt, symbolMeta, isUpdate) {
  var bundle = bar.__pictorialBundle;
  var mainPath = bar.__pictorialMainPath;
  if (!mainPath) {
    mainPath = bar.__pictorialMainPath = createPath(symbolMeta);
    bundle.add(mainPath);
    updateAttr(mainPath, {
      x: symbolMeta.pathPosition[0],
      y: symbolMeta.pathPosition[1],
      scaleX: 0,
      scaleY: 0,
      rotation: symbolMeta.rotation
    }, {
      scaleX: symbolMeta.symbolScale[0],
      scaleY: symbolMeta.symbolScale[1]
    }, symbolMeta, isUpdate);
  } else {
    updateAttr(mainPath, null, {
      x: symbolMeta.pathPosition[0],
      y: symbolMeta.pathPosition[1],
      scaleX: symbolMeta.symbolScale[0],
      scaleY: symbolMeta.symbolScale[1],
      rotation: symbolMeta.rotation
    }, symbolMeta, isUpdate);
  }
}
// bar rect is used for label.
function createOrUpdateBarRect(bar, symbolMeta, isUpdate) {
  var rectShape = zrUtil.extend({}, symbolMeta.barRectShape);
  var barRect = bar.__pictorialBarRect;
  if (!barRect) {
    barRect = bar.__pictorialBarRect = new graphic.Rect({
      z2: 2,
      shape: rectShape,
      silent: true,
      style: {
        stroke: 'transparent',
        fill: 'transparent',
        lineWidth: 0
      }
    });
    barRect.disableMorphing = true;
    bar.add(barRect);
  } else {
    updateAttr(barRect, null, {
      shape: rectShape
    }, symbolMeta, isUpdate);
  }
}
function createOrUpdateClip(bar, opt, symbolMeta, isUpdate) {
  // If not clip, symbol will be remove and rebuilt.
  if (symbolMeta.symbolClip) {
    var clipPath = bar.__pictorialClipPath;
    var clipShape = zrUtil.extend({}, symbolMeta.clipShape);
    var valueDim = opt.valueDim;
    var animationModel = symbolMeta.animationModel;
    var dataIndex = symbolMeta.dataIndex;
    if (clipPath) {
      graphic.updateProps(clipPath, {
        shape: clipShape
      }, animationModel, dataIndex);
    } else {
      clipShape[valueDim.wh] = 0;
      clipPath = new graphic.Rect({
        shape: clipShape
      });
      bar.__pictorialBundle.setClipPath(clipPath);
      bar.__pictorialClipPath = clipPath;
      var target = {};
      target[valueDim.wh] = symbolMeta.clipShape[valueDim.wh];
      graphic[isUpdate ? 'updateProps' : 'initProps'](clipPath, {
        shape: target
      }, animationModel, dataIndex);
    }
  }
}
function getItemModel(data, dataIndex) {
  var itemModel = data.getItemModel(dataIndex);
  itemModel.getAnimationDelayParams = getAnimationDelayParams;
  itemModel.isAnimationEnabled = isAnimationEnabled;
  return itemModel;
}
function getAnimationDelayParams(path) {
  // The order is the same as the z-order, see `symbolRepeatDiretion`.
  return {
    index: path.__pictorialAnimationIndex,
    count: path.__pictorialRepeatTimes
  };
}
function isAnimationEnabled() {
  // `animation` prop can be set on itemModel in pictorial bar chart.
  return this.parentModel.isAnimationEnabled() && !!this.getShallow('animation');
}
function createBar(data, opt, symbolMeta, isUpdate) {
  // bar is the main element for each data.
  var bar = new graphic.Group();
  // bundle is used for location and clip.
  var bundle = new graphic.Group();
  bar.add(bundle);
  bar.__pictorialBundle = bundle;
  bundle.x = symbolMeta.bundlePosition[0];
  bundle.y = symbolMeta.bundlePosition[1];
  if (symbolMeta.symbolRepeat) {
    createOrUpdateRepeatSymbols(bar, opt, symbolMeta);
  } else {
    createOrUpdateSingleSymbol(bar, opt, symbolMeta);
  }
  createOrUpdateBarRect(bar, symbolMeta, isUpdate);
  createOrUpdateClip(bar, opt, symbolMeta, isUpdate);
  bar.__pictorialShapeStr = getShapeStr(data, symbolMeta);
  bar.__pictorialSymbolMeta = symbolMeta;
  return bar;
}
function updateBar(bar, opt, symbolMeta) {
  var animationModel = symbolMeta.animationModel;
  var dataIndex = symbolMeta.dataIndex;
  var bundle = bar.__pictorialBundle;
  graphic.updateProps(bundle, {
    x: symbolMeta.bundlePosition[0],
    y: symbolMeta.bundlePosition[1]
  }, animationModel, dataIndex);
  if (symbolMeta.symbolRepeat) {
    createOrUpdateRepeatSymbols(bar, opt, symbolMeta, true);
  } else {
    createOrUpdateSingleSymbol(bar, opt, symbolMeta, true);
  }
  createOrUpdateBarRect(bar, symbolMeta, true);
  createOrUpdateClip(bar, opt, symbolMeta, true);
}
function removeBar(data, dataIndex, animationModel, bar) {
  // Not show text when animating
  var labelRect = bar.__pictorialBarRect;
  labelRect && labelRect.removeTextContent();
  var paths = [];
  eachPath(bar, function (path) {
    paths.push(path);
  });
  bar.__pictorialMainPath && paths.push(bar.__pictorialMainPath);
  // I do not find proper remove animation for clip yet.
  bar.__pictorialClipPath && (animationModel = null);
  zrUtil.each(paths, function (path) {
    graphic.removeElement(path, {
      scaleX: 0,
      scaleY: 0
    }, animationModel, dataIndex, function () {
      bar.parent && bar.parent.remove(bar);
    });
  });
  data.setItemGraphicEl(dataIndex, null);
}
function getShapeStr(data, symbolMeta) {
  return [data.getItemVisual(symbolMeta.dataIndex, 'symbol') || 'none', !!symbolMeta.symbolRepeat, !!symbolMeta.symbolClip].join(':');
}
function eachPath(bar, cb, context) {
  // Do not use Group#eachChild, because it do not support remove.
  zrUtil.each(bar.__pictorialBundle.children(), function (el) {
    el !== bar.__pictorialBarRect && cb.call(context, el);
  });
}
function updateAttr(el, immediateAttrs, animationAttrs, symbolMeta, isUpdate, cb) {
  immediateAttrs && el.attr(immediateAttrs);
  // when symbolCip used, only clip path has init animation, otherwise it would be weird effect.
  if (symbolMeta.symbolClip && !isUpdate) {
    animationAttrs && el.attr(animationAttrs);
  } else {
    animationAttrs && graphic[isUpdate ? 'updateProps' : 'initProps'](el, animationAttrs, symbolMeta.animationModel, symbolMeta.dataIndex, cb);
  }
}
function updateCommon(bar, opt, symbolMeta) {
  var dataIndex = symbolMeta.dataIndex;
  var itemModel = symbolMeta.itemModel;
  // Color must be excluded.
  // Because symbol provide setColor individually to set fill and stroke
  var emphasisModel = itemModel.getModel('emphasis');
  var emphasisStyle = emphasisModel.getModel('itemStyle').getItemStyle();
  var blurStyle = itemModel.getModel(['blur', 'itemStyle']).getItemStyle();
  var selectStyle = itemModel.getModel(['select', 'itemStyle']).getItemStyle();
  var cursorStyle = itemModel.getShallow('cursor');
  var focus = emphasisModel.get('focus');
  var blurScope = emphasisModel.get('blurScope');
  var hoverScale = emphasisModel.get('scale');
  eachPath(bar, function (path) {
    if (path instanceof ZRImage) {
      var pathStyle = path.style;
      path.useStyle(zrUtil.extend({
        // TODO other properties like dx, dy ?
        image: pathStyle.image,
        x: pathStyle.x,
        y: pathStyle.y,
        width: pathStyle.width,
        height: pathStyle.height
      }, symbolMeta.style));
    } else {
      path.useStyle(symbolMeta.style);
    }
    var emphasisState = path.ensureState('emphasis');
    emphasisState.style = emphasisStyle;
    if (hoverScale) {
      // NOTE: Must after scale is set after updateAttr
      emphasisState.scaleX = path.scaleX * 1.1;
      emphasisState.scaleY = path.scaleY * 1.1;
    }
    path.ensureState('blur').style = blurStyle;
    path.ensureState('select').style = selectStyle;
    cursorStyle && (path.cursor = cursorStyle);
    path.z2 = symbolMeta.z2;
  });
  var barPositionOutside = opt.valueDim.posDesc[+(symbolMeta.boundingLength > 0)];
  var barRect = bar.__pictorialBarRect;
  barRect.ignoreClip = true;
  setLabelStyle(barRect, getLabelStatesModels(itemModel), {
    labelFetcher: opt.seriesModel,
    labelDataIndex: dataIndex,
    defaultText: getDefaultLabel(opt.seriesModel.getData(), dataIndex),
    inheritColor: symbolMeta.style.fill,
    defaultOpacity: symbolMeta.style.opacity,
    defaultOutsidePosition: barPositionOutside
  });
  toggleHoverEmphasis(bar, focus, blurScope, emphasisModel.get('disabled'));
}
function toIntTimes(times) {
  var roundedTimes = Math.round(times);
  // Escapse accurate error
  return Math.abs(times - roundedTimes) < 1e-4 ? roundedTimes : Math.ceil(times);
}
export default PictorialBarView;