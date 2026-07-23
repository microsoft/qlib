
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
import Path from 'zrender/lib/graphic/Path.js';
import Group from 'zrender/lib/graphic/Group.js';
import { extend, each, map } from 'zrender/lib/core/util.js';
import { Rect, Sector, updateProps, initProps, removeElementWithFadeOut, traverseElements } from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import { setStatesStylesFromModel, toggleHoverEmphasis } from '../../util/states.js';
import { setLabelStyle, getLabelStatesModels, setLabelValueAnimation, labelInner } from '../../label/labelStyle.js';
import { throttle } from '../../util/throttle.js';
import { createClipPath } from '../helper/createClipPathFromCoordSys.js';
import Sausage from '../../util/shape/sausage.js';
import ChartView from '../../view/Chart.js';
import { isCoordinateSystemType } from '../../coord/CoordinateSystem.js';
import { getDefaultLabel, getDefaultInterpolatedLabel } from '../helper/labelHelper.js';
import { warn } from '../../util/log.js';
import { createSectorCalculateTextPosition, setSectorTextRotation } from '../../label/sectorLabel.js';
import { saveOldStyle } from '../../animation/basicTransition.js';
import { getSectorCornerRadius } from '../helper/sectorHelper.js';
var mathMax = Math.max;
var mathMin = Math.min;
function getClipArea(coord, data) {
  var coordSysClipArea = coord.getArea && coord.getArea();
  if (isCoordinateSystemType(coord, 'cartesian2d')) {
    var baseAxis = coord.getBaseAxis();
    // When boundaryGap is false or using time axis. bar may exceed the grid.
    // We should not clip this part.
    // See test/bar2.html
    if (baseAxis.type !== 'category' || !baseAxis.onBand) {
      var expandWidth = data.getLayout('bandWidth');
      if (baseAxis.isHorizontal()) {
        coordSysClipArea.x -= expandWidth;
        coordSysClipArea.width += expandWidth * 2;
      } else {
        coordSysClipArea.y -= expandWidth;
        coordSysClipArea.height += expandWidth * 2;
      }
    }
  }
  return coordSysClipArea;
}
var BarView = /** @class */function (_super) {
  __extends(BarView, _super);
  function BarView() {
    var _this = _super.call(this) || this;
    _this.type = BarView.type;
    _this._isFirstFrame = true;
    return _this;
  }
  BarView.prototype.render = function (seriesModel, ecModel, api, payload) {
    this._model = seriesModel;
    this._removeOnRenderedListener(api);
    this._updateDrawMode(seriesModel);
    var coordinateSystemType = seriesModel.get('coordinateSystem');
    if (coordinateSystemType === 'cartesian2d' || coordinateSystemType === 'polar') {
      // Clear previously rendered progressive elements.
      this._progressiveEls = null;
      this._isLargeDraw ? this._renderLarge(seriesModel, ecModel, api) : this._renderNormal(seriesModel, ecModel, api, payload);
    } else if (process.env.NODE_ENV !== 'production') {
      warn('Only cartesian2d and polar supported for bar.');
    }
  };
  BarView.prototype.incrementalPrepareRender = function (seriesModel) {
    this._clear();
    this._updateDrawMode(seriesModel);
    // incremental also need to clip, otherwise might be overlow.
    // But must not set clip in each frame, otherwise all of the children will be marked redraw.
    this._updateLargeClip(seriesModel);
  };
  BarView.prototype.incrementalRender = function (params, seriesModel) {
    // Reset
    this._progressiveEls = [];
    // Do not support progressive in normal mode.
    this._incrementalRenderLarge(params, seriesModel);
  };
  BarView.prototype.eachRendered = function (cb) {
    traverseElements(this._progressiveEls || this.group, cb);
  };
  BarView.prototype._updateDrawMode = function (seriesModel) {
    var isLargeDraw = seriesModel.pipelineContext.large;
    if (this._isLargeDraw == null || isLargeDraw !== this._isLargeDraw) {
      this._isLargeDraw = isLargeDraw;
      this._clear();
    }
  };
  BarView.prototype._renderNormal = function (seriesModel, ecModel, api, payload) {
    var group = this.group;
    var data = seriesModel.getData();
    var oldData = this._data;
    var coord = seriesModel.coordinateSystem;
    var baseAxis = coord.getBaseAxis();
    var isHorizontalOrRadial;
    if (coord.type === 'cartesian2d') {
      isHorizontalOrRadial = baseAxis.isHorizontal();
    } else if (coord.type === 'polar') {
      isHorizontalOrRadial = baseAxis.dim === 'angle';
    }
    var animationModel = seriesModel.isAnimationEnabled() ? seriesModel : null;
    var realtimeSortCfg = shouldRealtimeSort(seriesModel, coord);
    if (realtimeSortCfg) {
      this._enableRealtimeSort(realtimeSortCfg, data, api);
    }
    var needsClip = seriesModel.get('clip', true) || realtimeSortCfg;
    var coordSysClipArea = getClipArea(coord, data);
    // If there is clipPath created in large mode. Remove it.
    group.removeClipPath();
    // We don't use clipPath in normal mode because we needs a perfect animation
    // And don't want the label are clipped.
    var roundCap = seriesModel.get('roundCap', true);
    var drawBackground = seriesModel.get('showBackground', true);
    var backgroundModel = seriesModel.getModel('backgroundStyle');
    var barBorderRadius = backgroundModel.get('borderRadius') || 0;
    var bgEls = [];
    var oldBgEls = this._backgroundEls;
    var isInitSort = payload && payload.isInitSort;
    var isChangeOrder = payload && payload.type === 'changeAxisOrder';
    function createBackground(dataIndex) {
      var bgLayout = getLayout[coord.type](data, dataIndex);
      var bgEl = createBackgroundEl(coord, isHorizontalOrRadial, bgLayout);
      bgEl.useStyle(backgroundModel.getItemStyle());
      // Only cartesian2d support borderRadius.
      if (coord.type === 'cartesian2d') {
        bgEl.setShape('r', barBorderRadius);
      } else {
        bgEl.setShape('cornerRadius', barBorderRadius);
      }
      bgEls[dataIndex] = bgEl;
      return bgEl;
    }
    ;
    data.diff(oldData).add(function (dataIndex) {
      var itemModel = data.getItemModel(dataIndex);
      var layout = getLayout[coord.type](data, dataIndex, itemModel);
      if (drawBackground) {
        createBackground(dataIndex);
      }
      // If dataZoom in filteMode: 'empty', the baseValue can be set as NaN in "axisProxy".
      if (!data.hasValue(dataIndex) || !isValidLayout[coord.type](layout)) {
        return;
      }
      var isClipped = false;
      if (needsClip) {
        // Clip will modify the layout params.
        // And return a boolean to determine if the shape are fully clipped.
        isClipped = clip[coord.type](coordSysClipArea, layout);
      }
      var el = elementCreator[coord.type](seriesModel, data, dataIndex, layout, isHorizontalOrRadial, animationModel, baseAxis.model, false, roundCap);
      if (realtimeSortCfg) {
        /**
         * Force label animation because even if the element is
         * ignored because it's clipped, it may not be clipped after
         * changing order. Then, if not using forceLabelAnimation,
         * the label animation was never started, in which case,
         * the label will be the final value and doesn't have label
         * animation.
         */
        el.forceLabelAnimation = true;
      }
      updateStyle(el, data, dataIndex, itemModel, layout, seriesModel, isHorizontalOrRadial, coord.type === 'polar');
      if (isInitSort) {
        el.attr({
          shape: layout
        });
      } else if (realtimeSortCfg) {
        updateRealtimeAnimation(realtimeSortCfg, animationModel, el, layout, dataIndex, isHorizontalOrRadial, false, false);
      } else {
        initProps(el, {
          shape: layout
        }, seriesModel, dataIndex);
      }
      data.setItemGraphicEl(dataIndex, el);
      group.add(el);
      el.ignore = isClipped;
    }).update(function (newIndex, oldIndex) {
      var itemModel = data.getItemModel(newIndex);
      var layout = getLayout[coord.type](data, newIndex, itemModel);
      if (drawBackground) {
        var bgEl = void 0;
        if (oldBgEls.length === 0) {
          bgEl = createBackground(oldIndex);
        } else {
          bgEl = oldBgEls[oldIndex];
          bgEl.useStyle(backgroundModel.getItemStyle());
          // Only cartesian2d support borderRadius.
          if (coord.type === 'cartesian2d') {
            bgEl.setShape('r', barBorderRadius);
          } else {
            bgEl.setShape('cornerRadius', barBorderRadius);
          }
          bgEls[newIndex] = bgEl;
        }
        var bgLayout = getLayout[coord.type](data, newIndex);
        var shape = createBackgroundShape(isHorizontalOrRadial, bgLayout, coord);
        updateProps(bgEl, {
          shape: shape
        }, animationModel, newIndex);
      }
      var el = oldData.getItemGraphicEl(oldIndex);
      if (!data.hasValue(newIndex) || !isValidLayout[coord.type](layout)) {
        group.remove(el);
        return;
      }
      var isClipped = false;
      if (needsClip) {
        isClipped = clip[coord.type](coordSysClipArea, layout);
        if (isClipped) {
          group.remove(el);
        }
      }
      if (!el) {
        el = elementCreator[coord.type](seriesModel, data, newIndex, layout, isHorizontalOrRadial, animationModel, baseAxis.model, !!el, roundCap);
      } else {
        saveOldStyle(el);
      }
      if (realtimeSortCfg) {
        el.forceLabelAnimation = true;
      }
      if (isChangeOrder) {
        var textEl = el.getTextContent();
        if (textEl) {
          var labelInnerStore = labelInner(textEl);
          if (labelInnerStore.prevValue != null) {
            /**
             * Set preValue to be value so that no new label
             * should be started, otherwise, it will take a full
             * `animationDurationUpdate` time to finish the
             * animation, which is not expected.
             */
            labelInnerStore.prevValue = labelInnerStore.value;
          }
        }
      }
      // Not change anything if only order changed.
      // Especially not change label.
      else {
        updateStyle(el, data, newIndex, itemModel, layout, seriesModel, isHorizontalOrRadial, coord.type === 'polar');
      }
      if (isInitSort) {
        el.attr({
          shape: layout
        });
      } else if (realtimeSortCfg) {
        updateRealtimeAnimation(realtimeSortCfg, animationModel, el, layout, newIndex, isHorizontalOrRadial, true, isChangeOrder);
      } else {
        updateProps(el, {
          shape: layout
        }, seriesModel, newIndex, null);
      }
      data.setItemGraphicEl(newIndex, el);
      el.ignore = isClipped;
      group.add(el);
    }).remove(function (dataIndex) {
      var el = oldData.getItemGraphicEl(dataIndex);
      el && removeElementWithFadeOut(el, seriesModel, dataIndex);
    }).execute();
    var bgGroup = this._backgroundGroup || (this._backgroundGroup = new Group());
    bgGroup.removeAll();
    for (var i = 0; i < bgEls.length; ++i) {
      bgGroup.add(bgEls[i]);
    }
    group.add(bgGroup);
    this._backgroundEls = bgEls;
    this._data = data;
  };
  BarView.prototype._renderLarge = function (seriesModel, ecModel, api) {
    this._clear();
    createLarge(seriesModel, this.group);
    this._updateLargeClip(seriesModel);
  };
  BarView.prototype._incrementalRenderLarge = function (params, seriesModel) {
    this._removeBackground();
    createLarge(seriesModel, this.group, this._progressiveEls, true);
  };
  BarView.prototype._updateLargeClip = function (seriesModel) {
    // Use clipPath in large mode.
    var clipPath = seriesModel.get('clip', true) && createClipPath(seriesModel.coordinateSystem, false, seriesModel);
    var group = this.group;
    if (clipPath) {
      group.setClipPath(clipPath);
    } else {
      group.removeClipPath();
    }
  };
  BarView.prototype._enableRealtimeSort = function (realtimeSortCfg, data, api) {
    var _this = this;
    // If no data in the first frame, wait for data to initSort
    if (!data.count()) {
      return;
    }
    var baseAxis = realtimeSortCfg.baseAxis;
    if (this._isFirstFrame) {
      this._dispatchInitSort(data, realtimeSortCfg, api);
      this._isFirstFrame = false;
    } else {
      var orderMapping_1 = function (idx) {
        var el = data.getItemGraphicEl(idx);
        var shape = el && el.shape;
        return shape &&
        // The result should be consistent with the initial sort by data value.
        // Do not support the case that both positive and negative exist.
        Math.abs(baseAxis.isHorizontal() ? shape.height : shape.width)
        // If data is NaN, shape.xxx may be NaN, so use || 0 here in case
        || 0;
      };
      this._onRendered = function () {
        _this._updateSortWithinSameData(data, orderMapping_1, baseAxis, api);
      };
      api.getZr().on('rendered', this._onRendered);
    }
  };
  BarView.prototype._dataSort = function (data, baseAxis, orderMapping) {
    var info = [];
    data.each(data.mapDimension(baseAxis.dim), function (ordinalNumber, dataIdx) {
      var mappedValue = orderMapping(dataIdx);
      mappedValue = mappedValue == null ? NaN : mappedValue;
      info.push({
        dataIndex: dataIdx,
        mappedValue: mappedValue,
        ordinalNumber: ordinalNumber
      });
    });
    info.sort(function (a, b) {
      // If NaN, it will be treated as min val.
      return b.mappedValue - a.mappedValue;
    });
    return {
      ordinalNumbers: map(info, function (item) {
        return item.ordinalNumber;
      })
    };
  };
  BarView.prototype._isOrderChangedWithinSameData = function (data, orderMapping, baseAxis) {
    var scale = baseAxis.scale;
    var ordinalDataDim = data.mapDimension(baseAxis.dim);
    var lastValue = Number.MAX_VALUE;
    for (var tickNum = 0, len = scale.getOrdinalMeta().categories.length; tickNum < len; ++tickNum) {
      var rawIdx = data.rawIndexOf(ordinalDataDim, scale.getRawOrdinalNumber(tickNum));
      var value = rawIdx < 0
      // If some tick have no bar, the tick will be treated as min.
      ? Number.MIN_VALUE
      // PENDING: if dataZoom on baseAxis exits, is it a performance issue?
      : orderMapping(data.indexOfRawIndex(rawIdx));
      if (value > lastValue) {
        return true;
      }
      lastValue = value;
    }
    return false;
  };
  /*
   * Consider the case when A and B changed order, whose representing
   * bars are both out of sight, we don't wish to trigger reorder action
   * as long as the order in the view doesn't change.
   */
  BarView.prototype._isOrderDifferentInView = function (orderInfo, baseAxis) {
    var scale = baseAxis.scale;
    var extent = scale.getExtent();
    var tickNum = Math.max(0, extent[0]);
    var tickMax = Math.min(extent[1], scale.getOrdinalMeta().categories.length - 1);
    for (; tickNum <= tickMax; ++tickNum) {
      if (orderInfo.ordinalNumbers[tickNum] !== scale.getRawOrdinalNumber(tickNum)) {
        return true;
      }
    }
  };
  BarView.prototype._updateSortWithinSameData = function (data, orderMapping, baseAxis, api) {
    if (!this._isOrderChangedWithinSameData(data, orderMapping, baseAxis)) {
      return;
    }
    var sortInfo = this._dataSort(data, baseAxis, orderMapping);
    if (this._isOrderDifferentInView(sortInfo, baseAxis)) {
      this._removeOnRenderedListener(api);
      api.dispatchAction({
        type: 'changeAxisOrder',
        componentType: baseAxis.dim + 'Axis',
        axisId: baseAxis.index,
        sortInfo: sortInfo
      });
    }
  };
  BarView.prototype._dispatchInitSort = function (data, realtimeSortCfg, api) {
    var baseAxis = realtimeSortCfg.baseAxis;
    var sortResult = this._dataSort(data, baseAxis, function (dataIdx) {
      return data.get(data.mapDimension(realtimeSortCfg.otherAxis.dim), dataIdx);
    });
    api.dispatchAction({
      type: 'changeAxisOrder',
      componentType: baseAxis.dim + 'Axis',
      isInitSort: true,
      axisId: baseAxis.index,
      sortInfo: sortResult
    });
  };
  BarView.prototype.remove = function (ecModel, api) {
    this._clear(this._model);
    this._removeOnRenderedListener(api);
  };
  BarView.prototype.dispose = function (ecModel, api) {
    this._removeOnRenderedListener(api);
  };
  BarView.prototype._removeOnRenderedListener = function (api) {
    if (this._onRendered) {
      api.getZr().off('rendered', this._onRendered);
      this._onRendered = null;
    }
  };
  BarView.prototype._clear = function (model) {
    var group = this.group;
    var data = this._data;
    if (model && model.isAnimationEnabled() && data && !this._isLargeDraw) {
      this._removeBackground();
      this._backgroundEls = [];
      data.eachItemGraphicEl(function (el) {
        removeElementWithFadeOut(el, model, getECData(el).dataIndex);
      });
    } else {
      group.removeAll();
    }
    this._data = null;
    this._isFirstFrame = true;
  };
  BarView.prototype._removeBackground = function () {
    this.group.remove(this._backgroundGroup);
    this._backgroundGroup = null;
  };
  BarView.type = 'bar';
  return BarView;
}(ChartView);
var clip = {
  cartesian2d: function (coordSysBoundingRect, layout) {
    var signWidth = layout.width < 0 ? -1 : 1;
    var signHeight = layout.height < 0 ? -1 : 1;
    // Needs positive width and height
    if (signWidth < 0) {
      layout.x += layout.width;
      layout.width = -layout.width;
    }
    if (signHeight < 0) {
      layout.y += layout.height;
      layout.height = -layout.height;
    }
    var coordSysX2 = coordSysBoundingRect.x + coordSysBoundingRect.width;
    var coordSysY2 = coordSysBoundingRect.y + coordSysBoundingRect.height;
    var x = mathMax(layout.x, coordSysBoundingRect.x);
    var x2 = mathMin(layout.x + layout.width, coordSysX2);
    var y = mathMax(layout.y, coordSysBoundingRect.y);
    var y2 = mathMin(layout.y + layout.height, coordSysY2);
    var xClipped = x2 < x;
    var yClipped = y2 < y;
    // When xClipped or yClipped, the element will be marked as `ignore`.
    // But we should also place the element at the edge of the coord sys bounding rect.
    // Because if data changed and the bar shows again, its transition animation
    // will begin at this place.
    layout.x = xClipped && x > coordSysX2 ? x2 : x;
    layout.y = yClipped && y > coordSysY2 ? y2 : y;
    layout.width = xClipped ? 0 : x2 - x;
    layout.height = yClipped ? 0 : y2 - y;
    // Reverse back
    if (signWidth < 0) {
      layout.x += layout.width;
      layout.width = -layout.width;
    }
    if (signHeight < 0) {
      layout.y += layout.height;
      layout.height = -layout.height;
    }
    return xClipped || yClipped;
  },
  polar: function (coordSysClipArea, layout) {
    var signR = layout.r0 <= layout.r ? 1 : -1;
    // Make sure r is larger than r0
    if (signR < 0) {
      var tmp = layout.r;
      layout.r = layout.r0;
      layout.r0 = tmp;
    }
    var r = mathMin(layout.r, coordSysClipArea.r);
    var r0 = mathMax(layout.r0, coordSysClipArea.r0);
    layout.r = r;
    layout.r0 = r0;
    var clipped = r - r0 < 0;
    // Reverse back
    if (signR < 0) {
      var tmp = layout.r;
      layout.r = layout.r0;
      layout.r0 = tmp;
    }
    return clipped;
  }
};
var elementCreator = {
  cartesian2d: function (seriesModel, data, newIndex, layout, isHorizontal, animationModel, axisModel, isUpdate, roundCap) {
    var rect = new Rect({
      shape: extend({}, layout),
      z2: 1
    });
    rect.__dataIndex = newIndex;
    rect.name = 'item';
    if (animationModel) {
      var rectShape = rect.shape;
      var animateProperty = isHorizontal ? 'height' : 'width';
      rectShape[animateProperty] = 0;
    }
    return rect;
  },
  polar: function (seriesModel, data, newIndex, layout, isRadial, animationModel, axisModel, isUpdate, roundCap) {
    var ShapeClass = !isRadial && roundCap ? Sausage : Sector;
    var sector = new ShapeClass({
      shape: layout,
      z2: 1
    });
    sector.name = 'item';
    var positionMap = createPolarPositionMapping(isRadial);
    sector.calculateTextPosition = createSectorCalculateTextPosition(positionMap, {
      isRoundCap: ShapeClass === Sausage
    });
    // Animation
    if (animationModel) {
      var sectorShape = sector.shape;
      var animateProperty = isRadial ? 'r' : 'endAngle';
      var animateTarget = {};
      sectorShape[animateProperty] = isRadial ? layout.r0 : layout.startAngle;
      animateTarget[animateProperty] = layout[animateProperty];
      (isUpdate ? updateProps : initProps)(sector, {
        shape: animateTarget
        // __value: typeof dataValue === 'string' ? parseInt(dataValue, 10) : dataValue
      }, animationModel);
    }
    return sector;
  }
};
function shouldRealtimeSort(seriesModel, coordSys) {
  var realtimeSortOption = seriesModel.get('realtimeSort', true);
  var baseAxis = coordSys.getBaseAxis();
  if (process.env.NODE_ENV !== 'production') {
    if (realtimeSortOption) {
      if (baseAxis.type !== 'category') {
        warn('`realtimeSort` will not work because this bar series is not based on a category axis.');
      }
      if (coordSys.type !== 'cartesian2d') {
        warn('`realtimeSort` will not work because this bar series is not on cartesian2d.');
      }
    }
  }
  if (realtimeSortOption && baseAxis.type === 'category' && coordSys.type === 'cartesian2d') {
    return {
      baseAxis: baseAxis,
      otherAxis: coordSys.getOtherAxis(baseAxis)
    };
  }
}
function updateRealtimeAnimation(realtimeSortCfg, seriesAnimationModel, el, layout, newIndex, isHorizontal, isUpdate, isChangeOrder) {
  var seriesTarget;
  var axisTarget;
  if (isHorizontal) {
    axisTarget = {
      x: layout.x,
      width: layout.width
    };
    seriesTarget = {
      y: layout.y,
      height: layout.height
    };
  } else {
    axisTarget = {
      y: layout.y,
      height: layout.height
    };
    seriesTarget = {
      x: layout.x,
      width: layout.width
    };
  }
  if (!isChangeOrder) {
    // Keep the original growth animation if only axis order changed.
    // Not start a new animation.
    (isUpdate ? updateProps : initProps)(el, {
      shape: seriesTarget
    }, seriesAnimationModel, newIndex, null);
  }
  var axisAnimationModel = seriesAnimationModel ? realtimeSortCfg.baseAxis.model : null;
  (isUpdate ? updateProps : initProps)(el, {
    shape: axisTarget
  }, axisAnimationModel, newIndex);
}
function checkPropertiesNotValid(obj, props) {
  for (var i = 0; i < props.length; i++) {
    if (!isFinite(obj[props[i]])) {
      return true;
    }
  }
  return false;
}
var rectPropties = ['x', 'y', 'width', 'height'];
var polarPropties = ['cx', 'cy', 'r', 'startAngle', 'endAngle'];
var isValidLayout = {
  cartesian2d: function (layout) {
    return !checkPropertiesNotValid(layout, rectPropties);
  },
  polar: function (layout) {
    return !checkPropertiesNotValid(layout, polarPropties);
  }
};
var getLayout = {
  // itemModel is only used to get borderWidth, which is not needed
  // when calculating bar background layout.
  cartesian2d: function (data, dataIndex, itemModel) {
    var layout = data.getItemLayout(dataIndex);
    var fixedLineWidth = itemModel ? getLineWidth(itemModel, layout) : 0;
    // fix layout with lineWidth
    var signX = layout.width > 0 ? 1 : -1;
    var signY = layout.height > 0 ? 1 : -1;
    return {
      x: layout.x + signX * fixedLineWidth / 2,
      y: layout.y + signY * fixedLineWidth / 2,
      width: layout.width - signX * fixedLineWidth,
      height: layout.height - signY * fixedLineWidth
    };
  },
  polar: function (data, dataIndex, itemModel) {
    var layout = data.getItemLayout(dataIndex);
    return {
      cx: layout.cx,
      cy: layout.cy,
      r0: layout.r0,
      r: layout.r,
      startAngle: layout.startAngle,
      endAngle: layout.endAngle,
      clockwise: layout.clockwise
    };
  }
};
function isZeroOnPolar(layout) {
  return layout.startAngle != null && layout.endAngle != null && layout.startAngle === layout.endAngle;
}
function createPolarPositionMapping(isRadial) {
  return function (isRadial) {
    var arcOrAngle = isRadial ? 'Arc' : 'Angle';
    return function (position) {
      switch (position) {
        case 'start':
        case 'insideStart':
        case 'end':
        case 'insideEnd':
          return position + arcOrAngle;
        default:
          return position;
      }
    };
  }(isRadial);
}
function updateStyle(el, data, dataIndex, itemModel, layout, seriesModel, isHorizontalOrRadial, isPolar) {
  var style = data.getItemVisual(dataIndex, 'style');
  if (!isPolar) {
    var borderRadius = itemModel.get(['itemStyle', 'borderRadius']) || 0;
    el.setShape('r', borderRadius);
  } else if (!seriesModel.get('roundCap')) {
    var sectorShape = el.shape;
    var cornerRadius = getSectorCornerRadius(itemModel.getModel('itemStyle'), sectorShape, true);
    extend(sectorShape, cornerRadius);
    el.setShape(sectorShape);
  }
  el.useStyle(style);
  var cursorStyle = itemModel.getShallow('cursor');
  cursorStyle && el.attr('cursor', cursorStyle);
  var labelPositionOutside = isPolar ? isHorizontalOrRadial ? layout.r >= layout.r0 ? 'endArc' : 'startArc' : layout.endAngle >= layout.startAngle ? 'endAngle' : 'startAngle' : isHorizontalOrRadial ? layout.height >= 0 ? 'bottom' : 'top' : layout.width >= 0 ? 'right' : 'left';
  var labelStatesModels = getLabelStatesModels(itemModel);
  setLabelStyle(el, labelStatesModels, {
    labelFetcher: seriesModel,
    labelDataIndex: dataIndex,
    defaultText: getDefaultLabel(seriesModel.getData(), dataIndex),
    inheritColor: style.fill,
    defaultOpacity: style.opacity,
    defaultOutsidePosition: labelPositionOutside
  });
  var label = el.getTextContent();
  if (isPolar && label) {
    var position = itemModel.get(['label', 'position']);
    el.textConfig.inside = position === 'middle' ? true : null;
    setSectorTextRotation(el, position === 'outside' ? labelPositionOutside : position, createPolarPositionMapping(isHorizontalOrRadial), itemModel.get(['label', 'rotate']));
  }
  setLabelValueAnimation(label, labelStatesModels, seriesModel.getRawValue(dataIndex), function (value) {
    return getDefaultInterpolatedLabel(data, value);
  });
  var emphasisModel = itemModel.getModel(['emphasis']);
  toggleHoverEmphasis(el, emphasisModel.get('focus'), emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
  setStatesStylesFromModel(el, itemModel);
  if (isZeroOnPolar(layout)) {
    el.style.fill = 'none';
    el.style.stroke = 'none';
    each(el.states, function (state) {
      if (state.style) {
        state.style.fill = state.style.stroke = 'none';
      }
    });
  }
}
// In case width or height are too small.
function getLineWidth(itemModel, rawLayout) {
  // Has no border.
  var borderColor = itemModel.get(['itemStyle', 'borderColor']);
  if (!borderColor || borderColor === 'none') {
    return 0;
  }
  var lineWidth = itemModel.get(['itemStyle', 'borderWidth']) || 0;
  // width or height may be NaN for empty data
  var width = isNaN(rawLayout.width) ? Number.MAX_VALUE : Math.abs(rawLayout.width);
  var height = isNaN(rawLayout.height) ? Number.MAX_VALUE : Math.abs(rawLayout.height);
  return Math.min(lineWidth, width, height);
}
var LagePathShape = /** @class */function () {
  function LagePathShape() {}
  return LagePathShape;
}();
var LargePath = /** @class */function (_super) {
  __extends(LargePath, _super);
  function LargePath(opts) {
    var _this = _super.call(this, opts) || this;
    _this.type = 'largeBar';
    return _this;
  }
  LargePath.prototype.getDefaultShape = function () {
    return new LagePathShape();
  };
  LargePath.prototype.buildPath = function (ctx, shape) {
    // Drawing lines is more efficient than drawing
    // a whole line or drawing rects.
    var points = shape.points;
    var baseDimIdx = this.baseDimIdx;
    var valueDimIdx = 1 - this.baseDimIdx;
    var startPoint = [];
    var size = [];
    var barWidth = this.barWidth;
    for (var i = 0; i < points.length; i += 3) {
      size[baseDimIdx] = barWidth;
      size[valueDimIdx] = points[i + 2];
      startPoint[baseDimIdx] = points[i + baseDimIdx];
      startPoint[valueDimIdx] = points[i + valueDimIdx];
      ctx.rect(startPoint[0], startPoint[1], size[0], size[1]);
    }
  };
  return LargePath;
}(Path);
function createLarge(seriesModel, group, progressiveEls, incremental) {
  // TODO support polar
  var data = seriesModel.getData();
  var baseDimIdx = data.getLayout('valueAxisHorizontal') ? 1 : 0;
  var largeDataIndices = data.getLayout('largeDataIndices');
  var barWidth = data.getLayout('size');
  var backgroundModel = seriesModel.getModel('backgroundStyle');
  var bgPoints = data.getLayout('largeBackgroundPoints');
  if (bgPoints) {
    var bgEl = new LargePath({
      shape: {
        points: bgPoints
      },
      incremental: !!incremental,
      silent: true,
      z2: 0
    });
    bgEl.baseDimIdx = baseDimIdx;
    bgEl.largeDataIndices = largeDataIndices;
    bgEl.barWidth = barWidth;
    bgEl.useStyle(backgroundModel.getItemStyle());
    group.add(bgEl);
    progressiveEls && progressiveEls.push(bgEl);
  }
  var el = new LargePath({
    shape: {
      points: data.getLayout('largePoints')
    },
    incremental: !!incremental,
    ignoreCoarsePointer: true,
    z2: 1
  });
  el.baseDimIdx = baseDimIdx;
  el.largeDataIndices = largeDataIndices;
  el.barWidth = barWidth;
  group.add(el);
  el.useStyle(data.getVisual('style'));
  // Stroke is rendered first to avoid overlapping with fill
  el.style.stroke = null;
  // Enable tooltip and user mouse/touch event handlers.
  getECData(el).seriesIndex = seriesModel.seriesIndex;
  if (!seriesModel.get('silent')) {
    el.on('mousedown', largePathUpdateDataIndex);
    el.on('mousemove', largePathUpdateDataIndex);
  }
  progressiveEls && progressiveEls.push(el);
}
// Use throttle to avoid frequently traverse to find dataIndex.
var largePathUpdateDataIndex = throttle(function (event) {
  var largePath = this;
  var dataIndex = largePathFindDataIndex(largePath, event.offsetX, event.offsetY);
  getECData(largePath).dataIndex = dataIndex >= 0 ? dataIndex : null;
}, 30, false);
function largePathFindDataIndex(largePath, x, y) {
  var baseDimIdx = largePath.baseDimIdx;
  var valueDimIdx = 1 - baseDimIdx;
  var points = largePath.shape.points;
  var largeDataIndices = largePath.largeDataIndices;
  var startPoint = [];
  var size = [];
  var barWidth = largePath.barWidth;
  for (var i = 0, len = points.length / 3; i < len; i++) {
    var ii = i * 3;
    size[baseDimIdx] = barWidth;
    size[valueDimIdx] = points[ii + 2];
    startPoint[baseDimIdx] = points[ii + baseDimIdx];
    startPoint[valueDimIdx] = points[ii + valueDimIdx];
    if (size[valueDimIdx] < 0) {
      startPoint[valueDimIdx] += size[valueDimIdx];
      size[valueDimIdx] = -size[valueDimIdx];
    }
    if (x >= startPoint[0] && x <= startPoint[0] + size[0] && y >= startPoint[1] && y <= startPoint[1] + size[1]) {
      return largeDataIndices[i];
    }
  }
  return -1;
}
function createBackgroundShape(isHorizontalOrRadial, layout, coord) {
  if (isCoordinateSystemType(coord, 'cartesian2d')) {
    var rectShape = layout;
    var coordLayout = coord.getArea();
    return {
      x: isHorizontalOrRadial ? rectShape.x : coordLayout.x,
      y: isHorizontalOrRadial ? coordLayout.y : rectShape.y,
      width: isHorizontalOrRadial ? rectShape.width : coordLayout.width,
      height: isHorizontalOrRadial ? coordLayout.height : rectShape.height
    };
  } else {
    var coordLayout = coord.getArea();
    var sectorShape = layout;
    return {
      cx: coordLayout.cx,
      cy: coordLayout.cy,
      r0: isHorizontalOrRadial ? coordLayout.r0 : sectorShape.r0,
      r: isHorizontalOrRadial ? coordLayout.r : sectorShape.r,
      startAngle: isHorizontalOrRadial ? sectorShape.startAngle : 0,
      endAngle: isHorizontalOrRadial ? sectorShape.endAngle : Math.PI * 2
    };
  }
}
function createBackgroundEl(coord, isHorizontalOrRadial, layout) {
  var ElementClz = coord.type === 'polar' ? Sector : Rect;
  return new ElementClz({
    shape: createBackgroundShape(isHorizontalOrRadial, layout, coord),
    silent: true,
    z2: 0
  });
}
export default BarView;