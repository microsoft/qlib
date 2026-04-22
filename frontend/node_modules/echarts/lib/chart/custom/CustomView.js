
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
import { hasOwn, assert, isString, retrieve2, retrieve3, defaults, each, indexOf } from 'zrender/lib/core/util.js';
import * as graphicUtil from '../../util/graphic.js';
import { setDefaultStateProxy, toggleHoverEmphasis } from '../../util/states.js';
import * as labelStyleHelper from '../../label/labelStyle.js';
import { getDefaultLabel } from '../helper/labelHelper.js';
import { getLayoutOnAxis } from '../../layout/barGrid.js';
import DataDiffer from '../../data/DataDiffer.js';
import ChartView from '../../view/Chart.js';
import { createClipPath } from '../helper/createClipPathFromCoordSys.js';
import prepareCartesian2d from '../../coord/cartesian/prepareCustom.js';
import prepareGeo from '../../coord/geo/prepareCustom.js';
import prepareSingleAxis from '../../coord/single/prepareCustom.js';
import preparePolar from '../../coord/polar/prepareCustom.js';
import prepareCalendar from '../../coord/calendar/prepareCustom.js';
import Displayable from 'zrender/lib/graphic/Displayable.js';
import { convertToEC4StyleForCustomSerise, isEC4CompatibleStyle, convertFromEC4CompatibleStyle, warnDeprecated } from '../../util/styleCompat.js';
import { throwError } from '../../util/log.js';
import { createOrUpdatePatternFromDecal } from '../../util/decal.js';
import { STYLE_VISUAL_TYPE, NON_STYLE_VISUAL_PROPS, customInnerStore } from './CustomSeries.js';
import { applyLeaveTransition, applyUpdateTransition } from '../../animation/customGraphicTransition.js';
import { applyKeyframeAnimation, stopPreviousKeyframeAnimationAndRestore } from '../../animation/customGraphicKeyframeAnimation.js';
var EMPHASIS = 'emphasis';
var NORMAL = 'normal';
var BLUR = 'blur';
var SELECT = 'select';
var STATES = [NORMAL, EMPHASIS, BLUR, SELECT];
var PATH_ITEM_STYLE = {
  normal: ['itemStyle'],
  emphasis: [EMPHASIS, 'itemStyle'],
  blur: [BLUR, 'itemStyle'],
  select: [SELECT, 'itemStyle']
};
var PATH_LABEL = {
  normal: ['label'],
  emphasis: [EMPHASIS, 'label'],
  blur: [BLUR, 'label'],
  select: [SELECT, 'label']
};
var DEFAULT_TRANSITION = ['x', 'y'];
// Use prefix to avoid index to be the same as el.name,
// which will cause weird update animation.
var GROUP_DIFF_PREFIX = 'e\0\0';
var attachedTxInfoTmp = {
  normal: {},
  emphasis: {},
  blur: {},
  select: {}
};
/**
 * To reduce total package size of each coordinate systems, the modules `prepareCustom`
 * of each coordinate systems are not required by each coordinate systems directly, but
 * required by the module `custom`.
 *
 * prepareInfoForCustomSeries {Function}: optional
 *     @return {Object} {coordSys: {...}, api: {
 *         coord: function (data, clamp) {}, // return point in global.
 *         size: function (dataSize, dataItem) {} // return size of each axis in coordSys.
 *     }}
 */
var prepareCustoms = {
  cartesian2d: prepareCartesian2d,
  geo: prepareGeo,
  single: prepareSingleAxis,
  polar: preparePolar,
  calendar: prepareCalendar
};
function isPath(el) {
  return el instanceof graphicUtil.Path;
}
function isDisplayable(el) {
  return el instanceof Displayable;
}
function copyElement(sourceEl, targetEl) {
  targetEl.copyTransform(sourceEl);
  if (isDisplayable(targetEl) && isDisplayable(sourceEl)) {
    targetEl.setStyle(sourceEl.style);
    targetEl.z = sourceEl.z;
    targetEl.z2 = sourceEl.z2;
    targetEl.zlevel = sourceEl.zlevel;
    targetEl.invisible = sourceEl.invisible;
    targetEl.ignore = sourceEl.ignore;
    if (isPath(targetEl) && isPath(sourceEl)) {
      targetEl.setShape(sourceEl.shape);
    }
  }
}
var CustomChartView = /** @class */function (_super) {
  __extends(CustomChartView, _super);
  function CustomChartView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = CustomChartView.type;
    return _this;
  }
  CustomChartView.prototype.render = function (customSeries, ecModel, api, payload) {
    // Clear previously rendered progressive elements.
    this._progressiveEls = null;
    var oldData = this._data;
    var data = customSeries.getData();
    var group = this.group;
    var renderItem = makeRenderItem(customSeries, data, ecModel, api);
    if (!oldData) {
      // Previous render is incremental render or first render.
      // Needs remove the incremental rendered elements.
      group.removeAll();
    }
    data.diff(oldData).add(function (newIdx) {
      createOrUpdateItem(api, null, newIdx, renderItem(newIdx, payload), customSeries, group, data);
    }).remove(function (oldIdx) {
      var el = oldData.getItemGraphicEl(oldIdx);
      el && applyLeaveTransition(el, customInnerStore(el).option, customSeries);
    }).update(function (newIdx, oldIdx) {
      var oldEl = oldData.getItemGraphicEl(oldIdx);
      createOrUpdateItem(api, oldEl, newIdx, renderItem(newIdx, payload), customSeries, group, data);
    }).execute();
    // Do clipping
    var clipPath = customSeries.get('clip', true) ? createClipPath(customSeries.coordinateSystem, false, customSeries) : null;
    if (clipPath) {
      group.setClipPath(clipPath);
    } else {
      group.removeClipPath();
    }
    this._data = data;
  };
  CustomChartView.prototype.incrementalPrepareRender = function (customSeries, ecModel, api) {
    this.group.removeAll();
    this._data = null;
  };
  CustomChartView.prototype.incrementalRender = function (params, customSeries, ecModel, api, payload) {
    var data = customSeries.getData();
    var renderItem = makeRenderItem(customSeries, data, ecModel, api);
    var progressiveEls = this._progressiveEls = [];
    function setIncrementalAndHoverLayer(el) {
      if (!el.isGroup) {
        el.incremental = true;
        el.ensureState('emphasis').hoverLayer = true;
      }
    }
    for (var idx = params.start; idx < params.end; idx++) {
      var el = createOrUpdateItem(null, null, idx, renderItem(idx, payload), customSeries, this.group, data);
      if (el) {
        el.traverse(setIncrementalAndHoverLayer);
        progressiveEls.push(el);
      }
    }
  };
  CustomChartView.prototype.eachRendered = function (cb) {
    graphicUtil.traverseElements(this._progressiveEls || this.group, cb);
  };
  CustomChartView.prototype.filterForExposedEvent = function (eventType, query, targetEl, packedEvent) {
    var elementName = query.element;
    if (elementName == null || targetEl.name === elementName) {
      return true;
    }
    // Enable to give a name on a group made by `renderItem`, and listen
    // events that are triggered by its descendents.
    while ((targetEl = targetEl.__hostTarget || targetEl.parent) && targetEl !== this.group) {
      if (targetEl.name === elementName) {
        return true;
      }
    }
    return false;
  };
  CustomChartView.type = 'custom';
  return CustomChartView;
}(ChartView);
export default CustomChartView;
function createEl(elOption) {
  var graphicType = elOption.type;
  var el;
  // Those graphic elements are not shapes. They should not be
  // overwritten by users, so do them first.
  if (graphicType === 'path') {
    var shape = elOption.shape;
    // Using pathRect brings convenience to users sacle svg path.
    var pathRect = shape.width != null && shape.height != null ? {
      x: shape.x || 0,
      y: shape.y || 0,
      width: shape.width,
      height: shape.height
    } : null;
    var pathData = getPathData(shape);
    // Path is also used for icon, so layout 'center' by default.
    el = graphicUtil.makePath(pathData, null, pathRect, shape.layout || 'center');
    customInnerStore(el).customPathData = pathData;
  } else if (graphicType === 'image') {
    el = new graphicUtil.Image({});
    customInnerStore(el).customImagePath = elOption.style.image;
  } else if (graphicType === 'text') {
    el = new graphicUtil.Text({});
    // customInnerStore(el).customText = (elOption.style as TextStyleProps).text;
  } else if (graphicType === 'group') {
    el = new graphicUtil.Group();
  } else if (graphicType === 'compoundPath') {
    throw new Error('"compoundPath" is not supported yet.');
  } else {
    var Clz = graphicUtil.getShapeClass(graphicType);
    if (!Clz) {
      var errMsg = '';
      if (process.env.NODE_ENV !== 'production') {
        errMsg = 'graphic type "' + graphicType + '" can not be found.';
      }
      throwError(errMsg);
    }
    el = new Clz();
  }
  customInnerStore(el).customGraphicType = graphicType;
  el.name = elOption.name;
  // Compat ec4: the default z2 lift is 1. If changing the number,
  // some cases probably be broken: hierarchy layout along z, like circle packing,
  // where emphasis only intending to modify color/border rather than lift z2.
  el.z2EmphasisLift = 1;
  el.z2SelectLift = 1;
  return el;
}
function updateElNormal(
// Can be null/undefined
api, el, dataIndex, elOption, attachedTxInfo, seriesModel, isInit) {
  // Stop and restore before update any other attributes.
  stopPreviousKeyframeAnimationAndRestore(el);
  var txCfgOpt = attachedTxInfo && attachedTxInfo.normal.cfg;
  if (txCfgOpt) {
    // PENDING: whether use user object directly rather than clone?
    // TODO:5.0 textConfig transition animation?
    el.setTextConfig(txCfgOpt);
  }
  // Default transition ['x', 'y']
  if (elOption && elOption.transition == null) {
    elOption.transition = DEFAULT_TRANSITION;
  }
  // Do some normalization on style.
  var styleOpt = elOption && elOption.style;
  if (styleOpt) {
    if (el.type === 'text') {
      var textOptionStyle = styleOpt;
      // Compatible with ec4: if `textFill` or `textStroke` exists use them.
      hasOwn(textOptionStyle, 'textFill') && (textOptionStyle.fill = textOptionStyle.textFill);
      hasOwn(textOptionStyle, 'textStroke') && (textOptionStyle.stroke = textOptionStyle.textStroke);
    }
    var decalPattern = void 0;
    var decalObj = isPath(el) ? styleOpt.decal : null;
    if (api && decalObj) {
      decalObj.dirty = true;
      decalPattern = createOrUpdatePatternFromDecal(decalObj, api);
    }
    // Always overwrite in case user specify this prop.
    styleOpt.__decalPattern = decalPattern;
  }
  if (isDisplayable(el)) {
    if (styleOpt) {
      var decalPattern = styleOpt.__decalPattern;
      if (decalPattern) {
        styleOpt.decal = decalPattern;
      }
    }
  }
  applyUpdateTransition(el, elOption, seriesModel, {
    dataIndex: dataIndex,
    isInit: isInit,
    clearStyle: true
  });
  applyKeyframeAnimation(el, elOption.keyframeAnimation, seriesModel);
}
function updateElOnState(state, el, elStateOpt, styleOpt, attachedTxInfo) {
  var elDisplayable = el.isGroup ? null : el;
  var txCfgOpt = attachedTxInfo && attachedTxInfo[state].cfg;
  // PENDING:5.0 support customize scale change and transition animation?
  if (elDisplayable) {
    // By default support auto lift color when hover whether `emphasis` specified.
    var stateObj = elDisplayable.ensureState(state);
    if (styleOpt === false) {
      var existingEmphasisState = elDisplayable.getState(state);
      if (existingEmphasisState) {
        existingEmphasisState.style = null;
      }
    } else {
      // style is needed to enable default emphasis.
      stateObj.style = styleOpt || null;
    }
    // If `elOption.styleEmphasis` or `elOption.emphasis.style` is `false`,
    // remove hover style.
    // If `elOption.textConfig` or `elOption.emphasis.textConfig` is null/undefined, it does not
    // make sense. So for simplicity, we do not ditinguish `hasOwnProperty` and null/undefined.
    if (txCfgOpt) {
      stateObj.textConfig = txCfgOpt;
    }
    setDefaultStateProxy(elDisplayable);
  }
}
function updateZ(el, elOption, seriesModel) {
  // Group not support textContent and not support z yet.
  if (el.isGroup) {
    return;
  }
  var elDisplayable = el;
  var currentZ = seriesModel.currentZ;
  var currentZLevel = seriesModel.currentZLevel;
  // Always erase.
  elDisplayable.z = currentZ;
  elDisplayable.zlevel = currentZLevel;
  // z2 must not be null/undefined, otherwise sort error may occur.
  var optZ2 = elOption.z2;
  optZ2 != null && (elDisplayable.z2 = optZ2 || 0);
  for (var i = 0; i < STATES.length; i++) {
    updateZForEachState(elDisplayable, elOption, STATES[i]);
  }
}
function updateZForEachState(elDisplayable, elOption, state) {
  var isNormal = state === NORMAL;
  var elStateOpt = isNormal ? elOption : retrieveStateOption(elOption, state);
  var optZ2 = elStateOpt ? elStateOpt.z2 : null;
  var stateObj;
  if (optZ2 != null) {
    // Do not `ensureState` until required.
    stateObj = isNormal ? elDisplayable : elDisplayable.ensureState(state);
    stateObj.z2 = optZ2 || 0;
  }
}
function makeRenderItem(customSeries, data, ecModel, api) {
  var renderItem = customSeries.get('renderItem');
  var coordSys = customSeries.coordinateSystem;
  var prepareResult = {};
  if (coordSys) {
    if (process.env.NODE_ENV !== 'production') {
      assert(renderItem, 'series.render is required.');
      assert(coordSys.prepareCustoms || prepareCustoms[coordSys.type], 'This coordSys does not support custom series.');
    }
    // `coordSys.prepareCustoms` is used for external coord sys like bmap.
    prepareResult = coordSys.prepareCustoms ? coordSys.prepareCustoms(coordSys) : prepareCustoms[coordSys.type](coordSys);
  }
  var userAPI = defaults({
    getWidth: api.getWidth,
    getHeight: api.getHeight,
    getZr: api.getZr,
    getDevicePixelRatio: api.getDevicePixelRatio,
    value: value,
    style: style,
    ordinalRawValue: ordinalRawValue,
    styleEmphasis: styleEmphasis,
    visual: visual,
    barLayout: barLayout,
    currentSeriesIndices: currentSeriesIndices,
    font: font
  }, prepareResult.api || {});
  var userParams = {
    // The life cycle of context: current round of rendering.
    // The global life cycle is probably not necessary, because
    // user can store global status by themselves.
    context: {},
    seriesId: customSeries.id,
    seriesName: customSeries.name,
    seriesIndex: customSeries.seriesIndex,
    coordSys: prepareResult.coordSys,
    dataInsideLength: data.count(),
    encode: wrapEncodeDef(customSeries.getData())
  };
  // If someday intending to refactor them to a class, should consider do not
  // break change: currently these attribute member are encapsulated in a closure
  // so that do not need to force user to call these method with a scope.
  // Do not support call `api` asynchronously without dataIndexInside input.
  var currDataIndexInside;
  var currItemModel;
  var currItemStyleModels = {};
  var currLabelModels = {};
  var seriesItemStyleModels = {};
  var seriesLabelModels = {};
  for (var i = 0; i < STATES.length; i++) {
    var stateName = STATES[i];
    seriesItemStyleModels[stateName] = customSeries.getModel(PATH_ITEM_STYLE[stateName]);
    seriesLabelModels[stateName] = customSeries.getModel(PATH_LABEL[stateName]);
  }
  function getItemModel(dataIndexInside) {
    return dataIndexInside === currDataIndexInside ? currItemModel || (currItemModel = data.getItemModel(dataIndexInside)) : data.getItemModel(dataIndexInside);
  }
  function getItemStyleModel(dataIndexInside, state) {
    return !data.hasItemOption ? seriesItemStyleModels[state] : dataIndexInside === currDataIndexInside ? currItemStyleModels[state] || (currItemStyleModels[state] = getItemModel(dataIndexInside).getModel(PATH_ITEM_STYLE[state])) : getItemModel(dataIndexInside).getModel(PATH_ITEM_STYLE[state]);
  }
  function getLabelModel(dataIndexInside, state) {
    return !data.hasItemOption ? seriesLabelModels[state] : dataIndexInside === currDataIndexInside ? currLabelModels[state] || (currLabelModels[state] = getItemModel(dataIndexInside).getModel(PATH_LABEL[state])) : getItemModel(dataIndexInside).getModel(PATH_LABEL[state]);
  }
  return function (dataIndexInside, payload) {
    currDataIndexInside = dataIndexInside;
    currItemModel = null;
    currItemStyleModels = {};
    currLabelModels = {};
    return renderItem && renderItem(defaults({
      dataIndexInside: dataIndexInside,
      dataIndex: data.getRawIndex(dataIndexInside),
      // Can be used for optimization when zoom or roam.
      actionType: payload ? payload.type : null
    }, userParams), userAPI);
  };
  /**
   * @public
   * @param dim by default 0.
   * @param dataIndexInside by default `currDataIndexInside`.
   */
  function value(dim, dataIndexInside) {
    dataIndexInside == null && (dataIndexInside = currDataIndexInside);
    return data.getStore().get(data.getDimensionIndex(dim || 0), dataIndexInside);
  }
  /**
   * @public
   * @param dim by default 0.
   * @param dataIndexInside by default `currDataIndexInside`.
   */
  function ordinalRawValue(dim, dataIndexInside) {
    dataIndexInside == null && (dataIndexInside = currDataIndexInside);
    dim = dim || 0;
    var dimInfo = data.getDimensionInfo(dim);
    if (!dimInfo) {
      var dimIndex = data.getDimensionIndex(dim);
      return dimIndex >= 0 ? data.getStore().get(dimIndex, dataIndexInside) : undefined;
    }
    var val = data.get(dimInfo.name, dataIndexInside);
    var ordinalMeta = dimInfo && dimInfo.ordinalMeta;
    return ordinalMeta ? ordinalMeta.categories[val] : val;
  }
  /**
   * @deprecated The original intention of `api.style` is enable to set itemStyle
   * like other series. But it is not necessary and not easy to give a strict definition
   * of what it returns. And since echarts5 it needs to be make compat work. So
   * deprecates it since echarts5.
   *
   * By default, `visual` is applied to style (to support visualMap).
   * `visual.color` is applied at `fill`. If user want apply visual.color on `stroke`,
   * it can be implemented as:
   * `api.style({stroke: api.visual('color'), fill: null})`;
   *
   * [Compat]: since ec5, RectText has been separated from its hosts el.
   * so `api.style()` will only return the style from `itemStyle` but not handle `label`
   * any more. But `series.label` config is never published in doc.
   * We still compat it in `api.style()`. But not encourage to use it and will still not
   * to pulish it to doc.
   * @public
   * @param dataIndexInside by default `currDataIndexInside`.
   */
  function style(userProps, dataIndexInside) {
    if (process.env.NODE_ENV !== 'production') {
      warnDeprecated('api.style', 'Please write literal style directly instead.');
    }
    dataIndexInside == null && (dataIndexInside = currDataIndexInside);
    var style = data.getItemVisual(dataIndexInside, 'style');
    var visualColor = style && style.fill;
    var opacity = style && style.opacity;
    var itemStyle = getItemStyleModel(dataIndexInside, NORMAL).getItemStyle();
    visualColor != null && (itemStyle.fill = visualColor);
    opacity != null && (itemStyle.opacity = opacity);
    var opt = {
      inheritColor: isString(visualColor) ? visualColor : '#000'
    };
    var labelModel = getLabelModel(dataIndexInside, NORMAL);
    // Now that the feature of "auto adjust text fill/stroke" has been migrated to zrender
    // since ec5, we should set `isAttached` as `false` here and make compat in
    // `convertToEC4StyleForCustomSerise`.
    var textStyle = labelStyleHelper.createTextStyle(labelModel, null, opt, false, true);
    textStyle.text = labelModel.getShallow('show') ? retrieve2(customSeries.getFormattedLabel(dataIndexInside, NORMAL), getDefaultLabel(data, dataIndexInside)) : null;
    var textConfig = labelStyleHelper.createTextConfig(labelModel, opt, false);
    preFetchFromExtra(userProps, itemStyle);
    itemStyle = convertToEC4StyleForCustomSerise(itemStyle, textStyle, textConfig);
    userProps && applyUserPropsAfter(itemStyle, userProps);
    itemStyle.legacy = true;
    return itemStyle;
  }
  /**
   * @deprecated The reason see `api.style()`
   * @public
   * @param dataIndexInside by default `currDataIndexInside`.
   */
  function styleEmphasis(userProps, dataIndexInside) {
    if (process.env.NODE_ENV !== 'production') {
      warnDeprecated('api.styleEmphasis', 'Please write literal style directly instead.');
    }
    dataIndexInside == null && (dataIndexInside = currDataIndexInside);
    var itemStyle = getItemStyleModel(dataIndexInside, EMPHASIS).getItemStyle();
    var labelModel = getLabelModel(dataIndexInside, EMPHASIS);
    var textStyle = labelStyleHelper.createTextStyle(labelModel, null, null, true, true);
    textStyle.text = labelModel.getShallow('show') ? retrieve3(customSeries.getFormattedLabel(dataIndexInside, EMPHASIS), customSeries.getFormattedLabel(dataIndexInside, NORMAL), getDefaultLabel(data, dataIndexInside)) : null;
    var textConfig = labelStyleHelper.createTextConfig(labelModel, null, true);
    preFetchFromExtra(userProps, itemStyle);
    itemStyle = convertToEC4StyleForCustomSerise(itemStyle, textStyle, textConfig);
    userProps && applyUserPropsAfter(itemStyle, userProps);
    itemStyle.legacy = true;
    return itemStyle;
  }
  function applyUserPropsAfter(itemStyle, extra) {
    for (var key in extra) {
      if (hasOwn(extra, key)) {
        itemStyle[key] = extra[key];
      }
    }
  }
  function preFetchFromExtra(extra, itemStyle) {
    // A trick to retrieve those props firstly, which are used to
    // apply auto inside fill/stroke in `convertToEC4StyleForCustomSerise`.
    // (It's not reasonable but only for a degree of compat)
    if (extra) {
      extra.textFill && (itemStyle.textFill = extra.textFill);
      extra.textPosition && (itemStyle.textPosition = extra.textPosition);
    }
  }
  /**
   * @public
   * @param dataIndexInside by default `currDataIndexInside`.
   */
  function visual(visualType, dataIndexInside) {
    dataIndexInside == null && (dataIndexInside = currDataIndexInside);
    if (hasOwn(STYLE_VISUAL_TYPE, visualType)) {
      var style_1 = data.getItemVisual(dataIndexInside, 'style');
      return style_1 ? style_1[STYLE_VISUAL_TYPE[visualType]] : null;
    }
    // Only support these visuals. Other visual might be inner tricky
    // for performance (like `style`), do not expose to users.
    if (hasOwn(NON_STYLE_VISUAL_PROPS, visualType)) {
      return data.getItemVisual(dataIndexInside, visualType);
    }
  }
  /**
   * @public
   * @return If not support, return undefined.
   */
  function barLayout(opt) {
    if (coordSys.type === 'cartesian2d') {
      var baseAxis = coordSys.getBaseAxis();
      return getLayoutOnAxis(defaults({
        axis: baseAxis
      }, opt));
    }
  }
  /**
   * @public
   */
  function currentSeriesIndices() {
    return ecModel.getCurrentSeriesIndices();
  }
  /**
   * @public
   * @return font string
   */
  function font(opt) {
    return labelStyleHelper.getFont(opt, ecModel);
  }
}
function wrapEncodeDef(data) {
  var encodeDef = {};
  each(data.dimensions, function (dimName) {
    var dimInfo = data.getDimensionInfo(dimName);
    if (!dimInfo.isExtraCoord) {
      var coordDim = dimInfo.coordDim;
      var dataDims = encodeDef[coordDim] = encodeDef[coordDim] || [];
      dataDims[dimInfo.coordDimIndex] = data.getDimensionIndex(dimName);
    }
  });
  return encodeDef;
}
function createOrUpdateItem(api, existsEl, dataIndex, elOption, seriesModel, group, data) {
  // [Rule]
  // If `renderItem` returns `null`/`undefined`/`false`, remove the previous el if existing.
  //     (It seems that violate the "merge" principle, but most of users probably intuitively
  //     regard "return;" as "show nothing element whatever", so make a exception to meet the
  //     most cases.)
  // The rule or "merge" see [STRATEGY_MERGE].
  // If `elOption` is `null`/`undefined`/`false` (when `renderItem` returns nothing).
  if (!elOption) {
    group.remove(existsEl);
    return;
  }
  var el = doCreateOrUpdateEl(api, existsEl, dataIndex, elOption, seriesModel, group);
  el && data.setItemGraphicEl(dataIndex, el);
  el && toggleHoverEmphasis(el, elOption.focus, elOption.blurScope, elOption.emphasisDisabled);
  return el;
}
function doCreateOrUpdateEl(api, existsEl, dataIndex, elOption, seriesModel, group) {
  if (process.env.NODE_ENV !== 'production') {
    assert(elOption, 'should not have an null/undefined element setting');
  }
  var toBeReplacedIdx = -1;
  var oldEl = existsEl;
  if (existsEl && doesElNeedRecreate(existsEl, elOption, seriesModel)
  // || (
  //     // PENDING: even in one-to-one mapping case, if el is marked as morph,
  //     // do not sure whether the el will be mapped to another el with different
  //     // hierarchy in Group tree. So always recreate el rather than reuse the el.
  //     morphHelper && morphHelper.isOneToOneFrom(el)
  // )
  ) {
    // Should keep at the original index, otherwise "merge by index" will be incorrect.
    toBeReplacedIdx = indexOf(group.childrenRef(), existsEl);
    existsEl = null;
  }
  var isInit = !existsEl;
  var el = existsEl;
  if (!el) {
    el = createEl(elOption);
    if (oldEl) {
      copyElement(oldEl, el);
    }
  } else {
    // FIMXE:NEXT unified clearState?
    // If in some case the performance issue arised, consider
    // do not clearState but update cached normal state directly.
    el.clearStates();
  }
  // Need to set morph: false explictly to disable automatically morphing.
  if (elOption.morph === false) {
    el.disableMorphing = true;
  } else if (el.disableMorphing) {
    el.disableMorphing = false;
  }
  attachedTxInfoTmp.normal.cfg = attachedTxInfoTmp.normal.conOpt = attachedTxInfoTmp.emphasis.cfg = attachedTxInfoTmp.emphasis.conOpt = attachedTxInfoTmp.blur.cfg = attachedTxInfoTmp.blur.conOpt = attachedTxInfoTmp.select.cfg = attachedTxInfoTmp.select.conOpt = null;
  attachedTxInfoTmp.isLegacy = false;
  doCreateOrUpdateAttachedTx(el, dataIndex, elOption, seriesModel, isInit, attachedTxInfoTmp);
  doCreateOrUpdateClipPath(el, dataIndex, elOption, seriesModel, isInit);
  updateElNormal(api, el, dataIndex, elOption, attachedTxInfoTmp, seriesModel, isInit);
  // `elOption.info` enables user to mount some info on
  // elements and use them in event handlers.
  // Update them only when user specified, otherwise, remain.
  hasOwn(elOption, 'info') && (customInnerStore(el).info = elOption.info);
  for (var i = 0; i < STATES.length; i++) {
    var stateName = STATES[i];
    if (stateName !== NORMAL) {
      var otherStateOpt = retrieveStateOption(elOption, stateName);
      var otherStyleOpt = retrieveStyleOptionOnState(elOption, otherStateOpt, stateName);
      updateElOnState(stateName, el, otherStateOpt, otherStyleOpt, attachedTxInfoTmp);
    }
  }
  updateZ(el, elOption, seriesModel);
  if (elOption.type === 'group') {
    mergeChildren(api, el, dataIndex, elOption, seriesModel);
  }
  if (toBeReplacedIdx >= 0) {
    group.replaceAt(el, toBeReplacedIdx);
  } else {
    group.add(el);
  }
  return el;
}
// `el` must not be null/undefined.
function doesElNeedRecreate(el, elOption, seriesModel) {
  var elInner = customInnerStore(el);
  var elOptionType = elOption.type;
  var elOptionShape = elOption.shape;
  var elOptionStyle = elOption.style;
  return (
    // Always create new if universal transition is enabled.
    // Because we do transition after render. It needs to know what old element is. Replacement will loose it.
    seriesModel.isUniversalTransitionEnabled()
    // If `elOptionType` is `null`, follow the merge principle.
    || elOptionType != null && elOptionType !== elInner.customGraphicType || elOptionType === 'path' && hasOwnPathData(elOptionShape) && getPathData(elOptionShape) !== elInner.customPathData || elOptionType === 'image' && hasOwn(elOptionStyle, 'image') && elOptionStyle.image !== elInner.customImagePath
    // // FIXME test and remove this restriction?
    // || (elOptionType === 'text'
    //     && hasOwn(elOptionStyle, 'text')
    //     && (elOptionStyle as TextStyleProps).text !== elInner.customText
    // )
  );
}
function doCreateOrUpdateClipPath(el, dataIndex, elOption, seriesModel, isInit) {
  // Based on the "merge" principle, if no clipPath provided,
  // do nothing. The exists clip will be totally removed only if
  // `el.clipPath` is `false`. Otherwise it will be merged/replaced.
  var clipPathOpt = elOption.clipPath;
  if (clipPathOpt === false) {
    if (el && el.getClipPath()) {
      el.removeClipPath();
    }
  } else if (clipPathOpt) {
    var clipPath = el.getClipPath();
    if (clipPath && doesElNeedRecreate(clipPath, clipPathOpt, seriesModel)) {
      clipPath = null;
    }
    if (!clipPath) {
      clipPath = createEl(clipPathOpt);
      if (process.env.NODE_ENV !== 'production') {
        assert(isPath(clipPath), 'Only any type of `path` can be used in `clipPath`, rather than ' + clipPath.type + '.');
      }
      el.setClipPath(clipPath);
    }
    updateElNormal(null, clipPath, dataIndex, clipPathOpt, null, seriesModel, isInit);
  }
  // If not define `clipPath` in option, do nothing unnecessary.
}
function doCreateOrUpdateAttachedTx(el, dataIndex, elOption, seriesModel, isInit, attachedTxInfo) {
  // Group does not support textContent temporarily until necessary.
  if (el.isGroup) {
    return;
  }
  // Normal must be called before emphasis, for `isLegacy` detection.
  processTxInfo(elOption, null, attachedTxInfo);
  processTxInfo(elOption, EMPHASIS, attachedTxInfo);
  // If `elOption.textConfig` or `elOption.textContent` is null/undefined, it does not make sense.
  // So for simplicity, if "elOption hasOwnProperty of them but be null/undefined", we do not
  // trade them as set to null to el.
  // Especially:
  // `elOption.textContent: false` means remove textContent.
  // `elOption.textContent.emphasis.style: false` means remove the style from emphasis state.
  var txConOptNormal = attachedTxInfo.normal.conOpt;
  var txConOptEmphasis = attachedTxInfo.emphasis.conOpt;
  var txConOptBlur = attachedTxInfo.blur.conOpt;
  var txConOptSelect = attachedTxInfo.select.conOpt;
  if (txConOptNormal != null || txConOptEmphasis != null || txConOptSelect != null || txConOptBlur != null) {
    var textContent = el.getTextContent();
    if (txConOptNormal === false) {
      textContent && el.removeTextContent();
    } else {
      txConOptNormal = attachedTxInfo.normal.conOpt = txConOptNormal || {
        type: 'text'
      };
      if (!textContent) {
        textContent = createEl(txConOptNormal);
        el.setTextContent(textContent);
      } else {
        // If in some case the performance issue arised, consider
        // do not clearState but update cached normal state directly.
        textContent.clearStates();
      }
      updateElNormal(null, textContent, dataIndex, txConOptNormal, null, seriesModel, isInit);
      var txConStlOptNormal = txConOptNormal && txConOptNormal.style;
      for (var i = 0; i < STATES.length; i++) {
        var stateName = STATES[i];
        if (stateName !== NORMAL) {
          var txConOptOtherState = attachedTxInfo[stateName].conOpt;
          updateElOnState(stateName, textContent, txConOptOtherState, retrieveStyleOptionOnState(txConOptNormal, txConOptOtherState, stateName), null);
        }
      }
      txConStlOptNormal ? textContent.dirty() : textContent.markRedraw();
    }
  }
}
function processTxInfo(elOption, state, attachedTxInfo) {
  var stateOpt = !state ? elOption : retrieveStateOption(elOption, state);
  var styleOpt = !state ? elOption.style : retrieveStyleOptionOnState(elOption, stateOpt, EMPHASIS);
  var elType = elOption.type;
  var txCfg = stateOpt ? stateOpt.textConfig : null;
  var txConOptNormal = elOption.textContent;
  var txConOpt = !txConOptNormal ? null : !state ? txConOptNormal : retrieveStateOption(txConOptNormal, state);
  if (styleOpt && (
  // Because emphasis style has little info to detect legacy,
  // if normal is legacy, emphasis is trade as legacy.
  attachedTxInfo.isLegacy || isEC4CompatibleStyle(styleOpt, elType, !!txCfg, !!txConOpt))) {
    attachedTxInfo.isLegacy = true;
    var convertResult = convertFromEC4CompatibleStyle(styleOpt, elType, !state);
    // Explicitly specified `textConfig` and `textContent` has higher priority than
    // the ones generated by legacy style. Otherwise if users use them and `api.style`
    // at the same time, they not both work and hardly to known why.
    if (!txCfg && convertResult.textConfig) {
      txCfg = convertResult.textConfig;
    }
    if (!txConOpt && convertResult.textContent) {
      txConOpt = convertResult.textContent;
    }
  }
  if (!state && txConOpt) {
    var txConOptNormal_1 = txConOpt;
    // `textContent: {type: 'text'}`, the "type" is easy to be missing. So we tolerate it.
    !txConOptNormal_1.type && (txConOptNormal_1.type = 'text');
    if (process.env.NODE_ENV !== 'production') {
      // Do not tolerate incorrcet type for forward compat.
      assert(txConOptNormal_1.type === 'text', 'textContent.type must be "text"');
    }
  }
  var info = !state ? attachedTxInfo.normal : attachedTxInfo[state];
  info.cfg = txCfg;
  info.conOpt = txConOpt;
}
function retrieveStateOption(elOption, state) {
  return !state ? elOption : elOption ? elOption[state] : null;
}
function retrieveStyleOptionOnState(stateOptionNormal, stateOption, state) {
  var style = stateOption && stateOption.style;
  if (style == null && state === EMPHASIS && stateOptionNormal) {
    style = stateOptionNormal.styleEmphasis;
  }
  return style;
}
// Usage:
// (1) By default, `elOption.$mergeChildren` is `'byIndex'`, which indicates
//     that the existing children will not be removed, and enables the feature
//     that update some of the props of some of the children simply by construct
//     the returned children of `renderItem` like:
//     `var children = group.children = []; children[3] = {opacity: 0.5};`
// (2) If `elOption.$mergeChildren` is `'byName'`, add/update/remove children
//     by child.name. But that might be lower performance.
// (3) If `elOption.$mergeChildren` is `false`, the existing children will be
//     replaced totally.
// (4) If `!elOption.children`, following the "merge" principle, nothing will
//     happen.
// (5) If `elOption.$mergeChildren` is not `false` neither `'byName'` and the
//     `el` is a group, and if any of the new child is null, it means to remove
//     the element at the same index, if exists. On the other hand, if the new
//     child is and empty object `{}`, it means to keep the element not changed.
//
// For implementation simpleness, do not provide a direct way to remove single
// child (otherwise the total indices of the children array have to be modified).
// User can remove a single child by setting its `ignore` to `true`.
function mergeChildren(api, el, dataIndex, elOption, seriesModel) {
  var newChildren = elOption.children;
  var newLen = newChildren ? newChildren.length : 0;
  var mergeChildren = elOption.$mergeChildren;
  // `diffChildrenByName` has been deprecated.
  var byName = mergeChildren === 'byName' || elOption.diffChildrenByName;
  var notMerge = mergeChildren === false;
  // For better performance on roam update, only enter if necessary.
  if (!newLen && !byName && !notMerge) {
    return;
  }
  if (byName) {
    diffGroupChildren({
      api: api,
      oldChildren: el.children() || [],
      newChildren: newChildren || [],
      dataIndex: dataIndex,
      seriesModel: seriesModel,
      group: el
    });
    return;
  }
  notMerge && el.removeAll();
  // Mapping children of a group simply by index, which
  // might be better performance.
  var index = 0;
  for (; index < newLen; index++) {
    var newChild = newChildren[index];
    var oldChild = el.childAt(index);
    if (newChild) {
      if (newChild.ignore == null) {
        // The old child is set to be ignored if null (see comments
        // below). So we need to set ignore to be false back.
        newChild.ignore = false;
      }
      doCreateOrUpdateEl(api, oldChild, dataIndex, newChild, seriesModel, el);
    } else {
      if (process.env.NODE_ENV !== 'production') {
        assert(oldChild, 'renderItem should not return a group containing elements' + ' as null/undefined/{} if they do not exist before.');
      }
      // If the new element option is null, it means to remove the old
      // element. But we cannot really remove the element from the group
      // directly, because the element order may not be stable when this
      // element is added back. So we set the element to be ignored.
      oldChild.ignore = true;
    }
  }
  for (var i = el.childCount() - 1; i >= index; i--) {
    var child = el.childAt(i);
    removeChildFromGroup(el, child, seriesModel);
  }
}
function removeChildFromGroup(group, child, seriesModel) {
  // Do not support leave elements that are not mentioned in the latest
  // `renderItem` return. Otherwise users may not have a clear and simple
  // concept that how to control all of the elements.
  child && applyLeaveTransition(child, customInnerStore(group).option, seriesModel);
}
function diffGroupChildren(context) {
  new DataDiffer(context.oldChildren, context.newChildren, getKey, getKey, context).add(processAddUpdate).update(processAddUpdate).remove(processRemove).execute();
}
function getKey(item, idx) {
  var name = item && item.name;
  return name != null ? name : GROUP_DIFF_PREFIX + idx;
}
function processAddUpdate(newIndex, oldIndex) {
  var context = this.context;
  var childOption = newIndex != null ? context.newChildren[newIndex] : null;
  var child = oldIndex != null ? context.oldChildren[oldIndex] : null;
  doCreateOrUpdateEl(context.api, child, context.dataIndex, childOption, context.seriesModel, context.group);
}
function processRemove(oldIndex) {
  var context = this.context;
  var child = context.oldChildren[oldIndex];
  child && applyLeaveTransition(child, customInnerStore(child).option, context.seriesModel);
}
/**
 * @return SVG Path data.
 */
function getPathData(shape) {
  // "d" follows the SVG convention.
  return shape && (shape.pathData || shape.d);
}
function hasOwnPathData(shape) {
  return shape && (hasOwn(shape, 'pathData') || hasOwn(shape, 'd'));
}