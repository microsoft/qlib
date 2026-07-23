
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
import { extend, indexOf, isArrayLike, isObject, keys, isArray, each } from 'zrender/lib/core/util.js';
import { getECData } from './innerStore.js';
import { liftColor } from 'zrender/lib/tool/color.js';
import { queryDataIndex, makeInner } from './model.js';
import Path from 'zrender/lib/graphic/Path.js';
import { error } from './log.js';
// Reserve 0 as default.
var _highlightNextDigit = 1;
var _highlightKeyMap = {};
var getSavedStates = makeInner();
var getComponentStates = makeInner();
export var HOVER_STATE_NORMAL = 0;
export var HOVER_STATE_BLUR = 1;
export var HOVER_STATE_EMPHASIS = 2;
export var SPECIAL_STATES = ['emphasis', 'blur', 'select'];
export var DISPLAY_STATES = ['normal', 'emphasis', 'blur', 'select'];
export var Z2_EMPHASIS_LIFT = 10;
export var Z2_SELECT_LIFT = 9;
export var HIGHLIGHT_ACTION_TYPE = 'highlight';
export var DOWNPLAY_ACTION_TYPE = 'downplay';
export var SELECT_ACTION_TYPE = 'select';
export var UNSELECT_ACTION_TYPE = 'unselect';
export var TOGGLE_SELECT_ACTION_TYPE = 'toggleSelect';
function hasFillOrStroke(fillOrStroke) {
  return fillOrStroke != null && fillOrStroke !== 'none';
}
function doChangeHoverState(el, stateName, hoverStateEnum) {
  if (el.onHoverStateChange && (el.hoverState || 0) !== hoverStateEnum) {
    el.onHoverStateChange(stateName);
  }
  el.hoverState = hoverStateEnum;
}
function singleEnterEmphasis(el) {
  // Only mark the flag.
  // States will be applied in the echarts.ts in next frame.
  doChangeHoverState(el, 'emphasis', HOVER_STATE_EMPHASIS);
}
function singleLeaveEmphasis(el) {
  // Only mark the flag.
  // States will be applied in the echarts.ts in next frame.
  if (el.hoverState === HOVER_STATE_EMPHASIS) {
    doChangeHoverState(el, 'normal', HOVER_STATE_NORMAL);
  }
}
function singleEnterBlur(el) {
  doChangeHoverState(el, 'blur', HOVER_STATE_BLUR);
}
function singleLeaveBlur(el) {
  if (el.hoverState === HOVER_STATE_BLUR) {
    doChangeHoverState(el, 'normal', HOVER_STATE_NORMAL);
  }
}
function singleEnterSelect(el) {
  el.selected = true;
}
function singleLeaveSelect(el) {
  el.selected = false;
}
function updateElementState(el, updater, commonParam) {
  updater(el, commonParam);
}
function traverseUpdateState(el, updater, commonParam) {
  updateElementState(el, updater, commonParam);
  el.isGroup && el.traverse(function (child) {
    updateElementState(child, updater, commonParam);
  });
}
export function setStatesFlag(el, stateName) {
  switch (stateName) {
    case 'emphasis':
      el.hoverState = HOVER_STATE_EMPHASIS;
      break;
    case 'normal':
      el.hoverState = HOVER_STATE_NORMAL;
      break;
    case 'blur':
      el.hoverState = HOVER_STATE_BLUR;
      break;
    case 'select':
      el.selected = true;
  }
}
/**
 * If we reuse elements when rerender.
 * DON'T forget to clearStates before we update the style and shape.
 * Or we may update on the wrong state instead of normal state.
 */
export function clearStates(el) {
  if (el.isGroup) {
    el.traverse(function (child) {
      child.clearStates();
    });
  } else {
    el.clearStates();
  }
}
function getFromStateStyle(el, props, toStateName, defaultValue) {
  var style = el.style;
  var fromState = {};
  for (var i = 0; i < props.length; i++) {
    var propName = props[i];
    var val = style[propName];
    fromState[propName] = val == null ? defaultValue && defaultValue[propName] : val;
  }
  for (var i = 0; i < el.animators.length; i++) {
    var animator = el.animators[i];
    if (animator.__fromStateTransition
    // Don't consider the animation to emphasis state.
    && animator.__fromStateTransition.indexOf(toStateName) < 0 && animator.targetName === 'style') {
      animator.saveTo(fromState, props);
    }
  }
  return fromState;
}
function createEmphasisDefaultState(el, stateName, targetStates, state) {
  var hasSelect = targetStates && indexOf(targetStates, 'select') >= 0;
  var cloned = false;
  if (el instanceof Path) {
    var store = getSavedStates(el);
    var fromFill = hasSelect ? store.selectFill || store.normalFill : store.normalFill;
    var fromStroke = hasSelect ? store.selectStroke || store.normalStroke : store.normalStroke;
    if (hasFillOrStroke(fromFill) || hasFillOrStroke(fromStroke)) {
      state = state || {};
      var emphasisStyle = state.style || {};
      // inherit case
      if (emphasisStyle.fill === 'inherit') {
        cloned = true;
        state = extend({}, state);
        emphasisStyle = extend({}, emphasisStyle);
        emphasisStyle.fill = fromFill;
      }
      // Apply default color lift
      else if (!hasFillOrStroke(emphasisStyle.fill) && hasFillOrStroke(fromFill)) {
        cloned = true;
        // Not modify the original value.
        state = extend({}, state);
        emphasisStyle = extend({}, emphasisStyle);
        // Already being applied 'emphasis'. DON'T lift color multiple times.
        emphasisStyle.fill = liftColor(fromFill);
      }
      // Not highlight stroke if fill has been highlighted.
      else if (!hasFillOrStroke(emphasisStyle.stroke) && hasFillOrStroke(fromStroke)) {
        if (!cloned) {
          state = extend({}, state);
          emphasisStyle = extend({}, emphasisStyle);
        }
        emphasisStyle.stroke = liftColor(fromStroke);
      }
      state.style = emphasisStyle;
    }
  }
  if (state) {
    // TODO Share with textContent?
    if (state.z2 == null) {
      if (!cloned) {
        state = extend({}, state);
      }
      var z2EmphasisLift = el.z2EmphasisLift;
      state.z2 = el.z2 + (z2EmphasisLift != null ? z2EmphasisLift : Z2_EMPHASIS_LIFT);
    }
  }
  return state;
}
function createSelectDefaultState(el, stateName, state) {
  // const hasSelect = indexOf(el.currentStates, stateName) >= 0;
  if (state) {
    // TODO Share with textContent?
    if (state.z2 == null) {
      state = extend({}, state);
      var z2SelectLift = el.z2SelectLift;
      state.z2 = el.z2 + (z2SelectLift != null ? z2SelectLift : Z2_SELECT_LIFT);
    }
  }
  return state;
}
function createBlurDefaultState(el, stateName, state) {
  var hasBlur = indexOf(el.currentStates, stateName) >= 0;
  var currentOpacity = el.style.opacity;
  var fromState = !hasBlur ? getFromStateStyle(el, ['opacity'], stateName, {
    opacity: 1
  }) : null;
  state = state || {};
  var blurStyle = state.style || {};
  if (blurStyle.opacity == null) {
    // clone state
    state = extend({}, state);
    blurStyle = extend({
      // Already being applied 'emphasis'. DON'T mul opacity multiple times.
      opacity: hasBlur ? currentOpacity : fromState.opacity * 0.1
    }, blurStyle);
    state.style = blurStyle;
  }
  return state;
}
function elementStateProxy(stateName, targetStates) {
  var state = this.states[stateName];
  if (this.style) {
    if (stateName === 'emphasis') {
      return createEmphasisDefaultState(this, stateName, targetStates, state);
    } else if (stateName === 'blur') {
      return createBlurDefaultState(this, stateName, state);
    } else if (stateName === 'select') {
      return createSelectDefaultState(this, stateName, state);
    }
  }
  return state;
}
/**
 * Set hover style (namely "emphasis style") of element.
 * @param el Should not be `zrender/graphic/Group`.
 * @param focus 'self' | 'selfInSeries' | 'series'
 */
export function setDefaultStateProxy(el) {
  el.stateProxy = elementStateProxy;
  var textContent = el.getTextContent();
  var textGuide = el.getTextGuideLine();
  if (textContent) {
    textContent.stateProxy = elementStateProxy;
  }
  if (textGuide) {
    textGuide.stateProxy = elementStateProxy;
  }
}
export function enterEmphasisWhenMouseOver(el, e) {
  !shouldSilent(el, e)
  // "emphasis" event highlight has higher priority than mouse highlight.
  && !el.__highByOuter && traverseUpdateState(el, singleEnterEmphasis);
}
export function leaveEmphasisWhenMouseOut(el, e) {
  !shouldSilent(el, e)
  // "emphasis" event highlight has higher priority than mouse highlight.
  && !el.__highByOuter && traverseUpdateState(el, singleLeaveEmphasis);
}
export function enterEmphasis(el, highlightDigit) {
  el.__highByOuter |= 1 << (highlightDigit || 0);
  traverseUpdateState(el, singleEnterEmphasis);
}
export function leaveEmphasis(el, highlightDigit) {
  !(el.__highByOuter &= ~(1 << (highlightDigit || 0))) && traverseUpdateState(el, singleLeaveEmphasis);
}
export function enterBlur(el) {
  traverseUpdateState(el, singleEnterBlur);
}
export function leaveBlur(el) {
  traverseUpdateState(el, singleLeaveBlur);
}
export function enterSelect(el) {
  traverseUpdateState(el, singleEnterSelect);
}
export function leaveSelect(el) {
  traverseUpdateState(el, singleLeaveSelect);
}
function shouldSilent(el, e) {
  return el.__highDownSilentOnTouch && e.zrByTouch;
}
export function allLeaveBlur(api) {
  var model = api.getModel();
  var leaveBlurredSeries = [];
  var allComponentViews = [];
  model.eachComponent(function (componentType, componentModel) {
    var componentStates = getComponentStates(componentModel);
    var isSeries = componentType === 'series';
    var view = isSeries ? api.getViewOfSeriesModel(componentModel) : api.getViewOfComponentModel(componentModel);
    !isSeries && allComponentViews.push(view);
    if (componentStates.isBlured) {
      // Leave blur anyway
      view.group.traverse(function (child) {
        singleLeaveBlur(child);
      });
      isSeries && leaveBlurredSeries.push(componentModel);
    }
    componentStates.isBlured = false;
  });
  each(allComponentViews, function (view) {
    if (view && view.toggleBlurSeries) {
      view.toggleBlurSeries(leaveBlurredSeries, false, model);
    }
  });
}
export function blurSeries(targetSeriesIndex, focus, blurScope, api) {
  var ecModel = api.getModel();
  blurScope = blurScope || 'coordinateSystem';
  function leaveBlurOfIndices(data, dataIndices) {
    for (var i = 0; i < dataIndices.length; i++) {
      var itemEl = data.getItemGraphicEl(dataIndices[i]);
      itemEl && leaveBlur(itemEl);
    }
  }
  if (targetSeriesIndex == null) {
    return;
  }
  if (!focus || focus === 'none') {
    return;
  }
  var targetSeriesModel = ecModel.getSeriesByIndex(targetSeriesIndex);
  var targetCoordSys = targetSeriesModel.coordinateSystem;
  if (targetCoordSys && targetCoordSys.master) {
    targetCoordSys = targetCoordSys.master;
  }
  var blurredSeries = [];
  ecModel.eachSeries(function (seriesModel) {
    var sameSeries = targetSeriesModel === seriesModel;
    var coordSys = seriesModel.coordinateSystem;
    if (coordSys && coordSys.master) {
      coordSys = coordSys.master;
    }
    var sameCoordSys = coordSys && targetCoordSys ? coordSys === targetCoordSys : sameSeries; // If there is no coordinate system. use sameSeries instead.
    if (!(
    // Not blur other series if blurScope series
    blurScope === 'series' && !sameSeries
    // Not blur other coordinate system if blurScope is coordinateSystem
    || blurScope === 'coordinateSystem' && !sameCoordSys
    // Not blur self series if focus is series.
    || focus === 'series' && sameSeries
    // TODO blurScope: coordinate system
    )) {
      var view = api.getViewOfSeriesModel(seriesModel);
      view.group.traverse(function (child) {
        // For the elements that have been triggered by other components,
        // and are still required to be highlighted,
        // because the current is directly forced to blur the element,
        // it will cause the focus self to be unable to highlight, so skip the blur of this element.
        if (child.__highByOuter && sameSeries && focus === 'self') {
          return;
        }
        singleEnterBlur(child);
      });
      if (isArrayLike(focus)) {
        leaveBlurOfIndices(seriesModel.getData(), focus);
      } else if (isObject(focus)) {
        var dataTypes = keys(focus);
        for (var d = 0; d < dataTypes.length; d++) {
          leaveBlurOfIndices(seriesModel.getData(dataTypes[d]), focus[dataTypes[d]]);
        }
      }
      blurredSeries.push(seriesModel);
      getComponentStates(seriesModel).isBlured = true;
    }
  });
  ecModel.eachComponent(function (componentType, componentModel) {
    if (componentType === 'series') {
      return;
    }
    var view = api.getViewOfComponentModel(componentModel);
    if (view && view.toggleBlurSeries) {
      view.toggleBlurSeries(blurredSeries, true, ecModel);
    }
  });
}
export function blurComponent(componentMainType, componentIndex, api) {
  if (componentMainType == null || componentIndex == null) {
    return;
  }
  var componentModel = api.getModel().getComponent(componentMainType, componentIndex);
  if (!componentModel) {
    return;
  }
  getComponentStates(componentModel).isBlured = true;
  var view = api.getViewOfComponentModel(componentModel);
  if (!view || !view.focusBlurEnabled) {
    return;
  }
  view.group.traverse(function (child) {
    singleEnterBlur(child);
  });
}
export function blurSeriesFromHighlightPayload(seriesModel, payload, api) {
  var seriesIndex = seriesModel.seriesIndex;
  var data = seriesModel.getData(payload.dataType);
  if (!data) {
    if (process.env.NODE_ENV !== 'production') {
      error("Unknown dataType " + payload.dataType);
    }
    return;
  }
  var dataIndex = queryDataIndex(data, payload);
  // Pick the first one if there is multiple/none exists.
  dataIndex = (isArray(dataIndex) ? dataIndex[0] : dataIndex) || 0;
  var el = data.getItemGraphicEl(dataIndex);
  if (!el) {
    var count = data.count();
    var current = 0;
    // If data on dataIndex is NaN.
    while (!el && current < count) {
      el = data.getItemGraphicEl(current++);
    }
  }
  if (el) {
    var ecData = getECData(el);
    blurSeries(seriesIndex, ecData.focus, ecData.blurScope, api);
  } else {
    // If there is no element put on the data. Try getting it from raw option
    // TODO Should put it on seriesModel?
    var focus_1 = seriesModel.get(['emphasis', 'focus']);
    var blurScope = seriesModel.get(['emphasis', 'blurScope']);
    if (focus_1 != null) {
      blurSeries(seriesIndex, focus_1, blurScope, api);
    }
  }
}
export function findComponentHighDownDispatchers(componentMainType, componentIndex, name, api) {
  var ret = {
    focusSelf: false,
    dispatchers: null
  };
  if (componentMainType == null || componentMainType === 'series' || componentIndex == null || name == null) {
    return ret;
  }
  var componentModel = api.getModel().getComponent(componentMainType, componentIndex);
  if (!componentModel) {
    return ret;
  }
  var view = api.getViewOfComponentModel(componentModel);
  if (!view || !view.findHighDownDispatchers) {
    return ret;
  }
  var dispatchers = view.findHighDownDispatchers(name);
  // At presnet, the component (like Geo) only blur inside itself.
  // So we do not use `blurScope` in component.
  var focusSelf;
  for (var i = 0; i < dispatchers.length; i++) {
    if (process.env.NODE_ENV !== 'production' && !isHighDownDispatcher(dispatchers[i])) {
      error('param should be highDownDispatcher');
    }
    if (getECData(dispatchers[i]).focus === 'self') {
      focusSelf = true;
      break;
    }
  }
  return {
    focusSelf: focusSelf,
    dispatchers: dispatchers
  };
}
export function handleGlobalMouseOverForHighDown(dispatcher, e, api) {
  if (process.env.NODE_ENV !== 'production' && !isHighDownDispatcher(dispatcher)) {
    error('param should be highDownDispatcher');
  }
  var ecData = getECData(dispatcher);
  var _a = findComponentHighDownDispatchers(ecData.componentMainType, ecData.componentIndex, ecData.componentHighDownName, api),
    dispatchers = _a.dispatchers,
    focusSelf = _a.focusSelf;
  // If `findHighDownDispatchers` is supported on the component,
  // highlight/downplay elements with the same name.
  if (dispatchers) {
    if (focusSelf) {
      blurComponent(ecData.componentMainType, ecData.componentIndex, api);
    }
    each(dispatchers, function (dispatcher) {
      return enterEmphasisWhenMouseOver(dispatcher, e);
    });
  } else {
    // Try blur all in the related series. Then emphasis the hoverred.
    // TODO. progressive mode.
    blurSeries(ecData.seriesIndex, ecData.focus, ecData.blurScope, api);
    if (ecData.focus === 'self') {
      blurComponent(ecData.componentMainType, ecData.componentIndex, api);
    }
    // Other than series, component that not support `findHighDownDispatcher` will
    // also use it. But in this case, highlight/downplay are only supported in
    // mouse hover but not in dispatchAction.
    enterEmphasisWhenMouseOver(dispatcher, e);
  }
}
export function handleGlobalMouseOutForHighDown(dispatcher, e, api) {
  if (process.env.NODE_ENV !== 'production' && !isHighDownDispatcher(dispatcher)) {
    error('param should be highDownDispatcher');
  }
  allLeaveBlur(api);
  var ecData = getECData(dispatcher);
  var dispatchers = findComponentHighDownDispatchers(ecData.componentMainType, ecData.componentIndex, ecData.componentHighDownName, api).dispatchers;
  if (dispatchers) {
    each(dispatchers, function (dispatcher) {
      return leaveEmphasisWhenMouseOut(dispatcher, e);
    });
  } else {
    leaveEmphasisWhenMouseOut(dispatcher, e);
  }
}
export function toggleSelectionFromPayload(seriesModel, payload, api) {
  if (!isSelectChangePayload(payload)) {
    return;
  }
  var dataType = payload.dataType;
  var data = seriesModel.getData(dataType);
  var dataIndex = queryDataIndex(data, payload);
  if (!isArray(dataIndex)) {
    dataIndex = [dataIndex];
  }
  seriesModel[payload.type === TOGGLE_SELECT_ACTION_TYPE ? 'toggleSelect' : payload.type === SELECT_ACTION_TYPE ? 'select' : 'unselect'](dataIndex, dataType);
}
export function updateSeriesElementSelection(seriesModel) {
  var allData = seriesModel.getAllData();
  each(allData, function (_a) {
    var data = _a.data,
      type = _a.type;
    data.eachItemGraphicEl(function (el, idx) {
      seriesModel.isSelected(idx, type) ? enterSelect(el) : leaveSelect(el);
    });
  });
}
export function getAllSelectedIndices(ecModel) {
  var ret = [];
  ecModel.eachSeries(function (seriesModel) {
    var allData = seriesModel.getAllData();
    each(allData, function (_a) {
      var data = _a.data,
        type = _a.type;
      var dataIndices = seriesModel.getSelectedDataIndices();
      if (dataIndices.length > 0) {
        var item = {
          dataIndex: dataIndices,
          seriesIndex: seriesModel.seriesIndex
        };
        if (type != null) {
          item.dataType = type;
        }
        ret.push(item);
      }
    });
  });
  return ret;
}
/**
 * Enable the function that mouseover will trigger the emphasis state.
 *
 * NOTE:
 * This function should be used on the element with dataIndex, seriesIndex.
 *
 */
export function enableHoverEmphasis(el, focus, blurScope) {
  setAsHighDownDispatcher(el, true);
  traverseUpdateState(el, setDefaultStateProxy);
  enableHoverFocus(el, focus, blurScope);
}
export function disableHoverEmphasis(el) {
  setAsHighDownDispatcher(el, false);
}
export function toggleHoverEmphasis(el, focus, blurScope, isDisabled) {
  isDisabled ? disableHoverEmphasis(el) : enableHoverEmphasis(el, focus, blurScope);
}
export function enableHoverFocus(el, focus, blurScope) {
  var ecData = getECData(el);
  if (focus != null) {
    // TODO dataIndex may be set after this function. This check is not useful.
    // if (ecData.dataIndex == null) {
    //     if (__DEV__) {
    //         console.warn('focus can only been set on element with dataIndex');
    //     }
    // }
    // else {
    ecData.focus = focus;
    ecData.blurScope = blurScope;
    // }
  } else if (ecData.focus) {
    ecData.focus = null;
  }
}
var OTHER_STATES = ['emphasis', 'blur', 'select'];
var defaultStyleGetterMap = {
  itemStyle: 'getItemStyle',
  lineStyle: 'getLineStyle',
  areaStyle: 'getAreaStyle'
};
/**
 * Set emphasis/blur/selected states of element.
 */
export function setStatesStylesFromModel(el, itemModel, styleType,
// default itemStyle
getter) {
  styleType = styleType || 'itemStyle';
  for (var i = 0; i < OTHER_STATES.length; i++) {
    var stateName = OTHER_STATES[i];
    var model = itemModel.getModel([stateName, styleType]);
    var state = el.ensureState(stateName);
    // Let it throw error if getterType is not found.
    state.style = getter ? getter(model) : model[defaultStyleGetterMap[styleType]]();
  }
}
/**
 *
 * Set element as highlight / downplay dispatcher.
 * It will be checked when element received mouseover event or from highlight action.
 * It's in change of all highlight/downplay behavior of it's children.
 *
 * @param el
 * @param el.highDownSilentOnTouch
 *        In touch device, mouseover event will be trigger on touchstart event
 *        (see module:zrender/dom/HandlerProxy). By this mechanism, we can
 *        conveniently use hoverStyle when tap on touch screen without additional
 *        code for compatibility.
 *        But if the chart/component has select feature, which usually also use
 *        hoverStyle, there might be conflict between 'select-highlight' and
 *        'hover-highlight' especially when roam is enabled (see geo for example).
 *        In this case, `highDownSilentOnTouch` should be used to disable
 *        hover-highlight on touch device.
 * @param asDispatcher If `false`, do not set as "highDownDispatcher".
 */
export function setAsHighDownDispatcher(el, asDispatcher) {
  var disable = asDispatcher === false;
  var extendedEl = el;
  // Make `highDownSilentOnTouch` and `onStateChange` only work after
  // `setAsHighDownDispatcher` called. Avoid it is modified by user unexpectedly.
  if (el.highDownSilentOnTouch) {
    extendedEl.__highDownSilentOnTouch = el.highDownSilentOnTouch;
  }
  // Simple optimize, since this method might be
  // called for each elements of a group in some cases.
  if (!disable || extendedEl.__highDownDispatcher) {
    // Emphasis, normal can be triggered manually by API or other components like hover link.
    // el[method]('emphasis', onElementEmphasisEvent)[method]('normal', onElementNormalEvent);
    // Also keep previous record.
    extendedEl.__highByOuter = extendedEl.__highByOuter || 0;
    extendedEl.__highDownDispatcher = !disable;
  }
}
export function isHighDownDispatcher(el) {
  return !!(el && el.__highDownDispatcher);
}
/**
 * Enable component highlight/downplay features:
 * + hover link (within the same name)
 * + focus blur in component
 */
export function enableComponentHighDownFeatures(el, componentModel, componentHighDownName) {
  var ecData = getECData(el);
  ecData.componentMainType = componentModel.mainType;
  ecData.componentIndex = componentModel.componentIndex;
  ecData.componentHighDownName = componentHighDownName;
}
/**
 * Support highlight/downplay record on each elements.
 * For the case: hover highlight/downplay (legend, visualMap, ...) and
 * user triggered highlight/downplay should not conflict.
 * Only all of the highlightDigit cleared, return to normal.
 * @param {string} highlightKey
 * @return {number} highlightDigit
 */
export function getHighlightDigit(highlightKey) {
  var highlightDigit = _highlightKeyMap[highlightKey];
  if (highlightDigit == null && _highlightNextDigit <= 32) {
    highlightDigit = _highlightKeyMap[highlightKey] = _highlightNextDigit++;
  }
  return highlightDigit;
}
export function isSelectChangePayload(payload) {
  var payloadType = payload.type;
  return payloadType === SELECT_ACTION_TYPE || payloadType === UNSELECT_ACTION_TYPE || payloadType === TOGGLE_SELECT_ACTION_TYPE;
}
export function isHighDownPayload(payload) {
  var payloadType = payload.type;
  return payloadType === HIGHLIGHT_ACTION_TYPE || payloadType === DOWNPLAY_ACTION_TYPE;
}
export function savePathStates(el) {
  var store = getSavedStates(el);
  store.normalFill = el.style.fill;
  store.normalStroke = el.style.stroke;
  var selectState = el.states.select || {};
  store.selectFill = selectState.style && selectState.style.fill || null;
  store.selectStroke = selectState.style && selectState.style.stroke || null;
}