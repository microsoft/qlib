
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
import BoundingRect from 'zrender/lib/core/BoundingRect.js';
import * as matrix from 'zrender/lib/core/matrix.js';
import * as graphic from '../../util/graphic.js';
import { createTextStyle } from '../../label/labelStyle.js';
import * as layout from '../../util/layout.js';
import TimelineView from './TimelineView.js';
import TimelineAxis from './TimelineAxis.js';
import { createSymbol, normalizeSymbolOffset, normalizeSymbolSize } from '../../util/symbol.js';
import * as numberUtil from '../../util/number.js';
import { merge, each, extend, isString, bind, defaults, retrieve2 } from 'zrender/lib/core/util.js';
import OrdinalScale from '../../scale/Ordinal.js';
import TimeScale from '../../scale/Time.js';
import IntervalScale from '../../scale/Interval.js';
import { parsePercent } from 'zrender/lib/contain/text.js';
import { makeInner } from '../../util/model.js';
import { getECData } from '../../util/innerStore.js';
import { enableHoverEmphasis } from '../../util/states.js';
import { createTooltipMarkup } from '../tooltip/tooltipMarkup.js';
var PI = Math.PI;
var labelDataIndexStore = makeInner();
var SliderTimelineView = /** @class */function (_super) {
  __extends(SliderTimelineView, _super);
  function SliderTimelineView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SliderTimelineView.type;
    return _this;
  }
  SliderTimelineView.prototype.init = function (ecModel, api) {
    this.api = api;
  };
  /**
   * @override
   */
  SliderTimelineView.prototype.render = function (timelineModel, ecModel, api) {
    this.model = timelineModel;
    this.api = api;
    this.ecModel = ecModel;
    this.group.removeAll();
    if (timelineModel.get('show', true)) {
      var layoutInfo_1 = this._layout(timelineModel, api);
      var mainGroup_1 = this._createGroup('_mainGroup');
      var labelGroup = this._createGroup('_labelGroup');
      var axis_1 = this._axis = this._createAxis(layoutInfo_1, timelineModel);
      timelineModel.formatTooltip = function (dataIndex) {
        var name = axis_1.scale.getLabel({
          value: dataIndex
        });
        return createTooltipMarkup('nameValue', {
          noName: true,
          value: name
        });
      };
      each(['AxisLine', 'AxisTick', 'Control', 'CurrentPointer'], function (name) {
        this['_render' + name](layoutInfo_1, mainGroup_1, axis_1, timelineModel);
      }, this);
      this._renderAxisLabel(layoutInfo_1, labelGroup, axis_1, timelineModel);
      this._position(layoutInfo_1, timelineModel);
    }
    this._doPlayStop();
    this._updateTicksStatus();
  };
  /**
   * @override
   */
  SliderTimelineView.prototype.remove = function () {
    this._clearTimer();
    this.group.removeAll();
  };
  /**
   * @override
   */
  SliderTimelineView.prototype.dispose = function () {
    this._clearTimer();
  };
  SliderTimelineView.prototype._layout = function (timelineModel, api) {
    var labelPosOpt = timelineModel.get(['label', 'position']);
    var orient = timelineModel.get('orient');
    var viewRect = getViewRect(timelineModel, api);
    var parsedLabelPos;
    // Auto label offset.
    if (labelPosOpt == null || labelPosOpt === 'auto') {
      parsedLabelPos = orient === 'horizontal' ? viewRect.y + viewRect.height / 2 < api.getHeight() / 2 ? '-' : '+' : viewRect.x + viewRect.width / 2 < api.getWidth() / 2 ? '+' : '-';
    } else if (isString(labelPosOpt)) {
      parsedLabelPos = {
        horizontal: {
          top: '-',
          bottom: '+'
        },
        vertical: {
          left: '-',
          right: '+'
        }
      }[orient][labelPosOpt];
    } else {
      // is number
      parsedLabelPos = labelPosOpt;
    }
    var labelAlignMap = {
      horizontal: 'center',
      vertical: parsedLabelPos >= 0 || parsedLabelPos === '+' ? 'left' : 'right'
    };
    var labelBaselineMap = {
      horizontal: parsedLabelPos >= 0 || parsedLabelPos === '+' ? 'top' : 'bottom',
      vertical: 'middle'
    };
    var rotationMap = {
      horizontal: 0,
      vertical: PI / 2
    };
    // Position
    var mainLength = orient === 'vertical' ? viewRect.height : viewRect.width;
    var controlModel = timelineModel.getModel('controlStyle');
    var showControl = controlModel.get('show', true);
    var controlSize = showControl ? controlModel.get('itemSize') : 0;
    var controlGap = showControl ? controlModel.get('itemGap') : 0;
    var sizePlusGap = controlSize + controlGap;
    // Special label rotate.
    var labelRotation = timelineModel.get(['label', 'rotate']) || 0;
    labelRotation = labelRotation * PI / 180; // To radian.
    var playPosition;
    var prevBtnPosition;
    var nextBtnPosition;
    var controlPosition = controlModel.get('position', true);
    var showPlayBtn = showControl && controlModel.get('showPlayBtn', true);
    var showPrevBtn = showControl && controlModel.get('showPrevBtn', true);
    var showNextBtn = showControl && controlModel.get('showNextBtn', true);
    var xLeft = 0;
    var xRight = mainLength;
    // position[0] means left, position[1] means middle.
    if (controlPosition === 'left' || controlPosition === 'bottom') {
      showPlayBtn && (playPosition = [0, 0], xLeft += sizePlusGap);
      showPrevBtn && (prevBtnPosition = [xLeft, 0], xLeft += sizePlusGap);
      showNextBtn && (nextBtnPosition = [xRight - controlSize, 0], xRight -= sizePlusGap);
    } else {
      // 'top' 'right'
      showPlayBtn && (playPosition = [xRight - controlSize, 0], xRight -= sizePlusGap);
      showPrevBtn && (prevBtnPosition = [0, 0], xLeft += sizePlusGap);
      showNextBtn && (nextBtnPosition = [xRight - controlSize, 0], xRight -= sizePlusGap);
    }
    var axisExtent = [xLeft, xRight];
    if (timelineModel.get('inverse')) {
      axisExtent.reverse();
    }
    return {
      viewRect: viewRect,
      mainLength: mainLength,
      orient: orient,
      rotation: rotationMap[orient],
      labelRotation: labelRotation,
      labelPosOpt: parsedLabelPos,
      labelAlign: timelineModel.get(['label', 'align']) || labelAlignMap[orient],
      labelBaseline: timelineModel.get(['label', 'verticalAlign']) || timelineModel.get(['label', 'baseline']) || labelBaselineMap[orient],
      // Based on mainGroup.
      playPosition: playPosition,
      prevBtnPosition: prevBtnPosition,
      nextBtnPosition: nextBtnPosition,
      axisExtent: axisExtent,
      controlSize: controlSize,
      controlGap: controlGap
    };
  };
  SliderTimelineView.prototype._position = function (layoutInfo, timelineModel) {
    // Position is be called finally, because bounding rect is needed for
    // adapt content to fill viewRect (auto adapt offset).
    // Timeline may be not all in the viewRect when 'offset' is specified
    // as a number, because it is more appropriate that label aligns at
    // 'offset' but not the other edge defined by viewRect.
    var mainGroup = this._mainGroup;
    var labelGroup = this._labelGroup;
    var viewRect = layoutInfo.viewRect;
    if (layoutInfo.orient === 'vertical') {
      // transform to horizontal, inverse rotate by left-top point.
      var m = matrix.create();
      var rotateOriginX = viewRect.x;
      var rotateOriginY = viewRect.y + viewRect.height;
      matrix.translate(m, m, [-rotateOriginX, -rotateOriginY]);
      matrix.rotate(m, m, -PI / 2);
      matrix.translate(m, m, [rotateOriginX, rotateOriginY]);
      viewRect = viewRect.clone();
      viewRect.applyTransform(m);
    }
    var viewBound = getBound(viewRect);
    var mainBound = getBound(mainGroup.getBoundingRect());
    var labelBound = getBound(labelGroup.getBoundingRect());
    var mainPosition = [mainGroup.x, mainGroup.y];
    var labelsPosition = [labelGroup.x, labelGroup.y];
    labelsPosition[0] = mainPosition[0] = viewBound[0][0];
    var labelPosOpt = layoutInfo.labelPosOpt;
    if (labelPosOpt == null || isString(labelPosOpt)) {
      // '+' or '-'
      var mainBoundIdx = labelPosOpt === '+' ? 0 : 1;
      toBound(mainPosition, mainBound, viewBound, 1, mainBoundIdx);
      toBound(labelsPosition, labelBound, viewBound, 1, 1 - mainBoundIdx);
    } else {
      var mainBoundIdx = labelPosOpt >= 0 ? 0 : 1;
      toBound(mainPosition, mainBound, viewBound, 1, mainBoundIdx);
      labelsPosition[1] = mainPosition[1] + labelPosOpt;
    }
    mainGroup.setPosition(mainPosition);
    labelGroup.setPosition(labelsPosition);
    mainGroup.rotation = labelGroup.rotation = layoutInfo.rotation;
    setOrigin(mainGroup);
    setOrigin(labelGroup);
    function setOrigin(targetGroup) {
      targetGroup.originX = viewBound[0][0] - targetGroup.x;
      targetGroup.originY = viewBound[1][0] - targetGroup.y;
    }
    function getBound(rect) {
      // [[xmin, xmax], [ymin, ymax]]
      return [[rect.x, rect.x + rect.width], [rect.y, rect.y + rect.height]];
    }
    function toBound(fromPos, from, to, dimIdx, boundIdx) {
      fromPos[dimIdx] += to[dimIdx][boundIdx] - from[dimIdx][boundIdx];
    }
  };
  SliderTimelineView.prototype._createAxis = function (layoutInfo, timelineModel) {
    var data = timelineModel.getData();
    var axisType = timelineModel.get('axisType');
    var scale = createScaleByModel(timelineModel, axisType);
    // Customize scale. The `tickValue` is `dataIndex`.
    scale.getTicks = function () {
      return data.mapArray(['value'], function (value) {
        return {
          value: value
        };
      });
    };
    var dataExtent = data.getDataExtent('value');
    scale.setExtent(dataExtent[0], dataExtent[1]);
    scale.calcNiceTicks();
    var axis = new TimelineAxis('value', scale, layoutInfo.axisExtent, axisType);
    axis.model = timelineModel;
    return axis;
  };
  SliderTimelineView.prototype._createGroup = function (key) {
    var newGroup = this[key] = new graphic.Group();
    this.group.add(newGroup);
    return newGroup;
  };
  SliderTimelineView.prototype._renderAxisLine = function (layoutInfo, group, axis, timelineModel) {
    var axisExtent = axis.getExtent();
    if (!timelineModel.get(['lineStyle', 'show'])) {
      return;
    }
    var line = new graphic.Line({
      shape: {
        x1: axisExtent[0],
        y1: 0,
        x2: axisExtent[1],
        y2: 0
      },
      style: extend({
        lineCap: 'round'
      }, timelineModel.getModel('lineStyle').getLineStyle()),
      silent: true,
      z2: 1
    });
    group.add(line);
    var progressLine = this._progressLine = new graphic.Line({
      shape: {
        x1: axisExtent[0],
        x2: this._currentPointer ? this._currentPointer.x : axisExtent[0],
        y1: 0,
        y2: 0
      },
      style: defaults({
        lineCap: 'round',
        lineWidth: line.style.lineWidth
      }, timelineModel.getModel(['progress', 'lineStyle']).getLineStyle()),
      silent: true,
      z2: 1
    });
    group.add(progressLine);
  };
  SliderTimelineView.prototype._renderAxisTick = function (layoutInfo, group, axis, timelineModel) {
    var _this = this;
    var data = timelineModel.getData();
    // Show all ticks, despite ignoring strategy.
    var ticks = axis.scale.getTicks();
    this._tickSymbols = [];
    // The value is dataIndex, see the customized scale.
    each(ticks, function (tick) {
      var tickCoord = axis.dataToCoord(tick.value);
      var itemModel = data.getItemModel(tick.value);
      var itemStyleModel = itemModel.getModel('itemStyle');
      var hoverStyleModel = itemModel.getModel(['emphasis', 'itemStyle']);
      var progressStyleModel = itemModel.getModel(['progress', 'itemStyle']);
      var symbolOpt = {
        x: tickCoord,
        y: 0,
        onclick: bind(_this._changeTimeline, _this, tick.value)
      };
      var el = giveSymbol(itemModel, itemStyleModel, group, symbolOpt);
      el.ensureState('emphasis').style = hoverStyleModel.getItemStyle();
      el.ensureState('progress').style = progressStyleModel.getItemStyle();
      enableHoverEmphasis(el);
      var ecData = getECData(el);
      if (itemModel.get('tooltip')) {
        ecData.dataIndex = tick.value;
        ecData.dataModel = timelineModel;
      } else {
        ecData.dataIndex = ecData.dataModel = null;
      }
      _this._tickSymbols.push(el);
    });
  };
  SliderTimelineView.prototype._renderAxisLabel = function (layoutInfo, group, axis, timelineModel) {
    var _this = this;
    var labelModel = axis.getLabelModel();
    if (!labelModel.get('show')) {
      return;
    }
    var data = timelineModel.getData();
    var labels = axis.getViewLabels();
    this._tickLabels = [];
    each(labels, function (labelItem) {
      // The tickValue is dataIndex, see the customized scale.
      var dataIndex = labelItem.tickValue;
      var itemModel = data.getItemModel(dataIndex);
      var normalLabelModel = itemModel.getModel('label');
      var hoverLabelModel = itemModel.getModel(['emphasis', 'label']);
      var progressLabelModel = itemModel.getModel(['progress', 'label']);
      var tickCoord = axis.dataToCoord(labelItem.tickValue);
      var textEl = new graphic.Text({
        x: tickCoord,
        y: 0,
        rotation: layoutInfo.labelRotation - layoutInfo.rotation,
        onclick: bind(_this._changeTimeline, _this, dataIndex),
        silent: false,
        style: createTextStyle(normalLabelModel, {
          text: labelItem.formattedLabel,
          align: layoutInfo.labelAlign,
          verticalAlign: layoutInfo.labelBaseline
        })
      });
      textEl.ensureState('emphasis').style = createTextStyle(hoverLabelModel);
      textEl.ensureState('progress').style = createTextStyle(progressLabelModel);
      group.add(textEl);
      enableHoverEmphasis(textEl);
      labelDataIndexStore(textEl).dataIndex = dataIndex;
      _this._tickLabels.push(textEl);
    });
  };
  SliderTimelineView.prototype._renderControl = function (layoutInfo, group, axis, timelineModel) {
    var controlSize = layoutInfo.controlSize;
    var rotation = layoutInfo.rotation;
    var itemStyle = timelineModel.getModel('controlStyle').getItemStyle();
    var hoverStyle = timelineModel.getModel(['emphasis', 'controlStyle']).getItemStyle();
    var playState = timelineModel.getPlayState();
    var inverse = timelineModel.get('inverse', true);
    makeBtn(layoutInfo.nextBtnPosition, 'next', bind(this._changeTimeline, this, inverse ? '-' : '+'));
    makeBtn(layoutInfo.prevBtnPosition, 'prev', bind(this._changeTimeline, this, inverse ? '+' : '-'));
    makeBtn(layoutInfo.playPosition, playState ? 'stop' : 'play', bind(this._handlePlayClick, this, !playState), true);
    function makeBtn(position, iconName, onclick, willRotate) {
      if (!position) {
        return;
      }
      var iconSize = parsePercent(retrieve2(timelineModel.get(['controlStyle', iconName + 'BtnSize']), controlSize), controlSize);
      var rect = [0, -iconSize / 2, iconSize, iconSize];
      var btn = makeControlIcon(timelineModel, iconName + 'Icon', rect, {
        x: position[0],
        y: position[1],
        originX: controlSize / 2,
        originY: 0,
        rotation: willRotate ? -rotation : 0,
        rectHover: true,
        style: itemStyle,
        onclick: onclick
      });
      btn.ensureState('emphasis').style = hoverStyle;
      group.add(btn);
      enableHoverEmphasis(btn);
    }
  };
  SliderTimelineView.prototype._renderCurrentPointer = function (layoutInfo, group, axis, timelineModel) {
    var data = timelineModel.getData();
    var currentIndex = timelineModel.getCurrentIndex();
    var pointerModel = data.getItemModel(currentIndex).getModel('checkpointStyle');
    var me = this;
    var callback = {
      onCreate: function (pointer) {
        pointer.draggable = true;
        pointer.drift = bind(me._handlePointerDrag, me);
        pointer.ondragend = bind(me._handlePointerDragend, me);
        pointerMoveTo(pointer, me._progressLine, currentIndex, axis, timelineModel, true);
      },
      onUpdate: function (pointer) {
        pointerMoveTo(pointer, me._progressLine, currentIndex, axis, timelineModel);
      }
    };
    // Reuse when exists, for animation and drag.
    this._currentPointer = giveSymbol(pointerModel, pointerModel, this._mainGroup, {}, this._currentPointer, callback);
  };
  SliderTimelineView.prototype._handlePlayClick = function (nextState) {
    this._clearTimer();
    this.api.dispatchAction({
      type: 'timelinePlayChange',
      playState: nextState,
      from: this.uid
    });
  };
  SliderTimelineView.prototype._handlePointerDrag = function (dx, dy, e) {
    this._clearTimer();
    this._pointerChangeTimeline([e.offsetX, e.offsetY]);
  };
  SliderTimelineView.prototype._handlePointerDragend = function (e) {
    this._pointerChangeTimeline([e.offsetX, e.offsetY], true);
  };
  SliderTimelineView.prototype._pointerChangeTimeline = function (mousePos, trigger) {
    var toCoord = this._toAxisCoord(mousePos)[0];
    var axis = this._axis;
    var axisExtent = numberUtil.asc(axis.getExtent().slice());
    toCoord > axisExtent[1] && (toCoord = axisExtent[1]);
    toCoord < axisExtent[0] && (toCoord = axisExtent[0]);
    this._currentPointer.x = toCoord;
    this._currentPointer.markRedraw();
    var progressLine = this._progressLine;
    if (progressLine) {
      progressLine.shape.x2 = toCoord;
      progressLine.dirty();
    }
    var targetDataIndex = this._findNearestTick(toCoord);
    var timelineModel = this.model;
    if (trigger || targetDataIndex !== timelineModel.getCurrentIndex() && timelineModel.get('realtime')) {
      this._changeTimeline(targetDataIndex);
    }
  };
  SliderTimelineView.prototype._doPlayStop = function () {
    var _this = this;
    this._clearTimer();
    if (this.model.getPlayState()) {
      this._timer = setTimeout(function () {
        // Do not cache
        var timelineModel = _this.model;
        _this._changeTimeline(timelineModel.getCurrentIndex() + (timelineModel.get('rewind', true) ? -1 : 1));
      }, this.model.get('playInterval'));
    }
  };
  SliderTimelineView.prototype._toAxisCoord = function (vertex) {
    var trans = this._mainGroup.getLocalTransform();
    return graphic.applyTransform(vertex, trans, true);
  };
  SliderTimelineView.prototype._findNearestTick = function (axisCoord) {
    var data = this.model.getData();
    var dist = Infinity;
    var targetDataIndex;
    var axis = this._axis;
    data.each(['value'], function (value, dataIndex) {
      var coord = axis.dataToCoord(value);
      var d = Math.abs(coord - axisCoord);
      if (d < dist) {
        dist = d;
        targetDataIndex = dataIndex;
      }
    });
    return targetDataIndex;
  };
  SliderTimelineView.prototype._clearTimer = function () {
    if (this._timer) {
      clearTimeout(this._timer);
      this._timer = null;
    }
  };
  SliderTimelineView.prototype._changeTimeline = function (nextIndex) {
    var currentIndex = this.model.getCurrentIndex();
    if (nextIndex === '+') {
      nextIndex = currentIndex + 1;
    } else if (nextIndex === '-') {
      nextIndex = currentIndex - 1;
    }
    this.api.dispatchAction({
      type: 'timelineChange',
      currentIndex: nextIndex,
      from: this.uid
    });
  };
  SliderTimelineView.prototype._updateTicksStatus = function () {
    var currentIndex = this.model.getCurrentIndex();
    var tickSymbols = this._tickSymbols;
    var tickLabels = this._tickLabels;
    if (tickSymbols) {
      for (var i = 0; i < tickSymbols.length; i++) {
        tickSymbols && tickSymbols[i] && tickSymbols[i].toggleState('progress', i < currentIndex);
      }
    }
    if (tickLabels) {
      for (var i = 0; i < tickLabels.length; i++) {
        tickLabels && tickLabels[i] && tickLabels[i].toggleState('progress', labelDataIndexStore(tickLabels[i]).dataIndex <= currentIndex);
      }
    }
  };
  SliderTimelineView.type = 'timeline.slider';
  return SliderTimelineView;
}(TimelineView);
function createScaleByModel(model, axisType) {
  axisType = axisType || model.get('type');
  if (axisType) {
    switch (axisType) {
      // Buildin scale
      case 'category':
        return new OrdinalScale({
          ordinalMeta: model.getCategories(),
          extent: [Infinity, -Infinity]
        });
      case 'time':
        return new TimeScale({
          locale: model.ecModel.getLocaleModel(),
          useUTC: model.ecModel.get('useUTC')
        });
      default:
        // default to be value
        return new IntervalScale();
    }
  }
}
function getViewRect(model, api) {
  return layout.getLayoutRect(model.getBoxLayoutParams(), {
    width: api.getWidth(),
    height: api.getHeight()
  }, model.get('padding'));
}
function makeControlIcon(timelineModel, objPath, rect, opts) {
  var style = opts.style;
  var icon = graphic.createIcon(timelineModel.get(['controlStyle', objPath]), opts || {}, new BoundingRect(rect[0], rect[1], rect[2], rect[3]));
  // TODO createIcon won't use style in opt.
  if (style) {
    icon.setStyle(style);
  }
  return icon;
}
/**
 * Create symbol or update symbol
 * opt: basic position and event handlers
 */
function giveSymbol(hostModel, itemStyleModel, group, opt, symbol, callback) {
  var color = itemStyleModel.get('color');
  if (!symbol) {
    var symbolType = hostModel.get('symbol');
    symbol = createSymbol(symbolType, -1, -1, 2, 2, color);
    symbol.setStyle('strokeNoScale', true);
    group.add(symbol);
    callback && callback.onCreate(symbol);
  } else {
    symbol.setColor(color);
    group.add(symbol); // Group may be new, also need to add.
    callback && callback.onUpdate(symbol);
  }
  // Style
  var itemStyle = itemStyleModel.getItemStyle(['color']);
  symbol.setStyle(itemStyle);
  // Transform and events.
  opt = merge({
    rectHover: true,
    z2: 100
  }, opt, true);
  var symbolSize = normalizeSymbolSize(hostModel.get('symbolSize'));
  opt.scaleX = symbolSize[0] / 2;
  opt.scaleY = symbolSize[1] / 2;
  var symbolOffset = normalizeSymbolOffset(hostModel.get('symbolOffset'), symbolSize);
  if (symbolOffset) {
    opt.x = (opt.x || 0) + symbolOffset[0];
    opt.y = (opt.y || 0) + symbolOffset[1];
  }
  var symbolRotate = hostModel.get('symbolRotate');
  opt.rotation = (symbolRotate || 0) * Math.PI / 180 || 0;
  symbol.attr(opt);
  // FIXME
  // (1) When symbol.style.strokeNoScale is true and updateTransform is not performed,
  // getBoundingRect will return wrong result.
  // (This is supposed to be resolved in zrender, but it is a little difficult to
  // leverage performance and auto updateTransform)
  // (2) All of ancesters of symbol do not scale, so we can just updateTransform symbol.
  symbol.updateTransform();
  return symbol;
}
function pointerMoveTo(pointer, progressLine, dataIndex, axis, timelineModel, noAnimation) {
  if (pointer.dragging) {
    return;
  }
  var pointerModel = timelineModel.getModel('checkpointStyle');
  var toCoord = axis.dataToCoord(timelineModel.getData().get('value', dataIndex));
  if (noAnimation || !pointerModel.get('animation', true)) {
    pointer.attr({
      x: toCoord,
      y: 0
    });
    progressLine && progressLine.attr({
      shape: {
        x2: toCoord
      }
    });
  } else {
    var animationCfg = {
      duration: pointerModel.get('animationDuration', true),
      easing: pointerModel.get('animationEasing', true)
    };
    pointer.stopAnimation(null, true);
    pointer.animateTo({
      x: toCoord,
      y: 0
    }, animationCfg);
    progressLine && progressLine.animateTo({
      shape: {
        x2: toCoord
      }
    }, animationCfg);
  }
}
export default SliderTimelineView;