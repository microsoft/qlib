
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
import LinearGradient from 'zrender/lib/graphic/LinearGradient.js';
import * as eventTool from 'zrender/lib/core/event.js';
import VisualMapView from './VisualMapView.js';
import * as graphic from '../../util/graphic.js';
import * as numberUtil from '../../util/number.js';
import sliderMove from '../helper/sliderMove.js';
import * as helper from './helper.js';
import * as modelUtil from '../../util/model.js';
import { parsePercent } from 'zrender/lib/contain/text.js';
import { setAsHighDownDispatcher } from '../../util/states.js';
import { createSymbol } from '../../util/symbol.js';
import ZRImage from 'zrender/lib/graphic/Image.js';
import { getECData } from '../../util/innerStore.js';
import { createTextStyle } from '../../label/labelStyle.js';
import { findEventDispatcher } from '../../util/event.js';
var linearMap = numberUtil.linearMap;
var each = zrUtil.each;
var mathMin = Math.min;
var mathMax = Math.max;
// Arbitrary value
var HOVER_LINK_SIZE = 12;
var HOVER_LINK_OUT = 6;
// Notice:
// Any "interval" should be by the order of [low, high].
// "handle0" (handleIndex === 0) maps to
// low data value: this._dataInterval[0] and has low coord.
// "handle1" (handleIndex === 1) maps to
// high data value: this._dataInterval[1] and has high coord.
// The logic of transform is implemented in this._createBarGroup.
var ContinuousView = /** @class */function (_super) {
  __extends(ContinuousView, _super);
  function ContinuousView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ContinuousView.type;
    _this._shapes = {};
    _this._dataInterval = [];
    _this._handleEnds = [];
    _this._hoverLinkDataIndices = [];
    return _this;
  }
  ContinuousView.prototype.init = function (ecModel, api) {
    _super.prototype.init.call(this, ecModel, api);
    this._hoverLinkFromSeriesMouseOver = zrUtil.bind(this._hoverLinkFromSeriesMouseOver, this);
    this._hideIndicator = zrUtil.bind(this._hideIndicator, this);
  };
  ContinuousView.prototype.doRender = function (visualMapModel, ecModel, api, payload) {
    if (!payload || payload.type !== 'selectDataRange' || payload.from !== this.uid) {
      this._buildView();
    }
  };
  ContinuousView.prototype._buildView = function () {
    this.group.removeAll();
    var visualMapModel = this.visualMapModel;
    var thisGroup = this.group;
    this._orient = visualMapModel.get('orient');
    this._useHandle = visualMapModel.get('calculable');
    this._resetInterval();
    this._renderBar(thisGroup);
    var dataRangeText = visualMapModel.get('text');
    this._renderEndsText(thisGroup, dataRangeText, 0);
    this._renderEndsText(thisGroup, dataRangeText, 1);
    // Do this for background size calculation.
    this._updateView(true);
    // After updating view, inner shapes is built completely,
    // and then background can be rendered.
    this.renderBackground(thisGroup);
    // Real update view
    this._updateView();
    this._enableHoverLinkToSeries();
    this._enableHoverLinkFromSeries();
    this.positionGroup(thisGroup);
  };
  ContinuousView.prototype._renderEndsText = function (group, dataRangeText, endsIndex) {
    if (!dataRangeText) {
      return;
    }
    // Compatible with ec2, text[0] map to high value, text[1] map low value.
    var text = dataRangeText[1 - endsIndex];
    text = text != null ? text + '' : '';
    var visualMapModel = this.visualMapModel;
    var textGap = visualMapModel.get('textGap');
    var itemSize = visualMapModel.itemSize;
    var barGroup = this._shapes.mainGroup;
    var position = this._applyTransform([itemSize[0] / 2, endsIndex === 0 ? -textGap : itemSize[1] + textGap], barGroup);
    var align = this._applyTransform(endsIndex === 0 ? 'bottom' : 'top', barGroup);
    var orient = this._orient;
    var textStyleModel = this.visualMapModel.textStyleModel;
    this.group.add(new graphic.Text({
      style: createTextStyle(textStyleModel, {
        x: position[0],
        y: position[1],
        verticalAlign: orient === 'horizontal' ? 'middle' : align,
        align: orient === 'horizontal' ? align : 'center',
        text: text
      })
    }));
  };
  ContinuousView.prototype._renderBar = function (targetGroup) {
    var visualMapModel = this.visualMapModel;
    var shapes = this._shapes;
    var itemSize = visualMapModel.itemSize;
    var orient = this._orient;
    var useHandle = this._useHandle;
    var itemAlign = helper.getItemAlign(visualMapModel, this.api, itemSize);
    var mainGroup = shapes.mainGroup = this._createBarGroup(itemAlign);
    var gradientBarGroup = new graphic.Group();
    mainGroup.add(gradientBarGroup);
    // Bar
    gradientBarGroup.add(shapes.outOfRange = createPolygon());
    gradientBarGroup.add(shapes.inRange = createPolygon(null, useHandle ? getCursor(this._orient) : null, zrUtil.bind(this._dragHandle, this, 'all', false), zrUtil.bind(this._dragHandle, this, 'all', true)));
    // A border radius clip.
    gradientBarGroup.setClipPath(new graphic.Rect({
      shape: {
        x: 0,
        y: 0,
        width: itemSize[0],
        height: itemSize[1],
        r: 3
      }
    }));
    var textRect = visualMapModel.textStyleModel.getTextRect('国');
    var textSize = mathMax(textRect.width, textRect.height);
    // Handle
    if (useHandle) {
      shapes.handleThumbs = [];
      shapes.handleLabels = [];
      shapes.handleLabelPoints = [];
      this._createHandle(visualMapModel, mainGroup, 0, itemSize, textSize, orient);
      this._createHandle(visualMapModel, mainGroup, 1, itemSize, textSize, orient);
    }
    this._createIndicator(visualMapModel, mainGroup, itemSize, textSize, orient);
    targetGroup.add(mainGroup);
  };
  ContinuousView.prototype._createHandle = function (visualMapModel, mainGroup, handleIndex, itemSize, textSize, orient) {
    var onDrift = zrUtil.bind(this._dragHandle, this, handleIndex, false);
    var onDragEnd = zrUtil.bind(this._dragHandle, this, handleIndex, true);
    var handleSize = parsePercent(visualMapModel.get('handleSize'), itemSize[0]);
    var handleThumb = createSymbol(visualMapModel.get('handleIcon'), -handleSize / 2, -handleSize / 2, handleSize, handleSize, null, true);
    var cursor = getCursor(this._orient);
    handleThumb.attr({
      cursor: cursor,
      draggable: true,
      drift: onDrift,
      ondragend: onDragEnd,
      onmousemove: function (e) {
        eventTool.stop(e.event);
      }
    });
    handleThumb.x = itemSize[0] / 2;
    handleThumb.useStyle(visualMapModel.getModel('handleStyle').getItemStyle());
    handleThumb.setStyle({
      strokeNoScale: true,
      strokeFirst: true
    });
    handleThumb.style.lineWidth *= 2;
    handleThumb.ensureState('emphasis').style = visualMapModel.getModel(['emphasis', 'handleStyle']).getItemStyle();
    setAsHighDownDispatcher(handleThumb, true);
    mainGroup.add(handleThumb);
    // Text is always horizontal layout but should not be effected by
    // transform (orient/inverse). So label is built separately but not
    // use zrender/graphic/helper/RectText, and is located based on view
    // group (according to handleLabelPoint) but not barGroup.
    var textStyleModel = this.visualMapModel.textStyleModel;
    var handleLabel = new graphic.Text({
      cursor: cursor,
      draggable: true,
      drift: onDrift,
      onmousemove: function (e) {
        // For mobile device, prevent screen slider on the button.
        eventTool.stop(e.event);
      },
      ondragend: onDragEnd,
      style: createTextStyle(textStyleModel, {
        x: 0,
        y: 0,
        text: ''
      })
    });
    handleLabel.ensureState('blur').style = {
      opacity: 0.1
    };
    handleLabel.stateTransition = {
      duration: 200
    };
    this.group.add(handleLabel);
    var handleLabelPoint = [handleSize, 0];
    var shapes = this._shapes;
    shapes.handleThumbs[handleIndex] = handleThumb;
    shapes.handleLabelPoints[handleIndex] = handleLabelPoint;
    shapes.handleLabels[handleIndex] = handleLabel;
  };
  ContinuousView.prototype._createIndicator = function (visualMapModel, mainGroup, itemSize, textSize, orient) {
    var scale = parsePercent(visualMapModel.get('indicatorSize'), itemSize[0]);
    var indicator = createSymbol(visualMapModel.get('indicatorIcon'), -scale / 2, -scale / 2, scale, scale, null, true);
    indicator.attr({
      cursor: 'move',
      invisible: true,
      silent: true,
      x: itemSize[0] / 2
    });
    var indicatorStyle = visualMapModel.getModel('indicatorStyle').getItemStyle();
    if (indicator instanceof ZRImage) {
      var pathStyle = indicator.style;
      indicator.useStyle(zrUtil.extend({
        // TODO other properties like x, y ?
        image: pathStyle.image,
        x: pathStyle.x,
        y: pathStyle.y,
        width: pathStyle.width,
        height: pathStyle.height
      }, indicatorStyle));
    } else {
      indicator.useStyle(indicatorStyle);
    }
    mainGroup.add(indicator);
    var textStyleModel = this.visualMapModel.textStyleModel;
    var indicatorLabel = new graphic.Text({
      silent: true,
      invisible: true,
      style: createTextStyle(textStyleModel, {
        x: 0,
        y: 0,
        text: ''
      })
    });
    this.group.add(indicatorLabel);
    var indicatorLabelPoint = [(orient === 'horizontal' ? textSize / 2 : HOVER_LINK_OUT) + itemSize[0] / 2, 0];
    var shapes = this._shapes;
    shapes.indicator = indicator;
    shapes.indicatorLabel = indicatorLabel;
    shapes.indicatorLabelPoint = indicatorLabelPoint;
    this._firstShowIndicator = true;
  };
  ContinuousView.prototype._dragHandle = function (handleIndex, isEnd,
  // dx is event from ondragend if isEnd is true. It's not used
  dx, dy) {
    if (!this._useHandle) {
      return;
    }
    this._dragging = !isEnd;
    if (!isEnd) {
      // Transform dx, dy to bar coordination.
      var vertex = this._applyTransform([dx, dy], this._shapes.mainGroup, true);
      this._updateInterval(handleIndex, vertex[1]);
      this._hideIndicator();
      // Considering realtime, update view should be executed
      // before dispatch action.
      this._updateView();
    }
    // dragEnd do not dispatch action when realtime.
    if (isEnd === !this.visualMapModel.get('realtime')) {
      // jshint ignore:line
      this.api.dispatchAction({
        type: 'selectDataRange',
        from: this.uid,
        visualMapId: this.visualMapModel.id,
        selected: this._dataInterval.slice()
      });
    }
    if (isEnd) {
      !this._hovering && this._clearHoverLinkToSeries();
    } else if (useHoverLinkOnHandle(this.visualMapModel)) {
      this._doHoverLinkToSeries(this._handleEnds[handleIndex], false);
    }
  };
  ContinuousView.prototype._resetInterval = function () {
    var visualMapModel = this.visualMapModel;
    var dataInterval = this._dataInterval = visualMapModel.getSelected();
    var dataExtent = visualMapModel.getExtent();
    var sizeExtent = [0, visualMapModel.itemSize[1]];
    this._handleEnds = [linearMap(dataInterval[0], dataExtent, sizeExtent, true), linearMap(dataInterval[1], dataExtent, sizeExtent, true)];
  };
  /**
   * @private
   * @param {(number|string)} handleIndex 0 or 1 or 'all'
   * @param {number} dx
   * @param {number} dy
   */
  ContinuousView.prototype._updateInterval = function (handleIndex, delta) {
    delta = delta || 0;
    var visualMapModel = this.visualMapModel;
    var handleEnds = this._handleEnds;
    var sizeExtent = [0, visualMapModel.itemSize[1]];
    sliderMove(delta, handleEnds, sizeExtent, handleIndex,
    // cross is forbidden
    0);
    var dataExtent = visualMapModel.getExtent();
    // Update data interval.
    this._dataInterval = [linearMap(handleEnds[0], sizeExtent, dataExtent, true), linearMap(handleEnds[1], sizeExtent, dataExtent, true)];
  };
  ContinuousView.prototype._updateView = function (forSketch) {
    var visualMapModel = this.visualMapModel;
    var dataExtent = visualMapModel.getExtent();
    var shapes = this._shapes;
    var outOfRangeHandleEnds = [0, visualMapModel.itemSize[1]];
    var inRangeHandleEnds = forSketch ? outOfRangeHandleEnds : this._handleEnds;
    var visualInRange = this._createBarVisual(this._dataInterval, dataExtent, inRangeHandleEnds, 'inRange');
    var visualOutOfRange = this._createBarVisual(dataExtent, dataExtent, outOfRangeHandleEnds, 'outOfRange');
    shapes.inRange.setStyle({
      fill: visualInRange.barColor
      // opacity: visualInRange.opacity
    }).setShape('points', visualInRange.barPoints);
    shapes.outOfRange.setStyle({
      fill: visualOutOfRange.barColor
      // opacity: visualOutOfRange.opacity
    }).setShape('points', visualOutOfRange.barPoints);
    this._updateHandle(inRangeHandleEnds, visualInRange);
  };
  ContinuousView.prototype._createBarVisual = function (dataInterval, dataExtent, handleEnds, forceState) {
    var opts = {
      forceState: forceState,
      convertOpacityToAlpha: true
    };
    var colorStops = this._makeColorGradient(dataInterval, opts);
    var symbolSizes = [this.getControllerVisual(dataInterval[0], 'symbolSize', opts), this.getControllerVisual(dataInterval[1], 'symbolSize', opts)];
    var barPoints = this._createBarPoints(handleEnds, symbolSizes);
    return {
      barColor: new LinearGradient(0, 0, 0, 1, colorStops),
      barPoints: barPoints,
      handlesColor: [colorStops[0].color, colorStops[colorStops.length - 1].color]
    };
  };
  ContinuousView.prototype._makeColorGradient = function (dataInterval, opts) {
    // Considering colorHue, which is not linear, so we have to sample
    // to calculate gradient color stops, but not only calculate head
    // and tail.
    var sampleNumber = 100; // Arbitrary value.
    var colorStops = [];
    var step = (dataInterval[1] - dataInterval[0]) / sampleNumber;
    colorStops.push({
      color: this.getControllerVisual(dataInterval[0], 'color', opts),
      offset: 0
    });
    for (var i = 1; i < sampleNumber; i++) {
      var currValue = dataInterval[0] + step * i;
      if (currValue > dataInterval[1]) {
        break;
      }
      colorStops.push({
        color: this.getControllerVisual(currValue, 'color', opts),
        offset: i / sampleNumber
      });
    }
    colorStops.push({
      color: this.getControllerVisual(dataInterval[1], 'color', opts),
      offset: 1
    });
    return colorStops;
  };
  ContinuousView.prototype._createBarPoints = function (handleEnds, symbolSizes) {
    var itemSize = this.visualMapModel.itemSize;
    return [[itemSize[0] - symbolSizes[0], handleEnds[0]], [itemSize[0], handleEnds[0]], [itemSize[0], handleEnds[1]], [itemSize[0] - symbolSizes[1], handleEnds[1]]];
  };
  ContinuousView.prototype._createBarGroup = function (itemAlign) {
    var orient = this._orient;
    var inverse = this.visualMapModel.get('inverse');
    return new graphic.Group(orient === 'horizontal' && !inverse ? {
      scaleX: itemAlign === 'bottom' ? 1 : -1,
      rotation: Math.PI / 2
    } : orient === 'horizontal' && inverse ? {
      scaleX: itemAlign === 'bottom' ? -1 : 1,
      rotation: -Math.PI / 2
    } : orient === 'vertical' && !inverse ? {
      scaleX: itemAlign === 'left' ? 1 : -1,
      scaleY: -1
    } : {
      scaleX: itemAlign === 'left' ? 1 : -1
    });
  };
  ContinuousView.prototype._updateHandle = function (handleEnds, visualInRange) {
    if (!this._useHandle) {
      return;
    }
    var shapes = this._shapes;
    var visualMapModel = this.visualMapModel;
    var handleThumbs = shapes.handleThumbs;
    var handleLabels = shapes.handleLabels;
    var itemSize = visualMapModel.itemSize;
    var dataExtent = visualMapModel.getExtent();
    var align = this._applyTransform('left', shapes.mainGroup);
    each([0, 1], function (handleIndex) {
      var handleThumb = handleThumbs[handleIndex];
      handleThumb.setStyle('fill', visualInRange.handlesColor[handleIndex]);
      handleThumb.y = handleEnds[handleIndex];
      var val = linearMap(handleEnds[handleIndex], [0, itemSize[1]], dataExtent, true);
      var symbolSize = this.getControllerVisual(val, 'symbolSize');
      handleThumb.scaleX = handleThumb.scaleY = symbolSize / itemSize[0];
      handleThumb.x = itemSize[0] - symbolSize / 2;
      // Update handle label position.
      var textPoint = graphic.applyTransform(shapes.handleLabelPoints[handleIndex], graphic.getTransform(handleThumb, this.group));
      if (this._orient === 'horizontal') {
        // If visualMap controls symbol size, an additional offset needs to be added to labels to avoid collision at minimum size.
        // Offset reaches value of 0 at "maximum" position, so maximum position is not altered at all.
        var minimumOffset = align === 'left' || align === 'top' ? (itemSize[0] - symbolSize) / 2 : (itemSize[0] - symbolSize) / -2;
        textPoint[1] += minimumOffset;
      }
      handleLabels[handleIndex].setStyle({
        x: textPoint[0],
        y: textPoint[1],
        text: visualMapModel.formatValueText(this._dataInterval[handleIndex]),
        verticalAlign: 'middle',
        align: this._orient === 'vertical' ? this._applyTransform('left', shapes.mainGroup) : 'center'
      });
    }, this);
  };
  ContinuousView.prototype._showIndicator = function (cursorValue, textValue, rangeSymbol, halfHoverLinkSize) {
    var visualMapModel = this.visualMapModel;
    var dataExtent = visualMapModel.getExtent();
    var itemSize = visualMapModel.itemSize;
    var sizeExtent = [0, itemSize[1]];
    var shapes = this._shapes;
    var indicator = shapes.indicator;
    if (!indicator) {
      return;
    }
    indicator.attr('invisible', false);
    var opts = {
      convertOpacityToAlpha: true
    };
    var color = this.getControllerVisual(cursorValue, 'color', opts);
    var symbolSize = this.getControllerVisual(cursorValue, 'symbolSize');
    var y = linearMap(cursorValue, dataExtent, sizeExtent, true);
    var x = itemSize[0] - symbolSize / 2;
    var oldIndicatorPos = {
      x: indicator.x,
      y: indicator.y
    };
    // Update handle label position.
    indicator.y = y;
    indicator.x = x;
    var textPoint = graphic.applyTransform(shapes.indicatorLabelPoint, graphic.getTransform(indicator, this.group));
    var indicatorLabel = shapes.indicatorLabel;
    indicatorLabel.attr('invisible', false);
    var align = this._applyTransform('left', shapes.mainGroup);
    var orient = this._orient;
    var isHorizontal = orient === 'horizontal';
    indicatorLabel.setStyle({
      text: (rangeSymbol ? rangeSymbol : '') + visualMapModel.formatValueText(textValue),
      verticalAlign: isHorizontal ? align : 'middle',
      align: isHorizontal ? 'center' : align
    });
    var indicatorNewProps = {
      x: x,
      y: y,
      style: {
        fill: color
      }
    };
    var labelNewProps = {
      style: {
        x: textPoint[0],
        y: textPoint[1]
      }
    };
    if (visualMapModel.ecModel.isAnimationEnabled() && !this._firstShowIndicator) {
      var animationCfg = {
        duration: 100,
        easing: 'cubicInOut',
        additive: true
      };
      indicator.x = oldIndicatorPos.x;
      indicator.y = oldIndicatorPos.y;
      indicator.animateTo(indicatorNewProps, animationCfg);
      indicatorLabel.animateTo(labelNewProps, animationCfg);
    } else {
      indicator.attr(indicatorNewProps);
      indicatorLabel.attr(labelNewProps);
    }
    this._firstShowIndicator = false;
    var handleLabels = this._shapes.handleLabels;
    if (handleLabels) {
      for (var i = 0; i < handleLabels.length; i++) {
        // Fade out handle labels.
        // NOTE: Must use api enter/leave on emphasis/blur/select state. Or the global states manager will change it.
        this.api.enterBlur(handleLabels[i]);
      }
    }
  };
  ContinuousView.prototype._enableHoverLinkToSeries = function () {
    var self = this;
    this._shapes.mainGroup.on('mousemove', function (e) {
      self._hovering = true;
      if (!self._dragging) {
        var itemSize = self.visualMapModel.itemSize;
        var pos = self._applyTransform([e.offsetX, e.offsetY], self._shapes.mainGroup, true, true);
        // For hover link show when hover handle, which might be
        // below or upper than sizeExtent.
        pos[1] = mathMin(mathMax(0, pos[1]), itemSize[1]);
        self._doHoverLinkToSeries(pos[1], 0 <= pos[0] && pos[0] <= itemSize[0]);
      }
    }).on('mouseout', function () {
      // When mouse is out of handle, hoverLink still need
      // to be displayed when realtime is set as false.
      self._hovering = false;
      !self._dragging && self._clearHoverLinkToSeries();
    });
  };
  ContinuousView.prototype._enableHoverLinkFromSeries = function () {
    var zr = this.api.getZr();
    if (this.visualMapModel.option.hoverLink) {
      zr.on('mouseover', this._hoverLinkFromSeriesMouseOver, this);
      zr.on('mouseout', this._hideIndicator, this);
    } else {
      this._clearHoverLinkFromSeries();
    }
  };
  ContinuousView.prototype._doHoverLinkToSeries = function (cursorPos, hoverOnBar) {
    var visualMapModel = this.visualMapModel;
    var itemSize = visualMapModel.itemSize;
    if (!visualMapModel.option.hoverLink) {
      return;
    }
    var sizeExtent = [0, itemSize[1]];
    var dataExtent = visualMapModel.getExtent();
    // For hover link show when hover handle, which might be below or upper than sizeExtent.
    cursorPos = mathMin(mathMax(sizeExtent[0], cursorPos), sizeExtent[1]);
    var halfHoverLinkSize = getHalfHoverLinkSize(visualMapModel, dataExtent, sizeExtent);
    var hoverRange = [cursorPos - halfHoverLinkSize, cursorPos + halfHoverLinkSize];
    var cursorValue = linearMap(cursorPos, sizeExtent, dataExtent, true);
    var valueRange = [linearMap(hoverRange[0], sizeExtent, dataExtent, true), linearMap(hoverRange[1], sizeExtent, dataExtent, true)];
    // Consider data range is out of visualMap range, see test/visualMap-continuous.html,
    // where china and india has very large population.
    hoverRange[0] < sizeExtent[0] && (valueRange[0] = -Infinity);
    hoverRange[1] > sizeExtent[1] && (valueRange[1] = Infinity);
    // Do not show indicator when mouse is over handle,
    // otherwise labels overlap, especially when dragging.
    if (hoverOnBar) {
      if (valueRange[0] === -Infinity) {
        this._showIndicator(cursorValue, valueRange[1], '< ', halfHoverLinkSize);
      } else if (valueRange[1] === Infinity) {
        this._showIndicator(cursorValue, valueRange[0], '> ', halfHoverLinkSize);
      } else {
        this._showIndicator(cursorValue, cursorValue, '≈ ', halfHoverLinkSize);
      }
    }
    // When realtime is set as false, handles, which are in barGroup,
    // also trigger hoverLink, which help user to realize where they
    // focus on when dragging. (see test/heatmap-large.html)
    // When realtime is set as true, highlight will not show when hover
    // handle, because the label on handle, which displays a exact value
    // but not range, might mislead users.
    var oldBatch = this._hoverLinkDataIndices;
    var newBatch = [];
    if (hoverOnBar || useHoverLinkOnHandle(visualMapModel)) {
      newBatch = this._hoverLinkDataIndices = visualMapModel.findTargetDataIndices(valueRange);
    }
    var resultBatches = modelUtil.compressBatches(oldBatch, newBatch);
    this._dispatchHighDown('downplay', helper.makeHighDownBatch(resultBatches[0], visualMapModel));
    this._dispatchHighDown('highlight', helper.makeHighDownBatch(resultBatches[1], visualMapModel));
  };
  ContinuousView.prototype._hoverLinkFromSeriesMouseOver = function (e) {
    var ecData;
    findEventDispatcher(e.target, function (target) {
      var currECData = getECData(target);
      if (currECData.dataIndex != null) {
        ecData = currECData;
        return true;
      }
    }, true);
    if (!ecData) {
      return;
    }
    var dataModel = this.ecModel.getSeriesByIndex(ecData.seriesIndex);
    var visualMapModel = this.visualMapModel;
    if (!visualMapModel.isTargetSeries(dataModel)) {
      return;
    }
    var data = dataModel.getData(ecData.dataType);
    var value = data.getStore().get(visualMapModel.getDataDimensionIndex(data), ecData.dataIndex);
    if (!isNaN(value)) {
      this._showIndicator(value, value);
    }
  };
  ContinuousView.prototype._hideIndicator = function () {
    var shapes = this._shapes;
    shapes.indicator && shapes.indicator.attr('invisible', true);
    shapes.indicatorLabel && shapes.indicatorLabel.attr('invisible', true);
    var handleLabels = this._shapes.handleLabels;
    if (handleLabels) {
      for (var i = 0; i < handleLabels.length; i++) {
        // Fade out handle labels.
        // NOTE: Must use api enter/leave on emphasis/blur/select state. Or the global states manager will change it.
        this.api.leaveBlur(handleLabels[i]);
      }
    }
  };
  ContinuousView.prototype._clearHoverLinkToSeries = function () {
    this._hideIndicator();
    var indices = this._hoverLinkDataIndices;
    this._dispatchHighDown('downplay', helper.makeHighDownBatch(indices, this.visualMapModel));
    indices.length = 0;
  };
  ContinuousView.prototype._clearHoverLinkFromSeries = function () {
    this._hideIndicator();
    var zr = this.api.getZr();
    zr.off('mouseover', this._hoverLinkFromSeriesMouseOver);
    zr.off('mouseout', this._hideIndicator);
  };
  ContinuousView.prototype._applyTransform = function (vertex, element, inverse, global) {
    var transform = graphic.getTransform(element, global ? null : this.group);
    return zrUtil.isArray(vertex) ? graphic.applyTransform(vertex, transform, inverse) : graphic.transformDirection(vertex, transform, inverse);
  };
  // TODO: TYPE more specified payload types.
  ContinuousView.prototype._dispatchHighDown = function (type, batch) {
    batch && batch.length && this.api.dispatchAction({
      type: type,
      batch: batch
    });
  };
  /**
   * @override
   */
  ContinuousView.prototype.dispose = function () {
    this._clearHoverLinkFromSeries();
    this._clearHoverLinkToSeries();
  };
  ContinuousView.type = 'visualMap.continuous';
  return ContinuousView;
}(VisualMapView);
function createPolygon(points, cursor, onDrift, onDragEnd) {
  return new graphic.Polygon({
    shape: {
      points: points
    },
    draggable: !!onDrift,
    cursor: cursor,
    drift: onDrift,
    onmousemove: function (e) {
      // For mobile device, prevent screen slider on the button.
      eventTool.stop(e.event);
    },
    ondragend: onDragEnd
  });
}
function getHalfHoverLinkSize(visualMapModel, dataExtent, sizeExtent) {
  var halfHoverLinkSize = HOVER_LINK_SIZE / 2;
  var hoverLinkDataSize = visualMapModel.get('hoverLinkDataSize');
  if (hoverLinkDataSize) {
    halfHoverLinkSize = linearMap(hoverLinkDataSize, dataExtent, sizeExtent, true) / 2;
  }
  return halfHoverLinkSize;
}
function useHoverLinkOnHandle(visualMapModel) {
  var hoverLinkOnHandle = visualMapModel.get('hoverLinkOnHandle');
  return !!(hoverLinkOnHandle == null ? visualMapModel.get('realtime') : hoverLinkOnHandle);
}
function getCursor(orient) {
  return orient === 'vertical' ? 'ns-resize' : 'ew-resize';
}
export default ContinuousView;