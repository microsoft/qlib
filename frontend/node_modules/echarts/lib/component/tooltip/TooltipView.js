
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

import { __extends } from "tslib";
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
import { bind, each, clone, trim, isString, isFunction, isArray, isObject, extend } from 'zrender/lib/core/util.js';
import env from 'zrender/lib/core/env.js';
import TooltipHTMLContent from './TooltipHTMLContent.js';
import TooltipRichContent from './TooltipRichContent.js';
import { convertToColorString, encodeHTML, formatTpl } from '../../util/format.js';
import { parsePercent } from '../../util/number.js';
import { Rect } from '../../util/graphic.js';
import findPointFromSeries from '../axisPointer/findPointFromSeries.js';
import { getLayoutRect } from '../../util/layout.js';
import Model from '../../model/Model.js';
import * as globalListener from '../axisPointer/globalListener.js';
import * as axisHelper from '../../coord/axisHelper.js';
import * as axisPointerViewHelper from '../axisPointer/viewHelper.js';
import { getTooltipRenderMode, preParseFinder, queryReferringComponents } from '../../util/model.js';
import ComponentView from '../../view/Component.js';
import { format as timeFormat } from '../../util/time.js';
import { getECData } from '../../util/innerStore.js';
import { shouldTooltipConfine } from './helper.js';
import { normalizeTooltipFormatResult } from '../../model/mixin/dataFormat.js';
import { createTooltipMarkup, buildTooltipMarkup, TooltipMarkupStyleCreator } from './tooltipMarkup.js';
import { findEventDispatcher } from '../../util/event.js';
import { clear, createOrUpdate } from '../../util/throttle.js';
var proxyRect = new Rect({
  shape: {
    x: -1,
    y: -1,
    width: 2,
    height: 2
  }
});
var TooltipView = /** @class */function (_super) {
  __extends(TooltipView, _super);
  function TooltipView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TooltipView.type;
    return _this;
  }
  TooltipView.prototype.init = function (ecModel, api) {
    if (env.node || !api.getDom()) {
      return;
    }
    var tooltipModel = ecModel.getComponent('tooltip');
    var renderMode = this._renderMode = getTooltipRenderMode(tooltipModel.get('renderMode'));
    this._tooltipContent = renderMode === 'richText' ? new TooltipRichContent(api) : new TooltipHTMLContent(api, {
      appendTo: tooltipModel.get('appendToBody', true) ? 'body' : tooltipModel.get('appendTo', true)
    });
  };
  TooltipView.prototype.render = function (tooltipModel, ecModel, api) {
    if (env.node || !api.getDom()) {
      return;
    }
    // Reset
    this.group.removeAll();
    this._tooltipModel = tooltipModel;
    this._ecModel = ecModel;
    this._api = api;
    var tooltipContent = this._tooltipContent;
    tooltipContent.update(tooltipModel);
    tooltipContent.setEnterable(tooltipModel.get('enterable'));
    this._initGlobalListener();
    this._keepShow();
    // PENDING
    // `mousemove` event will be triggered very frequently when the mouse moves fast,
    // which causes that the `updatePosition` function was also called frequently.
    // In Chrome with devtools open and Firefox, tooltip looks laggy and shakes. See #14695 #16101
    // To avoid frequent triggering,
    // consider throttling it in 50ms when transition is enabled
    if (this._renderMode !== 'richText' && tooltipModel.get('transitionDuration')) {
      createOrUpdate(this, '_updatePosition', 50, 'fixRate');
    } else {
      clear(this, '_updatePosition');
    }
  };
  TooltipView.prototype._initGlobalListener = function () {
    var tooltipModel = this._tooltipModel;
    var triggerOn = tooltipModel.get('triggerOn');
    globalListener.register('itemTooltip', this._api, bind(function (currTrigger, e, dispatchAction) {
      // If 'none', it is not controlled by mouse totally.
      if (triggerOn !== 'none') {
        if (triggerOn.indexOf(currTrigger) >= 0) {
          this._tryShow(e, dispatchAction);
        } else if (currTrigger === 'leave') {
          this._hide(dispatchAction);
        }
      }
    }, this));
  };
  TooltipView.prototype._keepShow = function () {
    var tooltipModel = this._tooltipModel;
    var ecModel = this._ecModel;
    var api = this._api;
    var triggerOn = tooltipModel.get('triggerOn');
    // Try to keep the tooltip show when refreshing
    if (this._lastX != null && this._lastY != null
    // When user is willing to control tooltip totally using API,
    // self.manuallyShowTip({x, y}) might cause tooltip hide,
    // which is not expected.
    && triggerOn !== 'none' && triggerOn !== 'click') {
      var self_1 = this;
      clearTimeout(this._refreshUpdateTimeout);
      this._refreshUpdateTimeout = setTimeout(function () {
        // Show tip next tick after other charts are rendered
        // In case highlight action has wrong result
        // FIXME
        !api.isDisposed() && self_1.manuallyShowTip(tooltipModel, ecModel, api, {
          x: self_1._lastX,
          y: self_1._lastY,
          dataByCoordSys: self_1._lastDataByCoordSys
        });
      });
    }
  };
  /**
   * Show tip manually by
   * dispatchAction({
   *     type: 'showTip',
   *     x: 10,
   *     y: 10
   * });
   * Or
   * dispatchAction({
   *      type: 'showTip',
   *      seriesIndex: 0,
   *      dataIndex or dataIndexInside or name
   * });
   *
   *  TODO Batch
   */
  TooltipView.prototype.manuallyShowTip = function (tooltipModel, ecModel, api, payload) {
    if (payload.from === this.uid || env.node || !api.getDom()) {
      return;
    }
    var dispatchAction = makeDispatchAction(payload, api);
    // Reset ticket
    this._ticket = '';
    // When triggered from axisPointer.
    var dataByCoordSys = payload.dataByCoordSys;
    var cmptRef = findComponentReference(payload, ecModel, api);
    if (cmptRef) {
      var rect = cmptRef.el.getBoundingRect().clone();
      rect.applyTransform(cmptRef.el.transform);
      this._tryShow({
        offsetX: rect.x + rect.width / 2,
        offsetY: rect.y + rect.height / 2,
        target: cmptRef.el,
        position: payload.position,
        // When manully trigger, the mouse is not on the el, so we'd better to
        // position tooltip on the bottom of the el and display arrow is possible.
        positionDefault: 'bottom'
      }, dispatchAction);
    } else if (payload.tooltip && payload.x != null && payload.y != null) {
      var el = proxyRect;
      el.x = payload.x;
      el.y = payload.y;
      el.update();
      getECData(el).tooltipConfig = {
        name: null,
        option: payload.tooltip
      };
      // Manually show tooltip while view is not using zrender elements.
      this._tryShow({
        offsetX: payload.x,
        offsetY: payload.y,
        target: el
      }, dispatchAction);
    } else if (dataByCoordSys) {
      this._tryShow({
        offsetX: payload.x,
        offsetY: payload.y,
        position: payload.position,
        dataByCoordSys: dataByCoordSys,
        tooltipOption: payload.tooltipOption
      }, dispatchAction);
    } else if (payload.seriesIndex != null) {
      if (this._manuallyAxisShowTip(tooltipModel, ecModel, api, payload)) {
        return;
      }
      var pointInfo = findPointFromSeries(payload, ecModel);
      var cx = pointInfo.point[0];
      var cy = pointInfo.point[1];
      if (cx != null && cy != null) {
        this._tryShow({
          offsetX: cx,
          offsetY: cy,
          target: pointInfo.el,
          position: payload.position,
          // When manully trigger, the mouse is not on the el, so we'd better to
          // position tooltip on the bottom of the el and display arrow is possible.
          positionDefault: 'bottom'
        }, dispatchAction);
      }
    } else if (payload.x != null && payload.y != null) {
      // FIXME
      // should wrap dispatchAction like `axisPointer/globalListener` ?
      api.dispatchAction({
        type: 'updateAxisPointer',
        x: payload.x,
        y: payload.y
      });
      this._tryShow({
        offsetX: payload.x,
        offsetY: payload.y,
        position: payload.position,
        target: api.getZr().findHover(payload.x, payload.y).target
      }, dispatchAction);
    }
  };
  TooltipView.prototype.manuallyHideTip = function (tooltipModel, ecModel, api, payload) {
    var tooltipContent = this._tooltipContent;
    if (this._tooltipModel) {
      tooltipContent.hideLater(this._tooltipModel.get('hideDelay'));
    }
    this._lastX = this._lastY = this._lastDataByCoordSys = null;
    if (payload.from !== this.uid) {
      this._hide(makeDispatchAction(payload, api));
    }
  };
  // Be compatible with previous design, that is, when tooltip.type is 'axis' and
  // dispatchAction 'showTip' with seriesIndex and dataIndex will trigger axis pointer
  // and tooltip.
  TooltipView.prototype._manuallyAxisShowTip = function (tooltipModel, ecModel, api, payload) {
    var seriesIndex = payload.seriesIndex;
    var dataIndex = payload.dataIndex;
    // @ts-ignore
    var coordSysAxesInfo = ecModel.getComponent('axisPointer').coordSysAxesInfo;
    if (seriesIndex == null || dataIndex == null || coordSysAxesInfo == null) {
      return;
    }
    var seriesModel = ecModel.getSeriesByIndex(seriesIndex);
    if (!seriesModel) {
      return;
    }
    var data = seriesModel.getData();
    var tooltipCascadedModel = buildTooltipModel([data.getItemModel(dataIndex), seriesModel, (seriesModel.coordinateSystem || {}).model], this._tooltipModel);
    if (tooltipCascadedModel.get('trigger') !== 'axis') {
      return;
    }
    api.dispatchAction({
      type: 'updateAxisPointer',
      seriesIndex: seriesIndex,
      dataIndex: dataIndex,
      position: payload.position
    });
    return true;
  };
  TooltipView.prototype._tryShow = function (e, dispatchAction) {
    var el = e.target;
    var tooltipModel = this._tooltipModel;
    if (!tooltipModel) {
      return;
    }
    // Save mouse x, mouse y. So we can try to keep showing the tip if chart is refreshed
    this._lastX = e.offsetX;
    this._lastY = e.offsetY;
    var dataByCoordSys = e.dataByCoordSys;
    if (dataByCoordSys && dataByCoordSys.length) {
      this._showAxisTooltip(dataByCoordSys, e);
    } else if (el) {
      var ecData = getECData(el);
      if (ecData.ssrType === 'legend') {
        // Don't trigger tooltip for legend tooltip item
        return;
      }
      this._lastDataByCoordSys = null;
      var seriesDispatcher_1;
      var cmptDispatcher_1;
      findEventDispatcher(el, function (target) {
        // Always show item tooltip if mouse is on the element with dataIndex
        if (getECData(target).dataIndex != null) {
          seriesDispatcher_1 = target;
          return true;
        }
        // Tooltip provided directly. Like legend.
        if (getECData(target).tooltipConfig != null) {
          cmptDispatcher_1 = target;
          return true;
        }
      }, true);
      if (seriesDispatcher_1) {
        this._showSeriesItemTooltip(e, seriesDispatcher_1, dispatchAction);
      } else if (cmptDispatcher_1) {
        this._showComponentItemTooltip(e, cmptDispatcher_1, dispatchAction);
      } else {
        this._hide(dispatchAction);
      }
    } else {
      this._lastDataByCoordSys = null;
      this._hide(dispatchAction);
    }
  };
  TooltipView.prototype._showOrMove = function (tooltipModel, cb) {
    // showDelay is used in this case: tooltip.enterable is set
    // as true. User intent to move mouse into tooltip and click
    // something. `showDelay` makes it easier to enter the content
    // but tooltip do not move immediately.
    var delay = tooltipModel.get('showDelay');
    cb = bind(cb, this);
    clearTimeout(this._showTimout);
    delay > 0 ? this._showTimout = setTimeout(cb, delay) : cb();
  };
  TooltipView.prototype._showAxisTooltip = function (dataByCoordSys, e) {
    var ecModel = this._ecModel;
    var globalTooltipModel = this._tooltipModel;
    var point = [e.offsetX, e.offsetY];
    var singleTooltipModel = buildTooltipModel([e.tooltipOption], globalTooltipModel);
    var renderMode = this._renderMode;
    var cbParamsList = [];
    var articleMarkup = createTooltipMarkup('section', {
      blocks: [],
      noHeader: true
    });
    // Only for legacy: `Serise['formatTooltip']` returns a string.
    var markupTextArrLegacy = [];
    var markupStyleCreator = new TooltipMarkupStyleCreator();
    each(dataByCoordSys, function (itemCoordSys) {
      each(itemCoordSys.dataByAxis, function (axisItem) {
        var axisModel = ecModel.getComponent(axisItem.axisDim + 'Axis', axisItem.axisIndex);
        var axisValue = axisItem.value;
        if (!axisModel || axisValue == null) {
          return;
        }
        var axisValueLabel = axisPointerViewHelper.getValueLabel(axisValue, axisModel.axis, ecModel, axisItem.seriesDataIndices, axisItem.valueLabelOpt);
        var axisSectionMarkup = createTooltipMarkup('section', {
          header: axisValueLabel,
          noHeader: !trim(axisValueLabel),
          sortBlocks: true,
          blocks: []
        });
        articleMarkup.blocks.push(axisSectionMarkup);
        each(axisItem.seriesDataIndices, function (idxItem) {
          var series = ecModel.getSeriesByIndex(idxItem.seriesIndex);
          var dataIndex = idxItem.dataIndexInside;
          var cbParams = series.getDataParams(dataIndex);
          // Can't find data.
          if (cbParams.dataIndex < 0) {
            return;
          }
          cbParams.axisDim = axisItem.axisDim;
          cbParams.axisIndex = axisItem.axisIndex;
          cbParams.axisType = axisItem.axisType;
          cbParams.axisId = axisItem.axisId;
          cbParams.axisValue = axisHelper.getAxisRawValue(axisModel.axis, {
            value: axisValue
          });
          cbParams.axisValueLabel = axisValueLabel;
          // Pre-create marker style for makers. Users can assemble richText
          // text in `formatter` callback and use those markers style.
          cbParams.marker = markupStyleCreator.makeTooltipMarker('item', convertToColorString(cbParams.color), renderMode);
          var seriesTooltipResult = normalizeTooltipFormatResult(series.formatTooltip(dataIndex, true, null));
          var frag = seriesTooltipResult.frag;
          if (frag) {
            var valueFormatter = buildTooltipModel([series], globalTooltipModel).get('valueFormatter');
            axisSectionMarkup.blocks.push(valueFormatter ? extend({
              valueFormatter: valueFormatter
            }, frag) : frag);
          }
          if (seriesTooltipResult.text) {
            markupTextArrLegacy.push(seriesTooltipResult.text);
          }
          cbParamsList.push(cbParams);
        });
      });
    });
    // In most cases, the second axis is displays upper on the first one.
    // So we reverse it to look better.
    articleMarkup.blocks.reverse();
    markupTextArrLegacy.reverse();
    var positionExpr = e.position;
    var orderMode = singleTooltipModel.get('order');
    var builtMarkupText = buildTooltipMarkup(articleMarkup, markupStyleCreator, renderMode, orderMode, ecModel.get('useUTC'), singleTooltipModel.get('textStyle'));
    builtMarkupText && markupTextArrLegacy.unshift(builtMarkupText);
    var blockBreak = renderMode === 'richText' ? '\n\n' : '<br/>';
    var allMarkupText = markupTextArrLegacy.join(blockBreak);
    this._showOrMove(singleTooltipModel, function () {
      if (this._updateContentNotChangedOnAxis(dataByCoordSys, cbParamsList)) {
        this._updatePosition(singleTooltipModel, positionExpr, point[0], point[1], this._tooltipContent, cbParamsList);
      } else {
        this._showTooltipContent(singleTooltipModel, allMarkupText, cbParamsList, Math.random() + '', point[0], point[1], positionExpr, null, markupStyleCreator);
      }
    });
    // Do not trigger events here, because this branch only be entered
    // from dispatchAction.
  };
  TooltipView.prototype._showSeriesItemTooltip = function (e, dispatcher, dispatchAction) {
    var ecModel = this._ecModel;
    var ecData = getECData(dispatcher);
    // Use dataModel in element if possible
    // Used when mouseover on a element like markPoint or edge
    // In which case, the data is not main data in series.
    var seriesIndex = ecData.seriesIndex;
    var seriesModel = ecModel.getSeriesByIndex(seriesIndex);
    // For example, graph link.
    var dataModel = ecData.dataModel || seriesModel;
    var dataIndex = ecData.dataIndex;
    var dataType = ecData.dataType;
    var data = dataModel.getData(dataType);
    var renderMode = this._renderMode;
    var positionDefault = e.positionDefault;
    var tooltipModel = buildTooltipModel([data.getItemModel(dataIndex), dataModel, seriesModel && (seriesModel.coordinateSystem || {}).model], this._tooltipModel, positionDefault ? {
      position: positionDefault
    } : null);
    var tooltipTrigger = tooltipModel.get('trigger');
    if (tooltipTrigger != null && tooltipTrigger !== 'item') {
      return;
    }
    var params = dataModel.getDataParams(dataIndex, dataType);
    var markupStyleCreator = new TooltipMarkupStyleCreator();
    // Pre-create marker style for makers. Users can assemble richText
    // text in `formatter` callback and use those markers style.
    params.marker = markupStyleCreator.makeTooltipMarker('item', convertToColorString(params.color), renderMode);
    var seriesTooltipResult = normalizeTooltipFormatResult(dataModel.formatTooltip(dataIndex, false, dataType));
    var orderMode = tooltipModel.get('order');
    var valueFormatter = tooltipModel.get('valueFormatter');
    var frag = seriesTooltipResult.frag;
    var markupText = frag ? buildTooltipMarkup(valueFormatter ? extend({
      valueFormatter: valueFormatter
    }, frag) : frag, markupStyleCreator, renderMode, orderMode, ecModel.get('useUTC'), tooltipModel.get('textStyle')) : seriesTooltipResult.text;
    var asyncTicket = 'item_' + dataModel.name + '_' + dataIndex;
    this._showOrMove(tooltipModel, function () {
      this._showTooltipContent(tooltipModel, markupText, params, asyncTicket, e.offsetX, e.offsetY, e.position, e.target, markupStyleCreator);
    });
    // FIXME
    // duplicated showtip if manuallyShowTip is called from dispatchAction.
    dispatchAction({
      type: 'showTip',
      dataIndexInside: dataIndex,
      dataIndex: data.getRawIndex(dataIndex),
      seriesIndex: seriesIndex,
      from: this.uid
    });
  };
  TooltipView.prototype._showComponentItemTooltip = function (e, el, dispatchAction) {
    var isHTMLRenderMode = this._renderMode === 'html';
    var ecData = getECData(el);
    var tooltipConfig = ecData.tooltipConfig;
    var tooltipOpt = tooltipConfig.option || {};
    var encodeHTMLContent = tooltipOpt.encodeHTMLContent;
    if (isString(tooltipOpt)) {
      var content = tooltipOpt;
      tooltipOpt = {
        content: content,
        // Fixed formatter
        formatter: content
      };
      // when `tooltipConfig.option` is a string rather than an object,
      // we can't know if the content needs to be encoded
      // for the sake of security, encode it by default.
      encodeHTMLContent = true;
    }
    if (encodeHTMLContent && isHTMLRenderMode && tooltipOpt.content) {
      // clone might be unnecessary?
      tooltipOpt = clone(tooltipOpt);
      tooltipOpt.content = encodeHTML(tooltipOpt.content);
    }
    var tooltipModelCascade = [tooltipOpt];
    var cmpt = this._ecModel.getComponent(ecData.componentMainType, ecData.componentIndex);
    if (cmpt) {
      tooltipModelCascade.push(cmpt);
    }
    // In most cases, component tooltip formatter has different params with series tooltip formatter,
    // so that they cannot share the same formatter. Since the global tooltip formatter is used for series
    // by convention, we do not use it as the default formatter for component.
    tooltipModelCascade.push({
      formatter: tooltipOpt.content
    });
    var positionDefault = e.positionDefault;
    var subTooltipModel = buildTooltipModel(tooltipModelCascade, this._tooltipModel, positionDefault ? {
      position: positionDefault
    } : null);
    var defaultHtml = subTooltipModel.get('content');
    var asyncTicket = Math.random() + '';
    // PENDING: this case do not support richText style yet.
    var markupStyleCreator = new TooltipMarkupStyleCreator();
    // Do not check whether `trigger` is 'none' here, because `trigger`
    // only works on coordinate system. In fact, we have not found case
    // that requires setting `trigger` nothing on component yet.
    this._showOrMove(subTooltipModel, function () {
      // Use formatterParams from element defined in component
      // Avoid users modify it.
      var formatterParams = clone(subTooltipModel.get('formatterParams') || {});
      this._showTooltipContent(subTooltipModel, defaultHtml, formatterParams, asyncTicket, e.offsetX, e.offsetY, e.position, el, markupStyleCreator);
    });
    // If not dispatch showTip, tip may be hide triggered by axis.
    dispatchAction({
      type: 'showTip',
      from: this.uid
    });
  };
  TooltipView.prototype._showTooltipContent = function (
  // Use Model<TooltipOption> insteadof TooltipModel because this model may be from series or other options.
  // Instead of top level tooltip.
  tooltipModel, defaultHtml, params, asyncTicket, x, y, positionExpr, el, markupStyleCreator) {
    // Reset ticket
    this._ticket = '';
    if (!tooltipModel.get('showContent') || !tooltipModel.get('show')) {
      return;
    }
    var tooltipContent = this._tooltipContent;
    tooltipContent.setEnterable(tooltipModel.get('enterable'));
    var formatter = tooltipModel.get('formatter');
    positionExpr = positionExpr || tooltipModel.get('position');
    var html = defaultHtml;
    var nearPoint = this._getNearestPoint([x, y], params, tooltipModel.get('trigger'), tooltipModel.get('borderColor'));
    var nearPointColor = nearPoint.color;
    if (formatter) {
      if (isString(formatter)) {
        var useUTC = tooltipModel.ecModel.get('useUTC');
        var params0 = isArray(params) ? params[0] : params;
        var isTimeAxis = params0 && params0.axisType && params0.axisType.indexOf('time') >= 0;
        html = formatter;
        if (isTimeAxis) {
          html = timeFormat(params0.axisValue, html, useUTC);
        }
        html = formatTpl(html, params, true);
      } else if (isFunction(formatter)) {
        var callback = bind(function (cbTicket, html) {
          if (cbTicket === this._ticket) {
            tooltipContent.setContent(html, markupStyleCreator, tooltipModel, nearPointColor, positionExpr);
            this._updatePosition(tooltipModel, positionExpr, x, y, tooltipContent, params, el);
          }
        }, this);
        this._ticket = asyncTicket;
        html = formatter(params, asyncTicket, callback);
      } else {
        html = formatter;
      }
    }
    tooltipContent.setContent(html, markupStyleCreator, tooltipModel, nearPointColor, positionExpr);
    tooltipContent.show(tooltipModel, nearPointColor);
    this._updatePosition(tooltipModel, positionExpr, x, y, tooltipContent, params, el);
  };
  TooltipView.prototype._getNearestPoint = function (point, tooltipDataParams, trigger, borderColor) {
    if (trigger === 'axis' || isArray(tooltipDataParams)) {
      return {
        color: borderColor || (this._renderMode === 'html' ? '#fff' : 'none')
      };
    }
    if (!isArray(tooltipDataParams)) {
      return {
        color: borderColor || tooltipDataParams.color || tooltipDataParams.borderColor
      };
    }
  };
  TooltipView.prototype._updatePosition = function (tooltipModel, positionExpr, x,
  // Mouse x
  y,
  // Mouse y
  content, params, el) {
    var viewWidth = this._api.getWidth();
    var viewHeight = this._api.getHeight();
    positionExpr = positionExpr || tooltipModel.get('position');
    var contentSize = content.getSize();
    var align = tooltipModel.get('align');
    var vAlign = tooltipModel.get('verticalAlign');
    var rect = el && el.getBoundingRect().clone();
    el && rect.applyTransform(el.transform);
    if (isFunction(positionExpr)) {
      // Callback of position can be an array or a string specify the position
      positionExpr = positionExpr([x, y], params, content.el, rect, {
        viewSize: [viewWidth, viewHeight],
        contentSize: contentSize.slice()
      });
    }
    if (isArray(positionExpr)) {
      x = parsePercent(positionExpr[0], viewWidth);
      y = parsePercent(positionExpr[1], viewHeight);
    } else if (isObject(positionExpr)) {
      var boxLayoutPosition = positionExpr;
      boxLayoutPosition.width = contentSize[0];
      boxLayoutPosition.height = contentSize[1];
      var layoutRect = getLayoutRect(boxLayoutPosition, {
        width: viewWidth,
        height: viewHeight
      });
      x = layoutRect.x;
      y = layoutRect.y;
      align = null;
      // When positionExpr is left/top/right/bottom,
      // align and verticalAlign will not work.
      vAlign = null;
    }
    // Specify tooltip position by string 'top' 'bottom' 'left' 'right' around graphic element
    else if (isString(positionExpr) && el) {
      var pos = calcTooltipPosition(positionExpr, rect, contentSize, tooltipModel.get('borderWidth'));
      x = pos[0];
      y = pos[1];
    } else {
      var pos = refixTooltipPosition(x, y, content, viewWidth, viewHeight, align ? null : 20, vAlign ? null : 20);
      x = pos[0];
      y = pos[1];
    }
    align && (x -= isCenterAlign(align) ? contentSize[0] / 2 : align === 'right' ? contentSize[0] : 0);
    vAlign && (y -= isCenterAlign(vAlign) ? contentSize[1] / 2 : vAlign === 'bottom' ? contentSize[1] : 0);
    if (shouldTooltipConfine(tooltipModel)) {
      var pos = confineTooltipPosition(x, y, content, viewWidth, viewHeight);
      x = pos[0];
      y = pos[1];
    }
    content.moveTo(x, y);
  };
  // FIXME
  // Should we remove this but leave this to user?
  TooltipView.prototype._updateContentNotChangedOnAxis = function (dataByCoordSys, cbParamsList) {
    var lastCoordSys = this._lastDataByCoordSys;
    var lastCbParamsList = this._cbParamsList;
    var contentNotChanged = !!lastCoordSys && lastCoordSys.length === dataByCoordSys.length;
    contentNotChanged && each(lastCoordSys, function (lastItemCoordSys, indexCoordSys) {
      var lastDataByAxis = lastItemCoordSys.dataByAxis || [];
      var thisItemCoordSys = dataByCoordSys[indexCoordSys] || {};
      var thisDataByAxis = thisItemCoordSys.dataByAxis || [];
      contentNotChanged = contentNotChanged && lastDataByAxis.length === thisDataByAxis.length;
      contentNotChanged && each(lastDataByAxis, function (lastItem, indexAxis) {
        var thisItem = thisDataByAxis[indexAxis] || {};
        var lastIndices = lastItem.seriesDataIndices || [];
        var newIndices = thisItem.seriesDataIndices || [];
        contentNotChanged = contentNotChanged && lastItem.value === thisItem.value && lastItem.axisType === thisItem.axisType && lastItem.axisId === thisItem.axisId && lastIndices.length === newIndices.length;
        contentNotChanged && each(lastIndices, function (lastIdxItem, j) {
          var newIdxItem = newIndices[j];
          contentNotChanged = contentNotChanged && lastIdxItem.seriesIndex === newIdxItem.seriesIndex && lastIdxItem.dataIndex === newIdxItem.dataIndex;
        });
        // check is cbParams data value changed
        lastCbParamsList && each(lastItem.seriesDataIndices, function (idxItem) {
          var seriesIdx = idxItem.seriesIndex;
          var cbParams = cbParamsList[seriesIdx];
          var lastCbParams = lastCbParamsList[seriesIdx];
          if (cbParams && lastCbParams && lastCbParams.data !== cbParams.data) {
            contentNotChanged = false;
          }
        });
      });
    });
    this._lastDataByCoordSys = dataByCoordSys;
    this._cbParamsList = cbParamsList;
    return !!contentNotChanged;
  };
  TooltipView.prototype._hide = function (dispatchAction) {
    // Do not directly hideLater here, because this behavior may be prevented
    // in dispatchAction when showTip is dispatched.
    // FIXME
    // duplicated hideTip if manuallyHideTip is called from dispatchAction.
    this._lastDataByCoordSys = null;
    dispatchAction({
      type: 'hideTip',
      from: this.uid
    });
  };
  TooltipView.prototype.dispose = function (ecModel, api) {
    if (env.node || !api.getDom()) {
      return;
    }
    clear(this, '_updatePosition');
    this._tooltipContent.dispose();
    globalListener.unregister('itemTooltip', api);
  };
  TooltipView.type = 'tooltip';
  return TooltipView;
}(ComponentView);
/**
 * From top to bottom. (the last one should be globalTooltipModel);
 */
function buildTooltipModel(modelCascade, globalTooltipModel, defaultTooltipOption) {
  // Last is always tooltip model.
  var ecModel = globalTooltipModel.ecModel;
  var resultModel;
  if (defaultTooltipOption) {
    resultModel = new Model(defaultTooltipOption, ecModel, ecModel);
    resultModel = new Model(globalTooltipModel.option, resultModel, ecModel);
  } else {
    resultModel = globalTooltipModel;
  }
  for (var i = modelCascade.length - 1; i >= 0; i--) {
    var tooltipOpt = modelCascade[i];
    if (tooltipOpt) {
      if (tooltipOpt instanceof Model) {
        tooltipOpt = tooltipOpt.get('tooltip', true);
      }
      // In each data item tooltip can be simply write:
      // {
      //  value: 10,
      //  tooltip: 'Something you need to know'
      // }
      if (isString(tooltipOpt)) {
        tooltipOpt = {
          formatter: tooltipOpt
        };
      }
      if (tooltipOpt) {
        resultModel = new Model(tooltipOpt, resultModel, ecModel);
      }
    }
  }
  return resultModel;
}
function makeDispatchAction(payload, api) {
  return payload.dispatchAction || bind(api.dispatchAction, api);
}
function refixTooltipPosition(x, y, content, viewWidth, viewHeight, gapH, gapV) {
  var size = content.getSize();
  var width = size[0];
  var height = size[1];
  if (gapH != null) {
    // Add extra 2 pixels for this case:
    // At present the "values" in default tooltip are using CSS `float: right`.
    // When the right edge of the tooltip box is on the right side of the
    // viewport, the `float` layout might push the "values" to the second line.
    if (x + width + gapH + 2 > viewWidth) {
      x -= width + gapH;
    } else {
      x += gapH;
    }
  }
  if (gapV != null) {
    if (y + height + gapV > viewHeight) {
      y -= height + gapV;
    } else {
      y += gapV;
    }
  }
  return [x, y];
}
function confineTooltipPosition(x, y, content, viewWidth, viewHeight) {
  var size = content.getSize();
  var width = size[0];
  var height = size[1];
  x = Math.min(x + width, viewWidth) - width;
  y = Math.min(y + height, viewHeight) - height;
  x = Math.max(x, 0);
  y = Math.max(y, 0);
  return [x, y];
}
function calcTooltipPosition(position, rect, contentSize, borderWidth) {
  var domWidth = contentSize[0];
  var domHeight = contentSize[1];
  var offset = Math.ceil(Math.SQRT2 * borderWidth) + 8;
  var x = 0;
  var y = 0;
  var rectWidth = rect.width;
  var rectHeight = rect.height;
  switch (position) {
    case 'inside':
      x = rect.x + rectWidth / 2 - domWidth / 2;
      y = rect.y + rectHeight / 2 - domHeight / 2;
      break;
    case 'top':
      x = rect.x + rectWidth / 2 - domWidth / 2;
      y = rect.y - domHeight - offset;
      break;
    case 'bottom':
      x = rect.x + rectWidth / 2 - domWidth / 2;
      y = rect.y + rectHeight + offset;
      break;
    case 'left':
      x = rect.x - domWidth - offset;
      y = rect.y + rectHeight / 2 - domHeight / 2;
      break;
    case 'right':
      x = rect.x + rectWidth + offset;
      y = rect.y + rectHeight / 2 - domHeight / 2;
  }
  return [x, y];
}
function isCenterAlign(align) {
  return align === 'center' || align === 'middle';
}
/**
 * Find target component by payload like:
 * ```js
 * { legendId: 'some_id', name: 'xxx' }
 * { toolboxIndex: 1, name: 'xxx' }
 * { geoName: 'some_name', name: 'xxx' }
 * ```
 * PENDING: at present only
 *
 * If not found, return null/undefined.
 */
function findComponentReference(payload, ecModel, api) {
  var queryOptionMap = preParseFinder(payload).queryOptionMap;
  var componentMainType = queryOptionMap.keys()[0];
  if (!componentMainType || componentMainType === 'series') {
    return;
  }
  var queryResult = queryReferringComponents(ecModel, componentMainType, queryOptionMap.get(componentMainType), {
    useDefault: false,
    enableAll: false,
    enableNone: false
  });
  var model = queryResult.models[0];
  if (!model) {
    return;
  }
  var view = api.getViewOfComponentModel(model);
  var el;
  view.group.traverse(function (subEl) {
    var tooltipConfig = getECData(subEl).tooltipConfig;
    if (tooltipConfig && tooltipConfig.name === payload.name) {
      el = subEl;
      return true; // stop
    }
  });
  if (el) {
    return {
      componentMainType: componentMainType,
      componentIndex: model.componentIndex,
      el: el
    };
  }
}
export default TooltipView;