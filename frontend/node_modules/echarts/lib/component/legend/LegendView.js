
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
import { parse, stringify } from 'zrender/lib/tool/color.js';
import * as graphic from '../../util/graphic.js';
import { enableHoverEmphasis } from '../../util/states.js';
import { setLabelStyle, createTextStyle } from '../../label/labelStyle.js';
import { makeBackground } from '../helper/listComponent.js';
import * as layoutUtil from '../../util/layout.js';
import ComponentView from '../../view/Component.js';
import { createSymbol } from '../../util/symbol.js';
import { createOrUpdatePatternFromDecal } from '../../util/decal.js';
import { getECData } from '../../util/innerStore.js';
var curry = zrUtil.curry;
var each = zrUtil.each;
var Group = graphic.Group;
var LegendView = /** @class */function (_super) {
  __extends(LegendView, _super);
  function LegendView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = LegendView.type;
    _this.newlineDisabled = false;
    return _this;
  }
  LegendView.prototype.init = function () {
    this.group.add(this._contentGroup = new Group());
    this.group.add(this._selectorGroup = new Group());
    this._isFirstRender = true;
  };
  /**
   * @protected
   */
  LegendView.prototype.getContentGroup = function () {
    return this._contentGroup;
  };
  /**
   * @protected
   */
  LegendView.prototype.getSelectorGroup = function () {
    return this._selectorGroup;
  };
  /**
   * @override
   */
  LegendView.prototype.render = function (legendModel, ecModel, api) {
    var isFirstRender = this._isFirstRender;
    this._isFirstRender = false;
    this.resetInner();
    if (!legendModel.get('show', true)) {
      return;
    }
    var itemAlign = legendModel.get('align');
    var orient = legendModel.get('orient');
    if (!itemAlign || itemAlign === 'auto') {
      itemAlign = legendModel.get('left') === 'right' && orient === 'vertical' ? 'right' : 'left';
    }
    // selector has been normalized to an array in model
    var selector = legendModel.get('selector', true);
    var selectorPosition = legendModel.get('selectorPosition', true);
    if (selector && (!selectorPosition || selectorPosition === 'auto')) {
      selectorPosition = orient === 'horizontal' ? 'end' : 'start';
    }
    this.renderInner(itemAlign, legendModel, ecModel, api, selector, orient, selectorPosition);
    // Perform layout.
    var positionInfo = legendModel.getBoxLayoutParams();
    var viewportSize = {
      width: api.getWidth(),
      height: api.getHeight()
    };
    var padding = legendModel.get('padding');
    var maxSize = layoutUtil.getLayoutRect(positionInfo, viewportSize, padding);
    var mainRect = this.layoutInner(legendModel, itemAlign, maxSize, isFirstRender, selector, selectorPosition);
    // Place mainGroup, based on the calculated `mainRect`.
    var layoutRect = layoutUtil.getLayoutRect(zrUtil.defaults({
      width: mainRect.width,
      height: mainRect.height
    }, positionInfo), viewportSize, padding);
    this.group.x = layoutRect.x - mainRect.x;
    this.group.y = layoutRect.y - mainRect.y;
    this.group.markRedraw();
    // Render background after group is layout.
    this.group.add(this._backgroundEl = makeBackground(mainRect, legendModel));
  };
  LegendView.prototype.resetInner = function () {
    this.getContentGroup().removeAll();
    this._backgroundEl && this.group.remove(this._backgroundEl);
    this.getSelectorGroup().removeAll();
  };
  LegendView.prototype.renderInner = function (itemAlign, legendModel, ecModel, api, selector, orient, selectorPosition) {
    var contentGroup = this.getContentGroup();
    var legendDrawnMap = zrUtil.createHashMap();
    var selectMode = legendModel.get('selectedMode');
    var excludeSeriesId = [];
    ecModel.eachRawSeries(function (seriesModel) {
      !seriesModel.get('legendHoverLink') && excludeSeriesId.push(seriesModel.id);
    });
    each(legendModel.getData(), function (legendItemModel, dataIndex) {
      var name = legendItemModel.get('name');
      // Use empty string or \n as a newline string
      if (!this.newlineDisabled && (name === '' || name === '\n')) {
        var g = new Group();
        // @ts-ignore
        g.newline = true;
        contentGroup.add(g);
        return;
      }
      // Representitive series.
      var seriesModel = ecModel.getSeriesByName(name)[0];
      if (legendDrawnMap.get(name)) {
        // Have been drawn
        return;
      }
      // Legend to control series.
      if (seriesModel) {
        var data = seriesModel.getData();
        var lineVisualStyle = data.getVisual('legendLineStyle') || {};
        var legendIcon = data.getVisual('legendIcon');
        /**
         * `data.getVisual('style')` may be the color from the register
         * in series. For example, for line series,
         */
        var style = data.getVisual('style');
        var itemGroup = this._createItem(seriesModel, name, dataIndex, legendItemModel, legendModel, itemAlign, lineVisualStyle, style, legendIcon, selectMode, api);
        itemGroup.on('click', curry(dispatchSelectAction, name, null, api, excludeSeriesId)).on('mouseover', curry(dispatchHighlightAction, seriesModel.name, null, api, excludeSeriesId)).on('mouseout', curry(dispatchDownplayAction, seriesModel.name, null, api, excludeSeriesId));
        if (ecModel.ssr) {
          itemGroup.eachChild(function (child) {
            var ecData = getECData(child);
            ecData.seriesIndex = seriesModel.seriesIndex;
            ecData.dataIndex = dataIndex;
            ecData.ssrType = 'legend';
          });
        }
        legendDrawnMap.set(name, true);
      } else {
        // Legend to control data. In pie and funnel.
        ecModel.eachRawSeries(function (seriesModel) {
          // In case multiple series has same data name
          if (legendDrawnMap.get(name)) {
            return;
          }
          if (seriesModel.legendVisualProvider) {
            var provider = seriesModel.legendVisualProvider;
            if (!provider.containName(name)) {
              return;
            }
            var idx = provider.indexOfName(name);
            var style = provider.getItemVisual(idx, 'style');
            var legendIcon = provider.getItemVisual(idx, 'legendIcon');
            var colorArr = parse(style.fill);
            // Color may be set to transparent in visualMap when data is out of range.
            // Do not show nothing.
            if (colorArr && colorArr[3] === 0) {
              colorArr[3] = 0.2;
              // TODO color is set to 0, 0, 0, 0. Should show correct RGBA
              style = zrUtil.extend(zrUtil.extend({}, style), {
                fill: stringify(colorArr, 'rgba')
              });
            }
            var itemGroup = this._createItem(seriesModel, name, dataIndex, legendItemModel, legendModel, itemAlign, {}, style, legendIcon, selectMode, api);
            // FIXME: consider different series has items with the same name.
            itemGroup.on('click', curry(dispatchSelectAction, null, name, api, excludeSeriesId))
            // Should not specify the series name, consider legend controls
            // more than one pie series.
            .on('mouseover', curry(dispatchHighlightAction, null, name, api, excludeSeriesId)).on('mouseout', curry(dispatchDownplayAction, null, name, api, excludeSeriesId));
            if (ecModel.ssr) {
              itemGroup.eachChild(function (child) {
                var ecData = getECData(child);
                ecData.seriesIndex = seriesModel.seriesIndex;
                ecData.dataIndex = dataIndex;
                ecData.ssrType = 'legend';
              });
            }
            legendDrawnMap.set(name, true);
          }
        }, this);
      }
      if (process.env.NODE_ENV !== 'production') {
        if (!legendDrawnMap.get(name)) {
          console.warn(name + ' series not exists. Legend data should be same with series name or data name.');
        }
      }
    }, this);
    if (selector) {
      this._createSelector(selector, legendModel, api, orient, selectorPosition);
    }
  };
  LegendView.prototype._createSelector = function (selector, legendModel, api, orient, selectorPosition) {
    var selectorGroup = this.getSelectorGroup();
    each(selector, function createSelectorButton(selectorItem) {
      var type = selectorItem.type;
      var labelText = new graphic.Text({
        style: {
          x: 0,
          y: 0,
          align: 'center',
          verticalAlign: 'middle'
        },
        onclick: function () {
          api.dispatchAction({
            type: type === 'all' ? 'legendAllSelect' : 'legendInverseSelect',
            legendId: legendModel.id
          });
        }
      });
      selectorGroup.add(labelText);
      var labelModel = legendModel.getModel('selectorLabel');
      var emphasisLabelModel = legendModel.getModel(['emphasis', 'selectorLabel']);
      setLabelStyle(labelText, {
        normal: labelModel,
        emphasis: emphasisLabelModel
      }, {
        defaultText: selectorItem.title
      });
      enableHoverEmphasis(labelText);
    });
  };
  LegendView.prototype._createItem = function (seriesModel, name, dataIndex, legendItemModel, legendModel, itemAlign, lineVisualStyle, itemVisualStyle, legendIcon, selectMode, api) {
    var drawType = seriesModel.visualDrawType;
    var itemWidth = legendModel.get('itemWidth');
    var itemHeight = legendModel.get('itemHeight');
    var isSelected = legendModel.isSelected(name);
    var iconRotate = legendItemModel.get('symbolRotate');
    var symbolKeepAspect = legendItemModel.get('symbolKeepAspect');
    var legendIconType = legendItemModel.get('icon');
    legendIcon = legendIconType || legendIcon || 'roundRect';
    var style = getLegendStyle(legendIcon, legendItemModel, lineVisualStyle, itemVisualStyle, drawType, isSelected, api);
    var itemGroup = new Group();
    var textStyleModel = legendItemModel.getModel('textStyle');
    if (zrUtil.isFunction(seriesModel.getLegendIcon) && (!legendIconType || legendIconType === 'inherit')) {
      // Series has specific way to define legend icon
      itemGroup.add(seriesModel.getLegendIcon({
        itemWidth: itemWidth,
        itemHeight: itemHeight,
        icon: legendIcon,
        iconRotate: iconRotate,
        itemStyle: style.itemStyle,
        lineStyle: style.lineStyle,
        symbolKeepAspect: symbolKeepAspect
      }));
    } else {
      // Use default legend icon policy for most series
      var rotate = legendIconType === 'inherit' && seriesModel.getData().getVisual('symbol') ? iconRotate === 'inherit' ? seriesModel.getData().getVisual('symbolRotate') : iconRotate : 0; // No rotation for no icon
      itemGroup.add(getDefaultLegendIcon({
        itemWidth: itemWidth,
        itemHeight: itemHeight,
        icon: legendIcon,
        iconRotate: rotate,
        itemStyle: style.itemStyle,
        lineStyle: style.lineStyle,
        symbolKeepAspect: symbolKeepAspect
      }));
    }
    var textX = itemAlign === 'left' ? itemWidth + 5 : -5;
    var textAlign = itemAlign;
    var formatter = legendModel.get('formatter');
    var content = name;
    if (zrUtil.isString(formatter) && formatter) {
      content = formatter.replace('{name}', name != null ? name : '');
    } else if (zrUtil.isFunction(formatter)) {
      content = formatter(name);
    }
    var textColor = isSelected ? textStyleModel.getTextColor() : legendItemModel.get('inactiveColor');
    itemGroup.add(new graphic.Text({
      style: createTextStyle(textStyleModel, {
        text: content,
        x: textX,
        y: itemHeight / 2,
        fill: textColor,
        align: textAlign,
        verticalAlign: 'middle'
      }, {
        inheritColor: textColor
      })
    }));
    // Add a invisible rect to increase the area of mouse hover
    var hitRect = new graphic.Rect({
      shape: itemGroup.getBoundingRect(),
      style: {
        // Cannot use 'invisible' because SVG SSR will miss the node
        fill: 'transparent'
      }
    });
    var tooltipModel = legendItemModel.getModel('tooltip');
    if (tooltipModel.get('show')) {
      graphic.setTooltipConfig({
        el: hitRect,
        componentModel: legendModel,
        itemName: name,
        itemTooltipOption: tooltipModel.option
      });
    }
    itemGroup.add(hitRect);
    itemGroup.eachChild(function (child) {
      child.silent = true;
    });
    hitRect.silent = !selectMode;
    this.getContentGroup().add(itemGroup);
    enableHoverEmphasis(itemGroup);
    // @ts-ignore
    itemGroup.__legendDataIndex = dataIndex;
    return itemGroup;
  };
  LegendView.prototype.layoutInner = function (legendModel, itemAlign, maxSize, isFirstRender, selector, selectorPosition) {
    var contentGroup = this.getContentGroup();
    var selectorGroup = this.getSelectorGroup();
    // Place items in contentGroup.
    layoutUtil.box(legendModel.get('orient'), contentGroup, legendModel.get('itemGap'), maxSize.width, maxSize.height);
    var contentRect = contentGroup.getBoundingRect();
    var contentPos = [-contentRect.x, -contentRect.y];
    selectorGroup.markRedraw();
    contentGroup.markRedraw();
    if (selector) {
      // Place buttons in selectorGroup
      layoutUtil.box(
      // Buttons in selectorGroup always layout horizontally
      'horizontal', selectorGroup, legendModel.get('selectorItemGap', true));
      var selectorRect = selectorGroup.getBoundingRect();
      var selectorPos = [-selectorRect.x, -selectorRect.y];
      var selectorButtonGap = legendModel.get('selectorButtonGap', true);
      var orientIdx = legendModel.getOrient().index;
      var wh = orientIdx === 0 ? 'width' : 'height';
      var hw = orientIdx === 0 ? 'height' : 'width';
      var yx = orientIdx === 0 ? 'y' : 'x';
      if (selectorPosition === 'end') {
        selectorPos[orientIdx] += contentRect[wh] + selectorButtonGap;
      } else {
        contentPos[orientIdx] += selectorRect[wh] + selectorButtonGap;
      }
      // Always align selector to content as 'middle'
      selectorPos[1 - orientIdx] += contentRect[hw] / 2 - selectorRect[hw] / 2;
      selectorGroup.x = selectorPos[0];
      selectorGroup.y = selectorPos[1];
      contentGroup.x = contentPos[0];
      contentGroup.y = contentPos[1];
      var mainRect = {
        x: 0,
        y: 0
      };
      mainRect[wh] = contentRect[wh] + selectorButtonGap + selectorRect[wh];
      mainRect[hw] = Math.max(contentRect[hw], selectorRect[hw]);
      mainRect[yx] = Math.min(0, selectorRect[yx] + selectorPos[1 - orientIdx]);
      return mainRect;
    } else {
      contentGroup.x = contentPos[0];
      contentGroup.y = contentPos[1];
      return this.group.getBoundingRect();
    }
  };
  /**
   * @protected
   */
  LegendView.prototype.remove = function () {
    this.getContentGroup().removeAll();
    this._isFirstRender = true;
  };
  LegendView.type = 'legend.plain';
  return LegendView;
}(ComponentView);
function getLegendStyle(iconType, legendItemModel, lineVisualStyle, itemVisualStyle, drawType, isSelected, api) {
  /**
   * Use series style if is inherit;
   * elsewise, use legend style
   */
  function handleCommonProps(style, visualStyle) {
    // If lineStyle.width is 'auto', it is set to be 2 if series has border
    if (style.lineWidth === 'auto') {
      style.lineWidth = visualStyle.lineWidth > 0 ? 2 : 0;
    }
    each(style, function (propVal, propName) {
      style[propName] === 'inherit' && (style[propName] = visualStyle[propName]);
    });
  }
  // itemStyle
  var itemStyleModel = legendItemModel.getModel('itemStyle');
  var itemStyle = itemStyleModel.getItemStyle();
  var iconBrushType = iconType.lastIndexOf('empty', 0) === 0 ? 'fill' : 'stroke';
  var decalStyle = itemStyleModel.getShallow('decal');
  itemStyle.decal = !decalStyle || decalStyle === 'inherit' ? itemVisualStyle.decal : createOrUpdatePatternFromDecal(decalStyle, api);
  if (itemStyle.fill === 'inherit') {
    /**
     * Series with visualDrawType as 'stroke' should have
     * series stroke as legend fill
     */
    itemStyle.fill = itemVisualStyle[drawType];
  }
  if (itemStyle.stroke === 'inherit') {
    /**
     * icon type with "emptyXXX" should use fill color
     * in visual style
     */
    itemStyle.stroke = itemVisualStyle[iconBrushType];
  }
  if (itemStyle.opacity === 'inherit') {
    /**
     * Use lineStyle.opacity if drawType is stroke
     */
    itemStyle.opacity = (drawType === 'fill' ? itemVisualStyle : lineVisualStyle).opacity;
  }
  handleCommonProps(itemStyle, itemVisualStyle);
  // lineStyle
  var legendLineModel = legendItemModel.getModel('lineStyle');
  var lineStyle = legendLineModel.getLineStyle();
  handleCommonProps(lineStyle, lineVisualStyle);
  // Fix auto color to real color
  itemStyle.fill === 'auto' && (itemStyle.fill = itemVisualStyle.fill);
  itemStyle.stroke === 'auto' && (itemStyle.stroke = itemVisualStyle.fill);
  lineStyle.stroke === 'auto' && (lineStyle.stroke = itemVisualStyle.fill);
  if (!isSelected) {
    var borderWidth = legendItemModel.get('inactiveBorderWidth');
    /**
     * Since stroke is set to be inactiveBorderColor, it may occur that
     * there is no border in series but border in legend, so we need to
     * use border only when series has border if is set to be auto
     */
    var visualHasBorder = itemStyle[iconBrushType];
    itemStyle.lineWidth = borderWidth === 'auto' ? itemVisualStyle.lineWidth > 0 && visualHasBorder ? 2 : 0 : itemStyle.lineWidth;
    itemStyle.fill = legendItemModel.get('inactiveColor');
    itemStyle.stroke = legendItemModel.get('inactiveBorderColor');
    lineStyle.stroke = legendLineModel.get('inactiveColor');
    lineStyle.lineWidth = legendLineModel.get('inactiveWidth');
  }
  return {
    itemStyle: itemStyle,
    lineStyle: lineStyle
  };
}
function getDefaultLegendIcon(opt) {
  var symboType = opt.icon || 'roundRect';
  var icon = createSymbol(symboType, 0, 0, opt.itemWidth, opt.itemHeight, opt.itemStyle.fill, opt.symbolKeepAspect);
  icon.setStyle(opt.itemStyle);
  icon.rotation = (opt.iconRotate || 0) * Math.PI / 180;
  icon.setOrigin([opt.itemWidth / 2, opt.itemHeight / 2]);
  if (symboType.indexOf('empty') > -1) {
    icon.style.stroke = icon.style.fill;
    icon.style.fill = '#fff';
    icon.style.lineWidth = 2;
  }
  return icon;
}
function dispatchSelectAction(seriesName, dataName, api, excludeSeriesId) {
  // downplay before unselect
  dispatchDownplayAction(seriesName, dataName, api, excludeSeriesId);
  api.dispatchAction({
    type: 'legendToggleSelect',
    name: seriesName != null ? seriesName : dataName
  });
  // highlight after select
  // TODO highlight immediately may cause animation loss.
  dispatchHighlightAction(seriesName, dataName, api, excludeSeriesId);
}
function isUseHoverLayer(api) {
  var list = api.getZr().storage.getDisplayList();
  var emphasisState;
  var i = 0;
  var len = list.length;
  while (i < len && !(emphasisState = list[i].states.emphasis)) {
    i++;
  }
  return emphasisState && emphasisState.hoverLayer;
}
function dispatchHighlightAction(seriesName, dataName, api, excludeSeriesId) {
  // If element hover will move to a hoverLayer.
  if (!isUseHoverLayer(api)) {
    api.dispatchAction({
      type: 'highlight',
      seriesName: seriesName,
      name: dataName,
      excludeSeriesId: excludeSeriesId
    });
  }
}
function dispatchDownplayAction(seriesName, dataName, api, excludeSeriesId) {
  // If element hover will move to a hoverLayer.
  if (!isUseHoverLayer(api)) {
    api.dispatchAction({
      type: 'downplay',
      seriesName: seriesName,
      name: dataName,
      excludeSeriesId: excludeSeriesId
    });
  }
}
export default LegendView;