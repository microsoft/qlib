
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
import { getTooltipMarker, encodeHTML, makeValueReadable, convertToColorString } from '../../util/format.js';
import { isString, each, hasOwn, isArray, map, assert, extend } from 'zrender/lib/core/util.js';
import { SortOrderComparator } from '../../data/helper/dataValueHelper.js';
import { getRandomIdBase } from '../../util/number.js';
var TOOLTIP_LINE_HEIGHT_CSS = 'line-height:1';
function getTooltipLineHeight(textStyle) {
  var lineHeight = textStyle.lineHeight;
  if (lineHeight == null) {
    return TOOLTIP_LINE_HEIGHT_CSS;
  } else {
    return "line-height:" + encodeHTML(lineHeight + '') + "px";
  }
}
// TODO: more textStyle option
function getTooltipTextStyle(textStyle, renderMode) {
  var nameFontColor = textStyle.color || '#6e7079';
  var nameFontSize = textStyle.fontSize || 12;
  var nameFontWeight = textStyle.fontWeight || '400';
  var valueFontColor = textStyle.color || '#464646';
  var valueFontSize = textStyle.fontSize || 14;
  var valueFontWeight = textStyle.fontWeight || '900';
  if (renderMode === 'html') {
    // `textStyle` is probably from user input, should be encoded to reduce security risk.
    return {
      // eslint-disable-next-line max-len
      nameStyle: "font-size:" + encodeHTML(nameFontSize + '') + "px;color:" + encodeHTML(nameFontColor) + ";font-weight:" + encodeHTML(nameFontWeight + ''),
      // eslint-disable-next-line max-len
      valueStyle: "font-size:" + encodeHTML(valueFontSize + '') + "px;color:" + encodeHTML(valueFontColor) + ";font-weight:" + encodeHTML(valueFontWeight + '')
    };
  } else {
    return {
      nameStyle: {
        fontSize: nameFontSize,
        fill: nameFontColor,
        fontWeight: nameFontWeight
      },
      valueStyle: {
        fontSize: valueFontSize,
        fill: valueFontColor,
        fontWeight: valueFontWeight
      }
    };
  }
}
// See `TooltipMarkupLayoutIntent['innerGapLevel']`.
// (value from UI design)
var HTML_GAPS = [0, 10, 20, 30];
var RICH_TEXT_GAPS = ['', '\n', '\n\n', '\n\n\n'];
// eslint-disable-next-line max-len
export function createTooltipMarkup(type, option) {
  option.type = type;
  return option;
}
function isSectionFragment(frag) {
  return frag.type === 'section';
}
function getBuilder(frag) {
  return isSectionFragment(frag) ? buildSection : buildNameValue;
}
function getBlockGapLevel(frag) {
  if (isSectionFragment(frag)) {
    var gapLevel_1 = 0;
    var subBlockLen = frag.blocks.length;
    var hasInnerGap_1 = subBlockLen > 1 || subBlockLen > 0 && !frag.noHeader;
    each(frag.blocks, function (subBlock) {
      var subGapLevel = getBlockGapLevel(subBlock);
      // If the some of the sub-blocks have some gaps (like 10px) inside, this block
      // should use a larger gap (like 20px) to distinguish those sub-blocks.
      if (subGapLevel >= gapLevel_1) {
        gapLevel_1 = subGapLevel + +(hasInnerGap_1 && (
        // 0 always can not be readable gap level.
        !subGapLevel
        // If no header, always keep the sub gap level. Otherwise
        // look weird in case `multipleSeries`.
        || isSectionFragment(subBlock) && !subBlock.noHeader));
      }
    });
    return gapLevel_1;
  }
  return 0;
}
function buildSection(ctx, fragment, topMarginForOuterGap, toolTipTextStyle) {
  var noHeader = fragment.noHeader;
  var gaps = getGap(getBlockGapLevel(fragment));
  var subMarkupTextList = [];
  var subBlocks = fragment.blocks || [];
  assert(!subBlocks || isArray(subBlocks));
  subBlocks = subBlocks || [];
  var orderMode = ctx.orderMode;
  if (fragment.sortBlocks && orderMode) {
    subBlocks = subBlocks.slice();
    var orderMap = {
      valueAsc: 'asc',
      valueDesc: 'desc'
    };
    if (hasOwn(orderMap, orderMode)) {
      var comparator_1 = new SortOrderComparator(orderMap[orderMode], null);
      subBlocks.sort(function (a, b) {
        return comparator_1.evaluate(a.sortParam, b.sortParam);
      });
    }
    // FIXME 'seriesDesc' necessary?
    else if (orderMode === 'seriesDesc') {
      subBlocks.reverse();
    }
  }
  each(subBlocks, function (subBlock, idx) {
    var valueFormatter = fragment.valueFormatter;
    var subMarkupText = getBuilder(subBlock)(
    // Inherit valueFormatter
    valueFormatter ? extend(extend({}, ctx), {
      valueFormatter: valueFormatter
    }) : ctx, subBlock, idx > 0 ? gaps.html : 0, toolTipTextStyle);
    subMarkupText != null && subMarkupTextList.push(subMarkupText);
  });
  var subMarkupText = ctx.renderMode === 'richText' ? subMarkupTextList.join(gaps.richText) : wrapBlockHTML(toolTipTextStyle, subMarkupTextList.join(''), noHeader ? topMarginForOuterGap : gaps.html);
  if (noHeader) {
    return subMarkupText;
  }
  var displayableHeader = makeValueReadable(fragment.header, 'ordinal', ctx.useUTC);
  var nameStyle = getTooltipTextStyle(toolTipTextStyle, ctx.renderMode).nameStyle;
  var tooltipLineHeight = getTooltipLineHeight(toolTipTextStyle);
  if (ctx.renderMode === 'richText') {
    return wrapInlineNameRichText(ctx, displayableHeader, nameStyle) + gaps.richText + subMarkupText;
  } else {
    return wrapBlockHTML(toolTipTextStyle, "<div style=\"" + nameStyle + ";" + tooltipLineHeight + ";\">" + encodeHTML(displayableHeader) + '</div>' + subMarkupText, topMarginForOuterGap);
  }
}
function buildNameValue(ctx, fragment, topMarginForOuterGap, toolTipTextStyle) {
  var renderMode = ctx.renderMode;
  var noName = fragment.noName;
  var noValue = fragment.noValue;
  var noMarker = !fragment.markerType;
  var name = fragment.name;
  var useUTC = ctx.useUTC;
  var valueFormatter = fragment.valueFormatter || ctx.valueFormatter || function (value) {
    value = isArray(value) ? value : [value];
    return map(value, function (val, idx) {
      return makeValueReadable(val, isArray(valueTypeOption) ? valueTypeOption[idx] : valueTypeOption, useUTC);
    });
  };
  if (noName && noValue) {
    return;
  }
  var markerStr = noMarker ? '' : ctx.markupStyleCreator.makeTooltipMarker(fragment.markerType, fragment.markerColor || '#333', renderMode);
  var readableName = noName ? '' : makeValueReadable(name, 'ordinal', useUTC);
  var valueTypeOption = fragment.valueType;
  var readableValueList = noValue ? [] : valueFormatter(fragment.value, fragment.dataIndex);
  var valueAlignRight = !noMarker || !noName;
  // It little weird if only value next to marker but far from marker.
  var valueCloseToMarker = !noMarker && noName;
  var _a = getTooltipTextStyle(toolTipTextStyle, renderMode),
    nameStyle = _a.nameStyle,
    valueStyle = _a.valueStyle;
  return renderMode === 'richText' ? (noMarker ? '' : markerStr) + (noName ? '' : wrapInlineNameRichText(ctx, readableName, nameStyle))
  // Value has commas inside, so use ' ' as delimiter for multiple values.
  + (noValue ? '' : wrapInlineValueRichText(ctx, readableValueList, valueAlignRight, valueCloseToMarker, valueStyle)) : wrapBlockHTML(toolTipTextStyle, (noMarker ? '' : markerStr) + (noName ? '' : wrapInlineNameHTML(readableName, !noMarker, nameStyle)) + (noValue ? '' : wrapInlineValueHTML(readableValueList, valueAlignRight, valueCloseToMarker, valueStyle)), topMarginForOuterGap);
}
/**
 * @return markupText. null/undefined means no content.
 */
export function buildTooltipMarkup(fragment, markupStyleCreator, renderMode, orderMode, useUTC, toolTipTextStyle) {
  if (!fragment) {
    return;
  }
  var builder = getBuilder(fragment);
  var ctx = {
    useUTC: useUTC,
    renderMode: renderMode,
    orderMode: orderMode,
    markupStyleCreator: markupStyleCreator,
    valueFormatter: fragment.valueFormatter
  };
  return builder(ctx, fragment, 0, toolTipTextStyle);
}
function getGap(gapLevel) {
  return {
    html: HTML_GAPS[gapLevel],
    richText: RICH_TEXT_GAPS[gapLevel]
  };
}
function wrapBlockHTML(textStyle, encodedContent, topGap) {
  var clearfix = '<div style="clear:both"></div>';
  var marginCSS = "margin: " + topGap + "px 0 0";
  var tooltipLineHeight = getTooltipLineHeight(textStyle);
  return "<div style=\"" + marginCSS + ";" + tooltipLineHeight + ";\">" + encodedContent + clearfix + '</div>';
}
function wrapInlineNameHTML(name, leftHasMarker, style) {
  var marginCss = leftHasMarker ? 'margin-left:2px' : '';
  return "<span style=\"" + style + ";" + marginCss + "\">" + encodeHTML(name) + '</span>';
}
function wrapInlineValueHTML(valueList, alignRight, valueCloseToMarker, style) {
  // Do not too close to marker, considering there are multiple values separated by spaces.
  var paddingStr = valueCloseToMarker ? '10px' : '20px';
  var alignCSS = alignRight ? "float:right;margin-left:" + paddingStr : '';
  valueList = isArray(valueList) ? valueList : [valueList];
  return "<span style=\"" + alignCSS + ";" + style + "\">"
  // Value has commas inside, so use '  ' as delimiter for multiple values.
  + map(valueList, function (value) {
    return encodeHTML(value);
  }).join('&nbsp;&nbsp;') + '</span>';
}
function wrapInlineNameRichText(ctx, name, style) {
  return ctx.markupStyleCreator.wrapRichTextStyle(name, style);
}
function wrapInlineValueRichText(ctx, values, alignRight, valueCloseToMarker, style) {
  var styles = [style];
  var paddingLeft = valueCloseToMarker ? 10 : 20;
  alignRight && styles.push({
    padding: [0, 0, 0, paddingLeft],
    align: 'right'
  });
  // Value has commas inside, so use '  ' as delimiter for multiple values.
  return ctx.markupStyleCreator.wrapRichTextStyle(isArray(values) ? values.join('  ') : values, styles);
}
export function retrieveVisualColorForTooltipMarker(series, dataIndex) {
  var style = series.getData().getItemVisual(dataIndex, 'style');
  var color = style[series.visualDrawType];
  return convertToColorString(color);
}
export function getPaddingFromTooltipModel(model, renderMode) {
  var padding = model.get('padding');
  return padding != null ? padding
  // We give slightly different to look pretty.
  : renderMode === 'richText' ? [8, 10] : 10;
}
/**
 * The major feature is generate styles for `renderMode: 'richText'`.
 * But it also serves `renderMode: 'html'` to provide
 * "renderMode-independent" API.
 */
var TooltipMarkupStyleCreator = /** @class */function () {
  function TooltipMarkupStyleCreator() {
    this.richTextStyles = {};
    // Notice that "generate a style name" usually happens repeatedly when mouse is moving and
    // a tooltip is displayed. So we put the `_nextStyleNameId` as a member of each creator
    // rather than static shared by all creators (which will cause it increase to fast).
    this._nextStyleNameId = getRandomIdBase();
  }
  TooltipMarkupStyleCreator.prototype._generateStyleName = function () {
    return '__EC_aUTo_' + this._nextStyleNameId++;
  };
  TooltipMarkupStyleCreator.prototype.makeTooltipMarker = function (markerType, colorStr, renderMode) {
    var markerId = renderMode === 'richText' ? this._generateStyleName() : null;
    var marker = getTooltipMarker({
      color: colorStr,
      type: markerType,
      renderMode: renderMode,
      markerId: markerId
    });
    if (isString(marker)) {
      return marker;
    } else {
      if (process.env.NODE_ENV !== 'production') {
        assert(markerId);
      }
      this.richTextStyles[markerId] = marker.style;
      return marker.content;
    }
  };
  /**
   * @usage
   * ```ts
   * const styledText = markupStyleCreator.wrapRichTextStyle([
   *     // The styles will be auto merged.
   *     {
   *         fontSize: 12,
   *         color: 'blue'
   *     },
   *     {
   *         padding: 20
   *     }
   * ]);
   * ```
   */
  TooltipMarkupStyleCreator.prototype.wrapRichTextStyle = function (text, styles) {
    var finalStl = {};
    if (isArray(styles)) {
      each(styles, function (stl) {
        return extend(finalStl, stl);
      });
    } else {
      extend(finalStl, styles);
    }
    var styleName = this._generateStyleName();
    this.richTextStyles[styleName] = finalStl;
    return "{" + styleName + "|" + text + "}";
  };
  return TooltipMarkupStyleCreator;
}();
export { TooltipMarkupStyleCreator };