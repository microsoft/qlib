
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
import * as zrUtil from 'zrender/lib/core/util.js';
import { encodeHTML } from 'zrender/lib/core/dom.js';
import { parseDate, isNumeric, numericToNumber } from './number.js';
import { format as timeFormat, pad } from './time.js';
import { deprecateReplaceLog } from './log.js';
/**
 * Add a comma each three digit.
 */
export function addCommas(x) {
  if (!isNumeric(x)) {
    return zrUtil.isString(x) ? x : '-';
  }
  var parts = (x + '').split('.');
  return parts[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, '$1,') + (parts.length > 1 ? '.' + parts[1] : '');
}
export function toCamelCase(str, upperCaseFirst) {
  str = (str || '').toLowerCase().replace(/-(.)/g, function (match, group1) {
    return group1.toUpperCase();
  });
  if (upperCaseFirst && str) {
    str = str.charAt(0).toUpperCase() + str.slice(1);
  }
  return str;
}
export var normalizeCssArray = zrUtil.normalizeCssArray;
export { encodeHTML };
/**
 * Make value user readable for tooltip and label.
 * "User readable":
 *     Try to not print programmer-specific text like NaN, Infinity, null, undefined.
 *     Avoid to display an empty string, which users can not recognize there is
 *     a value and it might look like a bug.
 */
export function makeValueReadable(value, valueType, useUTC) {
  var USER_READABLE_DEFUALT_TIME_PATTERN = '{yyyy}-{MM}-{dd} {HH}:{mm}:{ss}';
  function stringToUserReadable(str) {
    return str && zrUtil.trim(str) ? str : '-';
  }
  function isNumberUserReadable(num) {
    return !!(num != null && !isNaN(num) && isFinite(num));
  }
  var isTypeTime = valueType === 'time';
  var isValueDate = value instanceof Date;
  if (isTypeTime || isValueDate) {
    var date = isTypeTime ? parseDate(value) : value;
    if (!isNaN(+date)) {
      return timeFormat(date, USER_READABLE_DEFUALT_TIME_PATTERN, useUTC);
    } else if (isValueDate) {
      return '-';
    }
    // In other cases, continue to try to display the value in the following code.
  }
  if (valueType === 'ordinal') {
    return zrUtil.isStringSafe(value) ? stringToUserReadable(value) : zrUtil.isNumber(value) ? isNumberUserReadable(value) ? value + '' : '-' : '-';
  }
  // By default.
  var numericResult = numericToNumber(value);
  return isNumberUserReadable(numericResult) ? addCommas(numericResult) : zrUtil.isStringSafe(value) ? stringToUserReadable(value) : typeof value === 'boolean' ? value + '' : '-';
}
var TPL_VAR_ALIAS = ['a', 'b', 'c', 'd', 'e', 'f', 'g'];
var wrapVar = function (varName, seriesIdx) {
  return '{' + varName + (seriesIdx == null ? '' : seriesIdx) + '}';
};
/**
 * Template formatter
 * @param {Array.<Object>|Object} paramsList
 */
export function formatTpl(tpl, paramsList, encode) {
  if (!zrUtil.isArray(paramsList)) {
    paramsList = [paramsList];
  }
  var seriesLen = paramsList.length;
  if (!seriesLen) {
    return '';
  }
  var $vars = paramsList[0].$vars || [];
  for (var i = 0; i < $vars.length; i++) {
    var alias = TPL_VAR_ALIAS[i];
    tpl = tpl.replace(wrapVar(alias), wrapVar(alias, 0));
  }
  for (var seriesIdx = 0; seriesIdx < seriesLen; seriesIdx++) {
    for (var k = 0; k < $vars.length; k++) {
      var val = paramsList[seriesIdx][$vars[k]];
      tpl = tpl.replace(wrapVar(TPL_VAR_ALIAS[k], seriesIdx), encode ? encodeHTML(val) : val);
    }
  }
  return tpl;
}
/**
 * simple Template formatter
 */
export function formatTplSimple(tpl, param, encode) {
  zrUtil.each(param, function (value, key) {
    tpl = tpl.replace('{' + key + '}', encode ? encodeHTML(value) : value);
  });
  return tpl;
}
export function getTooltipMarker(inOpt, extraCssText) {
  var opt = zrUtil.isString(inOpt) ? {
    color: inOpt,
    extraCssText: extraCssText
  } : inOpt || {};
  var color = opt.color;
  var type = opt.type;
  extraCssText = opt.extraCssText;
  var renderMode = opt.renderMode || 'html';
  if (!color) {
    return '';
  }
  if (renderMode === 'html') {
    return type === 'subItem' ? '<span style="display:inline-block;vertical-align:middle;margin-right:8px;margin-left:3px;' + 'border-radius:4px;width:4px;height:4px;background-color:'
    // Only support string
    + encodeHTML(color) + ';' + (extraCssText || '') + '"></span>' : '<span style="display:inline-block;margin-right:4px;' + 'border-radius:10px;width:10px;height:10px;background-color:' + encodeHTML(color) + ';' + (extraCssText || '') + '"></span>';
  } else {
    // Should better not to auto generate style name by auto-increment number here.
    // Because this util is usually called in tooltip formatter, which is probably
    // called repeatedly when mouse move and the auto-increment number increases fast.
    // Users can make their own style name by theirselves, make it unique and readable.
    var markerId = opt.markerId || 'markerX';
    return {
      renderMode: renderMode,
      content: '{' + markerId + '|}  ',
      style: type === 'subItem' ? {
        width: 4,
        height: 4,
        borderRadius: 2,
        backgroundColor: color
      } : {
        width: 10,
        height: 10,
        borderRadius: 5,
        backgroundColor: color
      }
    };
  }
}
/**
 * @deprecated Use `time/format` instead.
 * ISO Date format
 * @param {string} tpl
 * @param {number} value
 * @param {boolean} [isUTC=false] Default in local time.
 *           see `module:echarts/scale/Time`
 *           and `module:echarts/util/number#parseDate`.
 * @inner
 */
export function formatTime(tpl, value, isUTC) {
  if (process.env.NODE_ENV !== 'production') {
    deprecateReplaceLog('echarts.format.formatTime', 'echarts.time.format');
  }
  if (tpl === 'week' || tpl === 'month' || tpl === 'quarter' || tpl === 'half-year' || tpl === 'year') {
    tpl = 'MM-dd\nyyyy';
  }
  var date = parseDate(value);
  var getUTC = isUTC ? 'getUTC' : 'get';
  var y = date[getUTC + 'FullYear']();
  var M = date[getUTC + 'Month']() + 1;
  var d = date[getUTC + 'Date']();
  var h = date[getUTC + 'Hours']();
  var m = date[getUTC + 'Minutes']();
  var s = date[getUTC + 'Seconds']();
  var S = date[getUTC + 'Milliseconds']();
  tpl = tpl.replace('MM', pad(M, 2)).replace('M', M).replace('yyyy', y).replace('yy', pad(y % 100 + '', 2)).replace('dd', pad(d, 2)).replace('d', d).replace('hh', pad(h, 2)).replace('h', h).replace('mm', pad(m, 2)).replace('m', m).replace('ss', pad(s, 2)).replace('s', s).replace('SSS', pad(S, 3));
  return tpl;
}
/**
 * Capital first
 * @param {string} str
 * @return {string}
 */
export function capitalFirst(str) {
  return str ? str.charAt(0).toUpperCase() + str.substr(1) : str;
}
/**
 * @return Never be null/undefined.
 */
export function convertToColorString(color, defaultColor) {
  defaultColor = defaultColor || 'transparent';
  return zrUtil.isString(color) ? color : zrUtil.isObject(color) ? color.colorStops && (color.colorStops[0] || {}).color || defaultColor : defaultColor;
}
export { truncateText } from 'zrender/lib/graphic/helper/parseText.js';
/**
 * open new tab
 * @param link url
 * @param target blank or self
 */
export function windowOpen(link, target) {
  /* global window */
  if (target === '_blank' || target === 'blank') {
    var blank = window.open();
    blank.opener = null;
    blank.location.href = link;
  } else {
    window.open(link, target);
  }
}
export { getTextRect } from '../legacy/getTextRect.js';