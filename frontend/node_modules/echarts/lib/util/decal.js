
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
import WeakMap from 'zrender/lib/core/WeakMap.js';
import LRU from 'zrender/lib/core/LRU.js';
import { defaults, map, isArray, isString, isNumber } from 'zrender/lib/core/util.js';
import { getLeastCommonMultiple } from './number.js';
import { createSymbol } from './symbol.js';
import { brushSingle } from 'zrender/lib/canvas/graphic.js';
import { platformApi } from 'zrender/lib/core/platform.js';
var decalMap = new WeakMap();
var decalCache = new LRU(100);
var decalKeys = ['symbol', 'symbolSize', 'symbolKeepAspect', 'color', 'backgroundColor', 'dashArrayX', 'dashArrayY', 'maxTileWidth', 'maxTileHeight'];
/**
 * Create or update pattern image from decal options
 *
 * @param {InnerDecalObject | 'none'} decalObject decal options, 'none' if no decal
 * @return {Pattern} pattern with generated image, null if no decal
 */
export function createOrUpdatePatternFromDecal(decalObject, api) {
  if (decalObject === 'none') {
    return null;
  }
  var dpr = api.getDevicePixelRatio();
  var zr = api.getZr();
  var isSVG = zr.painter.type === 'svg';
  if (decalObject.dirty) {
    decalMap["delete"](decalObject);
  }
  var oldPattern = decalMap.get(decalObject);
  if (oldPattern) {
    return oldPattern;
  }
  var decalOpt = defaults(decalObject, {
    symbol: 'rect',
    symbolSize: 1,
    symbolKeepAspect: true,
    color: 'rgba(0, 0, 0, 0.2)',
    backgroundColor: null,
    dashArrayX: 5,
    dashArrayY: 5,
    rotation: 0,
    maxTileWidth: 512,
    maxTileHeight: 512
  });
  if (decalOpt.backgroundColor === 'none') {
    decalOpt.backgroundColor = null;
  }
  var pattern = {
    repeat: 'repeat'
  };
  setPatternnSource(pattern);
  pattern.rotation = decalOpt.rotation;
  pattern.scaleX = pattern.scaleY = isSVG ? 1 : 1 / dpr;
  decalMap.set(decalObject, pattern);
  decalObject.dirty = false;
  return pattern;
  function setPatternnSource(pattern) {
    var keys = [dpr];
    var isValidKey = true;
    for (var i = 0; i < decalKeys.length; ++i) {
      var value = decalOpt[decalKeys[i]];
      if (value != null && !isArray(value) && !isString(value) && !isNumber(value) && typeof value !== 'boolean') {
        isValidKey = false;
        break;
      }
      keys.push(value);
    }
    var cacheKey;
    if (isValidKey) {
      cacheKey = keys.join(',') + (isSVG ? '-svg' : '');
      var cache = decalCache.get(cacheKey);
      if (cache) {
        isSVG ? pattern.svgElement = cache : pattern.image = cache;
      }
    }
    var dashArrayX = normalizeDashArrayX(decalOpt.dashArrayX);
    var dashArrayY = normalizeDashArrayY(decalOpt.dashArrayY);
    var symbolArray = normalizeSymbolArray(decalOpt.symbol);
    var lineBlockLengthsX = getLineBlockLengthX(dashArrayX);
    var lineBlockLengthY = getLineBlockLengthY(dashArrayY);
    var canvas = !isSVG && platformApi.createCanvas();
    var svgRoot = isSVG && {
      tag: 'g',
      attrs: {},
      key: 'dcl',
      children: []
    };
    var pSize = getPatternSize();
    var ctx;
    if (canvas) {
      canvas.width = pSize.width * dpr;
      canvas.height = pSize.height * dpr;
      ctx = canvas.getContext('2d');
    }
    brushDecal();
    if (isValidKey) {
      decalCache.put(cacheKey, canvas || svgRoot);
    }
    pattern.image = canvas;
    pattern.svgElement = svgRoot;
    pattern.svgWidth = pSize.width;
    pattern.svgHeight = pSize.height;
    /**
     * Get minimum length that can make a repeatable pattern.
     *
     * @return {Object} pattern width and height
     */
    function getPatternSize() {
      /**
       * For example, if dash is [[3, 2], [2, 1]] for X, it looks like
       * |---  ---  ---  ---  --- ...
       * |-- -- -- -- -- -- -- -- ...
       * |---  ---  ---  ---  --- ...
       * |-- -- -- -- -- -- -- -- ...
       * So the minimum length of X is 15,
       * which is the least common multiple of `3 + 2` and `2 + 1`
       * |---  ---  ---  |---  --- ...
       * |-- -- -- -- -- |-- -- -- ...
       */
      var width = 1;
      for (var i = 0, xlen = lineBlockLengthsX.length; i < xlen; ++i) {
        width = getLeastCommonMultiple(width, lineBlockLengthsX[i]);
      }
      var symbolRepeats = 1;
      for (var i = 0, xlen = symbolArray.length; i < xlen; ++i) {
        symbolRepeats = getLeastCommonMultiple(symbolRepeats, symbolArray[i].length);
      }
      width *= symbolRepeats;
      var height = lineBlockLengthY * lineBlockLengthsX.length * symbolArray.length;
      if (process.env.NODE_ENV !== 'production') {
        var warn = function (attrName) {
          /* eslint-disable-next-line */
          console.warn("Calculated decal size is greater than " + attrName + " due to decal option settings so " + attrName + " is used for the decal size. Please consider changing the decal option to make a smaller decal or set " + attrName + " to be larger to avoid incontinuity.");
        };
        if (width > decalOpt.maxTileWidth) {
          warn('maxTileWidth');
        }
        if (height > decalOpt.maxTileHeight) {
          warn('maxTileHeight');
        }
      }
      return {
        width: Math.max(1, Math.min(width, decalOpt.maxTileWidth)),
        height: Math.max(1, Math.min(height, decalOpt.maxTileHeight))
      };
    }
    function brushDecal() {
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (decalOpt.backgroundColor) {
          ctx.fillStyle = decalOpt.backgroundColor;
          ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
      }
      var ySum = 0;
      for (var i = 0; i < dashArrayY.length; ++i) {
        ySum += dashArrayY[i];
      }
      if (ySum <= 0) {
        // dashArrayY is 0, draw nothing
        return;
      }
      var y = -lineBlockLengthY;
      var yId = 0;
      var yIdTotal = 0;
      var xId0 = 0;
      while (y < pSize.height) {
        if (yId % 2 === 0) {
          var symbolYId = yIdTotal / 2 % symbolArray.length;
          var x = 0;
          var xId1 = 0;
          var xId1Total = 0;
          while (x < pSize.width * 2) {
            var xSum = 0;
            for (var i = 0; i < dashArrayX[xId0].length; ++i) {
              xSum += dashArrayX[xId0][i];
            }
            if (xSum <= 0) {
              // Skip empty line
              break;
            }
            // E.g., [15, 5, 20, 5] draws only for 15 and 20
            if (xId1 % 2 === 0) {
              var size = (1 - decalOpt.symbolSize) * 0.5;
              var left = x + dashArrayX[xId0][xId1] * size;
              var top_1 = y + dashArrayY[yId] * size;
              var width = dashArrayX[xId0][xId1] * decalOpt.symbolSize;
              var height = dashArrayY[yId] * decalOpt.symbolSize;
              var symbolXId = xId1Total / 2 % symbolArray[symbolYId].length;
              brushSymbol(left, top_1, width, height, symbolArray[symbolYId][symbolXId]);
            }
            x += dashArrayX[xId0][xId1];
            ++xId1Total;
            ++xId1;
            if (xId1 === dashArrayX[xId0].length) {
              xId1 = 0;
            }
          }
          ++xId0;
          if (xId0 === dashArrayX.length) {
            xId0 = 0;
          }
        }
        y += dashArrayY[yId];
        ++yIdTotal;
        ++yId;
        if (yId === dashArrayY.length) {
          yId = 0;
        }
      }
      function brushSymbol(x, y, width, height, symbolType) {
        var scale = isSVG ? 1 : dpr;
        var symbol = createSymbol(symbolType, x * scale, y * scale, width * scale, height * scale, decalOpt.color, decalOpt.symbolKeepAspect);
        if (isSVG) {
          var symbolVNode = zr.painter.renderOneToVNode(symbol);
          if (symbolVNode) {
            svgRoot.children.push(symbolVNode);
          }
        } else {
          // Paint to canvas for all other renderers.
          brushSingle(ctx, symbol);
        }
      }
    }
  }
}
/**
 * Convert symbol array into normalized array
 *
 * @param {string | (string | string[])[]} symbol symbol input
 * @return {string[][]} normolized symbol array
 */
function normalizeSymbolArray(symbol) {
  if (!symbol || symbol.length === 0) {
    return [['rect']];
  }
  if (isString(symbol)) {
    return [[symbol]];
  }
  var isAllString = true;
  for (var i = 0; i < symbol.length; ++i) {
    if (!isString(symbol[i])) {
      isAllString = false;
      break;
    }
  }
  if (isAllString) {
    return normalizeSymbolArray([symbol]);
  }
  var result = [];
  for (var i = 0; i < symbol.length; ++i) {
    if (isString(symbol[i])) {
      result.push([symbol[i]]);
    } else {
      result.push(symbol[i]);
    }
  }
  return result;
}
/**
 * Convert dash input into dashArray
 *
 * @param {DecalDashArrayX} dash dash input
 * @return {number[][]} normolized dash array
 */
function normalizeDashArrayX(dash) {
  if (!dash || dash.length === 0) {
    return [[0, 0]];
  }
  if (isNumber(dash)) {
    var dashValue = Math.ceil(dash);
    return [[dashValue, dashValue]];
  }
  /**
   * [20, 5] should be normalized into [[20, 5]],
   * while [20, [5, 10]] should be normalized into [[20, 20], [5, 10]]
   */
  var isAllNumber = true;
  for (var i = 0; i < dash.length; ++i) {
    if (!isNumber(dash[i])) {
      isAllNumber = false;
      break;
    }
  }
  if (isAllNumber) {
    return normalizeDashArrayX([dash]);
  }
  var result = [];
  for (var i = 0; i < dash.length; ++i) {
    if (isNumber(dash[i])) {
      var dashValue = Math.ceil(dash[i]);
      result.push([dashValue, dashValue]);
    } else {
      var dashValue = map(dash[i], function (n) {
        return Math.ceil(n);
      });
      if (dashValue.length % 2 === 1) {
        // [4, 2, 1] means |----  -    -- |----  -    -- |
        // so normalize it to be [4, 2, 1, 4, 2, 1]
        result.push(dashValue.concat(dashValue));
      } else {
        result.push(dashValue);
      }
    }
  }
  return result;
}
/**
 * Convert dash input into dashArray
 *
 * @param {DecalDashArrayY} dash dash input
 * @return {number[]} normolized dash array
 */
function normalizeDashArrayY(dash) {
  if (!dash || typeof dash === 'object' && dash.length === 0) {
    return [0, 0];
  }
  if (isNumber(dash)) {
    var dashValue_1 = Math.ceil(dash);
    return [dashValue_1, dashValue_1];
  }
  var dashValue = map(dash, function (n) {
    return Math.ceil(n);
  });
  return dash.length % 2 ? dashValue.concat(dashValue) : dashValue;
}
/**
 * Get block length of each line. A block is the length of dash line and space.
 * For example, a line with [4, 1] has a dash line of 4 and a space of 1 after
 * that, so the block length of this line is 5.
 *
 * @param {number[][]} dash dash array of X or Y
 * @return {number[]} block length of each line
 */
function getLineBlockLengthX(dash) {
  return map(dash, function (line) {
    return getLineBlockLengthY(line);
  });
}
function getLineBlockLengthY(dash) {
  var blockLength = 0;
  for (var i = 0; i < dash.length; ++i) {
    blockLength += dash[i];
  }
  if (dash.length % 2 === 1) {
    // [4, 2, 1] means |----  -    -- |----  -    -- |
    // So total length is (4 + 2 + 1) * 2
    return blockLength * 2;
  }
  return blockLength;
}