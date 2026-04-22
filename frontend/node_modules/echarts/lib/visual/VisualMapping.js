
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
import * as zrColor from 'zrender/lib/tool/color.js';
import { linearMap } from '../util/number.js';
import { warn } from '../util/log.js';
var each = zrUtil.each;
var isObject = zrUtil.isObject;
var CATEGORY_DEFAULT_VISUAL_INDEX = -1;
var VisualMapping = /** @class */function () {
  function VisualMapping(option) {
    var mappingMethod = option.mappingMethod;
    var visualType = option.type;
    var thisOption = this.option = zrUtil.clone(option);
    this.type = visualType;
    this.mappingMethod = mappingMethod;
    this._normalizeData = normalizers[mappingMethod];
    var visualHandler = VisualMapping.visualHandlers[visualType];
    this.applyVisual = visualHandler.applyVisual;
    this.getColorMapper = visualHandler.getColorMapper;
    this._normalizedToVisual = visualHandler._normalizedToVisual[mappingMethod];
    if (mappingMethod === 'piecewise') {
      normalizeVisualRange(thisOption);
      preprocessForPiecewise(thisOption);
    } else if (mappingMethod === 'category') {
      thisOption.categories ? preprocessForSpecifiedCategory(thisOption)
      // categories is ordinal when thisOption.categories not specified,
      // which need no more preprocess except normalize visual.
      : normalizeVisualRange(thisOption, true);
    } else {
      // mappingMethod === 'linear' or 'fixed'
      zrUtil.assert(mappingMethod !== 'linear' || thisOption.dataExtent);
      normalizeVisualRange(thisOption);
    }
  }
  VisualMapping.prototype.mapValueToVisual = function (value) {
    var normalized = this._normalizeData(value);
    return this._normalizedToVisual(normalized, value);
  };
  VisualMapping.prototype.getNormalizer = function () {
    return zrUtil.bind(this._normalizeData, this);
  };
  /**
   * List available visual types.
   *
   * @public
   * @return {Array.<string>}
   */
  VisualMapping.listVisualTypes = function () {
    return zrUtil.keys(VisualMapping.visualHandlers);
  };
  // /**
  //  * @public
  //  */
  // static addVisualHandler(name, handler) {
  //     visualHandlers[name] = handler;
  // }
  /**
   * @public
   */
  VisualMapping.isValidType = function (visualType) {
    return VisualMapping.visualHandlers.hasOwnProperty(visualType);
  };
  /**
   * Convenient method.
   * Visual can be Object or Array or primary type.
   */
  VisualMapping.eachVisual = function (visual, callback, context) {
    if (zrUtil.isObject(visual)) {
      zrUtil.each(visual, callback, context);
    } else {
      callback.call(context, visual);
    }
  };
  VisualMapping.mapVisual = function (visual, callback, context) {
    var isPrimary;
    var newVisual = zrUtil.isArray(visual) ? [] : zrUtil.isObject(visual) ? {} : (isPrimary = true, null);
    VisualMapping.eachVisual(visual, function (v, key) {
      var newVal = callback.call(context, v, key);
      isPrimary ? newVisual = newVal : newVisual[key] = newVal;
    });
    return newVisual;
  };
  /**
   * Retrieve visual properties from given object.
   */
  VisualMapping.retrieveVisuals = function (obj) {
    var ret = {};
    var hasVisual;
    obj && each(VisualMapping.visualHandlers, function (h, visualType) {
      if (obj.hasOwnProperty(visualType)) {
        ret[visualType] = obj[visualType];
        hasVisual = true;
      }
    });
    return hasVisual ? ret : null;
  };
  /**
   * Give order to visual types, considering colorSaturation, colorAlpha depends on color.
   *
   * @public
   * @param {(Object|Array)} visualTypes If Object, like: {color: ..., colorSaturation: ...}
   *                                     IF Array, like: ['color', 'symbol', 'colorSaturation']
   * @return {Array.<string>} Sorted visual types.
   */
  VisualMapping.prepareVisualTypes = function (visualTypes) {
    if (zrUtil.isArray(visualTypes)) {
      visualTypes = visualTypes.slice();
    } else if (isObject(visualTypes)) {
      var types_1 = [];
      each(visualTypes, function (item, type) {
        types_1.push(type);
      });
      visualTypes = types_1;
    } else {
      return [];
    }
    visualTypes.sort(function (type1, type2) {
      // color should be front of colorSaturation, colorAlpha, ...
      // symbol and symbolSize do not matter.
      return type2 === 'color' && type1 !== 'color' && type1.indexOf('color') === 0 ? 1 : -1;
    });
    return visualTypes;
  };
  /**
   * 'color', 'colorSaturation', 'colorAlpha', ... are depends on 'color'.
   * Other visuals are only depends on themself.
   */
  VisualMapping.dependsOn = function (visualType1, visualType2) {
    return visualType2 === 'color' ? !!(visualType1 && visualType1.indexOf(visualType2) === 0) : visualType1 === visualType2;
  };
  /**
   * @param value
   * @param pieceList [{value: ..., interval: [min, max]}, ...]
   *                         Always from small to big.
   * @param findClosestWhenOutside Default to be false
   * @return index
   */
  VisualMapping.findPieceIndex = function (value, pieceList, findClosestWhenOutside) {
    var possibleI;
    var abs = Infinity;
    // value has the higher priority.
    for (var i = 0, len = pieceList.length; i < len; i++) {
      var pieceValue = pieceList[i].value;
      if (pieceValue != null) {
        if (pieceValue === value
        // FIXME
        // It is supposed to compare value according to value type of dimension,
        // but currently value type can exactly be string or number.
        // Compromise for numeric-like string (like '12'), especially
        // in the case that visualMap.categories is ['22', '33'].
        || zrUtil.isString(pieceValue) && pieceValue === value + '') {
          return i;
        }
        findClosestWhenOutside && updatePossible(pieceValue, i);
      }
    }
    for (var i = 0, len = pieceList.length; i < len; i++) {
      var piece = pieceList[i];
      var interval = piece.interval;
      var close_1 = piece.close;
      if (interval) {
        if (interval[0] === -Infinity) {
          if (littleThan(close_1[1], value, interval[1])) {
            return i;
          }
        } else if (interval[1] === Infinity) {
          if (littleThan(close_1[0], interval[0], value)) {
            return i;
          }
        } else if (littleThan(close_1[0], interval[0], value) && littleThan(close_1[1], value, interval[1])) {
          return i;
        }
        findClosestWhenOutside && updatePossible(interval[0], i);
        findClosestWhenOutside && updatePossible(interval[1], i);
      }
    }
    if (findClosestWhenOutside) {
      return value === Infinity ? pieceList.length - 1 : value === -Infinity ? 0 : possibleI;
    }
    function updatePossible(val, index) {
      var newAbs = Math.abs(val - value);
      if (newAbs < abs) {
        abs = newAbs;
        possibleI = index;
      }
    }
  };
  VisualMapping.visualHandlers = {
    color: {
      applyVisual: makeApplyVisual('color'),
      getColorMapper: function () {
        var thisOption = this.option;
        return zrUtil.bind(thisOption.mappingMethod === 'category' ? function (value, isNormalized) {
          !isNormalized && (value = this._normalizeData(value));
          return doMapCategory.call(this, value);
        } : function (value, isNormalized, out) {
          // If output rgb array
          // which will be much faster and useful in pixel manipulation
          var returnRGBArray = !!out;
          !isNormalized && (value = this._normalizeData(value));
          out = zrColor.fastLerp(value, thisOption.parsedVisual, out);
          return returnRGBArray ? out : zrColor.stringify(out, 'rgba');
        }, this);
      },
      _normalizedToVisual: {
        linear: function (normalized) {
          return zrColor.stringify(zrColor.fastLerp(normalized, this.option.parsedVisual), 'rgba');
        },
        category: doMapCategory,
        piecewise: function (normalized, value) {
          var result = getSpecifiedVisual.call(this, value);
          if (result == null) {
            result = zrColor.stringify(zrColor.fastLerp(normalized, this.option.parsedVisual), 'rgba');
          }
          return result;
        },
        fixed: doMapFixed
      }
    },
    colorHue: makePartialColorVisualHandler(function (color, value) {
      return zrColor.modifyHSL(color, value);
    }),
    colorSaturation: makePartialColorVisualHandler(function (color, value) {
      return zrColor.modifyHSL(color, null, value);
    }),
    colorLightness: makePartialColorVisualHandler(function (color, value) {
      return zrColor.modifyHSL(color, null, null, value);
    }),
    colorAlpha: makePartialColorVisualHandler(function (color, value) {
      return zrColor.modifyAlpha(color, value);
    }),
    decal: {
      applyVisual: makeApplyVisual('decal'),
      _normalizedToVisual: {
        linear: null,
        category: doMapCategory,
        piecewise: null,
        fixed: null
      }
    },
    opacity: {
      applyVisual: makeApplyVisual('opacity'),
      _normalizedToVisual: createNormalizedToNumericVisual([0, 1])
    },
    liftZ: {
      applyVisual: makeApplyVisual('liftZ'),
      _normalizedToVisual: {
        linear: doMapFixed,
        category: doMapFixed,
        piecewise: doMapFixed,
        fixed: doMapFixed
      }
    },
    symbol: {
      applyVisual: function (value, getter, setter) {
        var symbolCfg = this.mapValueToVisual(value);
        setter('symbol', symbolCfg);
      },
      _normalizedToVisual: {
        linear: doMapToArray,
        category: doMapCategory,
        piecewise: function (normalized, value) {
          var result = getSpecifiedVisual.call(this, value);
          if (result == null) {
            result = doMapToArray.call(this, normalized);
          }
          return result;
        },
        fixed: doMapFixed
      }
    },
    symbolSize: {
      applyVisual: makeApplyVisual('symbolSize'),
      _normalizedToVisual: createNormalizedToNumericVisual([0, 1])
    }
  };
  return VisualMapping;
}();
function preprocessForPiecewise(thisOption) {
  var pieceList = thisOption.pieceList;
  thisOption.hasSpecialVisual = false;
  zrUtil.each(pieceList, function (piece, index) {
    piece.originIndex = index;
    // piece.visual is "result visual value" but not
    // a visual range, so it does not need to be normalized.
    if (piece.visual != null) {
      thisOption.hasSpecialVisual = true;
    }
  });
}
function preprocessForSpecifiedCategory(thisOption) {
  // Hash categories.
  var categories = thisOption.categories;
  var categoryMap = thisOption.categoryMap = {};
  var visual = thisOption.visual;
  each(categories, function (cate, index) {
    categoryMap[cate] = index;
  });
  // Process visual map input.
  if (!zrUtil.isArray(visual)) {
    var visualArr_1 = [];
    if (zrUtil.isObject(visual)) {
      each(visual, function (v, cate) {
        var index = categoryMap[cate];
        visualArr_1[index != null ? index : CATEGORY_DEFAULT_VISUAL_INDEX] = v;
      });
    } else {
      // Is primary type, represents default visual.
      visualArr_1[CATEGORY_DEFAULT_VISUAL_INDEX] = visual;
    }
    visual = setVisualToOption(thisOption, visualArr_1);
  }
  // Remove categories that has no visual,
  // then we can mapping them to CATEGORY_DEFAULT_VISUAL_INDEX.
  for (var i = categories.length - 1; i >= 0; i--) {
    if (visual[i] == null) {
      delete categoryMap[categories[i]];
      categories.pop();
    }
  }
}
function normalizeVisualRange(thisOption, isCategory) {
  var visual = thisOption.visual;
  var visualArr = [];
  if (zrUtil.isObject(visual)) {
    each(visual, function (v) {
      visualArr.push(v);
    });
  } else if (visual != null) {
    visualArr.push(visual);
  }
  var doNotNeedPair = {
    color: 1,
    symbol: 1
  };
  if (!isCategory && visualArr.length === 1 && !doNotNeedPair.hasOwnProperty(thisOption.type)) {
    // Do not care visualArr.length === 0, which is illegal.
    visualArr[1] = visualArr[0];
  }
  setVisualToOption(thisOption, visualArr);
}
function makePartialColorVisualHandler(applyValue) {
  return {
    applyVisual: function (value, getter, setter) {
      // Only used in HSL
      var colorChannel = this.mapValueToVisual(value);
      // Must not be array value
      setter('color', applyValue(getter('color'), colorChannel));
    },
    _normalizedToVisual: createNormalizedToNumericVisual([0, 1])
  };
}
function doMapToArray(normalized) {
  var visual = this.option.visual;
  return visual[Math.round(linearMap(normalized, [0, 1], [0, visual.length - 1], true))] || {}; // TODO {}?
}
function makeApplyVisual(visualType) {
  return function (value, getter, setter) {
    setter(visualType, this.mapValueToVisual(value));
  };
}
function doMapCategory(normalized) {
  var visual = this.option.visual;
  return visual[this.option.loop && normalized !== CATEGORY_DEFAULT_VISUAL_INDEX ? normalized % visual.length : normalized];
}
function doMapFixed() {
  // visual will be convert to array.
  return this.option.visual[0];
}
/**
 * Create mapped to numeric visual
 */
function createNormalizedToNumericVisual(sourceExtent) {
  return {
    linear: function (normalized) {
      return linearMap(normalized, sourceExtent, this.option.visual, true);
    },
    category: doMapCategory,
    piecewise: function (normalized, value) {
      var result = getSpecifiedVisual.call(this, value);
      if (result == null) {
        result = linearMap(normalized, sourceExtent, this.option.visual, true);
      }
      return result;
    },
    fixed: doMapFixed
  };
}
function getSpecifiedVisual(value) {
  var thisOption = this.option;
  var pieceList = thisOption.pieceList;
  if (thisOption.hasSpecialVisual) {
    var pieceIndex = VisualMapping.findPieceIndex(value, pieceList);
    var piece = pieceList[pieceIndex];
    if (piece && piece.visual) {
      return piece.visual[this.type];
    }
  }
}
function setVisualToOption(thisOption, visualArr) {
  thisOption.visual = visualArr;
  if (thisOption.type === 'color') {
    thisOption.parsedVisual = zrUtil.map(visualArr, function (item) {
      var color = zrColor.parse(item);
      if (!color && process.env.NODE_ENV !== 'production') {
        warn("'" + item + "' is an illegal color, fallback to '#000000'", true);
      }
      return color || [0, 0, 0, 1];
    });
  }
  return visualArr;
}
/**
 * Normalizers by mapping methods.
 */
var normalizers = {
  linear: function (value) {
    return linearMap(value, this.option.dataExtent, [0, 1], true);
  },
  piecewise: function (value) {
    var pieceList = this.option.pieceList;
    var pieceIndex = VisualMapping.findPieceIndex(value, pieceList, true);
    if (pieceIndex != null) {
      return linearMap(pieceIndex, [0, pieceList.length - 1], [0, 1], true);
    }
  },
  category: function (value) {
    var index = this.option.categories ? this.option.categoryMap[value] : value; // ordinal value
    return index == null ? CATEGORY_DEFAULT_VISUAL_INDEX : index;
  },
  fixed: zrUtil.noop
};
function littleThan(close, a, b) {
  return close ? a <= b : a < b;
}
export default VisualMapping;