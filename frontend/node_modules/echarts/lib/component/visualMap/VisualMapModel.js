
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
import visualDefault from '../../visual/visualDefault.js';
import VisualMapping from '../../visual/VisualMapping.js';
import * as visualSolution from '../../visual/visualSolution.js';
import * as modelUtil from '../../util/model.js';
import * as numberUtil from '../../util/number.js';
import ComponentModel from '../../model/Component.js';
var mapVisual = VisualMapping.mapVisual;
var eachVisual = VisualMapping.eachVisual;
var isArray = zrUtil.isArray;
var each = zrUtil.each;
var asc = numberUtil.asc;
var linearMap = numberUtil.linearMap;
var VisualMapModel = /** @class */function (_super) {
  __extends(VisualMapModel, _super);
  function VisualMapModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = VisualMapModel.type;
    _this.stateList = ['inRange', 'outOfRange'];
    _this.replacableOptionKeys = ['inRange', 'outOfRange', 'target', 'controller', 'color'];
    _this.layoutMode = {
      type: 'box',
      ignoreSize: true
    };
    /**
     * [lowerBound, upperBound]
     */
    _this.dataBound = [-Infinity, Infinity];
    _this.targetVisuals = {};
    _this.controllerVisuals = {};
    return _this;
  }
  VisualMapModel.prototype.init = function (option, parentModel, ecModel) {
    this.mergeDefaultAndTheme(option, ecModel);
  };
  /**
   * @protected
   */
  VisualMapModel.prototype.optionUpdated = function (newOption, isInit) {
    var thisOption = this.option;
    !isInit && visualSolution.replaceVisualOption(thisOption, newOption, this.replacableOptionKeys);
    this.textStyleModel = this.getModel('textStyle');
    this.resetItemSize();
    this.completeVisualOption();
  };
  /**
   * @protected
   */
  VisualMapModel.prototype.resetVisual = function (supplementVisualOption) {
    var stateList = this.stateList;
    supplementVisualOption = zrUtil.bind(supplementVisualOption, this);
    this.controllerVisuals = visualSolution.createVisualMappings(this.option.controller, stateList, supplementVisualOption);
    this.targetVisuals = visualSolution.createVisualMappings(this.option.target, stateList, supplementVisualOption);
  };
  /**
   * @public
   */
  VisualMapModel.prototype.getItemSymbol = function () {
    return null;
  };
  /**
   * @protected
   * @return {Array.<number>} An array of series indices.
   */
  VisualMapModel.prototype.getTargetSeriesIndices = function () {
    var optionSeriesIndex = this.option.seriesIndex;
    var seriesIndices = [];
    if (optionSeriesIndex == null || optionSeriesIndex === 'all') {
      this.ecModel.eachSeries(function (seriesModel, index) {
        seriesIndices.push(index);
      });
    } else {
      seriesIndices = modelUtil.normalizeToArray(optionSeriesIndex);
    }
    return seriesIndices;
  };
  /**
   * @public
   */
  VisualMapModel.prototype.eachTargetSeries = function (callback, context) {
    zrUtil.each(this.getTargetSeriesIndices(), function (seriesIndex) {
      var seriesModel = this.ecModel.getSeriesByIndex(seriesIndex);
      if (seriesModel) {
        callback.call(context, seriesModel);
      }
    }, this);
  };
  /**
   * @pubilc
   */
  VisualMapModel.prototype.isTargetSeries = function (seriesModel) {
    var is = false;
    this.eachTargetSeries(function (model) {
      model === seriesModel && (is = true);
    });
    return is;
  };
  /**
   * @example
   * this.formatValueText(someVal); // format single numeric value to text.
   * this.formatValueText(someVal, true); // format single category value to text.
   * this.formatValueText([min, max]); // format numeric min-max to text.
   * this.formatValueText([this.dataBound[0], max]); // using data lower bound.
   * this.formatValueText([min, this.dataBound[1]]); // using data upper bound.
   *
   * @param value Real value, or this.dataBound[0 or 1].
   * @param isCategory Only available when value is number.
   * @param edgeSymbols Open-close symbol when value is interval.
   * @protected
   */
  VisualMapModel.prototype.formatValueText = function (value, isCategory, edgeSymbols) {
    var option = this.option;
    var precision = option.precision;
    var dataBound = this.dataBound;
    var formatter = option.formatter;
    var isMinMax;
    edgeSymbols = edgeSymbols || ['<', '>'];
    if (zrUtil.isArray(value)) {
      value = value.slice();
      isMinMax = true;
    }
    var textValue = isCategory ? value // Value is string when isCategory
    : isMinMax ? [toFixed(value[0]), toFixed(value[1])] : toFixed(value);
    if (zrUtil.isString(formatter)) {
      return formatter.replace('{value}', isMinMax ? textValue[0] : textValue).replace('{value2}', isMinMax ? textValue[1] : textValue);
    } else if (zrUtil.isFunction(formatter)) {
      return isMinMax ? formatter(value[0], value[1]) : formatter(value);
    }
    if (isMinMax) {
      if (value[0] === dataBound[0]) {
        return edgeSymbols[0] + ' ' + textValue[1];
      } else if (value[1] === dataBound[1]) {
        return edgeSymbols[1] + ' ' + textValue[0];
      } else {
        return textValue[0] + ' - ' + textValue[1];
      }
    } else {
      // Format single value (includes category case).
      return textValue;
    }
    function toFixed(val) {
      return val === dataBound[0] ? 'min' : val === dataBound[1] ? 'max' : (+val).toFixed(Math.min(precision, 20));
    }
  };
  /**
   * @protected
   */
  VisualMapModel.prototype.resetExtent = function () {
    var thisOption = this.option;
    // Can not calculate data extent by data here.
    // Because series and data may be modified in processing stage.
    // So we do not support the feature "auto min/max".
    var extent = asc([thisOption.min, thisOption.max]);
    this._dataExtent = extent;
  };
  /**
   * PENDING:
   * delete this method if no outer usage.
   *
   * Return  Concrete dimension. If null/undefined is returned, no dimension is used.
   */
  // getDataDimension(data: SeriesData) {
  //     const optDim = this.option.dimension;
  //     if (optDim != null) {
  //         return data.getDimension(optDim);
  //     }
  //     const dimNames = data.dimensions;
  //     for (let i = dimNames.length - 1; i >= 0; i--) {
  //         const dimName = dimNames[i];
  //         const dimInfo = data.getDimensionInfo(dimName);
  //         if (!dimInfo.isCalculationCoord) {
  //             return dimName;
  //         }
  //     }
  // }
  VisualMapModel.prototype.getDataDimensionIndex = function (data) {
    var optDim = this.option.dimension;
    if (optDim != null) {
      return data.getDimensionIndex(optDim);
    }
    var dimNames = data.dimensions;
    for (var i = dimNames.length - 1; i >= 0; i--) {
      var dimName = dimNames[i];
      var dimInfo = data.getDimensionInfo(dimName);
      if (!dimInfo.isCalculationCoord) {
        return dimInfo.storeDimIndex;
      }
    }
  };
  VisualMapModel.prototype.getExtent = function () {
    return this._dataExtent.slice();
  };
  VisualMapModel.prototype.completeVisualOption = function () {
    var ecModel = this.ecModel;
    var thisOption = this.option;
    var base = {
      inRange: thisOption.inRange,
      outOfRange: thisOption.outOfRange
    };
    var target = thisOption.target || (thisOption.target = {});
    var controller = thisOption.controller || (thisOption.controller = {});
    zrUtil.merge(target, base); // Do not override
    zrUtil.merge(controller, base); // Do not override
    var isCategory = this.isCategory();
    completeSingle.call(this, target);
    completeSingle.call(this, controller);
    completeInactive.call(this, target, 'inRange', 'outOfRange');
    // completeInactive.call(this, target, 'outOfRange', 'inRange');
    completeController.call(this, controller);
    function completeSingle(base) {
      // Compatible with ec2 dataRange.color.
      // The mapping order of dataRange.color is: [high value, ..., low value]
      // whereas inRange.color and outOfRange.color is [low value, ..., high value]
      // Notice: ec2 has no inverse.
      if (isArray(thisOption.color)
      // If there has been inRange: {symbol: ...}, adding color is a mistake.
      // So adding color only when no inRange defined.
      && !base.inRange) {
        base.inRange = {
          color: thisOption.color.slice().reverse()
        };
      }
      // Compatible with previous logic, always give a default color, otherwise
      // simple config with no inRange and outOfRange will not work.
      // Originally we use visualMap.color as the default color, but setOption at
      // the second time the default color will be erased. So we change to use
      // constant DEFAULT_COLOR.
      // If user do not want the default color, set inRange: {color: null}.
      base.inRange = base.inRange || {
        color: ecModel.get('gradientColor')
      };
    }
    function completeInactive(base, stateExist, stateAbsent) {
      var optExist = base[stateExist];
      var optAbsent = base[stateAbsent];
      if (optExist && !optAbsent) {
        optAbsent = base[stateAbsent] = {};
        each(optExist, function (visualData, visualType) {
          if (!VisualMapping.isValidType(visualType)) {
            return;
          }
          var defa = visualDefault.get(visualType, 'inactive', isCategory);
          if (defa != null) {
            optAbsent[visualType] = defa;
            // Compatibable with ec2:
            // Only inactive color to rgba(0,0,0,0) can not
            // make label transparent, so use opacity also.
            if (visualType === 'color' && !optAbsent.hasOwnProperty('opacity') && !optAbsent.hasOwnProperty('colorAlpha')) {
              optAbsent.opacity = [0, 0];
            }
          }
        });
      }
    }
    function completeController(controller) {
      var symbolExists = (controller.inRange || {}).symbol || (controller.outOfRange || {}).symbol;
      var symbolSizeExists = (controller.inRange || {}).symbolSize || (controller.outOfRange || {}).symbolSize;
      var inactiveColor = this.get('inactiveColor');
      var itemSymbol = this.getItemSymbol();
      var defaultSymbol = itemSymbol || 'roundRect';
      each(this.stateList, function (state) {
        var itemSize = this.itemSize;
        var visuals = controller[state];
        // Set inactive color for controller if no other color
        // attr (like colorAlpha) specified.
        if (!visuals) {
          visuals = controller[state] = {
            color: isCategory ? inactiveColor : [inactiveColor]
          };
        }
        // Consistent symbol and symbolSize if not specified.
        if (visuals.symbol == null) {
          visuals.symbol = symbolExists && zrUtil.clone(symbolExists) || (isCategory ? defaultSymbol : [defaultSymbol]);
        }
        if (visuals.symbolSize == null) {
          visuals.symbolSize = symbolSizeExists && zrUtil.clone(symbolSizeExists) || (isCategory ? itemSize[0] : [itemSize[0], itemSize[0]]);
        }
        // Filter none
        visuals.symbol = mapVisual(visuals.symbol, function (symbol) {
          return symbol === 'none' ? defaultSymbol : symbol;
        });
        // Normalize symbolSize
        var symbolSize = visuals.symbolSize;
        if (symbolSize != null) {
          var max_1 = -Infinity;
          // symbolSize can be object when categories defined.
          eachVisual(symbolSize, function (value) {
            value > max_1 && (max_1 = value);
          });
          visuals.symbolSize = mapVisual(symbolSize, function (value) {
            return linearMap(value, [0, max_1], [0, itemSize[0]], true);
          });
        }
      }, this);
    }
  };
  VisualMapModel.prototype.resetItemSize = function () {
    this.itemSize = [parseFloat(this.get('itemWidth')), parseFloat(this.get('itemHeight'))];
  };
  VisualMapModel.prototype.isCategory = function () {
    return !!this.option.categories;
  };
  /**
   * @public
   * @abstract
   */
  VisualMapModel.prototype.setSelected = function (selected) {};
  VisualMapModel.prototype.getSelected = function () {
    return null;
  };
  /**
   * @public
   * @abstract
   */
  VisualMapModel.prototype.getValueState = function (value) {
    return null;
  };
  /**
   * FIXME
   * Do not publish to thirt-part-dev temporarily
   * util the interface is stable. (Should it return
   * a function but not visual meta?)
   *
   * @pubilc
   * @abstract
   * @param getColorVisual
   *        params: value, valueState
   *        return: color
   * @return {Object} visualMeta
   *        should includes {stops, outerColors}
   *        outerColor means [colorBeyondMinValue, colorBeyondMaxValue]
   */
  VisualMapModel.prototype.getVisualMeta = function (getColorVisual) {
    return null;
  };
  VisualMapModel.type = 'visualMap';
  VisualMapModel.dependencies = ['series'];
  VisualMapModel.defaultOption = {
    show: true,
    // zlevel: 0,
    z: 4,
    seriesIndex: 'all',
    min: 0,
    max: 200,
    left: 0,
    right: null,
    top: null,
    bottom: 0,
    itemWidth: null,
    itemHeight: null,
    inverse: false,
    orient: 'vertical',
    backgroundColor: 'rgba(0,0,0,0)',
    borderColor: '#ccc',
    contentColor: '#5793f3',
    inactiveColor: '#aaa',
    borderWidth: 0,
    padding: 5,
    // 接受数组分别设定上右下左边距，同css
    textGap: 10,
    precision: 0,
    textStyle: {
      color: '#333' // 值域文字颜色
    }
  };
  return VisualMapModel;
}(ComponentModel);
export default VisualMapModel;