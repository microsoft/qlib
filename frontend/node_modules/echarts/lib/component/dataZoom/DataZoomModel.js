
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
import { each, createHashMap, merge, assert } from 'zrender/lib/core/util.js';
import ComponentModel from '../../model/Component.js';
import { getAxisMainType, DATA_ZOOM_AXIS_DIMENSIONS } from './helper.js';
import { MULTIPLE_REFERRING, SINGLE_REFERRING } from '../../util/model.js';
var DataZoomAxisInfo = /** @class */function () {
  function DataZoomAxisInfo() {
    this.indexList = [];
    this.indexMap = [];
  }
  DataZoomAxisInfo.prototype.add = function (axisCmptIdx) {
    // Remove duplication.
    if (!this.indexMap[axisCmptIdx]) {
      this.indexList.push(axisCmptIdx);
      this.indexMap[axisCmptIdx] = true;
    }
  };
  return DataZoomAxisInfo;
}();
var DataZoomModel = /** @class */function (_super) {
  __extends(DataZoomModel, _super);
  function DataZoomModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = DataZoomModel.type;
    _this._autoThrottle = true;
    _this._noTarget = true;
    /**
     * It is `[rangeModeForMin, rangeModeForMax]`.
     * The optional values for `rangeMode`:
     * + `'value'` mode: the axis extent will always be determined by
     *     `dataZoom.startValue` and `dataZoom.endValue`, despite
     *     how data like and how `axis.min` and `axis.max` are.
     * + `'percent'` mode: `100` represents 100% of the `[dMin, dMax]`,
     *     where `dMin` is `axis.min` if `axis.min` specified, otherwise `data.extent[0]`,
     *     and `dMax` is `axis.max` if `axis.max` specified, otherwise `data.extent[1]`.
     *     Axis extent will be determined by the result of the percent of `[dMin, dMax]`.
     *
     * For example, when users are using dynamic data (update data periodically via `setOption`),
     * if in `'value`' mode, the window will be kept in a fixed value range despite how
     * data are appended, while if in `'percent'` mode, whe window range will be changed alone with
     * the appended data (suppose `axis.min` and `axis.max` are not specified).
     */
    _this._rangePropMode = ['percent', 'percent'];
    return _this;
  }
  DataZoomModel.prototype.init = function (option, parentModel, ecModel) {
    var inputRawOption = retrieveRawOption(option);
    /**
     * Suppose a "main process" start at the point that model prepared (that is,
     * model initialized or merged or method called in `action`).
     * We should keep the `main process` idempotent, that is, given a set of values
     * on `option`, we get the same result.
     *
     * But sometimes, values on `option` will be updated for providing users
     * a "final calculated value" (`dataZoomProcessor` will do that). Those value
     * should not be the base/input of the `main process`.
     *
     * So in that case we should save and keep the input of the `main process`
     * separately, called `settledOption`.
     *
     * For example, consider the case:
     * (Step_1) brush zoom the grid by `toolbox.dataZoom`,
     *     where the original input `option.startValue`, `option.endValue` are earsed by
     *     calculated value.
     * (Step)2) click the legend to hide and show a series,
     *     where the new range is calculated by the earsed `startValue` and `endValue`,
     *     which brings incorrect result.
     */
    this.settledOption = inputRawOption;
    this.mergeDefaultAndTheme(option, ecModel);
    this._doInit(inputRawOption);
  };
  DataZoomModel.prototype.mergeOption = function (newOption) {
    var inputRawOption = retrieveRawOption(newOption);
    // FIX #2591
    merge(this.option, newOption, true);
    merge(this.settledOption, inputRawOption, true);
    this._doInit(inputRawOption);
  };
  DataZoomModel.prototype._doInit = function (inputRawOption) {
    var thisOption = this.option;
    this._setDefaultThrottle(inputRawOption);
    this._updateRangeUse(inputRawOption);
    var settledOption = this.settledOption;
    each([['start', 'startValue'], ['end', 'endValue']], function (names, index) {
      // start/end has higher priority over startValue/endValue if they
      // both set, but we should make chart.setOption({endValue: 1000})
      // effective, rather than chart.setOption({endValue: 1000, end: null}).
      if (this._rangePropMode[index] === 'value') {
        thisOption[names[0]] = settledOption[names[0]] = null;
      }
      // Otherwise do nothing and use the merge result.
    }, this);
    this._resetTarget();
  };
  DataZoomModel.prototype._resetTarget = function () {
    var optionOrient = this.get('orient', true);
    var targetAxisIndexMap = this._targetAxisInfoMap = createHashMap();
    var hasAxisSpecified = this._fillSpecifiedTargetAxis(targetAxisIndexMap);
    if (hasAxisSpecified) {
      this._orient = optionOrient || this._makeAutoOrientByTargetAxis();
    } else {
      this._orient = optionOrient || 'horizontal';
      this._fillAutoTargetAxisByOrient(targetAxisIndexMap, this._orient);
    }
    this._noTarget = true;
    targetAxisIndexMap.each(function (axisInfo) {
      if (axisInfo.indexList.length) {
        this._noTarget = false;
      }
    }, this);
  };
  DataZoomModel.prototype._fillSpecifiedTargetAxis = function (targetAxisIndexMap) {
    var hasAxisSpecified = false;
    each(DATA_ZOOM_AXIS_DIMENSIONS, function (axisDim) {
      var refering = this.getReferringComponents(getAxisMainType(axisDim), MULTIPLE_REFERRING);
      // When user set axisIndex as a empty array, we think that user specify axisIndex
      // but do not want use auto mode. Because empty array may be encountered when
      // some error occurred.
      if (!refering.specified) {
        return;
      }
      hasAxisSpecified = true;
      var axisInfo = new DataZoomAxisInfo();
      each(refering.models, function (axisModel) {
        axisInfo.add(axisModel.componentIndex);
      });
      targetAxisIndexMap.set(axisDim, axisInfo);
    }, this);
    return hasAxisSpecified;
  };
  DataZoomModel.prototype._fillAutoTargetAxisByOrient = function (targetAxisIndexMap, orient) {
    var ecModel = this.ecModel;
    var needAuto = true;
    // Find axis that parallel to dataZoom as default.
    if (needAuto) {
      var axisDim = orient === 'vertical' ? 'y' : 'x';
      var axisModels = ecModel.findComponents({
        mainType: axisDim + 'Axis'
      });
      setParallelAxis(axisModels, axisDim);
    }
    // Find axis that parallel to dataZoom as default.
    if (needAuto) {
      var axisModels = ecModel.findComponents({
        mainType: 'singleAxis',
        filter: function (axisModel) {
          return axisModel.get('orient', true) === orient;
        }
      });
      setParallelAxis(axisModels, 'single');
    }
    function setParallelAxis(axisModels, axisDim) {
      // At least use the first parallel axis as the target axis.
      var axisModel = axisModels[0];
      if (!axisModel) {
        return;
      }
      var axisInfo = new DataZoomAxisInfo();
      axisInfo.add(axisModel.componentIndex);
      targetAxisIndexMap.set(axisDim, axisInfo);
      needAuto = false;
      // Find parallel axes in the same grid.
      if (axisDim === 'x' || axisDim === 'y') {
        var gridModel_1 = axisModel.getReferringComponents('grid', SINGLE_REFERRING).models[0];
        gridModel_1 && each(axisModels, function (axModel) {
          if (axisModel.componentIndex !== axModel.componentIndex && gridModel_1 === axModel.getReferringComponents('grid', SINGLE_REFERRING).models[0]) {
            axisInfo.add(axModel.componentIndex);
          }
        });
      }
    }
    if (needAuto) {
      // If no parallel axis, find the first category axis as default. (Also consider polar).
      each(DATA_ZOOM_AXIS_DIMENSIONS, function (axisDim) {
        if (!needAuto) {
          return;
        }
        var axisModels = ecModel.findComponents({
          mainType: getAxisMainType(axisDim),
          filter: function (axisModel) {
            return axisModel.get('type', true) === 'category';
          }
        });
        if (axisModels[0]) {
          var axisInfo = new DataZoomAxisInfo();
          axisInfo.add(axisModels[0].componentIndex);
          targetAxisIndexMap.set(axisDim, axisInfo);
          needAuto = false;
        }
      }, this);
    }
  };
  DataZoomModel.prototype._makeAutoOrientByTargetAxis = function () {
    var dim;
    // Find the first axis
    this.eachTargetAxis(function (axisDim) {
      !dim && (dim = axisDim);
    }, this);
    return dim === 'y' ? 'vertical' : 'horizontal';
  };
  DataZoomModel.prototype._setDefaultThrottle = function (inputRawOption) {
    // When first time user set throttle, auto throttle ends.
    if (inputRawOption.hasOwnProperty('throttle')) {
      this._autoThrottle = false;
    }
    if (this._autoThrottle) {
      var globalOption = this.ecModel.option;
      this.option.throttle = globalOption.animation && globalOption.animationDurationUpdate > 0 ? 100 : 20;
    }
  };
  DataZoomModel.prototype._updateRangeUse = function (inputRawOption) {
    var rangePropMode = this._rangePropMode;
    var rangeModeInOption = this.get('rangeMode');
    each([['start', 'startValue'], ['end', 'endValue']], function (names, index) {
      var percentSpecified = inputRawOption[names[0]] != null;
      var valueSpecified = inputRawOption[names[1]] != null;
      if (percentSpecified && !valueSpecified) {
        rangePropMode[index] = 'percent';
      } else if (!percentSpecified && valueSpecified) {
        rangePropMode[index] = 'value';
      } else if (rangeModeInOption) {
        rangePropMode[index] = rangeModeInOption[index];
      } else if (percentSpecified) {
        // percentSpecified && valueSpecified
        rangePropMode[index] = 'percent';
      }
      // else remain its original setting.
    });
  };
  DataZoomModel.prototype.noTarget = function () {
    return this._noTarget;
  };
  DataZoomModel.prototype.getFirstTargetAxisModel = function () {
    var firstAxisModel;
    this.eachTargetAxis(function (axisDim, axisIndex) {
      if (firstAxisModel == null) {
        firstAxisModel = this.ecModel.getComponent(getAxisMainType(axisDim), axisIndex);
      }
    }, this);
    return firstAxisModel;
  };
  /**
   * @param {Function} callback param: axisModel, dimNames, axisIndex, dataZoomModel, ecModel
   */
  DataZoomModel.prototype.eachTargetAxis = function (callback, context) {
    this._targetAxisInfoMap.each(function (axisInfo, axisDim) {
      each(axisInfo.indexList, function (axisIndex) {
        callback.call(context, axisDim, axisIndex);
      });
    });
  };
  /**
   * @return If not found, return null/undefined.
   */
  DataZoomModel.prototype.getAxisProxy = function (axisDim, axisIndex) {
    var axisModel = this.getAxisModel(axisDim, axisIndex);
    if (axisModel) {
      return axisModel.__dzAxisProxy;
    }
  };
  /**
   * @return If not found, return null/undefined.
   */
  DataZoomModel.prototype.getAxisModel = function (axisDim, axisIndex) {
    if (process.env.NODE_ENV !== 'production') {
      assert(axisDim && axisIndex != null);
    }
    var axisInfo = this._targetAxisInfoMap.get(axisDim);
    if (axisInfo && axisInfo.indexMap[axisIndex]) {
      return this.ecModel.getComponent(getAxisMainType(axisDim), axisIndex);
    }
  };
  /**
   * If not specified, set to undefined.
   */
  DataZoomModel.prototype.setRawRange = function (opt) {
    var thisOption = this.option;
    var settledOption = this.settledOption;
    each([['start', 'startValue'], ['end', 'endValue']], function (names) {
      // Consider the pair <start, startValue>:
      // If one has value and the other one is `null/undefined`, we both set them
      // to `settledOption`. This strategy enables the feature to clear the original
      // value in `settledOption` to `null/undefined`.
      // But if both of them are `null/undefined`, we do not set them to `settledOption`
      // and keep `settledOption` with the original value. This strategy enables users to
      // only set <end or endValue> but not set <start or startValue> when calling
      // `dispatchAction`.
      // The pair <end, endValue> is treated in the same way.
      if (opt[names[0]] != null || opt[names[1]] != null) {
        thisOption[names[0]] = settledOption[names[0]] = opt[names[0]];
        thisOption[names[1]] = settledOption[names[1]] = opt[names[1]];
      }
    }, this);
    this._updateRangeUse(opt);
  };
  DataZoomModel.prototype.setCalculatedRange = function (opt) {
    var option = this.option;
    each(['start', 'startValue', 'end', 'endValue'], function (name) {
      option[name] = opt[name];
    });
  };
  DataZoomModel.prototype.getPercentRange = function () {
    var axisProxy = this.findRepresentativeAxisProxy();
    if (axisProxy) {
      return axisProxy.getDataPercentWindow();
    }
  };
  /**
   * For example, chart.getModel().getComponent('dataZoom').getValueRange('y', 0);
   *
   * @return [startValue, endValue] value can only be '-' or finite number.
   */
  DataZoomModel.prototype.getValueRange = function (axisDim, axisIndex) {
    if (axisDim == null && axisIndex == null) {
      var axisProxy = this.findRepresentativeAxisProxy();
      if (axisProxy) {
        return axisProxy.getDataValueWindow();
      }
    } else {
      return this.getAxisProxy(axisDim, axisIndex).getDataValueWindow();
    }
  };
  /**
   * @param axisModel If axisModel given, find axisProxy
   *      corresponding to the axisModel
   */
  DataZoomModel.prototype.findRepresentativeAxisProxy = function (axisModel) {
    if (axisModel) {
      return axisModel.__dzAxisProxy;
    }
    // Find the first hosted axisProxy
    var firstProxy;
    var axisDimList = this._targetAxisInfoMap.keys();
    for (var i = 0; i < axisDimList.length; i++) {
      var axisDim = axisDimList[i];
      var axisInfo = this._targetAxisInfoMap.get(axisDim);
      for (var j = 0; j < axisInfo.indexList.length; j++) {
        var proxy = this.getAxisProxy(axisDim, axisInfo.indexList[j]);
        if (proxy.hostedBy(this)) {
          return proxy;
        }
        if (!firstProxy) {
          firstProxy = proxy;
        }
      }
    }
    // If no hosted proxy found, still need to return a proxy.
    // This case always happens in toolbox dataZoom, where axes are all hosted by
    // other dataZooms.
    return firstProxy;
  };
  DataZoomModel.prototype.getRangePropMode = function () {
    return this._rangePropMode.slice();
  };
  DataZoomModel.prototype.getOrient = function () {
    if (process.env.NODE_ENV !== 'production') {
      // Should not be called before initialized.
      assert(this._orient);
    }
    return this._orient;
  };
  DataZoomModel.type = 'dataZoom';
  DataZoomModel.dependencies = ['xAxis', 'yAxis', 'radiusAxis', 'angleAxis', 'singleAxis', 'series', 'toolbox'];
  DataZoomModel.defaultOption = {
    // zlevel: 0,
    z: 4,
    filterMode: 'filter',
    start: 0,
    end: 100
  };
  return DataZoomModel;
}(ComponentModel);
/**
 * Retrieve those raw params from option, which will be cached separately,
 * because they will be overwritten by normalized/calculated values in the main
 * process.
 */
function retrieveRawOption(option) {
  var ret = {};
  each(['start', 'end', 'startValue', 'endValue', 'throttle'], function (name) {
    option.hasOwnProperty(name) && (ret[name] = option[name]);
  });
  return ret;
}
export default DataZoomModel;