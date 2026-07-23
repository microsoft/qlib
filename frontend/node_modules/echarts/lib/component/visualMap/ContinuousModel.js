
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
import VisualMapModel from './VisualMapModel.js';
import * as numberUtil from '../../util/number.js';
import { inheritDefaultOption } from '../../util/component.js';
// Constant
var DEFAULT_BAR_BOUND = [20, 140];
var ContinuousModel = /** @class */function (_super) {
  __extends(ContinuousModel, _super);
  function ContinuousModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ContinuousModel.type;
    return _this;
  }
  /**
   * @override
   */
  ContinuousModel.prototype.optionUpdated = function (newOption, isInit) {
    _super.prototype.optionUpdated.apply(this, arguments);
    this.resetExtent();
    this.resetVisual(function (mappingOption) {
      mappingOption.mappingMethod = 'linear';
      mappingOption.dataExtent = this.getExtent();
    });
    this._resetRange();
  };
  /**
   * @protected
   * @override
   */
  ContinuousModel.prototype.resetItemSize = function () {
    _super.prototype.resetItemSize.apply(this, arguments);
    var itemSize = this.itemSize;
    (itemSize[0] == null || isNaN(itemSize[0])) && (itemSize[0] = DEFAULT_BAR_BOUND[0]);
    (itemSize[1] == null || isNaN(itemSize[1])) && (itemSize[1] = DEFAULT_BAR_BOUND[1]);
  };
  /**
   * @private
   */
  ContinuousModel.prototype._resetRange = function () {
    var dataExtent = this.getExtent();
    var range = this.option.range;
    if (!range || range.auto) {
      // `range` should always be array (so we don't use other
      // value like 'auto') for user-friend. (consider getOption).
      dataExtent.auto = 1;
      this.option.range = dataExtent;
    } else if (zrUtil.isArray(range)) {
      if (range[0] > range[1]) {
        range.reverse();
      }
      range[0] = Math.max(range[0], dataExtent[0]);
      range[1] = Math.min(range[1], dataExtent[1]);
    }
  };
  /**
   * @protected
   * @override
   */
  ContinuousModel.prototype.completeVisualOption = function () {
    _super.prototype.completeVisualOption.apply(this, arguments);
    zrUtil.each(this.stateList, function (state) {
      var symbolSize = this.option.controller[state].symbolSize;
      if (symbolSize && symbolSize[0] !== symbolSize[1]) {
        symbolSize[0] = symbolSize[1] / 3; // For good looking.
      }
    }, this);
  };
  /**
   * @override
   */
  ContinuousModel.prototype.setSelected = function (selected) {
    this.option.range = selected.slice();
    this._resetRange();
  };
  /**
   * @public
   */
  ContinuousModel.prototype.getSelected = function () {
    var dataExtent = this.getExtent();
    var dataInterval = numberUtil.asc((this.get('range') || []).slice());
    // Clamp
    dataInterval[0] > dataExtent[1] && (dataInterval[0] = dataExtent[1]);
    dataInterval[1] > dataExtent[1] && (dataInterval[1] = dataExtent[1]);
    dataInterval[0] < dataExtent[0] && (dataInterval[0] = dataExtent[0]);
    dataInterval[1] < dataExtent[0] && (dataInterval[1] = dataExtent[0]);
    return dataInterval;
  };
  /**
   * @override
   */
  ContinuousModel.prototype.getValueState = function (value) {
    var range = this.option.range;
    var dataExtent = this.getExtent();
    // When range[0] === dataExtent[0], any value larger than dataExtent[0] maps to 'inRange'.
    // range[1] is processed likewise.
    return (range[0] <= dataExtent[0] || range[0] <= value) && (range[1] >= dataExtent[1] || value <= range[1]) ? 'inRange' : 'outOfRange';
  };
  ContinuousModel.prototype.findTargetDataIndices = function (range) {
    var result = [];
    this.eachTargetSeries(function (seriesModel) {
      var dataIndices = [];
      var data = seriesModel.getData();
      data.each(this.getDataDimensionIndex(data), function (value, dataIndex) {
        range[0] <= value && value <= range[1] && dataIndices.push(dataIndex);
      }, this);
      result.push({
        seriesId: seriesModel.id,
        dataIndex: dataIndices
      });
    }, this);
    return result;
  };
  /**
   * @implement
   */
  ContinuousModel.prototype.getVisualMeta = function (getColorVisual) {
    var oVals = getColorStopValues(this, 'outOfRange', this.getExtent());
    var iVals = getColorStopValues(this, 'inRange', this.option.range.slice());
    var stops = [];
    function setStop(value, valueState) {
      stops.push({
        value: value,
        color: getColorVisual(value, valueState)
      });
    }
    // Format to: outOfRange -- inRange -- outOfRange.
    var iIdx = 0;
    var oIdx = 0;
    var iLen = iVals.length;
    var oLen = oVals.length;
    for (; oIdx < oLen && (!iVals.length || oVals[oIdx] <= iVals[0]); oIdx++) {
      // If oVal[oIdx] === iVals[iIdx], oVal[oIdx] should be ignored.
      if (oVals[oIdx] < iVals[iIdx]) {
        setStop(oVals[oIdx], 'outOfRange');
      }
    }
    for (var first = 1; iIdx < iLen; iIdx++, first = 0) {
      // If range is full, value beyond min, max will be clamped.
      // make a singularity
      first && stops.length && setStop(iVals[iIdx], 'outOfRange');
      setStop(iVals[iIdx], 'inRange');
    }
    for (var first = 1; oIdx < oLen; oIdx++) {
      if (!iVals.length || iVals[iVals.length - 1] < oVals[oIdx]) {
        // make a singularity
        if (first) {
          stops.length && setStop(stops[stops.length - 1].value, 'outOfRange');
          first = 0;
        }
        setStop(oVals[oIdx], 'outOfRange');
      }
    }
    var stopsLen = stops.length;
    return {
      stops: stops,
      outerColors: [stopsLen ? stops[0].color : 'transparent', stopsLen ? stops[stopsLen - 1].color : 'transparent']
    };
  };
  ContinuousModel.type = 'visualMap.continuous';
  ContinuousModel.defaultOption = inheritDefaultOption(VisualMapModel.defaultOption, {
    align: 'auto',
    calculable: false,
    hoverLink: true,
    realtime: true,
    handleIcon: 'path://M-11.39,9.77h0a3.5,3.5,0,0,1-3.5,3.5h-22a3.5,3.5,0,0,1-3.5-3.5h0a3.5,3.5,0,0,1,3.5-3.5h22A3.5,3.5,0,0,1-11.39,9.77Z',
    handleSize: '120%',
    handleStyle: {
      borderColor: '#fff',
      borderWidth: 1
    },
    indicatorIcon: 'circle',
    indicatorSize: '50%',
    indicatorStyle: {
      borderColor: '#fff',
      borderWidth: 2,
      shadowBlur: 2,
      shadowOffsetX: 1,
      shadowOffsetY: 1,
      shadowColor: 'rgba(0,0,0,0.2)'
    }
    // emphasis: {
    //     handleStyle: {
    //         shadowBlur: 3,
    //         shadowOffsetX: 1,
    //         shadowOffsetY: 1,
    //         shadowColor: 'rgba(0,0,0,0.2)'
    //     }
    // }
  });
  return ContinuousModel;
}(VisualMapModel);
function getColorStopValues(visualMapModel, valueState, dataExtent) {
  if (dataExtent[0] === dataExtent[1]) {
    return dataExtent.slice();
  }
  // When using colorHue mapping, it is not linear color any more.
  // Moreover, canvas gradient seems not to be accurate linear.
  // FIXME
  // Should be arbitrary value 100? or based on pixel size?
  var count = 200;
  var step = (dataExtent[1] - dataExtent[0]) / count;
  var value = dataExtent[0];
  var stopValues = [];
  for (var i = 0; i <= count && value < dataExtent[1]; i++) {
    stopValues.push(value);
    value += step;
  }
  stopValues.push(dataExtent[1]);
  return stopValues;
}
export default ContinuousModel;