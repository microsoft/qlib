
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
import * as numberUtil from '../../util/number.js';
import { isDimensionStacked } from '../../data/helper/dataStackHelper.js';
import { indexOf, curry, clone, isArray } from 'zrender/lib/core/util.js';
import { parseDataValue } from '../../data/helper/dataValueHelper.js';
function hasXOrY(item) {
  return !(isNaN(parseFloat(item.x)) && isNaN(parseFloat(item.y)));
}
function hasXAndY(item) {
  return !isNaN(parseFloat(item.x)) && !isNaN(parseFloat(item.y));
}
function markerTypeCalculatorWithExtent(markerType, data, otherDataDim, targetDataDim, otherCoordIndex, targetCoordIndex) {
  var coordArr = [];
  var stacked = isDimensionStacked(data, targetDataDim /* , otherDataDim */);
  var calcDataDim = stacked ? data.getCalculationInfo('stackResultDimension') : targetDataDim;
  var value = numCalculate(data, calcDataDim, markerType);
  var dataIndex = data.indicesOfNearest(calcDataDim, value)[0];
  coordArr[otherCoordIndex] = data.get(otherDataDim, dataIndex);
  coordArr[targetCoordIndex] = data.get(calcDataDim, dataIndex);
  var coordArrValue = data.get(targetDataDim, dataIndex);
  // Make it simple, do not visit all stacked value to count precision.
  var precision = numberUtil.getPrecision(data.get(targetDataDim, dataIndex));
  precision = Math.min(precision, 20);
  if (precision >= 0) {
    coordArr[targetCoordIndex] = +coordArr[targetCoordIndex].toFixed(precision);
  }
  return [coordArr, coordArrValue];
}
// TODO Specified percent
var markerTypeCalculator = {
  min: curry(markerTypeCalculatorWithExtent, 'min'),
  max: curry(markerTypeCalculatorWithExtent, 'max'),
  average: curry(markerTypeCalculatorWithExtent, 'average'),
  median: curry(markerTypeCalculatorWithExtent, 'median')
};
/**
 * Transform markPoint data item to format used in List by do the following
 * 1. Calculate statistic like `max`, `min`, `average`
 * 2. Convert `item.xAxis`, `item.yAxis` to `item.coord` array
 */
export function dataTransform(seriesModel, item) {
  if (!item) {
    return;
  }
  var data = seriesModel.getData();
  var coordSys = seriesModel.coordinateSystem;
  var dims = coordSys && coordSys.dimensions;
  // 1. If not specify the position with pixel directly
  // 2. If `coord` is not a data array. Which uses `xAxis`,
  // `yAxis` to specify the coord on each dimension
  // parseFloat first because item.x and item.y can be percent string like '20%'
  if (!hasXAndY(item) && !isArray(item.coord) && isArray(dims)) {
    var axisInfo = getAxisInfo(item, data, coordSys, seriesModel);
    // Clone the option
    // Transform the properties xAxis, yAxis, radiusAxis, angleAxis, geoCoord to value
    item = clone(item);
    if (item.type && markerTypeCalculator[item.type] && axisInfo.baseAxis && axisInfo.valueAxis) {
      var otherCoordIndex = indexOf(dims, axisInfo.baseAxis.dim);
      var targetCoordIndex = indexOf(dims, axisInfo.valueAxis.dim);
      var coordInfo = markerTypeCalculator[item.type](data, axisInfo.baseDataDim, axisInfo.valueDataDim, otherCoordIndex, targetCoordIndex);
      item.coord = coordInfo[0];
      // Force to use the value of calculated value.
      // let item use the value without stack.
      item.value = coordInfo[1];
    } else {
      // FIXME Only has one of xAxis and yAxis.
      item.coord = [item.xAxis != null ? item.xAxis : item.radiusAxis, item.yAxis != null ? item.yAxis : item.angleAxis];
    }
  }
  // x y is provided
  if (item.coord == null || !isArray(dims)) {
    item.coord = [];
  } else {
    // Each coord support max, min, average
    var coord = item.coord;
    for (var i = 0; i < 2; i++) {
      if (markerTypeCalculator[coord[i]]) {
        coord[i] = numCalculate(data, data.mapDimension(dims[i]), coord[i]);
      }
    }
  }
  return item;
}
export function getAxisInfo(item, data, coordSys, seriesModel) {
  var ret = {};
  if (item.valueIndex != null || item.valueDim != null) {
    ret.valueDataDim = item.valueIndex != null ? data.getDimension(item.valueIndex) : item.valueDim;
    ret.valueAxis = coordSys.getAxis(dataDimToCoordDim(seriesModel, ret.valueDataDim));
    ret.baseAxis = coordSys.getOtherAxis(ret.valueAxis);
    ret.baseDataDim = data.mapDimension(ret.baseAxis.dim);
  } else {
    ret.baseAxis = seriesModel.getBaseAxis();
    ret.valueAxis = coordSys.getOtherAxis(ret.baseAxis);
    ret.baseDataDim = data.mapDimension(ret.baseAxis.dim);
    ret.valueDataDim = data.mapDimension(ret.valueAxis.dim);
  }
  return ret;
}
function dataDimToCoordDim(seriesModel, dataDim) {
  var dimItem = seriesModel.getData().getDimensionInfo(dataDim);
  return dimItem && dimItem.coordDim;
}
/**
 * Filter data which is out of coordinateSystem range
 * [dataFilter description]
 */
export function dataFilter(
// Currently only polar and cartesian has containData.
coordSys, item) {
  // Always return true if there is no coordSys
  return coordSys && coordSys.containData && item.coord && !hasXOrY(item) ? coordSys.containData(item.coord) : true;
}
export function zoneFilter(
// Currently only polar and cartesian has containData.
coordSys, item1, item2) {
  // Always return true if there is no coordSys
  return coordSys && coordSys.containZone && item1.coord && item2.coord && !hasXOrY(item1) && !hasXOrY(item2) ? coordSys.containZone(item1.coord, item2.coord) : true;
}
export function createMarkerDimValueGetter(inCoordSys, dims) {
  return inCoordSys ? function (item, dimName, dataIndex, dimIndex) {
    var rawVal = dimIndex < 2
    // x, y, radius, angle
    ? item.coord && item.coord[dimIndex] : item.value;
    return parseDataValue(rawVal, dims[dimIndex]);
  } : function (item, dimName, dataIndex, dimIndex) {
    return parseDataValue(item.value, dims[dimIndex]);
  };
}
export function numCalculate(data, valueDataDim, type) {
  if (type === 'average') {
    var sum_1 = 0;
    var count_1 = 0;
    data.each(valueDataDim, function (val, idx) {
      if (!isNaN(val)) {
        sum_1 += val;
        count_1++;
      }
    });
    return sum_1 / count_1;
  } else if (type === 'median') {
    return data.getMedian(valueDataDim);
  } else {
    // max & min
    return data.getDataExtent(valueDataDim)[type === 'max' ? 1 : 0];
  }
}