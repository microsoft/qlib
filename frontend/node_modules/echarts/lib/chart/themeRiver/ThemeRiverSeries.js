
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
import SeriesModel from '../../model/Series.js';
import prepareSeriesDataSchema from '../../data/helper/createDimensions.js';
import { getDimensionTypeByAxis } from '../../data/helper/dimensionHelper.js';
import SeriesData from '../../data/SeriesData.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import { groupData, SINGLE_REFERRING } from '../../util/model.js';
import LegendVisualProvider from '../../visual/LegendVisualProvider.js';
import { createTooltipMarkup } from '../../component/tooltip/tooltipMarkup.js';
var DATA_NAME_INDEX = 2;
var ThemeRiverSeriesModel = /** @class */function (_super) {
  __extends(ThemeRiverSeriesModel, _super);
  function ThemeRiverSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ThemeRiverSeriesModel.type;
    return _this;
  }
  /**
   * @override
   */
  ThemeRiverSeriesModel.prototype.init = function (option) {
    // eslint-disable-next-line
    _super.prototype.init.apply(this, arguments);
    // Put this function here is for the sake of consistency of code style.
    // Enable legend selection for each data item
    // Use a function instead of direct access because data reference may changed
    this.legendVisualProvider = new LegendVisualProvider(zrUtil.bind(this.getData, this), zrUtil.bind(this.getRawData, this));
  };
  /**
   * If there is no value of a certain point in the time for some event,set it value to 0.
   *
   * @param {Array} data  initial data in the option
   * @return {Array}
   */
  ThemeRiverSeriesModel.prototype.fixData = function (data) {
    var rawDataLength = data.length;
    /**
     * Make sure every layer data get the same keys.
     * The value index tells which layer has visited.
     * {
     *  2014/01/01: -1
     * }
     */
    var timeValueKeys = {};
    // grouped data by name
    var groupResult = groupData(data, function (item) {
      if (!timeValueKeys.hasOwnProperty(item[0] + '')) {
        timeValueKeys[item[0] + ''] = -1;
      }
      return item[2];
    });
    var layerData = [];
    groupResult.buckets.each(function (items, key) {
      layerData.push({
        name: key,
        dataList: items
      });
    });
    var layerNum = layerData.length;
    for (var k = 0; k < layerNum; ++k) {
      var name_1 = layerData[k].name;
      for (var j = 0; j < layerData[k].dataList.length; ++j) {
        var timeValue = layerData[k].dataList[j][0] + '';
        timeValueKeys[timeValue] = k;
      }
      for (var timeValue in timeValueKeys) {
        if (timeValueKeys.hasOwnProperty(timeValue) && timeValueKeys[timeValue] !== k) {
          timeValueKeys[timeValue] = k;
          data[rawDataLength] = [timeValue, 0, name_1];
          rawDataLength++;
        }
      }
    }
    return data;
  };
  /**
   * @override
   * @param  option  the initial option that user gave
   * @param  ecModel  the model object for themeRiver option
   */
  ThemeRiverSeriesModel.prototype.getInitialData = function (option, ecModel) {
    var singleAxisModel = this.getReferringComponents('singleAxis', SINGLE_REFERRING).models[0];
    var axisType = singleAxisModel.get('type');
    // filter the data item with the value of label is undefined
    var filterData = zrUtil.filter(option.data, function (dataItem) {
      return dataItem[2] !== undefined;
    });
    // ??? TODO design a stage to transfer data for themeRiver and lines?
    var data = this.fixData(filterData || []);
    var nameList = [];
    var nameMap = this.nameMap = zrUtil.createHashMap();
    var count = 0;
    for (var i = 0; i < data.length; ++i) {
      nameList.push(data[i][DATA_NAME_INDEX]);
      if (!nameMap.get(data[i][DATA_NAME_INDEX])) {
        nameMap.set(data[i][DATA_NAME_INDEX], count);
        count++;
      }
    }
    var dimensions = prepareSeriesDataSchema(data, {
      coordDimensions: ['single'],
      dimensionsDefine: [{
        name: 'time',
        type: getDimensionTypeByAxis(axisType)
      }, {
        name: 'value',
        type: 'float'
      }, {
        name: 'name',
        type: 'ordinal'
      }],
      encodeDefine: {
        single: 0,
        value: 1,
        itemName: 2
      }
    }).dimensions;
    var list = new SeriesData(dimensions, this);
    list.initData(data);
    return list;
  };
  /**
   * The raw data is divided into multiple layers and each layer
   *     has same name.
   */
  ThemeRiverSeriesModel.prototype.getLayerSeries = function () {
    var data = this.getData();
    var lenCount = data.count();
    var indexArr = [];
    for (var i = 0; i < lenCount; ++i) {
      indexArr[i] = i;
    }
    var timeDim = data.mapDimension('single');
    // data group by name
    var groupResult = groupData(indexArr, function (index) {
      return data.get('name', index);
    });
    var layerSeries = [];
    groupResult.buckets.each(function (items, key) {
      items.sort(function (index1, index2) {
        return data.get(timeDim, index1) - data.get(timeDim, index2);
      });
      layerSeries.push({
        name: key,
        indices: items
      });
    });
    return layerSeries;
  };
  /**
   * Get data indices for show tooltip content
   */
  ThemeRiverSeriesModel.prototype.getAxisTooltipData = function (dim, value, baseAxis) {
    if (!zrUtil.isArray(dim)) {
      dim = dim ? [dim] : [];
    }
    var data = this.getData();
    var layerSeries = this.getLayerSeries();
    var indices = [];
    var layerNum = layerSeries.length;
    var nestestValue;
    for (var i = 0; i < layerNum; ++i) {
      var minDist = Number.MAX_VALUE;
      var nearestIdx = -1;
      var pointNum = layerSeries[i].indices.length;
      for (var j = 0; j < pointNum; ++j) {
        var theValue = data.get(dim[0], layerSeries[i].indices[j]);
        var dist = Math.abs(theValue - value);
        if (dist <= minDist) {
          nestestValue = theValue;
          minDist = dist;
          nearestIdx = layerSeries[i].indices[j];
        }
      }
      indices.push(nearestIdx);
    }
    return {
      dataIndices: indices,
      nestestValue: nestestValue
    };
  };
  ThemeRiverSeriesModel.prototype.formatTooltip = function (dataIndex, multipleSeries, dataType) {
    var data = this.getData();
    var name = data.getName(dataIndex);
    var value = data.get(data.mapDimension('value'), dataIndex);
    return createTooltipMarkup('nameValue', {
      name: name,
      value: value
    });
  };
  ThemeRiverSeriesModel.type = 'series.themeRiver';
  ThemeRiverSeriesModel.dependencies = ['singleAxis'];
  ThemeRiverSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    colorBy: 'data',
    coordinateSystem: 'singleAxis',
    // gap in axis's orthogonal orientation
    boundaryGap: ['10%', '10%'],
    // legendHoverLink: true,
    singleAxisIndex: 0,
    animationEasing: 'linear',
    label: {
      margin: 4,
      show: true,
      position: 'left',
      fontSize: 11
    },
    emphasis: {
      label: {
        show: true
      }
    }
  };
  return ThemeRiverSeriesModel;
}(SeriesModel);
export default ThemeRiverSeriesModel;