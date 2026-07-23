
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
import createSeriesDataSimply from '../helper/createSeriesDataSimply.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import * as modelUtil from '../../util/model.js';
import { getPercentSeats } from '../../util/number.js';
import { makeSeriesEncodeForNameBased } from '../../data/helper/sourceHelper.js';
import LegendVisualProvider from '../../visual/LegendVisualProvider.js';
import SeriesModel from '../../model/Series.js';
var innerData = modelUtil.makeInner();
var PieSeriesModel = /** @class */function (_super) {
  __extends(PieSeriesModel, _super);
  function PieSeriesModel() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  /**
   * @overwrite
   */
  PieSeriesModel.prototype.init = function (option) {
    _super.prototype.init.apply(this, arguments);
    // Enable legend selection for each data item
    // Use a function instead of direct access because data reference may changed
    this.legendVisualProvider = new LegendVisualProvider(zrUtil.bind(this.getData, this), zrUtil.bind(this.getRawData, this));
    this._defaultLabelLine(option);
  };
  /**
   * @overwrite
   */
  PieSeriesModel.prototype.mergeOption = function () {
    _super.prototype.mergeOption.apply(this, arguments);
  };
  /**
   * @overwrite
   */
  PieSeriesModel.prototype.getInitialData = function () {
    return createSeriesDataSimply(this, {
      coordDimensions: ['value'],
      encodeDefaulter: zrUtil.curry(makeSeriesEncodeForNameBased, this)
    });
  };
  /**
   * @overwrite
   */
  PieSeriesModel.prototype.getDataParams = function (dataIndex) {
    var data = this.getData();
    // update seats when data is changed
    var dataInner = innerData(data);
    var seats = dataInner.seats;
    if (!seats) {
      var valueList_1 = [];
      data.each(data.mapDimension('value'), function (value) {
        valueList_1.push(value);
      });
      seats = dataInner.seats = getPercentSeats(valueList_1, data.hostModel.get('percentPrecision'));
    }
    var params = _super.prototype.getDataParams.call(this, dataIndex);
    // seats may be empty when sum is 0
    params.percent = seats[dataIndex] || 0;
    params.$vars.push('percent');
    return params;
  };
  PieSeriesModel.prototype._defaultLabelLine = function (option) {
    // Extend labelLine emphasis
    modelUtil.defaultEmphasis(option, 'labelLine', ['show']);
    var labelLineNormalOpt = option.labelLine;
    var labelLineEmphasisOpt = option.emphasis.labelLine;
    // Not show label line if `label.normal.show = false`
    labelLineNormalOpt.show = labelLineNormalOpt.show && option.label.show;
    labelLineEmphasisOpt.show = labelLineEmphasisOpt.show && option.emphasis.label.show;
  };
  PieSeriesModel.type = 'series.pie';
  PieSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    legendHoverLink: true,
    colorBy: 'data',
    // 默认全局居中
    center: ['50%', '50%'],
    radius: [0, '75%'],
    // 默认顺时针
    clockwise: true,
    startAngle: 90,
    endAngle: 'auto',
    padAngle: 0,
    // 最小角度改为0
    minAngle: 0,
    // If the angle of a sector less than `minShowLabelAngle`,
    // the label will not be displayed.
    minShowLabelAngle: 0,
    // 选中时扇区偏移量
    selectedOffset: 10,
    // 选择模式，默认关闭，可选single，multiple
    // selectedMode: false,
    // 南丁格尔玫瑰图模式，'radius'（半径） | 'area'（面积）
    // roseType: null,
    percentPrecision: 2,
    // If still show when all data zero.
    stillShowZeroSum: true,
    // cursor: null,
    left: 0,
    top: 0,
    right: 0,
    bottom: 0,
    width: null,
    height: null,
    label: {
      // color: 'inherit',
      // If rotate around circle
      rotate: 0,
      show: true,
      overflow: 'truncate',
      // 'outer', 'inside', 'center'
      position: 'outer',
      // 'none', 'labelLine', 'edge'. Works only when position is 'outer'
      alignTo: 'none',
      // Closest distance between label and chart edge.
      // Works only position is 'outer' and alignTo is 'edge'.
      edgeDistance: '25%',
      // Works only position is 'outer' and alignTo is not 'edge'.
      bleedMargin: 10,
      // Distance between text and label line.
      distanceToLabelLine: 5
      // formatter: 标签文本格式器，同 tooltip.formatter，不支持异步回调
      // 默认使用全局文本样式，详见 textStyle
      // distance: 当position为inner时有效，为label位置到圆心的距离与圆半径(环状图为内外半径和)的比例系数
    },
    // Enabled when label.normal.position is 'outer'
    labelLine: {
      show: true,
      // 引导线两段中的第一段长度
      length: 15,
      // 引导线两段中的第二段长度
      length2: 15,
      smooth: false,
      minTurnAngle: 90,
      maxSurfaceAngle: 90,
      lineStyle: {
        // color: 各异,
        width: 1,
        type: 'solid'
      }
    },
    itemStyle: {
      borderWidth: 1,
      borderJoin: 'round'
    },
    showEmptyCircle: true,
    emptyCircleStyle: {
      color: 'lightgray',
      opacity: 1
    },
    labelLayout: {
      // Hide the overlapped label.
      hideOverlap: true
    },
    emphasis: {
      scale: true,
      scaleSize: 5
    },
    // If use strategy to avoid label overlapping
    avoidLabelOverlap: true,
    // Animation type. Valid values: expansion, scale
    animationType: 'expansion',
    animationDuration: 1000,
    // Animation type when update. Valid values: transition, expansion
    animationTypeUpdate: 'transition',
    animationEasingUpdate: 'cubicInOut',
    animationDurationUpdate: 500,
    animationEasing: 'cubicInOut'
  };
  return PieSeriesModel;
}(SeriesModel);
export default PieSeriesModel;