
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
import SeriesModel from '../../model/Series.js';
var GaugeSeriesModel = /** @class */function (_super) {
  __extends(GaugeSeriesModel, _super);
  function GaugeSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = GaugeSeriesModel.type;
    _this.visualStyleAccessPath = 'itemStyle';
    return _this;
  }
  GaugeSeriesModel.prototype.getInitialData = function (option, ecModel) {
    return createSeriesDataSimply(this, ['value']);
  };
  GaugeSeriesModel.type = 'series.gauge';
  GaugeSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    colorBy: 'data',
    // 默认全局居中
    center: ['50%', '50%'],
    legendHoverLink: true,
    radius: '75%',
    startAngle: 225,
    endAngle: -45,
    clockwise: true,
    // 最小值
    min: 0,
    // 最大值
    max: 100,
    // 分割段数，默认为10
    splitNumber: 10,
    // 坐标轴线
    axisLine: {
      // 默认显示，属性show控制显示与否
      show: true,
      roundCap: false,
      lineStyle: {
        color: [[1, '#E6EBF8']],
        width: 10
      }
    },
    // 坐标轴线
    progress: {
      // 默认显示，属性show控制显示与否
      show: false,
      overlap: true,
      width: 10,
      roundCap: false,
      clip: true
    },
    // 分隔线
    splitLine: {
      // 默认显示，属性show控制显示与否
      show: true,
      // 属性length控制线长
      length: 10,
      distance: 10,
      // 属性lineStyle（详见lineStyle）控制线条样式
      lineStyle: {
        color: '#63677A',
        width: 3,
        type: 'solid'
      }
    },
    // 坐标轴小标记
    axisTick: {
      // 属性show控制显示与否，默认不显示
      show: true,
      // 每份split细分多少段
      splitNumber: 5,
      // 属性length控制线长
      length: 6,
      distance: 10,
      // 属性lineStyle控制线条样式
      lineStyle: {
        color: '#63677A',
        width: 1,
        type: 'solid'
      }
    },
    axisLabel: {
      show: true,
      distance: 15,
      // formatter: null,
      color: '#464646',
      fontSize: 12,
      rotate: 0
    },
    pointer: {
      icon: null,
      offsetCenter: [0, 0],
      show: true,
      showAbove: true,
      length: '60%',
      width: 6,
      keepAspect: false
    },
    anchor: {
      show: false,
      showAbove: false,
      size: 6,
      icon: 'circle',
      offsetCenter: [0, 0],
      keepAspect: false,
      itemStyle: {
        color: '#fff',
        borderWidth: 0,
        borderColor: '#5470c6'
      }
    },
    title: {
      show: true,
      // x, y，单位px
      offsetCenter: [0, '20%'],
      // 其余属性默认使用全局文本样式，详见TEXTSTYLE
      color: '#464646',
      fontSize: 16,
      valueAnimation: false
    },
    detail: {
      show: true,
      backgroundColor: 'rgba(0,0,0,0)',
      borderWidth: 0,
      borderColor: '#ccc',
      width: 100,
      height: null,
      padding: [5, 10],
      // x, y，单位px
      offsetCenter: [0, '40%'],
      // formatter: null,
      // 其余属性默认使用全局文本样式，详见TEXTSTYLE
      color: '#464646',
      fontSize: 30,
      fontWeight: 'bold',
      lineHeight: 30,
      valueAnimation: false
    }
  };
  return GaugeSeriesModel;
}(SeriesModel);
export default GaugeSeriesModel;