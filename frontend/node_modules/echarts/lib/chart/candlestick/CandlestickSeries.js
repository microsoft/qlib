
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
import { WhiskerBoxCommonMixin } from '../helper/whiskerBoxCommon.js';
import { mixin } from 'zrender/lib/core/util.js';
var CandlestickSeriesModel = /** @class */function (_super) {
  __extends(CandlestickSeriesModel, _super);
  function CandlestickSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = CandlestickSeriesModel.type;
    _this.defaultValueDimensions = [{
      name: 'open',
      defaultTooltip: true
    }, {
      name: 'close',
      defaultTooltip: true
    }, {
      name: 'lowest',
      defaultTooltip: true
    }, {
      name: 'highest',
      defaultTooltip: true
    }];
    return _this;
  }
  /**
   * Get dimension for shadow in dataZoom
   * @return dimension name
   */
  CandlestickSeriesModel.prototype.getShadowDim = function () {
    return 'open';
  };
  CandlestickSeriesModel.prototype.brushSelector = function (dataIndex, data, selectors) {
    var itemLayout = data.getItemLayout(dataIndex);
    return itemLayout && selectors.rect(itemLayout.brushRect);
  };
  CandlestickSeriesModel.type = 'series.candlestick';
  CandlestickSeriesModel.dependencies = ['xAxis', 'yAxis', 'grid'];
  CandlestickSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    coordinateSystem: 'cartesian2d',
    legendHoverLink: true,
    // xAxisIndex: 0,
    // yAxisIndex: 0,
    layout: null,
    clip: true,
    itemStyle: {
      color: '#eb5454',
      color0: '#47b262',
      borderColor: '#eb5454',
      borderColor0: '#47b262',
      borderColorDoji: null,
      // borderColor: '#d24040',
      // borderColor0: '#398f4f',
      borderWidth: 1
    },
    emphasis: {
      itemStyle: {
        borderWidth: 2
      }
    },
    barMaxWidth: null,
    barMinWidth: null,
    barWidth: null,
    large: true,
    largeThreshold: 600,
    progressive: 3e3,
    progressiveThreshold: 1e4,
    progressiveChunkMode: 'mod',
    animationEasing: 'linear',
    animationDuration: 300
  };
  return CandlestickSeriesModel;
}(SeriesModel);
mixin(CandlestickSeriesModel, WhiskerBoxCommonMixin, true);
export default CandlestickSeriesModel;