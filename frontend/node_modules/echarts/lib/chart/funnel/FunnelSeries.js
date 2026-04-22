
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
import createSeriesDataSimply from '../helper/createSeriesDataSimply.js';
import { defaultEmphasis } from '../../util/model.js';
import { makeSeriesEncodeForNameBased } from '../../data/helper/sourceHelper.js';
import LegendVisualProvider from '../../visual/LegendVisualProvider.js';
import SeriesModel from '../../model/Series.js';
var FunnelSeriesModel = /** @class */function (_super) {
  __extends(FunnelSeriesModel, _super);
  function FunnelSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = FunnelSeriesModel.type;
    return _this;
  }
  FunnelSeriesModel.prototype.init = function (option) {
    _super.prototype.init.apply(this, arguments);
    // Enable legend selection for each data item
    // Use a function instead of direct access because data reference may changed
    this.legendVisualProvider = new LegendVisualProvider(zrUtil.bind(this.getData, this), zrUtil.bind(this.getRawData, this));
    // Extend labelLine emphasis
    this._defaultLabelLine(option);
  };
  FunnelSeriesModel.prototype.getInitialData = function (option, ecModel) {
    return createSeriesDataSimply(this, {
      coordDimensions: ['value'],
      encodeDefaulter: zrUtil.curry(makeSeriesEncodeForNameBased, this)
    });
  };
  FunnelSeriesModel.prototype._defaultLabelLine = function (option) {
    // Extend labelLine emphasis
    defaultEmphasis(option, 'labelLine', ['show']);
    var labelLineNormalOpt = option.labelLine;
    var labelLineEmphasisOpt = option.emphasis.labelLine;
    // Not show label line if `label.normal.show = false`
    labelLineNormalOpt.show = labelLineNormalOpt.show && option.label.show;
    labelLineEmphasisOpt.show = labelLineEmphasisOpt.show && option.emphasis.label.show;
  };
  // Overwrite
  FunnelSeriesModel.prototype.getDataParams = function (dataIndex) {
    var data = this.getData();
    var params = _super.prototype.getDataParams.call(this, dataIndex);
    var valueDim = data.mapDimension('value');
    var sum = data.getSum(valueDim);
    // Percent is 0 if sum is 0
    params.percent = !sum ? 0 : +(data.get(valueDim, dataIndex) / sum * 100).toFixed(2);
    params.$vars.push('percent');
    return params;
  };
  FunnelSeriesModel.type = 'series.funnel';
  FunnelSeriesModel.defaultOption = {
    // zlevel: 0,                  // 一级层叠
    z: 2,
    legendHoverLink: true,
    colorBy: 'data',
    left: 80,
    top: 60,
    right: 80,
    bottom: 60,
    // width: {totalWidth} - left - right,
    // height: {totalHeight} - top - bottom,
    // 默认取数据最小最大值
    // min: 0,
    // max: 100,
    minSize: '0%',
    maxSize: '100%',
    sort: 'descending',
    orient: 'vertical',
    gap: 0,
    funnelAlign: 'center',
    label: {
      show: true,
      position: 'outer'
      // formatter: 标签文本格式器，同Tooltip.formatter，不支持异步回调
    },
    labelLine: {
      show: true,
      length: 20,
      lineStyle: {
        // color: 各异,
        width: 1
      }
    },
    itemStyle: {
      // color: 各异,
      borderColor: '#fff',
      borderWidth: 1
    },
    emphasis: {
      label: {
        show: true
      }
    },
    select: {
      itemStyle: {
        borderColor: '#212121'
      }
    }
  };
  return FunnelSeriesModel;
}(SeriesModel);
export default FunnelSeriesModel;