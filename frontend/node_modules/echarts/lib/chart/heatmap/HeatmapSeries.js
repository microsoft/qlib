
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
import createSeriesData from '../helper/createSeriesData.js';
import CoordinateSystem from '../../core/CoordinateSystem.js';
var HeatmapSeriesModel = /** @class */function (_super) {
  __extends(HeatmapSeriesModel, _super);
  function HeatmapSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = HeatmapSeriesModel.type;
    return _this;
  }
  HeatmapSeriesModel.prototype.getInitialData = function (option, ecModel) {
    return createSeriesData(null, this, {
      generateCoord: 'value'
    });
  };
  HeatmapSeriesModel.prototype.preventIncremental = function () {
    var coordSysCreator = CoordinateSystem.get(this.get('coordinateSystem'));
    if (coordSysCreator && coordSysCreator.dimensions) {
      return coordSysCreator.dimensions[0] === 'lng' && coordSysCreator.dimensions[1] === 'lat';
    }
  };
  HeatmapSeriesModel.type = 'series.heatmap';
  HeatmapSeriesModel.dependencies = ['grid', 'geo', 'calendar'];
  HeatmapSeriesModel.defaultOption = {
    coordinateSystem: 'cartesian2d',
    // zlevel: 0,
    z: 2,
    // Cartesian coordinate system
    // xAxisIndex: 0,
    // yAxisIndex: 0,
    // Geo coordinate system
    geoIndex: 0,
    blurSize: 30,
    pointSize: 20,
    maxOpacity: 1,
    minOpacity: 0,
    select: {
      itemStyle: {
        borderColor: '#212121'
      }
    }
  };
  return HeatmapSeriesModel;
}(SeriesModel);
export default HeatmapSeriesModel;