
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
import { each } from 'zrender/lib/core/util.js';
var BaseBarSeriesModel = /** @class */function (_super) {
  __extends(BaseBarSeriesModel, _super);
  function BaseBarSeriesModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = BaseBarSeriesModel.type;
    return _this;
  }
  BaseBarSeriesModel.prototype.getInitialData = function (option, ecModel) {
    return createSeriesData(null, this, {
      useEncodeDefaulter: true
    });
  };
  BaseBarSeriesModel.prototype.getMarkerPosition = function (value, dims, startingAtTick) {
    var coordSys = this.coordinateSystem;
    if (coordSys && coordSys.clampData) {
      // PENDING if clamp ?
      var clampData_1 = coordSys.clampData(value);
      var pt_1 = coordSys.dataToPoint(clampData_1);
      if (startingAtTick) {
        each(coordSys.getAxes(), function (axis, idx) {
          // If axis type is category, use tick coords instead
          if (axis.type === 'category' && dims != null) {
            var tickCoords = axis.getTicksCoords();
            var alignTicksWithLabel = axis.getTickModel().get('alignWithLabel');
            var targetTickId = clampData_1[idx];
            // The index of rightmost tick of markArea is 1 larger than x1/y1 index
            var isEnd = dims[idx] === 'x1' || dims[idx] === 'y1';
            if (isEnd && !alignTicksWithLabel) {
              targetTickId += 1;
            }
            // The only contains one tick, tickCoords is
            // like [{coord: 0, tickValue: 0}, {coord: 0}]
            // to the length should always be larger than 1
            if (tickCoords.length < 2) {
              return;
            } else if (tickCoords.length === 2) {
              // The left value and right value of the axis are
              // the same. coord is 0 in both items. Use the max
              // value of the axis as the coord
              pt_1[idx] = axis.toGlobalCoord(axis.getExtent()[isEnd ? 1 : 0]);
              return;
            }
            var leftCoord = void 0;
            var coord = void 0;
            var stepTickValue = 1;
            for (var i = 0; i < tickCoords.length; i++) {
              var tickCoord = tickCoords[i].coord;
              // The last item of tickCoords doesn't contain
              // tickValue
              var tickValue = i === tickCoords.length - 1 ? tickCoords[i - 1].tickValue + stepTickValue : tickCoords[i].tickValue;
              if (tickValue === targetTickId) {
                coord = tickCoord;
                break;
              } else if (tickValue < targetTickId) {
                leftCoord = tickCoord;
              } else if (leftCoord != null && tickValue > targetTickId) {
                coord = (tickCoord + leftCoord) / 2;
                break;
              }
              if (i === 1) {
                // Here we assume the step of category axes is
                // the same
                stepTickValue = tickValue - tickCoords[0].tickValue;
              }
            }
            if (coord == null) {
              if (!leftCoord) {
                // targetTickId is smaller than all tick ids in the
                // visible area, use the leftmost tick coord
                coord = tickCoords[0].coord;
              } else if (leftCoord) {
                // targetTickId is larger than all tick ids in the
                // visible area, use the rightmost tick coord
                coord = tickCoords[tickCoords.length - 1].coord;
              }
            }
            pt_1[idx] = axis.toGlobalCoord(coord);
          }
        });
      } else {
        var data = this.getData();
        var offset = data.getLayout('offset');
        var size = data.getLayout('size');
        var offsetIndex = coordSys.getBaseAxis().isHorizontal() ? 0 : 1;
        pt_1[offsetIndex] += offset + size / 2;
      }
      return pt_1;
    }
    return [NaN, NaN];
  };
  BaseBarSeriesModel.type = 'series.__base_bar__';
  BaseBarSeriesModel.defaultOption = {
    // zlevel: 0,
    z: 2,
    coordinateSystem: 'cartesian2d',
    legendHoverLink: true,
    // stack: null
    // Cartesian coordinate system
    // xAxisIndex: 0,
    // yAxisIndex: 0,
    barMinHeight: 0,
    barMinAngle: 0,
    // cursor: null,
    large: false,
    largeThreshold: 400,
    progressive: 3e3,
    progressiveChunkMode: 'mod'
  };
  return BaseBarSeriesModel;
}(SeriesModel);
SeriesModel.registerClass(BaseBarSeriesModel);
export default BaseBarSeriesModel;