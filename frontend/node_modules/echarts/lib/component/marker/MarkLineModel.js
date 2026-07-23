
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
import MarkerModel from './MarkerModel.js';
var MarkLineModel = /** @class */function (_super) {
  __extends(MarkLineModel, _super);
  function MarkLineModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = MarkLineModel.type;
    return _this;
  }
  MarkLineModel.prototype.createMarkerModelFromSeries = function (markerOpt, masterMarkerModel, ecModel) {
    return new MarkLineModel(markerOpt, masterMarkerModel, ecModel);
  };
  MarkLineModel.type = 'markLine';
  MarkLineModel.defaultOption = {
    // zlevel: 0,
    z: 5,
    symbol: ['circle', 'arrow'],
    symbolSize: [8, 16],
    // symbolRotate: 0,
    symbolOffset: 0,
    precision: 2,
    tooltip: {
      trigger: 'item'
    },
    label: {
      show: true,
      position: 'end',
      distance: 5
    },
    lineStyle: {
      type: 'dashed'
    },
    emphasis: {
      label: {
        show: true
      },
      lineStyle: {
        width: 3
      }
    },
    animationEasing: 'linear'
  };
  return MarkLineModel;
}(MarkerModel);
export default MarkLineModel;