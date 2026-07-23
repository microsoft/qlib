
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
import ComponentModel from '../../model/Component.js';
import { AxisModelCommonMixin } from '../axisModelCommonMixin.js';
import { mixin } from 'zrender/lib/core/util.js';
var SingleAxisModel = /** @class */function (_super) {
  __extends(SingleAxisModel, _super);
  function SingleAxisModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = SingleAxisModel.type;
    return _this;
  }
  SingleAxisModel.prototype.getCoordSysModel = function () {
    return this;
  };
  SingleAxisModel.type = 'singleAxis';
  SingleAxisModel.layoutMode = 'box';
  SingleAxisModel.defaultOption = {
    left: '5%',
    top: '5%',
    right: '5%',
    bottom: '5%',
    type: 'value',
    position: 'bottom',
    orient: 'horizontal',
    axisLine: {
      show: true,
      lineStyle: {
        width: 1,
        type: 'solid'
      }
    },
    // Single coordinate system and single axis is the,
    // which is used as the parent tooltip model.
    // same model, so we set default tooltip show as true.
    tooltip: {
      show: true
    },
    axisTick: {
      show: true,
      length: 6,
      lineStyle: {
        width: 1
      }
    },
    axisLabel: {
      show: true,
      interval: 'auto'
    },
    splitLine: {
      show: true,
      lineStyle: {
        type: 'dashed',
        opacity: 0.2
      }
    }
  };
  return SingleAxisModel;
}(ComponentModel);
mixin(SingleAxisModel, AxisModelCommonMixin.prototype);
export default SingleAxisModel;