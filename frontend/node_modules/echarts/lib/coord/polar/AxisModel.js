
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
import ComponentModel from '../../model/Component.js';
import { AxisModelCommonMixin } from '../axisModelCommonMixin.js';
import { SINGLE_REFERRING } from '../../util/model.js';
var PolarAxisModel = /** @class */function (_super) {
  __extends(PolarAxisModel, _super);
  function PolarAxisModel() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  PolarAxisModel.prototype.getCoordSysModel = function () {
    return this.getReferringComponents('polar', SINGLE_REFERRING).models[0];
  };
  PolarAxisModel.type = 'polarAxis';
  return PolarAxisModel;
}(ComponentModel);
zrUtil.mixin(PolarAxisModel, AxisModelCommonMixin);
export { PolarAxisModel };
var AngleAxisModel = /** @class */function (_super) {
  __extends(AngleAxisModel, _super);
  function AngleAxisModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = AngleAxisModel.type;
    return _this;
  }
  AngleAxisModel.type = 'angleAxis';
  return AngleAxisModel;
}(PolarAxisModel);
export { AngleAxisModel };
var RadiusAxisModel = /** @class */function (_super) {
  __extends(RadiusAxisModel, _super);
  function RadiusAxisModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = RadiusAxisModel.type;
    return _this;
  }
  RadiusAxisModel.type = 'radiusAxis';
  return RadiusAxisModel;
}(PolarAxisModel);
export { RadiusAxisModel };