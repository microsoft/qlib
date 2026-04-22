
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
import { use } from '../../extension.js';
import AxisView from '../axis/AxisView.js';
import PolarAxisPointer from '../axisPointer/PolarAxisPointer.js';
import { install as installAxisPointer } from '../axisPointer/install.js';
import PolarModel from '../../coord/polar/PolarModel.js';
import axisModelCreator from '../../coord/axisModelCreator.js';
import { AngleAxisModel, RadiusAxisModel } from '../../coord/polar/AxisModel.js';
import polarCreator from '../../coord/polar/polarCreator.js';
import AngleAxisView from '../axis/AngleAxisView.js';
import RadiusAxisView from '../axis/RadiusAxisView.js';
import ComponentView from '../../view/Component.js';
import { curry } from 'zrender/lib/core/util.js';
import barLayoutPolar from '../../layout/barPolar.js';
var angleAxisExtraOption = {
  startAngle: 90,
  clockwise: true,
  splitNumber: 12,
  axisLabel: {
    rotate: 0
  }
};
var radiusAxisExtraOption = {
  splitNumber: 5
};
var PolarView = /** @class */function (_super) {
  __extends(PolarView, _super);
  function PolarView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = PolarView.type;
    return _this;
  }
  PolarView.type = 'polar';
  return PolarView;
}(ComponentView);
export function install(registers) {
  use(installAxisPointer);
  AxisView.registerAxisPointerClass('PolarAxisPointer', PolarAxisPointer);
  registers.registerCoordinateSystem('polar', polarCreator);
  registers.registerComponentModel(PolarModel);
  registers.registerComponentView(PolarView);
  // Model and view for angleAxis and radiusAxis
  axisModelCreator(registers, 'angle', AngleAxisModel, angleAxisExtraOption);
  axisModelCreator(registers, 'radius', RadiusAxisModel, radiusAxisExtraOption);
  registers.registerComponentView(AngleAxisView);
  registers.registerComponentView(RadiusAxisView);
  registers.registerLayout(curry(barLayoutPolar, 'bar'));
}