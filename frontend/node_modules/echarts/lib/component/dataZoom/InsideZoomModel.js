
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
import DataZoomModel from './DataZoomModel.js';
import { inheritDefaultOption } from '../../util/component.js';
var InsideZoomModel = /** @class */function (_super) {
  __extends(InsideZoomModel, _super);
  function InsideZoomModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = InsideZoomModel.type;
    return _this;
  }
  InsideZoomModel.type = 'dataZoom.inside';
  InsideZoomModel.defaultOption = inheritDefaultOption(DataZoomModel.defaultOption, {
    disabled: false,
    zoomLock: false,
    zoomOnMouseWheel: true,
    moveOnMouseMove: true,
    moveOnMouseWheel: false,
    preventDefaultMouseMove: true
  });
  return InsideZoomModel;
}(DataZoomModel);
export default InsideZoomModel;