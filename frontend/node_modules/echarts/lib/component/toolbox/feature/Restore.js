
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
import * as echarts from '../../../core/echarts.js';
import * as history from '../../dataZoom/history.js';
import { ToolboxFeature } from '../featureManager.js';
var RestoreOption = /** @class */function (_super) {
  __extends(RestoreOption, _super);
  function RestoreOption() {
    return _super !== null && _super.apply(this, arguments) || this;
  }
  RestoreOption.prototype.onclick = function (ecModel, api) {
    history.clear(ecModel);
    api.dispatchAction({
      type: 'restore',
      from: this.uid
    });
  };
  RestoreOption.getDefaultOption = function (ecModel) {
    var defaultOption = {
      show: true,
      // eslint-disable-next-line
      icon: 'M3.8,33.4 M47,18.9h9.8V8.7 M56.3,20.1 C52.1,9,40.5,0.6,26.8,2.1C12.6,3.7,1.6,16.2,2.1,30.6 M13,41.1H3.1v10.2 M3.7,39.9c4.2,11.1,15.8,19.5,29.5,18 c14.2-1.6,25.2-14.1,24.7-28.5',
      title: ecModel.getLocaleModel().get(['toolbox', 'restore', 'title'])
    };
    return defaultOption;
  };
  return RestoreOption;
}(ToolboxFeature);
// TODO: SELF REGISTERED.
echarts.registerAction({
  type: 'restore',
  event: 'restore',
  update: 'prepareAndUpdate'
}, function (payload, ecModel) {
  ecModel.resetOption('recreate');
});
export default RestoreOption;