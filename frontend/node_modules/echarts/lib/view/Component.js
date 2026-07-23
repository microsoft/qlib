
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
import Group from 'zrender/lib/graphic/Group.js';
import * as componentUtil from '../util/component.js';
import * as clazzUtil from '../util/clazz.js';
var ComponentView = /** @class */function () {
  function ComponentView() {
    this.group = new Group();
    this.uid = componentUtil.getUID('viewComponent');
  }
  ComponentView.prototype.init = function (ecModel, api) {};
  ComponentView.prototype.render = function (model, ecModel, api, payload) {};
  ComponentView.prototype.dispose = function (ecModel, api) {};
  ComponentView.prototype.updateView = function (model, ecModel, api, payload) {
    // Do nothing;
  };
  ComponentView.prototype.updateLayout = function (model, ecModel, api, payload) {
    // Do nothing;
  };
  ComponentView.prototype.updateVisual = function (model, ecModel, api, payload) {
    // Do nothing;
  };
  /**
   * Hook for toggle blur target series.
   * Can be used in marker for blur or leave blur the markers
   */
  ComponentView.prototype.toggleBlurSeries = function (seriesModels, isBlur, ecModel) {
    // Do nothing;
  };
  /**
   * Traverse the new rendered elements.
   *
   * It will traverse the new added element in progressive rendering.
   * And traverse all in normal rendering.
   */
  ComponentView.prototype.eachRendered = function (cb) {
    var group = this.group;
    if (group) {
      group.traverse(cb);
    }
  };
  return ComponentView;
}();
;
clazzUtil.enableClassExtend(ComponentView);
clazzUtil.enableClassManagement(ComponentView);
export default ComponentView;