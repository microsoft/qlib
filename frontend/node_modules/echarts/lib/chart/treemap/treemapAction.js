
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
import * as helper from '../helper/treeHelper.js';
import { noop } from 'zrender/lib/core/util.js';
var actionTypes = ['treemapZoomToNode', 'treemapRender', 'treemapMove'];
export function installTreemapAction(registers) {
  for (var i = 0; i < actionTypes.length; i++) {
    registers.registerAction({
      type: actionTypes[i],
      update: 'updateView'
    }, noop);
  }
  registers.registerAction({
    type: 'treemapRootToNode',
    update: 'updateView'
  }, function (payload, ecModel) {
    ecModel.eachComponent({
      mainType: 'series',
      subType: 'treemap',
      query: payload
    }, handleRootToNode);
    function handleRootToNode(model, index) {
      var types = ['treemapZoomToNode', 'treemapRootToNode'];
      var targetInfo = helper.retrieveTargetInfo(payload, types, model);
      if (targetInfo) {
        var originViewRoot = model.getViewRoot();
        if (originViewRoot) {
          payload.direction = helper.aboveViewRoot(originViewRoot, targetInfo.node) ? 'rollUp' : 'drillDown';
        }
        model.resetViewRoot(targetInfo.node);
      }
    }
  });
}