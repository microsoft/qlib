
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
import { extend } from 'zrender/lib/core/util.js';
import { deprecateReplaceLog } from '../../util/log.js';
import { retrieveTargetInfo, aboveViewRoot } from '../helper/treeHelper.js';
export var ROOT_TO_NODE_ACTION = 'sunburstRootToNode';
var HIGHLIGHT_ACTION = 'sunburstHighlight';
var UNHIGHLIGHT_ACTION = 'sunburstUnhighlight';
export function installSunburstAction(registers) {
  registers.registerAction({
    type: ROOT_TO_NODE_ACTION,
    update: 'updateView'
  }, function (payload, ecModel) {
    ecModel.eachComponent({
      mainType: 'series',
      subType: 'sunburst',
      query: payload
    }, handleRootToNode);
    function handleRootToNode(model, index) {
      var targetInfo = retrieveTargetInfo(payload, [ROOT_TO_NODE_ACTION], model);
      if (targetInfo) {
        var originViewRoot = model.getViewRoot();
        if (originViewRoot) {
          payload.direction = aboveViewRoot(originViewRoot, targetInfo.node) ? 'rollUp' : 'drillDown';
        }
        model.resetViewRoot(targetInfo.node);
      }
    }
  });
  registers.registerAction({
    type: HIGHLIGHT_ACTION,
    update: 'none'
  }, function (payload, ecModel, api) {
    // Clone
    payload = extend({}, payload);
    ecModel.eachComponent({
      mainType: 'series',
      subType: 'sunburst',
      query: payload
    }, handleHighlight);
    function handleHighlight(model) {
      var targetInfo = retrieveTargetInfo(payload, [HIGHLIGHT_ACTION], model);
      if (targetInfo) {
        payload.dataIndex = targetInfo.node.dataIndex;
      }
    }
    if (process.env.NODE_ENV !== 'production') {
      deprecateReplaceLog('sunburstHighlight', 'highlight');
    }
    // Fast forward action
    api.dispatchAction(extend(payload, {
      type: 'highlight'
    }));
  });
  registers.registerAction({
    type: UNHIGHLIGHT_ACTION,
    update: 'updateView'
  }, function (payload, ecModel, api) {
    payload = extend({}, payload);
    if (process.env.NODE_ENV !== 'production') {
      deprecateReplaceLog('sunburstUnhighlight', 'downplay');
    }
    api.dispatchAction(extend(payload, {
      type: 'downplay'
    }));
  });
}