
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
import brushPreprocessor from './preprocessor.js';
import BrushView from './BrushView.js';
import BrushModel from './BrushModel.js';
import brushVisual from './visualEncoding.js';
// TODO
import BrushFeature from '../toolbox/feature/Brush.js';
import { registerFeature } from '../toolbox/featureManager.js';
import { noop } from 'zrender/lib/core/util.js';
export function install(registers) {
  registers.registerComponentView(BrushView);
  registers.registerComponentModel(BrushModel);
  registers.registerPreprocessor(brushPreprocessor);
  registers.registerVisual(registers.PRIORITY.VISUAL.BRUSH, brushVisual);
  registers.registerAction({
    type: 'brush',
    event: 'brush',
    update: 'updateVisual'
  }, function (payload, ecModel) {
    ecModel.eachComponent({
      mainType: 'brush',
      query: payload
    }, function (brushModel) {
      brushModel.setAreas(payload.areas);
    });
  });
  /**
   * payload: {
   *      brushComponents: [
   *          {
   *              brushId,
   *              brushIndex,
   *              brushName,
   *              series: [
   *                  {
   *                      seriesId,
   *                      seriesIndex,
   *                      seriesName,
   *                      rawIndices: [21, 34, ...]
   *                  },
   *                  ...
   *              ]
   *          },
   *          ...
   *      ]
   * }
   */
  registers.registerAction({
    type: 'brushSelect',
    event: 'brushSelected',
    update: 'none'
  }, noop);
  registers.registerAction({
    type: 'brushEnd',
    event: 'brushEnd',
    update: 'none'
  }, noop);
  registerFeature('brush', BrushFeature);
}