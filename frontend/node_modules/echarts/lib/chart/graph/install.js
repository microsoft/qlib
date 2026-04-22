
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
import categoryFilter from './categoryFilter.js';
import categoryVisual from './categoryVisual.js';
import edgeVisual from './edgeVisual.js';
import simpleLayout from './simpleLayout.js';
import circularLayout from './circularLayout.js';
import forceLayout from './forceLayout.js';
import createView from './createView.js';
import View from '../../coord/View.js';
import GraphView from './GraphView.js';
import GraphSeriesModel from './GraphSeries.js';
import { updateCenterAndZoom } from '../../action/roamHelper.js';
import { noop } from 'zrender/lib/core/util.js';
var actionInfo = {
  type: 'graphRoam',
  event: 'graphRoam',
  update: 'none'
};
export function install(registers) {
  registers.registerChartView(GraphView);
  registers.registerSeriesModel(GraphSeriesModel);
  registers.registerProcessor(categoryFilter);
  registers.registerVisual(categoryVisual);
  registers.registerVisual(edgeVisual);
  registers.registerLayout(simpleLayout);
  registers.registerLayout(registers.PRIORITY.VISUAL.POST_CHART_LAYOUT, circularLayout);
  registers.registerLayout(forceLayout);
  registers.registerCoordinateSystem('graphView', {
    dimensions: View.dimensions,
    create: createView
  });
  // Register legacy focus actions
  registers.registerAction({
    type: 'focusNodeAdjacency',
    event: 'focusNodeAdjacency',
    update: 'series:focusNodeAdjacency'
  }, noop);
  registers.registerAction({
    type: 'unfocusNodeAdjacency',
    event: 'unfocusNodeAdjacency',
    update: 'series:unfocusNodeAdjacency'
  }, noop);
  // Register roam action.
  registers.registerAction(actionInfo, function (payload, ecModel, api) {
    ecModel.eachComponent({
      mainType: 'series',
      query: payload
    }, function (seriesModel) {
      var coordSys = seriesModel.coordinateSystem;
      var res = updateCenterAndZoom(coordSys, payload, undefined, api);
      seriesModel.setCenter && seriesModel.setCenter(res.center);
      seriesModel.setZoom && seriesModel.setZoom(res.zoom);
    });
  });
}