
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
/**
 * This module exposes helper functions for developing extensions.
 */
import * as zrUtil from 'zrender/lib/core/util.js';
import createSeriesData from '../../chart/helper/createSeriesData.js';
// import createGraphFromNodeEdge from './chart/helper/createGraphFromNodeEdge.js';
import * as axisHelper from '../../coord/axisHelper.js';
import { AxisModelCommonMixin } from '../../coord/axisModelCommonMixin.js';
import Model from '../../model/Model.js';
import { getLayoutRect } from '../../util/layout.js';
import { enableDataStack, isDimensionStacked, getStackedDimension } from '../../data/helper/dataStackHelper.js';
import { getECData } from '../../util/innerStore.js';
import { createTextStyle as innerCreateTextStyle } from '../../label/labelStyle.js';
/**
 * Create a multi dimension List structure from seriesModel.
 */
export function createList(seriesModel) {
  return createSeriesData(null, seriesModel);
}
// export function createGraph(seriesModel) {
//     let nodes = seriesModel.get('data');
//     let links = seriesModel.get('links');
//     return createGraphFromNodeEdge(nodes, links, seriesModel);
// }
export { getLayoutRect };
export { createDimensions } from '../../data/helper/createDimensions.js';
export var dataStack = {
  isDimensionStacked: isDimensionStacked,
  enableDataStack: enableDataStack,
  getStackedDimension: getStackedDimension
};
/**
 * Create a symbol element with given symbol configuration: shape, x, y, width, height, color
 * @param {string} symbolDesc
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {string} color
 */
export { createSymbol } from '../../util/symbol.js';
/**
 * Create scale
 * @param {Array.<number>} dataExtent
 * @param {Object|module:echarts/Model} option If `optoin.type`
 *        is secified, it can only be `'value'` currently.
 */
export function createScale(dataExtent, option) {
  var axisModel = option;
  if (!(option instanceof Model)) {
    axisModel = new Model(option);
    // FIXME
    // Currently AxisModelCommonMixin has nothing to do with the
    // the requirements of `axisHelper.createScaleByModel`. For
    // example the methods `getCategories` and `getOrdinalMeta`
    // are required for `'category'` axis, and ecModel is required
    // for `'time'` axis. But occasionally echarts-gl happened
    // to only use `'value'` axis.
    // zrUtil.mixin(axisModel, AxisModelCommonMixin);
  }
  var scale = axisHelper.createScaleByModel(axisModel);
  scale.setExtent(dataExtent[0], dataExtent[1]);
  axisHelper.niceScaleExtent(scale, axisModel);
  return scale;
}
/**
 * Mixin common methods to axis model,
 *
 * Include methods
 * `getFormattedLabels() => Array.<string>`
 * `getCategories() => Array.<string>`
 * `getMin(origin: boolean) => number`
 * `getMax(origin: boolean) => number`
 * `getNeedCrossZero() => boolean`
 */
export function mixinAxisModelCommonMethods(Model) {
  zrUtil.mixin(Model, AxisModelCommonMixin);
}
export { getECData };
export { enableHoverEmphasis } from '../../util/states.js';
export function createTextStyle(textStyleModel, opts) {
  opts = opts || {};
  return innerCreateTextStyle(textStyleModel, null, null, opts.state !== 'normal');
}