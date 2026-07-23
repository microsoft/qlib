
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
import { each } from 'zrender/lib/core/util.js';
import { simpleLayout, simpleLayoutEdge } from './simpleLayoutHelper.js';
export default function graphSimpleLayout(ecModel, api) {
  ecModel.eachSeriesByType('graph', function (seriesModel) {
    var layout = seriesModel.get('layout');
    var coordSys = seriesModel.coordinateSystem;
    if (coordSys && coordSys.type !== 'view') {
      var data_1 = seriesModel.getData();
      var dimensions_1 = [];
      each(coordSys.dimensions, function (coordDim) {
        dimensions_1 = dimensions_1.concat(data_1.mapDimensionsAll(coordDim));
      });
      for (var dataIndex = 0; dataIndex < data_1.count(); dataIndex++) {
        var value = [];
        var hasValue = false;
        for (var i = 0; i < dimensions_1.length; i++) {
          var val = data_1.get(dimensions_1[i], dataIndex);
          if (!isNaN(val)) {
            hasValue = true;
          }
          value.push(val);
        }
        if (hasValue) {
          data_1.setItemLayout(dataIndex, coordSys.dataToPoint(value));
        } else {
          // Also {Array.<number>}, not undefined to avoid if...else... statement
          data_1.setItemLayout(dataIndex, [NaN, NaN]);
        }
      }
      simpleLayoutEdge(data_1.graph, seriesModel);
    } else if (!layout || layout === 'none') {
      simpleLayout(seriesModel);
    }
  });
}