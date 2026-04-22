
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
var opacityAccessPath = ['lineStyle', 'opacity'];
var parallelVisual = {
  seriesType: 'parallel',
  reset: function (seriesModel, ecModel) {
    var coordSys = seriesModel.coordinateSystem;
    var opacityMap = {
      normal: seriesModel.get(['lineStyle', 'opacity']),
      active: seriesModel.get('activeOpacity'),
      inactive: seriesModel.get('inactiveOpacity')
    };
    return {
      progress: function (params, data) {
        coordSys.eachActiveState(data, function (activeState, dataIndex) {
          var opacity = opacityMap[activeState];
          if (activeState === 'normal' && data.hasItemOption) {
            var itemOpacity = data.getItemModel(dataIndex).get(opacityAccessPath, true);
            itemOpacity != null && (opacity = itemOpacity);
          }
          var existsStyle = data.ensureUniqueItemVisual(dataIndex, 'style');
          existsStyle.opacity = opacity;
        }, params.start, params.end);
      }
    };
  }
};
export default parallelVisual;