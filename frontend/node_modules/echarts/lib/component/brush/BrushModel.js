
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
import * as zrUtil from 'zrender/lib/core/util.js';
import * as visualSolution from '../../visual/visualSolution.js';
import Model from '../../model/Model.js';
import ComponentModel from '../../model/Component.js';
var DEFAULT_OUT_OF_BRUSH_COLOR = '#ddd';
var BrushModel = /** @class */function (_super) {
  __extends(BrushModel, _super);
  function BrushModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = BrushModel.type;
    /**
     * @readOnly
     */
    _this.areas = [];
    /**
     * Current brush painting area settings.
     * @readOnly
     */
    _this.brushOption = {};
    return _this;
  }
  BrushModel.prototype.optionUpdated = function (newOption, isInit) {
    var thisOption = this.option;
    !isInit && visualSolution.replaceVisualOption(thisOption, newOption, ['inBrush', 'outOfBrush']);
    var inBrush = thisOption.inBrush = thisOption.inBrush || {};
    // Always give default visual, consider setOption at the second time.
    thisOption.outOfBrush = thisOption.outOfBrush || {
      color: DEFAULT_OUT_OF_BRUSH_COLOR
    };
    if (!inBrush.hasOwnProperty('liftZ')) {
      // Bigger than the highlight z lift, otherwise it will
      // be effected by the highlight z when brush.
      inBrush.liftZ = 5;
    }
  };
  /**
   * If `areas` is null/undefined, range state remain.
   */
  BrushModel.prototype.setAreas = function (areas) {
    if (process.env.NODE_ENV !== 'production') {
      zrUtil.assert(zrUtil.isArray(areas));
      zrUtil.each(areas, function (area) {
        zrUtil.assert(area.brushType, 'Illegal areas');
      });
    }
    // If areas is null/undefined, range state remain.
    // This helps user to dispatchAction({type: 'brush'}) with no areas
    // set but just want to get the current brush select info from a `brush` event.
    if (!areas) {
      return;
    }
    this.areas = zrUtil.map(areas, function (area) {
      return generateBrushOption(this.option, area);
    }, this);
  };
  /**
   * Set the current painting brush option.
   */
  BrushModel.prototype.setBrushOption = function (brushOption) {
    this.brushOption = generateBrushOption(this.option, brushOption);
    this.brushType = this.brushOption.brushType;
  };
  BrushModel.type = 'brush';
  BrushModel.dependencies = ['geo', 'grid', 'xAxis', 'yAxis', 'parallel', 'series'];
  BrushModel.defaultOption = {
    seriesIndex: 'all',
    brushType: 'rect',
    brushMode: 'single',
    transformable: true,
    brushStyle: {
      borderWidth: 1,
      color: 'rgba(210,219,238,0.3)',
      borderColor: '#D2DBEE'
    },
    throttleType: 'fixRate',
    throttleDelay: 0,
    removeOnClick: true,
    z: 10000
  };
  return BrushModel;
}(ComponentModel);
function generateBrushOption(option, brushOption) {
  return zrUtil.merge({
    brushType: option.brushType,
    brushMode: option.brushMode,
    transformable: option.transformable,
    brushStyle: new Model(option.brushStyle).getItemStyle(),
    removeOnClick: option.removeOnClick,
    z: option.z
  }, brushOption, true);
}
export default BrushModel;