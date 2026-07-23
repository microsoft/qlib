
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
import ComponentModel from '../../model/Component.js';
var TooltipModel = /** @class */function (_super) {
  __extends(TooltipModel, _super);
  function TooltipModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = TooltipModel.type;
    return _this;
  }
  TooltipModel.type = 'tooltip';
  TooltipModel.dependencies = ['axisPointer'];
  TooltipModel.defaultOption = {
    // zlevel: 0,
    z: 60,
    show: true,
    // tooltip main content
    showContent: true,
    // 'trigger' only works on coordinate system.
    // 'item' | 'axis' | 'none'
    trigger: 'item',
    // 'click' | 'mousemove' | 'none'
    triggerOn: 'mousemove|click',
    alwaysShowContent: false,
    displayMode: 'single',
    renderMode: 'auto',
    // whether restraint content inside viewRect.
    // If renderMode: 'richText', default true.
    // If renderMode: 'html', defaut false (for backward compat).
    confine: null,
    showDelay: 0,
    hideDelay: 100,
    // Animation transition time, unit is second
    transitionDuration: 0.4,
    enterable: false,
    backgroundColor: '#fff',
    // box shadow
    shadowBlur: 10,
    shadowColor: 'rgba(0, 0, 0, .2)',
    shadowOffsetX: 1,
    shadowOffsetY: 2,
    // tooltip border radius, unit is px, default is 4
    borderRadius: 4,
    // tooltip border width, unit is px, default is 0 (no border)
    borderWidth: 1,
    // Tooltip inside padding, default is 5 for all direction
    // Array is allowed to set up, right, bottom, left, same with css
    // The default value: See `tooltip/tooltipMarkup.ts#getPaddingFromTooltipModel`.
    padding: null,
    // Extra css text
    extraCssText: '',
    // axis indicator, trigger by axis
    axisPointer: {
      // default is line
      // legal values: 'line' | 'shadow' | 'cross'
      type: 'line',
      // Valid when type is line, appoint tooltip line locate on which line. Optional
      // legal values: 'x' | 'y' | 'angle' | 'radius' | 'auto'
      // default is 'auto', chose the axis which type is category.
      // for multiply y axis, cartesian coord chose x axis, polar chose angle axis
      axis: 'auto',
      animation: 'auto',
      animationDurationUpdate: 200,
      animationEasingUpdate: 'exponentialOut',
      crossStyle: {
        color: '#999',
        width: 1,
        type: 'dashed',
        // TODO formatter
        textStyle: {}
      }
      // lineStyle and shadowStyle should not be specified here,
      // otherwise it will always override those styles on option.axisPointer.
    },
    textStyle: {
      color: '#666',
      fontSize: 14
    }
  };
  return TooltipModel;
}(ComponentModel);
export default TooltipModel;