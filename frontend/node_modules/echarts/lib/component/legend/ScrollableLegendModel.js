
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
import LegendModel from './LegendModel.js';
import { mergeLayoutParam, getLayoutParams } from '../../util/layout.js';
import { inheritDefaultOption } from '../../util/component.js';
var ScrollableLegendModel = /** @class */function (_super) {
  __extends(ScrollableLegendModel, _super);
  function ScrollableLegendModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ScrollableLegendModel.type;
    return _this;
  }
  /**
   * @param {number} scrollDataIndex
   */
  ScrollableLegendModel.prototype.setScrollDataIndex = function (scrollDataIndex) {
    this.option.scrollDataIndex = scrollDataIndex;
  };
  ScrollableLegendModel.prototype.init = function (option, parentModel, ecModel) {
    var inputPositionParams = getLayoutParams(option);
    _super.prototype.init.call(this, option, parentModel, ecModel);
    mergeAndNormalizeLayoutParams(this, option, inputPositionParams);
  };
  /**
   * @override
   */
  ScrollableLegendModel.prototype.mergeOption = function (option, ecModel) {
    _super.prototype.mergeOption.call(this, option, ecModel);
    mergeAndNormalizeLayoutParams(this, this.option, option);
  };
  ScrollableLegendModel.type = 'legend.scroll';
  ScrollableLegendModel.defaultOption = inheritDefaultOption(LegendModel.defaultOption, {
    scrollDataIndex: 0,
    pageButtonItemGap: 5,
    pageButtonGap: null,
    pageButtonPosition: 'end',
    pageFormatter: '{current}/{total}',
    pageIcons: {
      horizontal: ['M0,0L12,-10L12,10z', 'M0,0L-12,-10L-12,10z'],
      vertical: ['M0,0L20,0L10,-20z', 'M0,0L20,0L10,20z']
    },
    pageIconColor: '#2f4554',
    pageIconInactiveColor: '#aaa',
    pageIconSize: 15,
    pageTextStyle: {
      color: '#333'
    },
    animationDurationUpdate: 800
  });
  return ScrollableLegendModel;
}(LegendModel);
;
// Do not `ignoreSize` to enable setting {left: 10, right: 10}.
function mergeAndNormalizeLayoutParams(legendModel, target, raw) {
  var orient = legendModel.getOrient();
  var ignoreSize = [1, 1];
  ignoreSize[orient.index] = 0;
  mergeLayoutParam(target, raw, {
    type: 'box',
    ignoreSize: !!ignoreSize
  });
}
export default ScrollableLegendModel;