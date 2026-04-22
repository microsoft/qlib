
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
import * as featureManager from './featureManager.js';
import ComponentModel from '../../model/Component.js';
var ToolboxModel = /** @class */function (_super) {
  __extends(ToolboxModel, _super);
  function ToolboxModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ToolboxModel.type;
    return _this;
  }
  ToolboxModel.prototype.optionUpdated = function () {
    _super.prototype.optionUpdated.apply(this, arguments);
    var ecModel = this.ecModel;
    zrUtil.each(this.option.feature, function (featureOpt, featureName) {
      var Feature = featureManager.getFeature(featureName);
      if (Feature) {
        if (Feature.getDefaultOption) {
          Feature.defaultOption = Feature.getDefaultOption(ecModel);
        }
        zrUtil.merge(featureOpt, Feature.defaultOption);
      }
    });
  };
  ToolboxModel.type = 'toolbox';
  ToolboxModel.layoutMode = {
    type: 'box',
    ignoreSize: true
  };
  ToolboxModel.defaultOption = {
    show: true,
    z: 6,
    // zlevel: 0,
    orient: 'horizontal',
    left: 'right',
    top: 'top',
    // right
    // bottom
    backgroundColor: 'transparent',
    borderColor: '#ccc',
    borderRadius: 0,
    borderWidth: 0,
    padding: 5,
    itemSize: 15,
    itemGap: 8,
    showTitle: true,
    iconStyle: {
      borderColor: '#666',
      color: 'none'
    },
    emphasis: {
      iconStyle: {
        borderColor: '#3E98C5'
      }
    },
    // textStyle: {},
    // feature
    tooltip: {
      show: false,
      position: 'bottom'
    }
  };
  return ToolboxModel;
}(ComponentModel);
export default ToolboxModel;