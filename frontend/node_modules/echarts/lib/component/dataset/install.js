
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
/**
 * This module is imported by echarts directly.
 *
 * Notice:
 * Always keep this file exists for backward compatibility.
 * Because before 4.1.0, dataset is an optional component,
 * some users may import this module manually.
 */
import ComponentModel from '../../model/Component.js';
import ComponentView from '../../view/Component.js';
import { SERIES_LAYOUT_BY_COLUMN } from '../../util/types.js';
import { disableTransformOptionMerge, SourceManager } from '../../data/helper/sourceManager.js';
var DatasetModel = /** @class */function (_super) {
  __extends(DatasetModel, _super);
  function DatasetModel() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = 'dataset';
    return _this;
  }
  DatasetModel.prototype.init = function (option, parentModel, ecModel) {
    _super.prototype.init.call(this, option, parentModel, ecModel);
    this._sourceManager = new SourceManager(this);
    disableTransformOptionMerge(this);
  };
  DatasetModel.prototype.mergeOption = function (newOption, ecModel) {
    _super.prototype.mergeOption.call(this, newOption, ecModel);
    disableTransformOptionMerge(this);
  };
  DatasetModel.prototype.optionUpdated = function () {
    this._sourceManager.dirty();
  };
  DatasetModel.prototype.getSourceManager = function () {
    return this._sourceManager;
  };
  DatasetModel.type = 'dataset';
  DatasetModel.defaultOption = {
    seriesLayoutBy: SERIES_LAYOUT_BY_COLUMN
  };
  return DatasetModel;
}(ComponentModel);
export { DatasetModel };
var DatasetView = /** @class */function (_super) {
  __extends(DatasetView, _super);
  function DatasetView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = 'dataset';
    return _this;
  }
  DatasetView.type = 'dataset';
  return DatasetView;
}(ComponentView);
export function install(registers) {
  registers.registerComponentModel(DatasetModel);
  registers.registerComponentView(DatasetView);
}