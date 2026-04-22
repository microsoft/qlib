
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
import BrushController from '../helper/BrushController.js';
import { layoutCovers } from './visualEncoding.js';
import ComponentView from '../../view/Component.js';
var BrushView = /** @class */function (_super) {
  __extends(BrushView, _super);
  function BrushView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = BrushView.type;
    return _this;
  }
  BrushView.prototype.init = function (ecModel, api) {
    this.ecModel = ecModel;
    this.api = api;
    this.model;
    (this._brushController = new BrushController(api.getZr())).on('brush', zrUtil.bind(this._onBrush, this)).mount();
  };
  BrushView.prototype.render = function (brushModel, ecModel, api, payload) {
    this.model = brushModel;
    this._updateController(brushModel, ecModel, api, payload);
  };
  BrushView.prototype.updateTransform = function (brushModel, ecModel, api, payload) {
    // PENDING: `updateTransform` is a little tricky, whose layout need
    // to be calculate mandatorily and other stages will not be performed.
    // Take care the correctness of the logic. See #11754 .
    layoutCovers(ecModel);
    this._updateController(brushModel, ecModel, api, payload);
  };
  BrushView.prototype.updateVisual = function (brushModel, ecModel, api, payload) {
    this.updateTransform(brushModel, ecModel, api, payload);
  };
  BrushView.prototype.updateView = function (brushModel, ecModel, api, payload) {
    this._updateController(brushModel, ecModel, api, payload);
  };
  BrushView.prototype._updateController = function (brushModel, ecModel, api, payload) {
    // Do not update controller when drawing.
    (!payload || payload.$from !== brushModel.id) && this._brushController.setPanels(brushModel.brushTargetManager.makePanelOpts(api)).enableBrush(brushModel.brushOption).updateCovers(brushModel.areas.slice());
  };
  // updateLayout: updateController,
  // updateVisual: updateController,
  BrushView.prototype.dispose = function () {
    this._brushController.dispose();
  };
  BrushView.prototype._onBrush = function (eventParam) {
    var modelId = this.model.id;
    var areas = this.model.brushTargetManager.setOutputRanges(eventParam.areas, this.ecModel);
    // Action is not dispatched on drag end, because the drag end
    // emits the same params with the last drag move event, and
    // may have some delay when using touch pad, which makes
    // animation not smooth (when using debounce).
    (!eventParam.isEnd || eventParam.removeOnClick) && this.api.dispatchAction({
      type: 'brush',
      brushId: modelId,
      areas: zrUtil.clone(areas),
      $from: modelId
    });
    eventParam.isEnd && this.api.dispatchAction({
      type: 'brushEnd',
      brushId: modelId,
      areas: zrUtil.clone(areas),
      $from: modelId
    });
  };
  BrushView.type = 'brush';
  return BrushView;
}(ComponentView);
export default BrushView;