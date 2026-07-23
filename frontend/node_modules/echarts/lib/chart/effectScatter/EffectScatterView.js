
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
import SymbolDraw from '../helper/SymbolDraw.js';
import EffectSymbol from '../helper/EffectSymbol.js';
import * as matrix from 'zrender/lib/core/matrix.js';
import pointsLayout from '../../layout/points.js';
import ChartView from '../../view/Chart.js';
var EffectScatterView = /** @class */function (_super) {
  __extends(EffectScatterView, _super);
  function EffectScatterView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = EffectScatterView.type;
    return _this;
  }
  EffectScatterView.prototype.init = function () {
    this._symbolDraw = new SymbolDraw(EffectSymbol);
  };
  EffectScatterView.prototype.render = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    var effectSymbolDraw = this._symbolDraw;
    effectSymbolDraw.updateData(data, {
      clipShape: this._getClipShape(seriesModel)
    });
    this.group.add(effectSymbolDraw.group);
  };
  EffectScatterView.prototype._getClipShape = function (seriesModel) {
    var coordSys = seriesModel.coordinateSystem;
    var clipArea = coordSys && coordSys.getArea && coordSys.getArea();
    return seriesModel.get('clip', true) ? clipArea : null;
  };
  EffectScatterView.prototype.updateTransform = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    this.group.dirty();
    var res = pointsLayout('').reset(seriesModel, ecModel, api);
    if (res.progress) {
      res.progress({
        start: 0,
        end: data.count(),
        count: data.count()
      }, data);
    }
    this._symbolDraw.updateLayout();
  };
  EffectScatterView.prototype._updateGroupTransform = function (seriesModel) {
    var coordSys = seriesModel.coordinateSystem;
    if (coordSys && coordSys.getRoamTransform) {
      this.group.transform = matrix.clone(coordSys.getRoamTransform());
      this.group.decomposeTransform();
    }
  };
  EffectScatterView.prototype.remove = function (ecModel, api) {
    this._symbolDraw && this._symbolDraw.remove(true);
  };
  EffectScatterView.type = 'effectScatter';
  return EffectScatterView;
}(ChartView);
export default EffectScatterView;