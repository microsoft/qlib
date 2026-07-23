
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
import LargeSymbolDraw from '../helper/LargeSymbolDraw.js';
import pointsLayout from '../../layout/points.js';
import ChartView from '../../view/Chart.js';
var ScatterView = /** @class */function (_super) {
  __extends(ScatterView, _super);
  function ScatterView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = ScatterView.type;
    return _this;
  }
  ScatterView.prototype.render = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    var symbolDraw = this._updateSymbolDraw(data, seriesModel);
    symbolDraw.updateData(data, {
      // TODO
      // If this parameter should be a shape or a bounding volume
      // shape will be more general.
      // But bounding volume like bounding rect will be much faster in the contain calculation
      clipShape: this._getClipShape(seriesModel)
    });
    this._finished = true;
  };
  ScatterView.prototype.incrementalPrepareRender = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    var symbolDraw = this._updateSymbolDraw(data, seriesModel);
    symbolDraw.incrementalPrepareUpdate(data);
    this._finished = false;
  };
  ScatterView.prototype.incrementalRender = function (taskParams, seriesModel, ecModel) {
    this._symbolDraw.incrementalUpdate(taskParams, seriesModel.getData(), {
      clipShape: this._getClipShape(seriesModel)
    });
    this._finished = taskParams.end === seriesModel.getData().count();
  };
  ScatterView.prototype.updateTransform = function (seriesModel, ecModel, api) {
    var data = seriesModel.getData();
    // Must mark group dirty and make sure the incremental layer will be cleared
    // PENDING
    this.group.dirty();
    if (!this._finished || data.count() > 1e4) {
      return {
        update: true
      };
    } else {
      var res = pointsLayout('').reset(seriesModel, ecModel, api);
      if (res.progress) {
        res.progress({
          start: 0,
          end: data.count(),
          count: data.count()
        }, data);
      }
      this._symbolDraw.updateLayout(data);
    }
  };
  ScatterView.prototype.eachRendered = function (cb) {
    this._symbolDraw && this._symbolDraw.eachRendered(cb);
  };
  ScatterView.prototype._getClipShape = function (seriesModel) {
    if (!seriesModel.get('clip', true)) {
      return;
    }
    var coordSys = seriesModel.coordinateSystem;
    // PENDING make `0.1` configurable, for example, `clipTolerance`?
    return coordSys && coordSys.getArea && coordSys.getArea(.1);
  };
  ScatterView.prototype._updateSymbolDraw = function (data, seriesModel) {
    var symbolDraw = this._symbolDraw;
    var pipelineContext = seriesModel.pipelineContext;
    var isLargeDraw = pipelineContext.large;
    if (!symbolDraw || isLargeDraw !== this._isLargeDraw) {
      symbolDraw && symbolDraw.remove();
      symbolDraw = this._symbolDraw = isLargeDraw ? new LargeSymbolDraw() : new SymbolDraw();
      this._isLargeDraw = isLargeDraw;
      this.group.removeAll();
    }
    this.group.add(symbolDraw.group);
    return symbolDraw;
  };
  ScatterView.prototype.remove = function (ecModel, api) {
    this._symbolDraw && this._symbolDraw.remove(true);
    this._symbolDraw = null;
  };
  ScatterView.prototype.dispose = function () {};
  ScatterView.type = 'scatter';
  return ScatterView;
}(ChartView);
export default ScatterView;