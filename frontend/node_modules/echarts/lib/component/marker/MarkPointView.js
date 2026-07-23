
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
import SymbolDraw from '../../chart/helper/SymbolDraw.js';
import * as numberUtil from '../../util/number.js';
import SeriesData from '../../data/SeriesData.js';
import * as markerHelper from './markerHelper.js';
import MarkerView from './MarkerView.js';
import MarkerModel from './MarkerModel.js';
import { isFunction, map, filter, curry, extend } from 'zrender/lib/core/util.js';
import { getECData } from '../../util/innerStore.js';
import { getVisualFromData } from '../../visual/helper.js';
function updateMarkerLayout(mpData, seriesModel, api) {
  var coordSys = seriesModel.coordinateSystem;
  mpData.each(function (idx) {
    var itemModel = mpData.getItemModel(idx);
    var point;
    var xPx = numberUtil.parsePercent(itemModel.get('x'), api.getWidth());
    var yPx = numberUtil.parsePercent(itemModel.get('y'), api.getHeight());
    if (!isNaN(xPx) && !isNaN(yPx)) {
      point = [xPx, yPx];
    }
    // Chart like bar may have there own marker positioning logic
    else if (seriesModel.getMarkerPosition) {
      // Use the getMarkerPosition
      point = seriesModel.getMarkerPosition(mpData.getValues(mpData.dimensions, idx));
    } else if (coordSys) {
      var x = mpData.get(coordSys.dimensions[0], idx);
      var y = mpData.get(coordSys.dimensions[1], idx);
      point = coordSys.dataToPoint([x, y]);
    }
    // Use x, y if has any
    if (!isNaN(xPx)) {
      point[0] = xPx;
    }
    if (!isNaN(yPx)) {
      point[1] = yPx;
    }
    mpData.setItemLayout(idx, point);
  });
}
var MarkPointView = /** @class */function (_super) {
  __extends(MarkPointView, _super);
  function MarkPointView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = MarkPointView.type;
    return _this;
  }
  MarkPointView.prototype.updateTransform = function (markPointModel, ecModel, api) {
    ecModel.eachSeries(function (seriesModel) {
      var mpModel = MarkerModel.getMarkerModelFromSeries(seriesModel, 'markPoint');
      if (mpModel) {
        updateMarkerLayout(mpModel.getData(), seriesModel, api);
        this.markerGroupMap.get(seriesModel.id).updateLayout();
      }
    }, this);
  };
  MarkPointView.prototype.renderSeries = function (seriesModel, mpModel, ecModel, api) {
    var coordSys = seriesModel.coordinateSystem;
    var seriesId = seriesModel.id;
    var seriesData = seriesModel.getData();
    var symbolDrawMap = this.markerGroupMap;
    var symbolDraw = symbolDrawMap.get(seriesId) || symbolDrawMap.set(seriesId, new SymbolDraw());
    var mpData = createData(coordSys, seriesModel, mpModel);
    // FIXME
    mpModel.setData(mpData);
    updateMarkerLayout(mpModel.getData(), seriesModel, api);
    mpData.each(function (idx) {
      var itemModel = mpData.getItemModel(idx);
      var symbol = itemModel.getShallow('symbol');
      var symbolSize = itemModel.getShallow('symbolSize');
      var symbolRotate = itemModel.getShallow('symbolRotate');
      var symbolOffset = itemModel.getShallow('symbolOffset');
      var symbolKeepAspect = itemModel.getShallow('symbolKeepAspect');
      // TODO: refactor needed: single data item should not support callback function
      if (isFunction(symbol) || isFunction(symbolSize) || isFunction(symbolRotate) || isFunction(symbolOffset)) {
        var rawIdx = mpModel.getRawValue(idx);
        var dataParams = mpModel.getDataParams(idx);
        if (isFunction(symbol)) {
          symbol = symbol(rawIdx, dataParams);
        }
        if (isFunction(symbolSize)) {
          // FIXME 这里不兼容 ECharts 2.x，2.x 貌似参数是整个数据？
          symbolSize = symbolSize(rawIdx, dataParams);
        }
        if (isFunction(symbolRotate)) {
          symbolRotate = symbolRotate(rawIdx, dataParams);
        }
        if (isFunction(symbolOffset)) {
          symbolOffset = symbolOffset(rawIdx, dataParams);
        }
      }
      var style = itemModel.getModel('itemStyle').getItemStyle();
      var color = getVisualFromData(seriesData, 'color');
      if (!style.fill) {
        style.fill = color;
      }
      mpData.setItemVisual(idx, {
        symbol: symbol,
        symbolSize: symbolSize,
        symbolRotate: symbolRotate,
        symbolOffset: symbolOffset,
        symbolKeepAspect: symbolKeepAspect,
        style: style
      });
    });
    // TODO Text are wrong
    symbolDraw.updateData(mpData);
    this.group.add(symbolDraw.group);
    // Set host model for tooltip
    // FIXME
    mpData.eachItemGraphicEl(function (el) {
      el.traverse(function (child) {
        getECData(child).dataModel = mpModel;
      });
    });
    this.markKeep(symbolDraw);
    symbolDraw.group.silent = mpModel.get('silent') || seriesModel.get('silent');
  };
  MarkPointView.type = 'markPoint';
  return MarkPointView;
}(MarkerView);
function createData(coordSys, seriesModel, mpModel) {
  var coordDimsInfos;
  if (coordSys) {
    coordDimsInfos = map(coordSys && coordSys.dimensions, function (coordDim) {
      var info = seriesModel.getData().getDimensionInfo(seriesModel.getData().mapDimension(coordDim)) || {};
      // In map series data don't have lng and lat dimension. Fallback to same with coordSys
      return extend(extend({}, info), {
        name: coordDim,
        // DON'T use ordinalMeta to parse and collect ordinal.
        ordinalMeta: null
      });
    });
  } else {
    coordDimsInfos = [{
      name: 'value',
      type: 'float'
    }];
  }
  var mpData = new SeriesData(coordDimsInfos, mpModel);
  var dataOpt = map(mpModel.get('data'), curry(markerHelper.dataTransform, seriesModel));
  if (coordSys) {
    dataOpt = filter(dataOpt, curry(markerHelper.dataFilter, coordSys));
  }
  var dimValueGetter = markerHelper.createMarkerDimValueGetter(!!coordSys, coordDimsInfos);
  mpData.initData(dataOpt, null, dimValueGetter);
  return mpData;
}
export default MarkPointView;