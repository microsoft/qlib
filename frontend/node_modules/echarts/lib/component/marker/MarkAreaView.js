
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
// TODO Optimize on polar
import * as colorUtil from 'zrender/lib/tool/color.js';
import SeriesData from '../../data/SeriesData.js';
import * as numberUtil from '../../util/number.js';
import * as graphic from '../../util/graphic.js';
import { toggleHoverEmphasis, setStatesStylesFromModel } from '../../util/states.js';
import * as markerHelper from './markerHelper.js';
import MarkerView from './MarkerView.js';
import { retrieve, mergeAll, map, curry, filter, extend, isString } from 'zrender/lib/core/util.js';
import { isCoordinateSystemType } from '../../coord/CoordinateSystem.js';
import MarkerModel from './MarkerModel.js';
import { makeInner } from '../../util/model.js';
import { getVisualFromData } from '../../visual/helper.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import { getECData } from '../../util/innerStore.js';
import { parseDataValue } from '../../data/helper/dataValueHelper.js';
var inner = makeInner();
var markAreaTransform = function (seriesModel, coordSys, maModel, item) {
  // item may be null
  var item0 = item[0];
  var item1 = item[1];
  if (!item0 || !item1) {
    return;
  }
  var lt = markerHelper.dataTransform(seriesModel, item0);
  var rb = markerHelper.dataTransform(seriesModel, item1);
  // FIXME make sure lt is less than rb
  var ltCoord = lt.coord;
  var rbCoord = rb.coord;
  ltCoord[0] = retrieve(ltCoord[0], -Infinity);
  ltCoord[1] = retrieve(ltCoord[1], -Infinity);
  rbCoord[0] = retrieve(rbCoord[0], Infinity);
  rbCoord[1] = retrieve(rbCoord[1], Infinity);
  // Merge option into one
  var result = mergeAll([{}, lt, rb]);
  result.coord = [lt.coord, rb.coord];
  result.x0 = lt.x;
  result.y0 = lt.y;
  result.x1 = rb.x;
  result.y1 = rb.y;
  return result;
};
function isInfinity(val) {
  return !isNaN(val) && !isFinite(val);
}
// If a markArea has one dim
function ifMarkAreaHasOnlyDim(dimIndex, fromCoord, toCoord, coordSys) {
  var otherDimIndex = 1 - dimIndex;
  return isInfinity(fromCoord[otherDimIndex]) && isInfinity(toCoord[otherDimIndex]);
}
function markAreaFilter(coordSys, item) {
  var fromCoord = item.coord[0];
  var toCoord = item.coord[1];
  var item0 = {
    coord: fromCoord,
    x: item.x0,
    y: item.y0
  };
  var item1 = {
    coord: toCoord,
    x: item.x1,
    y: item.y1
  };
  if (isCoordinateSystemType(coordSys, 'cartesian2d')) {
    // In case
    // {
    //  markArea: {
    //    data: [{ yAxis: 2 }]
    //  }
    // }
    if (fromCoord && toCoord && (ifMarkAreaHasOnlyDim(1, fromCoord, toCoord, coordSys) || ifMarkAreaHasOnlyDim(0, fromCoord, toCoord, coordSys))) {
      return true;
    }
    // Directly returning true may also do the work,
    // because markArea will not be shown automatically
    // when it's not included in coordinate system.
    // But filtering ahead can avoid keeping rendering markArea
    // when there are too many of them.
    return markerHelper.zoneFilter(coordSys, item0, item1);
  }
  return markerHelper.dataFilter(coordSys, item0) || markerHelper.dataFilter(coordSys, item1);
}
// dims can be ['x0', 'y0'], ['x1', 'y1'], ['x0', 'y1'], ['x1', 'y0']
function getSingleMarkerEndPoint(data, idx, dims, seriesModel, api) {
  var coordSys = seriesModel.coordinateSystem;
  var itemModel = data.getItemModel(idx);
  var point;
  var xPx = numberUtil.parsePercent(itemModel.get(dims[0]), api.getWidth());
  var yPx = numberUtil.parsePercent(itemModel.get(dims[1]), api.getHeight());
  if (!isNaN(xPx) && !isNaN(yPx)) {
    point = [xPx, yPx];
  } else {
    // Chart like bar may have there own marker positioning logic
    if (seriesModel.getMarkerPosition) {
      // Consider the case that user input the right-bottom point first
      // Pick the larger x and y as 'x1' and 'y1'
      var pointValue0 = data.getValues(['x0', 'y0'], idx);
      var pointValue1 = data.getValues(['x1', 'y1'], idx);
      var clampPointValue0 = coordSys.clampData(pointValue0);
      var clampPointValue1 = coordSys.clampData(pointValue1);
      var pointValue = [];
      if (dims[0] === 'x0') {
        pointValue[0] = clampPointValue0[0] > clampPointValue1[0] ? pointValue1[0] : pointValue0[0];
      } else {
        pointValue[0] = clampPointValue0[0] > clampPointValue1[0] ? pointValue0[0] : pointValue1[0];
      }
      if (dims[1] === 'y0') {
        pointValue[1] = clampPointValue0[1] > clampPointValue1[1] ? pointValue1[1] : pointValue0[1];
      } else {
        pointValue[1] = clampPointValue0[1] > clampPointValue1[1] ? pointValue0[1] : pointValue1[1];
      }
      // Use the getMarkerPosition
      point = seriesModel.getMarkerPosition(pointValue, dims, true);
    } else {
      var x = data.get(dims[0], idx);
      var y = data.get(dims[1], idx);
      var pt = [x, y];
      coordSys.clampData && coordSys.clampData(pt, pt);
      point = coordSys.dataToPoint(pt, true);
    }
    if (isCoordinateSystemType(coordSys, 'cartesian2d')) {
      // TODO: TYPE ts@4.1 may still infer it as Axis instead of Axis2D. Not sure if it's a bug
      var xAxis = coordSys.getAxis('x');
      var yAxis = coordSys.getAxis('y');
      var x = data.get(dims[0], idx);
      var y = data.get(dims[1], idx);
      if (isInfinity(x)) {
        point[0] = xAxis.toGlobalCoord(xAxis.getExtent()[dims[0] === 'x0' ? 0 : 1]);
      } else if (isInfinity(y)) {
        point[1] = yAxis.toGlobalCoord(yAxis.getExtent()[dims[1] === 'y0' ? 0 : 1]);
      }
    }
    // Use x, y if has any
    if (!isNaN(xPx)) {
      point[0] = xPx;
    }
    if (!isNaN(yPx)) {
      point[1] = yPx;
    }
  }
  return point;
}
export var dimPermutations = [['x0', 'y0'], ['x1', 'y0'], ['x1', 'y1'], ['x0', 'y1']];
var MarkAreaView = /** @class */function (_super) {
  __extends(MarkAreaView, _super);
  function MarkAreaView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = MarkAreaView.type;
    return _this;
  }
  MarkAreaView.prototype.updateTransform = function (markAreaModel, ecModel, api) {
    ecModel.eachSeries(function (seriesModel) {
      var maModel = MarkerModel.getMarkerModelFromSeries(seriesModel, 'markArea');
      if (maModel) {
        var areaData_1 = maModel.getData();
        areaData_1.each(function (idx) {
          var points = map(dimPermutations, function (dim) {
            return getSingleMarkerEndPoint(areaData_1, idx, dim, seriesModel, api);
          });
          // Layout
          areaData_1.setItemLayout(idx, points);
          var el = areaData_1.getItemGraphicEl(idx);
          el.setShape('points', points);
        });
      }
    }, this);
  };
  MarkAreaView.prototype.renderSeries = function (seriesModel, maModel, ecModel, api) {
    var coordSys = seriesModel.coordinateSystem;
    var seriesId = seriesModel.id;
    var seriesData = seriesModel.getData();
    var areaGroupMap = this.markerGroupMap;
    var polygonGroup = areaGroupMap.get(seriesId) || areaGroupMap.set(seriesId, {
      group: new graphic.Group()
    });
    this.group.add(polygonGroup.group);
    this.markKeep(polygonGroup);
    var areaData = createList(coordSys, seriesModel, maModel);
    // Line data for tooltip and formatter
    maModel.setData(areaData);
    // Update visual and layout of line
    areaData.each(function (idx) {
      // Layout
      var points = map(dimPermutations, function (dim) {
        return getSingleMarkerEndPoint(areaData, idx, dim, seriesModel, api);
      });
      var xAxisScale = coordSys.getAxis('x').scale;
      var yAxisScale = coordSys.getAxis('y').scale;
      var xAxisExtent = xAxisScale.getExtent();
      var yAxisExtent = yAxisScale.getExtent();
      var xPointExtent = [xAxisScale.parse(areaData.get('x0', idx)), xAxisScale.parse(areaData.get('x1', idx))];
      var yPointExtent = [yAxisScale.parse(areaData.get('y0', idx)), yAxisScale.parse(areaData.get('y1', idx))];
      numberUtil.asc(xPointExtent);
      numberUtil.asc(yPointExtent);
      var overlapped = !(xAxisExtent[0] > xPointExtent[1] || xAxisExtent[1] < xPointExtent[0] || yAxisExtent[0] > yPointExtent[1] || yAxisExtent[1] < yPointExtent[0]);
      // If none of the area is inside coordSys, allClipped is set to be true
      // in layout so that label will not be displayed. See #12591
      var allClipped = !overlapped;
      areaData.setItemLayout(idx, {
        points: points,
        allClipped: allClipped
      });
      var style = areaData.getItemModel(idx).getModel('itemStyle').getItemStyle();
      var color = getVisualFromData(seriesData, 'color');
      if (!style.fill) {
        style.fill = color;
        if (isString(style.fill)) {
          style.fill = colorUtil.modifyAlpha(style.fill, 0.4);
        }
      }
      if (!style.stroke) {
        style.stroke = color;
      }
      // Visual
      areaData.setItemVisual(idx, 'style', style);
    });
    areaData.diff(inner(polygonGroup).data).add(function (idx) {
      var layout = areaData.getItemLayout(idx);
      if (!layout.allClipped) {
        var polygon = new graphic.Polygon({
          shape: {
            points: layout.points
          }
        });
        areaData.setItemGraphicEl(idx, polygon);
        polygonGroup.group.add(polygon);
      }
    }).update(function (newIdx, oldIdx) {
      var polygon = inner(polygonGroup).data.getItemGraphicEl(oldIdx);
      var layout = areaData.getItemLayout(newIdx);
      if (!layout.allClipped) {
        if (polygon) {
          graphic.updateProps(polygon, {
            shape: {
              points: layout.points
            }
          }, maModel, newIdx);
        } else {
          polygon = new graphic.Polygon({
            shape: {
              points: layout.points
            }
          });
        }
        areaData.setItemGraphicEl(newIdx, polygon);
        polygonGroup.group.add(polygon);
      } else if (polygon) {
        polygonGroup.group.remove(polygon);
      }
    }).remove(function (idx) {
      var polygon = inner(polygonGroup).data.getItemGraphicEl(idx);
      polygonGroup.group.remove(polygon);
    }).execute();
    areaData.eachItemGraphicEl(function (polygon, idx) {
      var itemModel = areaData.getItemModel(idx);
      var style = areaData.getItemVisual(idx, 'style');
      polygon.useStyle(areaData.getItemVisual(idx, 'style'));
      setLabelStyle(polygon, getLabelStatesModels(itemModel), {
        labelFetcher: maModel,
        labelDataIndex: idx,
        defaultText: areaData.getName(idx) || '',
        inheritColor: isString(style.fill) ? colorUtil.modifyAlpha(style.fill, 1) : '#000'
      });
      setStatesStylesFromModel(polygon, itemModel);
      toggleHoverEmphasis(polygon, null, null, itemModel.get(['emphasis', 'disabled']));
      getECData(polygon).dataModel = maModel;
    });
    inner(polygonGroup).data = areaData;
    polygonGroup.group.silent = maModel.get('silent') || seriesModel.get('silent');
  };
  MarkAreaView.type = 'markArea';
  return MarkAreaView;
}(MarkerView);
function createList(coordSys, seriesModel, maModel) {
  var areaData;
  var dataDims;
  var dims = ['x0', 'y0', 'x1', 'y1'];
  if (coordSys) {
    var coordDimsInfos_1 = map(coordSys && coordSys.dimensions, function (coordDim) {
      var data = seriesModel.getData();
      var info = data.getDimensionInfo(data.mapDimension(coordDim)) || {};
      // In map series data don't have lng and lat dimension. Fallback to same with coordSys
      return extend(extend({}, info), {
        name: coordDim,
        // DON'T use ordinalMeta to parse and collect ordinal.
        ordinalMeta: null
      });
    });
    dataDims = map(dims, function (dim, idx) {
      return {
        name: dim,
        type: coordDimsInfos_1[idx % 2].type
      };
    });
    areaData = new SeriesData(dataDims, maModel);
  } else {
    dataDims = [{
      name: 'value',
      type: 'float'
    }];
    areaData = new SeriesData(dataDims, maModel);
  }
  var optData = map(maModel.get('data'), curry(markAreaTransform, seriesModel, coordSys, maModel));
  if (coordSys) {
    optData = filter(optData, curry(markAreaFilter, coordSys));
  }
  var dimValueGetter = coordSys ? function (item, dimName, dataIndex, dimIndex) {
    // TODO should convert to ParsedValue?
    var rawVal = item.coord[Math.floor(dimIndex / 2)][dimIndex % 2];
    return parseDataValue(rawVal, dataDims[dimIndex]);
  } : function (item, dimName, dataIndex, dimIndex) {
    return parseDataValue(item.value, dataDims[dimIndex]);
  };
  areaData.initData(optData, null, dimValueGetter);
  areaData.hasItemOption = true;
  return areaData;
}
export default MarkAreaView;