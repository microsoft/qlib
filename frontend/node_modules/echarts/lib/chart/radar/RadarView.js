
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
import * as graphic from '../../util/graphic.js';
import { setStatesStylesFromModel, toggleHoverEmphasis } from '../../util/states.js';
import * as zrUtil from 'zrender/lib/core/util.js';
import * as symbolUtil from '../../util/symbol.js';
import ChartView from '../../view/Chart.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import ZRImage from 'zrender/lib/graphic/Image.js';
import { saveOldStyle } from '../../animation/basicTransition.js';
var RadarView = /** @class */function (_super) {
  __extends(RadarView, _super);
  function RadarView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = RadarView.type;
    return _this;
  }
  RadarView.prototype.render = function (seriesModel, ecModel, api) {
    var polar = seriesModel.coordinateSystem;
    var group = this.group;
    var data = seriesModel.getData();
    var oldData = this._data;
    function createSymbol(data, idx) {
      var symbolType = data.getItemVisual(idx, 'symbol') || 'circle';
      if (symbolType === 'none') {
        return;
      }
      var symbolSize = symbolUtil.normalizeSymbolSize(data.getItemVisual(idx, 'symbolSize'));
      var symbolPath = symbolUtil.createSymbol(symbolType, -1, -1, 2, 2);
      var symbolRotate = data.getItemVisual(idx, 'symbolRotate') || 0;
      symbolPath.attr({
        style: {
          strokeNoScale: true
        },
        z2: 100,
        scaleX: symbolSize[0] / 2,
        scaleY: symbolSize[1] / 2,
        rotation: symbolRotate * Math.PI / 180 || 0
      });
      return symbolPath;
    }
    function updateSymbols(oldPoints, newPoints, symbolGroup, data, idx, isInit) {
      // Simply rerender all
      symbolGroup.removeAll();
      for (var i = 0; i < newPoints.length - 1; i++) {
        var symbolPath = createSymbol(data, idx);
        if (symbolPath) {
          symbolPath.__dimIdx = i;
          if (oldPoints[i]) {
            symbolPath.setPosition(oldPoints[i]);
            graphic[isInit ? 'initProps' : 'updateProps'](symbolPath, {
              x: newPoints[i][0],
              y: newPoints[i][1]
            }, seriesModel, idx);
          } else {
            symbolPath.setPosition(newPoints[i]);
          }
          symbolGroup.add(symbolPath);
        }
      }
    }
    function getInitialPoints(points) {
      return zrUtil.map(points, function (pt) {
        return [polar.cx, polar.cy];
      });
    }
    data.diff(oldData).add(function (idx) {
      var points = data.getItemLayout(idx);
      if (!points) {
        return;
      }
      var polygon = new graphic.Polygon();
      var polyline = new graphic.Polyline();
      var target = {
        shape: {
          points: points
        }
      };
      polygon.shape.points = getInitialPoints(points);
      polyline.shape.points = getInitialPoints(points);
      graphic.initProps(polygon, target, seriesModel, idx);
      graphic.initProps(polyline, target, seriesModel, idx);
      var itemGroup = new graphic.Group();
      var symbolGroup = new graphic.Group();
      itemGroup.add(polyline);
      itemGroup.add(polygon);
      itemGroup.add(symbolGroup);
      updateSymbols(polyline.shape.points, points, symbolGroup, data, idx, true);
      data.setItemGraphicEl(idx, itemGroup);
    }).update(function (newIdx, oldIdx) {
      var itemGroup = oldData.getItemGraphicEl(oldIdx);
      var polyline = itemGroup.childAt(0);
      var polygon = itemGroup.childAt(1);
      var symbolGroup = itemGroup.childAt(2);
      var target = {
        shape: {
          points: data.getItemLayout(newIdx)
        }
      };
      if (!target.shape.points) {
        return;
      }
      updateSymbols(polyline.shape.points, target.shape.points, symbolGroup, data, newIdx, false);
      saveOldStyle(polygon);
      saveOldStyle(polyline);
      graphic.updateProps(polyline, target, seriesModel);
      graphic.updateProps(polygon, target, seriesModel);
      data.setItemGraphicEl(newIdx, itemGroup);
    }).remove(function (idx) {
      group.remove(oldData.getItemGraphicEl(idx));
    }).execute();
    data.eachItemGraphicEl(function (itemGroup, idx) {
      var itemModel = data.getItemModel(idx);
      var polyline = itemGroup.childAt(0);
      var polygon = itemGroup.childAt(1);
      var symbolGroup = itemGroup.childAt(2);
      // Radar uses the visual encoded from itemStyle.
      var itemStyle = data.getItemVisual(idx, 'style');
      var color = itemStyle.fill;
      group.add(itemGroup);
      polyline.useStyle(zrUtil.defaults(itemModel.getModel('lineStyle').getLineStyle(), {
        fill: 'none',
        stroke: color
      }));
      setStatesStylesFromModel(polyline, itemModel, 'lineStyle');
      setStatesStylesFromModel(polygon, itemModel, 'areaStyle');
      var areaStyleModel = itemModel.getModel('areaStyle');
      var polygonIgnore = areaStyleModel.isEmpty() && areaStyleModel.parentModel.isEmpty();
      polygon.ignore = polygonIgnore;
      zrUtil.each(['emphasis', 'select', 'blur'], function (stateName) {
        var stateModel = itemModel.getModel([stateName, 'areaStyle']);
        var stateIgnore = stateModel.isEmpty() && stateModel.parentModel.isEmpty();
        // Won't be ignore if normal state is not ignore.
        polygon.ensureState(stateName).ignore = stateIgnore && polygonIgnore;
      });
      polygon.useStyle(zrUtil.defaults(areaStyleModel.getAreaStyle(), {
        fill: color,
        opacity: 0.7,
        decal: itemStyle.decal
      }));
      var emphasisModel = itemModel.getModel('emphasis');
      var itemHoverStyle = emphasisModel.getModel('itemStyle').getItemStyle();
      symbolGroup.eachChild(function (symbolPath) {
        if (symbolPath instanceof ZRImage) {
          var pathStyle = symbolPath.style;
          symbolPath.useStyle(zrUtil.extend({
            // TODO other properties like x, y ?
            image: pathStyle.image,
            x: pathStyle.x,
            y: pathStyle.y,
            width: pathStyle.width,
            height: pathStyle.height
          }, itemStyle));
        } else {
          symbolPath.useStyle(itemStyle);
          symbolPath.setColor(color);
          symbolPath.style.strokeNoScale = true;
        }
        var pathEmphasisState = symbolPath.ensureState('emphasis');
        pathEmphasisState.style = zrUtil.clone(itemHoverStyle);
        var defaultText = data.getStore().get(data.getDimensionIndex(symbolPath.__dimIdx), idx);
        (defaultText == null || isNaN(defaultText)) && (defaultText = '');
        setLabelStyle(symbolPath, getLabelStatesModels(itemModel), {
          labelFetcher: data.hostModel,
          labelDataIndex: idx,
          labelDimIndex: symbolPath.__dimIdx,
          defaultText: defaultText,
          inheritColor: color,
          defaultOpacity: itemStyle.opacity
        });
      });
      toggleHoverEmphasis(itemGroup, emphasisModel.get('focus'), emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
    });
    this._data = data;
  };
  RadarView.prototype.remove = function () {
    this.group.removeAll();
    this._data = null;
  };
  RadarView.type = 'radar';
  return RadarView;
}(ChartView);
export default RadarView;