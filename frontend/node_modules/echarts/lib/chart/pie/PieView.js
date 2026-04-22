
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
import { extend, retrieve3 } from 'zrender/lib/core/util.js';
import * as graphic from '../../util/graphic.js';
import { setStatesStylesFromModel, toggleHoverEmphasis } from '../../util/states.js';
import ChartView from '../../view/Chart.js';
import labelLayout from './labelLayout.js';
import { setLabelLineStyle, getLabelLineStatesModels } from '../../label/labelGuideHelper.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import { getSectorCornerRadius } from '../helper/sectorHelper.js';
import { saveOldStyle } from '../../animation/basicTransition.js';
import { getBasicPieLayout, getSeriesLayoutData } from './pieLayout.js';
/**
 * Piece of pie including Sector, Label, LabelLine
 */
var PiePiece = /** @class */function (_super) {
  __extends(PiePiece, _super);
  function PiePiece(data, idx, startAngle) {
    var _this = _super.call(this) || this;
    _this.z2 = 2;
    var text = new graphic.Text();
    _this.setTextContent(text);
    _this.updateData(data, idx, startAngle, true);
    return _this;
  }
  PiePiece.prototype.updateData = function (data, idx, startAngle, firstCreate) {
    var sector = this;
    var seriesModel = data.hostModel;
    var itemModel = data.getItemModel(idx);
    var emphasisModel = itemModel.getModel('emphasis');
    var layout = data.getItemLayout(idx);
    // cornerRadius & innerCornerRadius doesn't exist in the item layout. Use `0` if null value is specified.
    // see `setItemLayout` in `pieLayout.ts`.
    var sectorShape = extend(getSectorCornerRadius(itemModel.getModel('itemStyle'), layout, true), layout);
    // Ignore NaN data.
    if (isNaN(sectorShape.startAngle)) {
      // Use NaN shape to avoid drawing shape.
      sector.setShape(sectorShape);
      return;
    }
    if (firstCreate) {
      sector.setShape(sectorShape);
      var animationType = seriesModel.getShallow('animationType');
      if (seriesModel.ecModel.ssr) {
        // Use scale animation in SSR mode(opacity?)
        // Because CSS SVG animation doesn't support very customized shape animation.
        graphic.initProps(sector, {
          scaleX: 0,
          scaleY: 0
        }, seriesModel, {
          dataIndex: idx,
          isFrom: true
        });
        sector.originX = sectorShape.cx;
        sector.originY = sectorShape.cy;
      } else if (animationType === 'scale') {
        sector.shape.r = layout.r0;
        graphic.initProps(sector, {
          shape: {
            r: layout.r
          }
        }, seriesModel, idx);
      }
      // Expansion
      else {
        if (startAngle != null) {
          sector.setShape({
            startAngle: startAngle,
            endAngle: startAngle
          });
          graphic.initProps(sector, {
            shape: {
              startAngle: layout.startAngle,
              endAngle: layout.endAngle
            }
          }, seriesModel, idx);
        } else {
          sector.shape.endAngle = layout.startAngle;
          graphic.updateProps(sector, {
            shape: {
              endAngle: layout.endAngle
            }
          }, seriesModel, idx);
        }
      }
    } else {
      saveOldStyle(sector);
      // Transition animation from the old shape
      graphic.updateProps(sector, {
        shape: sectorShape
      }, seriesModel, idx);
    }
    sector.useStyle(data.getItemVisual(idx, 'style'));
    setStatesStylesFromModel(sector, itemModel);
    var midAngle = (layout.startAngle + layout.endAngle) / 2;
    var offset = seriesModel.get('selectedOffset');
    var dx = Math.cos(midAngle) * offset;
    var dy = Math.sin(midAngle) * offset;
    var cursorStyle = itemModel.getShallow('cursor');
    cursorStyle && sector.attr('cursor', cursorStyle);
    this._updateLabel(seriesModel, data, idx);
    sector.ensureState('emphasis').shape = extend({
      r: layout.r + (emphasisModel.get('scale') ? emphasisModel.get('scaleSize') || 0 : 0)
    }, getSectorCornerRadius(emphasisModel.getModel('itemStyle'), layout));
    extend(sector.ensureState('select'), {
      x: dx,
      y: dy,
      shape: getSectorCornerRadius(itemModel.getModel(['select', 'itemStyle']), layout)
    });
    extend(sector.ensureState('blur'), {
      shape: getSectorCornerRadius(itemModel.getModel(['blur', 'itemStyle']), layout)
    });
    var labelLine = sector.getTextGuideLine();
    var labelText = sector.getTextContent();
    labelLine && extend(labelLine.ensureState('select'), {
      x: dx,
      y: dy
    });
    // TODO: needs dx, dy in zrender?
    extend(labelText.ensureState('select'), {
      x: dx,
      y: dy
    });
    toggleHoverEmphasis(this, emphasisModel.get('focus'), emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
  };
  PiePiece.prototype._updateLabel = function (seriesModel, data, idx) {
    var sector = this;
    var itemModel = data.getItemModel(idx);
    var labelLineModel = itemModel.getModel('labelLine');
    var style = data.getItemVisual(idx, 'style');
    var visualColor = style && style.fill;
    var visualOpacity = style && style.opacity;
    setLabelStyle(sector, getLabelStatesModels(itemModel), {
      labelFetcher: data.hostModel,
      labelDataIndex: idx,
      inheritColor: visualColor,
      defaultOpacity: visualOpacity,
      defaultText: seriesModel.getFormattedLabel(idx, 'normal') || data.getName(idx)
    });
    var labelText = sector.getTextContent();
    // Set textConfig on sector.
    sector.setTextConfig({
      // reset position, rotation
      position: null,
      rotation: null
    });
    // Make sure update style on labelText after setLabelStyle.
    // Because setLabelStyle will replace a new style on it.
    labelText.attr({
      z2: 10
    });
    var labelPosition = seriesModel.get(['label', 'position']);
    if (labelPosition !== 'outside' && labelPosition !== 'outer') {
      sector.removeTextGuideLine();
    } else {
      var polyline = this.getTextGuideLine();
      if (!polyline) {
        polyline = new graphic.Polyline();
        this.setTextGuideLine(polyline);
      }
      // Default use item visual color
      setLabelLineStyle(this, getLabelLineStatesModels(itemModel), {
        stroke: visualColor,
        opacity: retrieve3(labelLineModel.get(['lineStyle', 'opacity']), visualOpacity, 1)
      });
    }
  };
  return PiePiece;
}(graphic.Sector);
// Pie view
var PieView = /** @class */function (_super) {
  __extends(PieView, _super);
  function PieView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.ignoreLabelLineUpdate = true;
    return _this;
  }
  PieView.prototype.render = function (seriesModel, ecModel, api, payload) {
    var data = seriesModel.getData();
    var oldData = this._data;
    var group = this.group;
    var startAngle;
    // First render
    if (!oldData && data.count() > 0) {
      var shape = data.getItemLayout(0);
      for (var s = 1; isNaN(shape && shape.startAngle) && s < data.count(); ++s) {
        shape = data.getItemLayout(s);
      }
      if (shape) {
        startAngle = shape.startAngle;
      }
    }
    // remove empty-circle if it exists
    if (this._emptyCircleSector) {
      group.remove(this._emptyCircleSector);
    }
    // when all data are filtered, show lightgray empty circle
    if (data.count() === 0 && seriesModel.get('showEmptyCircle')) {
      var layoutData = getSeriesLayoutData(seriesModel);
      var sector = new graphic.Sector({
        shape: extend(getBasicPieLayout(seriesModel, api), layoutData)
      });
      sector.useStyle(seriesModel.getModel('emptyCircleStyle').getItemStyle());
      this._emptyCircleSector = sector;
      group.add(sector);
    }
    data.diff(oldData).add(function (idx) {
      var piePiece = new PiePiece(data, idx, startAngle);
      data.setItemGraphicEl(idx, piePiece);
      group.add(piePiece);
    }).update(function (newIdx, oldIdx) {
      var piePiece = oldData.getItemGraphicEl(oldIdx);
      piePiece.updateData(data, newIdx, startAngle);
      piePiece.off('click');
      group.add(piePiece);
      data.setItemGraphicEl(newIdx, piePiece);
    }).remove(function (idx) {
      var piePiece = oldData.getItemGraphicEl(idx);
      graphic.removeElementWithFadeOut(piePiece, seriesModel, idx);
    }).execute();
    labelLayout(seriesModel);
    // Always use initial animation.
    if (seriesModel.get('animationTypeUpdate') !== 'expansion') {
      this._data = data;
    }
  };
  PieView.prototype.dispose = function () {};
  PieView.prototype.containPoint = function (point, seriesModel) {
    var data = seriesModel.getData();
    var itemLayout = data.getItemLayout(0);
    if (itemLayout) {
      var dx = point[0] - itemLayout.cx;
      var dy = point[1] - itemLayout.cy;
      var radius = Math.sqrt(dx * dx + dy * dy);
      return radius <= itemLayout.r && radius >= itemLayout.r0;
    }
  };
  PieView.type = 'pie';
  return PieView;
}(ChartView);
export default PieView;