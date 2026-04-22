
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
import VisualMapView from './VisualMapView.js';
import * as graphic from '../../util/graphic.js';
import { createSymbol } from '../../util/symbol.js';
import * as layout from '../../util/layout.js';
import * as helper from './helper.js';
import { createTextStyle } from '../../label/labelStyle.js';
var PiecewiseVisualMapView = /** @class */function (_super) {
  __extends(PiecewiseVisualMapView, _super);
  function PiecewiseVisualMapView() {
    var _this = _super !== null && _super.apply(this, arguments) || this;
    _this.type = PiecewiseVisualMapView.type;
    return _this;
  }
  PiecewiseVisualMapView.prototype.doRender = function () {
    var thisGroup = this.group;
    thisGroup.removeAll();
    var visualMapModel = this.visualMapModel;
    var textGap = visualMapModel.get('textGap');
    var textStyleModel = visualMapModel.textStyleModel;
    var textFont = textStyleModel.getFont();
    var textFill = textStyleModel.getTextColor();
    var itemAlign = this._getItemAlign();
    var itemSize = visualMapModel.itemSize;
    var viewData = this._getViewData();
    var endsText = viewData.endsText;
    var showLabel = zrUtil.retrieve(visualMapModel.get('showLabel', true), !endsText);
    var silent = !visualMapModel.get('selectedMode');
    endsText && this._renderEndsText(thisGroup, endsText[0], itemSize, showLabel, itemAlign);
    zrUtil.each(viewData.viewPieceList, function (item) {
      var piece = item.piece;
      var itemGroup = new graphic.Group();
      itemGroup.onclick = zrUtil.bind(this._onItemClick, this, piece);
      this._enableHoverLink(itemGroup, item.indexInModelPieceList);
      // TODO Category
      var representValue = visualMapModel.getRepresentValue(piece);
      this._createItemSymbol(itemGroup, representValue, [0, 0, itemSize[0], itemSize[1]], silent);
      if (showLabel) {
        var visualState = this.visualMapModel.getValueState(representValue);
        itemGroup.add(new graphic.Text({
          style: {
            x: itemAlign === 'right' ? -textGap : itemSize[0] + textGap,
            y: itemSize[1] / 2,
            text: piece.text,
            verticalAlign: 'middle',
            align: itemAlign,
            font: textFont,
            fill: textFill,
            opacity: visualState === 'outOfRange' ? 0.5 : 1
          },
          silent: silent
        }));
      }
      thisGroup.add(itemGroup);
    }, this);
    endsText && this._renderEndsText(thisGroup, endsText[1], itemSize, showLabel, itemAlign);
    layout.box(visualMapModel.get('orient'), thisGroup, visualMapModel.get('itemGap'));
    this.renderBackground(thisGroup);
    this.positionGroup(thisGroup);
  };
  PiecewiseVisualMapView.prototype._enableHoverLink = function (itemGroup, pieceIndex) {
    var _this = this;
    itemGroup.on('mouseover', function () {
      return onHoverLink('highlight');
    }).on('mouseout', function () {
      return onHoverLink('downplay');
    });
    var onHoverLink = function (method) {
      var visualMapModel = _this.visualMapModel;
      // TODO: TYPE More detailed action types
      visualMapModel.option.hoverLink && _this.api.dispatchAction({
        type: method,
        batch: helper.makeHighDownBatch(visualMapModel.findTargetDataIndices(pieceIndex), visualMapModel)
      });
    };
  };
  PiecewiseVisualMapView.prototype._getItemAlign = function () {
    var visualMapModel = this.visualMapModel;
    var modelOption = visualMapModel.option;
    if (modelOption.orient === 'vertical') {
      return helper.getItemAlign(visualMapModel, this.api, visualMapModel.itemSize);
    } else {
      // horizontal, most case left unless specifying right.
      var align = modelOption.align;
      if (!align || align === 'auto') {
        align = 'left';
      }
      return align;
    }
  };
  PiecewiseVisualMapView.prototype._renderEndsText = function (group, text, itemSize, showLabel, itemAlign) {
    if (!text) {
      return;
    }
    var itemGroup = new graphic.Group();
    var textStyleModel = this.visualMapModel.textStyleModel;
    itemGroup.add(new graphic.Text({
      style: createTextStyle(textStyleModel, {
        x: showLabel ? itemAlign === 'right' ? itemSize[0] : 0 : itemSize[0] / 2,
        y: itemSize[1] / 2,
        verticalAlign: 'middle',
        align: showLabel ? itemAlign : 'center',
        text: text
      })
    }));
    group.add(itemGroup);
  };
  /**
   * @private
   * @return {Object} {peiceList, endsText} The order is the same as screen pixel order.
   */
  PiecewiseVisualMapView.prototype._getViewData = function () {
    var visualMapModel = this.visualMapModel;
    var viewPieceList = zrUtil.map(visualMapModel.getPieceList(), function (piece, index) {
      return {
        piece: piece,
        indexInModelPieceList: index
      };
    });
    var endsText = visualMapModel.get('text');
    // Consider orient and inverse.
    var orient = visualMapModel.get('orient');
    var inverse = visualMapModel.get('inverse');
    // Order of model pieceList is always [low, ..., high]
    if (orient === 'horizontal' ? inverse : !inverse) {
      viewPieceList.reverse();
    }
    // Origin order of endsText is [high, low]
    else if (endsText) {
      endsText = endsText.slice().reverse();
    }
    return {
      viewPieceList: viewPieceList,
      endsText: endsText
    };
  };
  PiecewiseVisualMapView.prototype._createItemSymbol = function (group, representValue, shapeParam, silent) {
    var itemSymbol = createSymbol(
    // symbol will be string
    this.getControllerVisual(representValue, 'symbol'), shapeParam[0], shapeParam[1], shapeParam[2], shapeParam[3],
    // color will be string
    this.getControllerVisual(representValue, 'color'));
    itemSymbol.silent = silent;
    group.add(itemSymbol);
  };
  PiecewiseVisualMapView.prototype._onItemClick = function (piece) {
    var visualMapModel = this.visualMapModel;
    var option = visualMapModel.option;
    var selectedMode = option.selectedMode;
    if (!selectedMode) {
      return;
    }
    var selected = zrUtil.clone(option.selected);
    var newKey = visualMapModel.getSelectedMapKey(piece);
    if (selectedMode === 'single' || selectedMode === true) {
      selected[newKey] = true;
      zrUtil.each(selected, function (o, key) {
        selected[key] = key === newKey;
      });
    } else {
      selected[newKey] = !selected[newKey];
    }
    this.api.dispatchAction({
      type: 'selectDataRange',
      from: this.uid,
      visualMapId: this.visualMapModel.id,
      selected: selected
    });
  };
  PiecewiseVisualMapView.type = 'visualMap.piecewise';
  return PiecewiseVisualMapView;
}(VisualMapView);
export default PiecewiseVisualMapView;