
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
import { createSymbol, normalizeSymbolOffset, normalizeSymbolSize } from '../../util/symbol.js';
import * as graphic from '../../util/graphic.js';
import { getECData } from '../../util/innerStore.js';
import { enterEmphasis, leaveEmphasis, toggleHoverEmphasis } from '../../util/states.js';
import { getDefaultLabel } from './labelHelper.js';
import { extend } from 'zrender/lib/core/util.js';
import { setLabelStyle, getLabelStatesModels } from '../../label/labelStyle.js';
import ZRImage from 'zrender/lib/graphic/Image.js';
import { saveOldStyle } from '../../animation/basicTransition.js';
var Symbol = /** @class */function (_super) {
  __extends(Symbol, _super);
  function Symbol(data, idx, seriesScope, opts) {
    var _this = _super.call(this) || this;
    _this.updateData(data, idx, seriesScope, opts);
    return _this;
  }
  Symbol.prototype._createSymbol = function (symbolType, data, idx, symbolSize, keepAspect) {
    // Remove paths created before
    this.removeAll();
    // let symbolPath = createSymbol(
    //     symbolType, -0.5, -0.5, 1, 1, color
    // );
    // If width/height are set too small (e.g., set to 1) on ios10
    // and macOS Sierra, a circle stroke become a rect, no matter what
    // the scale is set. So we set width/height as 2. See #4150.
    var symbolPath = createSymbol(symbolType, -1, -1, 2, 2, null, keepAspect);
    symbolPath.attr({
      z2: 100,
      culling: true,
      scaleX: symbolSize[0] / 2,
      scaleY: symbolSize[1] / 2
    });
    // Rewrite drift method
    symbolPath.drift = driftSymbol;
    this._symbolType = symbolType;
    this.add(symbolPath);
  };
  /**
   * Stop animation
   * @param {boolean} toLastFrame
   */
  Symbol.prototype.stopSymbolAnimation = function (toLastFrame) {
    this.childAt(0).stopAnimation(null, toLastFrame);
  };
  Symbol.prototype.getSymbolType = function () {
    return this._symbolType;
  };
  /**
   * FIXME:
   * Caution: This method breaks the encapsulation of this module,
   * but it indeed brings convenience. So do not use the method
   * unless you detailedly know all the implements of `Symbol`,
   * especially animation.
   *
   * Get symbol path element.
   */
  Symbol.prototype.getSymbolPath = function () {
    return this.childAt(0);
  };
  /**
   * Highlight symbol
   */
  Symbol.prototype.highlight = function () {
    enterEmphasis(this.childAt(0));
  };
  /**
   * Downplay symbol
   */
  Symbol.prototype.downplay = function () {
    leaveEmphasis(this.childAt(0));
  };
  /**
   * @param {number} zlevel
   * @param {number} z
   */
  Symbol.prototype.setZ = function (zlevel, z) {
    var symbolPath = this.childAt(0);
    symbolPath.zlevel = zlevel;
    symbolPath.z = z;
  };
  Symbol.prototype.setDraggable = function (draggable, hasCursorOption) {
    var symbolPath = this.childAt(0);
    symbolPath.draggable = draggable;
    symbolPath.cursor = !hasCursorOption && draggable ? 'move' : symbolPath.cursor;
  };
  /**
   * Update symbol properties
   */
  Symbol.prototype.updateData = function (data, idx, seriesScope, opts) {
    this.silent = false;
    var symbolType = data.getItemVisual(idx, 'symbol') || 'circle';
    var seriesModel = data.hostModel;
    var symbolSize = Symbol.getSymbolSize(data, idx);
    var isInit = symbolType !== this._symbolType;
    var disableAnimation = opts && opts.disableAnimation;
    if (isInit) {
      var keepAspect = data.getItemVisual(idx, 'symbolKeepAspect');
      this._createSymbol(symbolType, data, idx, symbolSize, keepAspect);
    } else {
      var symbolPath = this.childAt(0);
      symbolPath.silent = false;
      var target = {
        scaleX: symbolSize[0] / 2,
        scaleY: symbolSize[1] / 2
      };
      disableAnimation ? symbolPath.attr(target) : graphic.updateProps(symbolPath, target, seriesModel, idx);
      saveOldStyle(symbolPath);
    }
    this._updateCommon(data, idx, symbolSize, seriesScope, opts);
    if (isInit) {
      var symbolPath = this.childAt(0);
      if (!disableAnimation) {
        var target = {
          scaleX: this._sizeX,
          scaleY: this._sizeY,
          style: {
            // Always fadeIn. Because it has fadeOut animation when symbol is removed..
            opacity: symbolPath.style.opacity
          }
        };
        symbolPath.scaleX = symbolPath.scaleY = 0;
        symbolPath.style.opacity = 0;
        graphic.initProps(symbolPath, target, seriesModel, idx);
      }
    }
    if (disableAnimation) {
      // Must stop leave transition manually if don't call initProps or updateProps.
      this.childAt(0).stopAnimation('leave');
    }
  };
  Symbol.prototype._updateCommon = function (data, idx, symbolSize, seriesScope, opts) {
    var symbolPath = this.childAt(0);
    var seriesModel = data.hostModel;
    var emphasisItemStyle;
    var blurItemStyle;
    var selectItemStyle;
    var focus;
    var blurScope;
    var emphasisDisabled;
    var labelStatesModels;
    var hoverScale;
    var cursorStyle;
    if (seriesScope) {
      emphasisItemStyle = seriesScope.emphasisItemStyle;
      blurItemStyle = seriesScope.blurItemStyle;
      selectItemStyle = seriesScope.selectItemStyle;
      focus = seriesScope.focus;
      blurScope = seriesScope.blurScope;
      labelStatesModels = seriesScope.labelStatesModels;
      hoverScale = seriesScope.hoverScale;
      cursorStyle = seriesScope.cursorStyle;
      emphasisDisabled = seriesScope.emphasisDisabled;
    }
    if (!seriesScope || data.hasItemOption) {
      var itemModel = seriesScope && seriesScope.itemModel ? seriesScope.itemModel : data.getItemModel(idx);
      var emphasisModel = itemModel.getModel('emphasis');
      emphasisItemStyle = emphasisModel.getModel('itemStyle').getItemStyle();
      selectItemStyle = itemModel.getModel(['select', 'itemStyle']).getItemStyle();
      blurItemStyle = itemModel.getModel(['blur', 'itemStyle']).getItemStyle();
      focus = emphasisModel.get('focus');
      blurScope = emphasisModel.get('blurScope');
      emphasisDisabled = emphasisModel.get('disabled');
      labelStatesModels = getLabelStatesModels(itemModel);
      hoverScale = emphasisModel.getShallow('scale');
      cursorStyle = itemModel.getShallow('cursor');
    }
    var symbolRotate = data.getItemVisual(idx, 'symbolRotate');
    symbolPath.attr('rotation', (symbolRotate || 0) * Math.PI / 180 || 0);
    var symbolOffset = normalizeSymbolOffset(data.getItemVisual(idx, 'symbolOffset'), symbolSize);
    if (symbolOffset) {
      symbolPath.x = symbolOffset[0];
      symbolPath.y = symbolOffset[1];
    }
    cursorStyle && symbolPath.attr('cursor', cursorStyle);
    var symbolStyle = data.getItemVisual(idx, 'style');
    var visualColor = symbolStyle.fill;
    if (symbolPath instanceof ZRImage) {
      var pathStyle = symbolPath.style;
      symbolPath.useStyle(extend({
        // TODO other properties like x, y ?
        image: pathStyle.image,
        x: pathStyle.x,
        y: pathStyle.y,
        width: pathStyle.width,
        height: pathStyle.height
      }, symbolStyle));
    } else {
      if (symbolPath.__isEmptyBrush) {
        // fill and stroke will be swapped if it's empty.
        // So we cloned a new style to avoid it affecting the original style in visual storage.
        // TODO Better implementation. No empty logic!
        symbolPath.useStyle(extend({}, symbolStyle));
      } else {
        symbolPath.useStyle(symbolStyle);
      }
      // Disable decal because symbol scale will been applied on the decal.
      symbolPath.style.decal = null;
      symbolPath.setColor(visualColor, opts && opts.symbolInnerColor);
      symbolPath.style.strokeNoScale = true;
    }
    var liftZ = data.getItemVisual(idx, 'liftZ');
    var z2Origin = this._z2;
    if (liftZ != null) {
      if (z2Origin == null) {
        this._z2 = symbolPath.z2;
        symbolPath.z2 += liftZ;
      }
    } else if (z2Origin != null) {
      symbolPath.z2 = z2Origin;
      this._z2 = null;
    }
    var useNameLabel = opts && opts.useNameLabel;
    setLabelStyle(symbolPath, labelStatesModels, {
      labelFetcher: seriesModel,
      labelDataIndex: idx,
      defaultText: getLabelDefaultText,
      inheritColor: visualColor,
      defaultOpacity: symbolStyle.opacity
    });
    // Do not execute util needed.
    function getLabelDefaultText(idx) {
      return useNameLabel ? data.getName(idx) : getDefaultLabel(data, idx);
    }
    this._sizeX = symbolSize[0] / 2;
    this._sizeY = symbolSize[1] / 2;
    var emphasisState = symbolPath.ensureState('emphasis');
    emphasisState.style = emphasisItemStyle;
    symbolPath.ensureState('select').style = selectItemStyle;
    symbolPath.ensureState('blur').style = blurItemStyle;
    // null / undefined / true means to use default strategy.
    // 0 / false / negative number / NaN / Infinity means no scale.
    var scaleRatio = hoverScale == null || hoverScale === true ? Math.max(1.1, 3 / this._sizeY)
    // PENDING: restrict hoverScale > 1? It seems unreasonable to scale down
    : isFinite(hoverScale) && hoverScale > 0 ? +hoverScale : 1;
    // always set scale to allow resetting
    emphasisState.scaleX = this._sizeX * scaleRatio;
    emphasisState.scaleY = this._sizeY * scaleRatio;
    this.setSymbolScale(1);
    toggleHoverEmphasis(this, focus, blurScope, emphasisDisabled);
  };
  Symbol.prototype.setSymbolScale = function (scale) {
    this.scaleX = this.scaleY = scale;
  };
  Symbol.prototype.fadeOut = function (cb, seriesModel, opt) {
    var symbolPath = this.childAt(0);
    var dataIndex = getECData(this).dataIndex;
    var animationOpt = opt && opt.animation;
    // Avoid mistaken hover when fading out
    this.silent = symbolPath.silent = true;
    // Not show text when animating
    if (opt && opt.fadeLabel) {
      var textContent = symbolPath.getTextContent();
      if (textContent) {
        graphic.removeElement(textContent, {
          style: {
            opacity: 0
          }
        }, seriesModel, {
          dataIndex: dataIndex,
          removeOpt: animationOpt,
          cb: function () {
            symbolPath.removeTextContent();
          }
        });
      }
    } else {
      symbolPath.removeTextContent();
    }
    graphic.removeElement(symbolPath, {
      style: {
        opacity: 0
      },
      scaleX: 0,
      scaleY: 0
    }, seriesModel, {
      dataIndex: dataIndex,
      cb: cb,
      removeOpt: animationOpt
    });
  };
  Symbol.getSymbolSize = function (data, idx) {
    return normalizeSymbolSize(data.getItemVisual(idx, 'symbolSize'));
  };
  return Symbol;
}(graphic.Group);
function driftSymbol(dx, dy) {
  this.parent.drift(dx, dy);
}
export default Symbol;