
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
import { Group } from '../../util/graphic.js';
import { enterEmphasis, leaveEmphasis, toggleHoverEmphasis } from '../../util/states.js';
import SymbolClz from './Symbol.js';
function updateRipplePath(rippleGroup, effectCfg) {
  var color = effectCfg.rippleEffectColor || effectCfg.color;
  rippleGroup.eachChild(function (ripplePath) {
    ripplePath.attr({
      z: effectCfg.z,
      zlevel: effectCfg.zlevel,
      style: {
        stroke: effectCfg.brushType === 'stroke' ? color : null,
        fill: effectCfg.brushType === 'fill' ? color : null
      }
    });
  });
}
var EffectSymbol = /** @class */function (_super) {
  __extends(EffectSymbol, _super);
  function EffectSymbol(data, idx) {
    var _this = _super.call(this) || this;
    var symbol = new SymbolClz(data, idx);
    var rippleGroup = new Group();
    _this.add(symbol);
    _this.add(rippleGroup);
    _this.updateData(data, idx);
    return _this;
  }
  EffectSymbol.prototype.stopEffectAnimation = function () {
    this.childAt(1).removeAll();
  };
  EffectSymbol.prototype.startEffectAnimation = function (effectCfg) {
    var symbolType = effectCfg.symbolType;
    var color = effectCfg.color;
    var rippleNumber = effectCfg.rippleNumber;
    var rippleGroup = this.childAt(1);
    for (var i = 0; i < rippleNumber; i++) {
      // If width/height are set too small (e.g., set to 1) on ios10
      // and macOS Sierra, a circle stroke become a rect, no matter what
      // the scale is set. So we set width/height as 2. See #4136.
      var ripplePath = createSymbol(symbolType, -1, -1, 2, 2, color);
      ripplePath.attr({
        style: {
          strokeNoScale: true
        },
        z2: 99,
        silent: true,
        scaleX: 0.5,
        scaleY: 0.5
      });
      var delay = -i / rippleNumber * effectCfg.period + effectCfg.effectOffset;
      ripplePath.animate('', true).when(effectCfg.period, {
        scaleX: effectCfg.rippleScale / 2,
        scaleY: effectCfg.rippleScale / 2
      }).delay(delay).start();
      ripplePath.animateStyle(true).when(effectCfg.period, {
        opacity: 0
      }).delay(delay).start();
      rippleGroup.add(ripplePath);
    }
    updateRipplePath(rippleGroup, effectCfg);
  };
  /**
   * Update effect symbol
   */
  EffectSymbol.prototype.updateEffectAnimation = function (effectCfg) {
    var oldEffectCfg = this._effectCfg;
    var rippleGroup = this.childAt(1);
    // Must reinitialize effect if following configuration changed
    var DIFFICULT_PROPS = ['symbolType', 'period', 'rippleScale', 'rippleNumber'];
    for (var i = 0; i < DIFFICULT_PROPS.length; i++) {
      var propName = DIFFICULT_PROPS[i];
      if (oldEffectCfg[propName] !== effectCfg[propName]) {
        this.stopEffectAnimation();
        this.startEffectAnimation(effectCfg);
        return;
      }
    }
    updateRipplePath(rippleGroup, effectCfg);
  };
  /**
   * Highlight symbol
   */
  EffectSymbol.prototype.highlight = function () {
    enterEmphasis(this);
  };
  /**
   * Downplay symbol
   */
  EffectSymbol.prototype.downplay = function () {
    leaveEmphasis(this);
  };
  EffectSymbol.prototype.getSymbolType = function () {
    var symbol = this.childAt(0);
    return symbol && symbol.getSymbolType();
  };
  /**
   * Update symbol properties
   */
  EffectSymbol.prototype.updateData = function (data, idx) {
    var _this = this;
    var seriesModel = data.hostModel;
    this.childAt(0).updateData(data, idx);
    var rippleGroup = this.childAt(1);
    var itemModel = data.getItemModel(idx);
    var symbolType = data.getItemVisual(idx, 'symbol');
    var symbolSize = normalizeSymbolSize(data.getItemVisual(idx, 'symbolSize'));
    var symbolStyle = data.getItemVisual(idx, 'style');
    var color = symbolStyle && symbolStyle.fill;
    var emphasisModel = itemModel.getModel('emphasis');
    rippleGroup.setScale(symbolSize);
    rippleGroup.traverse(function (ripplePath) {
      ripplePath.setStyle('fill', color);
    });
    var symbolOffset = normalizeSymbolOffset(data.getItemVisual(idx, 'symbolOffset'), symbolSize);
    if (symbolOffset) {
      rippleGroup.x = symbolOffset[0];
      rippleGroup.y = symbolOffset[1];
    }
    var symbolRotate = data.getItemVisual(idx, 'symbolRotate');
    rippleGroup.rotation = (symbolRotate || 0) * Math.PI / 180 || 0;
    var effectCfg = {};
    effectCfg.showEffectOn = seriesModel.get('showEffectOn');
    effectCfg.rippleScale = itemModel.get(['rippleEffect', 'scale']);
    effectCfg.brushType = itemModel.get(['rippleEffect', 'brushType']);
    effectCfg.period = itemModel.get(['rippleEffect', 'period']) * 1000;
    effectCfg.effectOffset = idx / data.count();
    effectCfg.z = seriesModel.getShallow('z') || 0;
    effectCfg.zlevel = seriesModel.getShallow('zlevel') || 0;
    effectCfg.symbolType = symbolType;
    effectCfg.color = color;
    effectCfg.rippleEffectColor = itemModel.get(['rippleEffect', 'color']);
    effectCfg.rippleNumber = itemModel.get(['rippleEffect', 'number']);
    if (effectCfg.showEffectOn === 'render') {
      this._effectCfg ? this.updateEffectAnimation(effectCfg) : this.startEffectAnimation(effectCfg);
      this._effectCfg = effectCfg;
    } else {
      // Not keep old effect config
      this._effectCfg = null;
      this.stopEffectAnimation();
      this.onHoverStateChange = function (toState) {
        if (toState === 'emphasis') {
          if (effectCfg.showEffectOn !== 'render') {
            _this.startEffectAnimation(effectCfg);
          }
        } else if (toState === 'normal') {
          if (effectCfg.showEffectOn !== 'render') {
            _this.stopEffectAnimation();
          }
        }
      };
    }
    this._effectCfg = effectCfg;
    toggleHoverEmphasis(this, emphasisModel.get('focus'), emphasisModel.get('blurScope'), emphasisModel.get('disabled'));
  };
  ;
  EffectSymbol.prototype.fadeOut = function (cb) {
    cb && cb();
  };
  ;
  return EffectSymbol;
}(Group);
export default EffectSymbol;