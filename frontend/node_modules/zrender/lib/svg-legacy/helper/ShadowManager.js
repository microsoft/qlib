import { __extends } from "tslib";
import Definable from './Definable.js';
import { getIdURL, getShadowKey, hasShadow, normalizeColor } from '../../svg/helper.js';
import { createElement } from '../../svg/core.js';
var ShadowManager = (function (_super) {
    __extends(ShadowManager, _super);
    function ShadowManager(zrId, svgRoot) {
        var _this = _super.call(this, zrId, svgRoot, ['filter'], '__filter_in_use__', '_shadowDom') || this;
        _this._shadowDomMap = {};
        _this._shadowDomPool = [];
        return _this;
    }
    ShadowManager.prototype._getFromPool = function () {
        var shadowDom = this._shadowDomPool.pop();
        if (!shadowDom) {
            shadowDom = createElement('filter');
            shadowDom.setAttribute('id', 'zr' + this._zrId + '-shadow-' + this.nextId++);
            var domChild = createElement('feDropShadow');
            shadowDom.appendChild(domChild);
            this.addDom(shadowDom);
        }
        return shadowDom;
    };
    ShadowManager.prototype.update = function (svgElement, displayable) {
        var style = displayable.style;
        if (hasShadow(style)) {
            var shadowKey = getShadowKey(displayable);
            var shadowDom = displayable._shadowDom = this._shadowDomMap[shadowKey];
            if (!shadowDom) {
                shadowDom = this._getFromPool();
                this._shadowDomMap[shadowKey] = shadowDom;
            }
            this.updateDom(svgElement, displayable, shadowDom);
        }
        else {
            this.remove(svgElement, displayable);
        }
    };
    ShadowManager.prototype.remove = function (svgElement, displayable) {
        if (displayable._shadowDom != null) {
            displayable._shadowDom = null;
            svgElement.removeAttribute('filter');
        }
    };
    ShadowManager.prototype.updateDom = function (svgElement, displayable, shadowDom) {
        var domChild = shadowDom.children[0];
        var style = displayable.style;
        var globalScale = displayable.getGlobalScale();
        var scaleX = globalScale[0];
        var scaleY = globalScale[1];
        if (!scaleX || !scaleY) {
            return;
        }
        var offsetX = style.shadowOffsetX || 0;
        var offsetY = style.shadowOffsetY || 0;
        var blur = style.shadowBlur;
        var normalizedColor = normalizeColor(style.shadowColor);
        domChild.setAttribute('dx', offsetX / scaleX + '');
        domChild.setAttribute('dy', offsetY / scaleY + '');
        domChild.setAttribute('flood-color', normalizedColor.color);
        domChild.setAttribute('flood-opacity', normalizedColor.opacity + '');
        var stdDx = blur / 2 / scaleX;
        var stdDy = blur / 2 / scaleY;
        var stdDeviation = stdDx + ' ' + stdDy;
        domChild.setAttribute('stdDeviation', stdDeviation);
        shadowDom.setAttribute('x', '-100%');
        shadowDom.setAttribute('y', '-100%');
        shadowDom.setAttribute('width', '300%');
        shadowDom.setAttribute('height', '300%');
        displayable._shadowDom = shadowDom;
        svgElement.setAttribute('filter', getIdURL(shadowDom.getAttribute('id')));
    };
    ShadowManager.prototype.removeUnused = function () {
        var defs = this.getDefs(false);
        if (!defs) {
            return;
        }
        var shadowDomsPool = this._shadowDomPool;
        var shadowDomMap = this._shadowDomMap;
        for (var key in shadowDomMap) {
            if (shadowDomMap.hasOwnProperty(key)) {
                shadowDomsPool.push(shadowDomMap[key]);
            }
        }
        this._shadowDomMap = {};
    };
    return ShadowManager;
}(Definable));
export default ShadowManager;
