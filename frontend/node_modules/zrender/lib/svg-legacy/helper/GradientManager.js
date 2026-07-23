import { __extends } from "tslib";
import Definable from './Definable.js';
import * as zrUtil from '../../core/util.js';
import { getIdURL, isGradient, isLinearGradient, isRadialGradient, normalizeColor, round4 } from '../../svg/helper.js';
import { createElement } from '../../svg/core.js';
var GradientManager = (function (_super) {
    __extends(GradientManager, _super);
    function GradientManager(zrId, svgRoot) {
        return _super.call(this, zrId, svgRoot, ['linearGradient', 'radialGradient'], '__gradient_in_use__') || this;
    }
    GradientManager.prototype.addWithoutUpdate = function (svgElement, displayable) {
        if (displayable && displayable.style) {
            var that_1 = this;
            zrUtil.each(['fill', 'stroke'], function (fillOrStroke) {
                var value = displayable.style[fillOrStroke];
                if (isGradient(value)) {
                    var gradient = value;
                    var defs = that_1.getDefs(true);
                    var dom = void 0;
                    if (gradient.__dom) {
                        dom = gradient.__dom;
                        if (!defs.contains(gradient.__dom)) {
                            that_1.addDom(dom);
                        }
                    }
                    else {
                        dom = that_1.add(gradient);
                    }
                    that_1.markUsed(displayable);
                    svgElement.setAttribute(fillOrStroke, getIdURL(dom.getAttribute('id')));
                }
            });
        }
    };
    GradientManager.prototype.add = function (gradient) {
        var dom;
        if (isLinearGradient(gradient)) {
            dom = createElement('linearGradient');
        }
        else if (isRadialGradient(gradient)) {
            dom = createElement('radialGradient');
        }
        else {
            if (process.env.NODE_ENV !== 'production') {
                zrUtil.logError('Illegal gradient type.');
            }
            return null;
        }
        gradient.id = gradient.id || this.nextId++;
        dom.setAttribute('id', 'zr' + this._zrId
            + '-gradient-' + gradient.id);
        this.updateDom(gradient, dom);
        this.addDom(dom);
        return dom;
    };
    GradientManager.prototype.update = function (gradient) {
        if (!isGradient(gradient)) {
            return;
        }
        var that = this;
        this.doUpdate(gradient, function () {
            var dom = gradient.__dom;
            if (!dom) {
                return;
            }
            var tagName = dom.tagName;
            var type = gradient.type;
            if (type === 'linear' && tagName === 'linearGradient'
                || type === 'radial' && tagName === 'radialGradient') {
                that.updateDom(gradient, gradient.__dom);
            }
            else {
                that.removeDom(gradient);
                that.add(gradient);
            }
        });
    };
    GradientManager.prototype.updateDom = function (gradient, dom) {
        if (isLinearGradient(gradient)) {
            dom.setAttribute('x1', gradient.x);
            dom.setAttribute('y1', gradient.y);
            dom.setAttribute('x2', gradient.x2);
            dom.setAttribute('y2', gradient.y2);
        }
        else if (isRadialGradient(gradient)) {
            dom.setAttribute('cx', gradient.x);
            dom.setAttribute('cy', gradient.y);
            dom.setAttribute('r', gradient.r);
        }
        else {
            if (process.env.NODE_ENV !== 'production') {
                zrUtil.logError('Illegal gradient type.');
            }
            return;
        }
        dom.setAttribute('gradientUnits', gradient.global
            ? 'userSpaceOnUse'
            : 'objectBoundingBox');
        dom.innerHTML = '';
        var colors = gradient.colorStops;
        for (var i = 0, len = colors.length; i < len; ++i) {
            var stop_1 = createElement('stop');
            stop_1.setAttribute('offset', round4(colors[i].offset) * 100 + '%');
            var stopColor = colors[i].color;
            var _a = normalizeColor(stopColor), color = _a.color, opacity = _a.opacity;
            stop_1.setAttribute('stop-color', color);
            if (opacity < 1) {
                stop_1.setAttribute('stop-opacity', opacity);
            }
            dom.appendChild(stop_1);
        }
        gradient.__dom = dom;
    };
    GradientManager.prototype.markUsed = function (displayable) {
        if (displayable.style) {
            var gradient = displayable.style.fill;
            if (gradient && gradient.__dom) {
                _super.prototype.markDomUsed.call(this, gradient.__dom);
            }
            gradient = displayable.style.stroke;
            if (gradient && gradient.__dom) {
                _super.prototype.markDomUsed.call(this, gradient.__dom);
            }
        }
    };
    return GradientManager;
}(Definable));
export default GradientManager;
