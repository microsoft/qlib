import { __extends } from "tslib";
import Element from '../Element.js';
import BoundingRect from '../core/BoundingRect.js';
import { keys, extend, createObject } from '../core/util.js';
import { REDRAW_BIT, STYLE_CHANGED_BIT } from './constants.js';
var STYLE_MAGIC_KEY = '__zr_style_' + Math.round((Math.random() * 10));
export var DEFAULT_COMMON_STYLE = {
    shadowBlur: 0,
    shadowOffsetX: 0,
    shadowOffsetY: 0,
    shadowColor: '#000',
    opacity: 1,
    blend: 'source-over'
};
export var DEFAULT_COMMON_ANIMATION_PROPS = {
    style: {
        shadowBlur: true,
        shadowOffsetX: true,
        shadowOffsetY: true,
        shadowColor: true,
        opacity: true
    }
};
DEFAULT_COMMON_STYLE[STYLE_MAGIC_KEY] = true;
var PRIMARY_STATES_KEYS = ['z', 'z2', 'invisible'];
var PRIMARY_STATES_KEYS_IN_HOVER_LAYER = ['invisible'];
var Displayable = (function (_super) {
    __extends(Displayable, _super);
    function Displayable(props) {
        return _super.call(this, props) || this;
    }
    Displayable.prototype._init = function (props) {
        var keysArr = keys(props);
        for (var i = 0; i < keysArr.length; i++) {
            var key = keysArr[i];
            if (key === 'style') {
                this.useStyle(props[key]);
            }
            else {
                _super.prototype.attrKV.call(this, key, props[key]);
            }
        }
        if (!this.style) {
            this.useStyle({});
        }
    };
    Displayable.prototype.beforeBrush = function () { };
    Displayable.prototype.afterBrush = function () { };
    Displayable.prototype.innerBeforeBrush = function () { };
    Displayable.prototype.innerAfterBrush = function () { };
    Displayable.prototype.shouldBePainted = function (viewWidth, viewHeight, considerClipPath, considerAncestors) {
        var m = this.transform;
        if (this.ignore
            || this.invisible
            || this.style.opacity === 0
            || (this.culling
                && isDisplayableCulled(this, viewWidth, viewHeight))
            || (m && !m[0] && !m[3])) {
            return false;
        }
        if (considerClipPath && this.__clipPaths) {
            for (var i = 0; i < this.__clipPaths.length; ++i) {
                if (this.__clipPaths[i].isZeroArea()) {
                    return false;
                }
            }
        }
        if (considerAncestors && this.parent) {
            var parent_1 = this.parent;
            while (parent_1) {
                if (parent_1.ignore) {
                    return false;
                }
                parent_1 = parent_1.parent;
            }
        }
        return true;
    };
    Displayable.prototype.contain = function (x, y) {
        return this.rectContain(x, y);
    };
    Displayable.prototype.traverse = function (cb, context) {
        cb.call(context, this);
    };
    Displayable.prototype.rectContain = function (x, y) {
        var coord = this.transformCoordToLocal(x, y);
        var rect = this.getBoundingRect();
        return rect.contain(coord[0], coord[1]);
    };
    Displayable.prototype.getPaintRect = function () {
        var rect = this._paintRect;
        if (!this._paintRect || this.__dirty) {
            var transform = this.transform;
            var elRect = this.getBoundingRect();
            var style = this.style;
            var shadowSize = style.shadowBlur || 0;
            var shadowOffsetX = style.shadowOffsetX || 0;
            var shadowOffsetY = style.shadowOffsetY || 0;
            rect = this._paintRect || (this._paintRect = new BoundingRect(0, 0, 0, 0));
            if (transform) {
                BoundingRect.applyTransform(rect, elRect, transform);
            }
            else {
                rect.copy(elRect);
            }
            if (shadowSize || shadowOffsetX || shadowOffsetY) {
                rect.width += shadowSize * 2 + Math.abs(shadowOffsetX);
                rect.height += shadowSize * 2 + Math.abs(shadowOffsetY);
                rect.x = Math.min(rect.x, rect.x + shadowOffsetX - shadowSize);
                rect.y = Math.min(rect.y, rect.y + shadowOffsetY - shadowSize);
            }
            var tolerance = this.dirtyRectTolerance;
            if (!rect.isZero()) {
                rect.x = Math.floor(rect.x - tolerance);
                rect.y = Math.floor(rect.y - tolerance);
                rect.width = Math.ceil(rect.width + 1 + tolerance * 2);
                rect.height = Math.ceil(rect.height + 1 + tolerance * 2);
            }
        }
        return rect;
    };
    Displayable.prototype.setPrevPaintRect = function (paintRect) {
        if (paintRect) {
            this._prevPaintRect = this._prevPaintRect || new BoundingRect(0, 0, 0, 0);
            this._prevPaintRect.copy(paintRect);
        }
        else {
            this._prevPaintRect = null;
        }
    };
    Displayable.prototype.getPrevPaintRect = function () {
        return this._prevPaintRect;
    };
    Displayable.prototype.animateStyle = function (loop) {
        return this.animate('style', loop);
    };
    Displayable.prototype.updateDuringAnimation = function (targetKey) {
        if (targetKey === 'style') {
            this.dirtyStyle();
        }
        else {
            this.markRedraw();
        }
    };
    Displayable.prototype.attrKV = function (key, value) {
        if (key !== 'style') {
            _super.prototype.attrKV.call(this, key, value);
        }
        else {
            if (!this.style) {
                this.useStyle(value);
            }
            else {
                this.setStyle(value);
            }
        }
    };
    Displayable.prototype.setStyle = function (keyOrObj, value) {
        if (typeof keyOrObj === 'string') {
            this.style[keyOrObj] = value;
        }
        else {
            extend(this.style, keyOrObj);
        }
        this.dirtyStyle();
        return this;
    };
    Displayable.prototype.dirtyStyle = function (notRedraw) {
        if (!notRedraw) {
            this.markRedraw();
        }
        this.__dirty |= STYLE_CHANGED_BIT;
        if (this._rect) {
            this._rect = null;
        }
    };
    Displayable.prototype.dirty = function () {
        this.dirtyStyle();
    };
    Displayable.prototype.styleChanged = function () {
        return !!(this.__dirty & STYLE_CHANGED_BIT);
    };
    Displayable.prototype.styleUpdated = function () {
        this.__dirty &= ~STYLE_CHANGED_BIT;
    };
    Displayable.prototype.createStyle = function (obj) {
        return createObject(DEFAULT_COMMON_STYLE, obj);
    };
    Displayable.prototype.useStyle = function (obj) {
        if (!obj[STYLE_MAGIC_KEY]) {
            obj = this.createStyle(obj);
        }
        if (this.__inHover) {
            this.__hoverStyle = obj;
        }
        else {
            this.style = obj;
        }
        this.dirtyStyle();
    };
    Displayable.prototype.isStyleObject = function (obj) {
        return obj[STYLE_MAGIC_KEY];
    };
    Displayable.prototype._innerSaveToNormal = function (toState) {
        _super.prototype._innerSaveToNormal.call(this, toState);
        var normalState = this._normalState;
        if (toState.style && !normalState.style) {
            normalState.style = this._mergeStyle(this.createStyle(), this.style);
        }
        this._savePrimaryToNormal(toState, normalState, PRIMARY_STATES_KEYS);
    };
    Displayable.prototype._applyStateObj = function (stateName, state, normalState, keepCurrentStates, transition, animationCfg) {
        _super.prototype._applyStateObj.call(this, stateName, state, normalState, keepCurrentStates, transition, animationCfg);
        var needsRestoreToNormal = !(state && keepCurrentStates);
        var targetStyle;
        if (state && state.style) {
            if (transition) {
                if (keepCurrentStates) {
                    targetStyle = state.style;
                }
                else {
                    targetStyle = this._mergeStyle(this.createStyle(), normalState.style);
                    this._mergeStyle(targetStyle, state.style);
                }
            }
            else {
                targetStyle = this._mergeStyle(this.createStyle(), keepCurrentStates ? this.style : normalState.style);
                this._mergeStyle(targetStyle, state.style);
            }
        }
        else if (needsRestoreToNormal) {
            targetStyle = normalState.style;
        }
        if (targetStyle) {
            if (transition) {
                var sourceStyle = this.style;
                this.style = this.createStyle(needsRestoreToNormal ? {} : sourceStyle);
                if (needsRestoreToNormal) {
                    var changedKeys = keys(sourceStyle);
                    for (var i = 0; i < changedKeys.length; i++) {
                        var key = changedKeys[i];
                        if (key in targetStyle) {
                            targetStyle[key] = targetStyle[key];
                            this.style[key] = sourceStyle[key];
                        }
                    }
                }
                var targetKeys = keys(targetStyle);
                for (var i = 0; i < targetKeys.length; i++) {
                    var key = targetKeys[i];
                    this.style[key] = this.style[key];
                }
                this._transitionState(stateName, {
                    style: targetStyle
                }, animationCfg, this.getAnimationStyleProps());
            }
            else {
                this.useStyle(targetStyle);
            }
        }
        var statesKeys = this.__inHover ? PRIMARY_STATES_KEYS_IN_HOVER_LAYER : PRIMARY_STATES_KEYS;
        for (var i = 0; i < statesKeys.length; i++) {
            var key = statesKeys[i];
            if (state && state[key] != null) {
                this[key] = state[key];
            }
            else if (needsRestoreToNormal) {
                if (normalState[key] != null) {
                    this[key] = normalState[key];
                }
            }
        }
    };
    Displayable.prototype._mergeStates = function (states) {
        var mergedState = _super.prototype._mergeStates.call(this, states);
        var mergedStyle;
        for (var i = 0; i < states.length; i++) {
            var state = states[i];
            if (state.style) {
                mergedStyle = mergedStyle || {};
                this._mergeStyle(mergedStyle, state.style);
            }
        }
        if (mergedStyle) {
            mergedState.style = mergedStyle;
        }
        return mergedState;
    };
    Displayable.prototype._mergeStyle = function (targetStyle, sourceStyle) {
        extend(targetStyle, sourceStyle);
        return targetStyle;
    };
    Displayable.prototype.getAnimationStyleProps = function () {
        return DEFAULT_COMMON_ANIMATION_PROPS;
    };
    Displayable.initDefaultProps = (function () {
        var dispProto = Displayable.prototype;
        dispProto.type = 'displayable';
        dispProto.invisible = false;
        dispProto.z = 0;
        dispProto.z2 = 0;
        dispProto.zlevel = 0;
        dispProto.culling = false;
        dispProto.cursor = 'pointer';
        dispProto.rectHover = false;
        dispProto.incremental = false;
        dispProto._rect = null;
        dispProto.dirtyRectTolerance = 0;
        dispProto.__dirty = REDRAW_BIT | STYLE_CHANGED_BIT;
    })();
    return Displayable;
}(Element));
var tmpRect = new BoundingRect(0, 0, 0, 0);
var viewRect = new BoundingRect(0, 0, 0, 0);
function isDisplayableCulled(el, width, height) {
    tmpRect.copy(el.getBoundingRect());
    if (el.transform) {
        tmpRect.applyTransform(el.transform);
    }
    viewRect.width = width;
    viewRect.height = height;
    return !tmpRect.intersect(viewRect);
}
export default Displayable;
