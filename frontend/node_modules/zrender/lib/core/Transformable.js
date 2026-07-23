import * as matrix from './matrix.js';
import * as vector from './vector.js';
var mIdentity = matrix.identity;
var EPSILON = 5e-5;
function isNotAroundZero(val) {
    return val > EPSILON || val < -EPSILON;
}
var scaleTmp = [];
var tmpTransform = [];
var originTransform = matrix.create();
var abs = Math.abs;
var Transformable = (function () {
    function Transformable() {
    }
    Transformable.prototype.getLocalTransform = function (m) {
        return Transformable.getLocalTransform(this, m);
    };
    Transformable.prototype.setPosition = function (arr) {
        this.x = arr[0];
        this.y = arr[1];
    };
    Transformable.prototype.setScale = function (arr) {
        this.scaleX = arr[0];
        this.scaleY = arr[1];
    };
    Transformable.prototype.setSkew = function (arr) {
        this.skewX = arr[0];
        this.skewY = arr[1];
    };
    Transformable.prototype.setOrigin = function (arr) {
        this.originX = arr[0];
        this.originY = arr[1];
    };
    Transformable.prototype.needLocalTransform = function () {
        return isNotAroundZero(this.rotation)
            || isNotAroundZero(this.x)
            || isNotAroundZero(this.y)
            || isNotAroundZero(this.scaleX - 1)
            || isNotAroundZero(this.scaleY - 1)
            || isNotAroundZero(this.skewX)
            || isNotAroundZero(this.skewY);
    };
    Transformable.prototype.updateTransform = function () {
        var parentTransform = this.parent && this.parent.transform;
        var needLocalTransform = this.needLocalTransform();
        var m = this.transform;
        if (!(needLocalTransform || parentTransform)) {
            if (m) {
                mIdentity(m);
                this.invTransform = null;
            }
            return;
        }
        m = m || matrix.create();
        if (needLocalTransform) {
            this.getLocalTransform(m);
        }
        else {
            mIdentity(m);
        }
        if (parentTransform) {
            if (needLocalTransform) {
                matrix.mul(m, parentTransform, m);
            }
            else {
                matrix.copy(m, parentTransform);
            }
        }
        this.transform = m;
        this._resolveGlobalScaleRatio(m);
    };
    Transformable.prototype._resolveGlobalScaleRatio = function (m) {
        var globalScaleRatio = this.globalScaleRatio;
        if (globalScaleRatio != null && globalScaleRatio !== 1) {
            this.getGlobalScale(scaleTmp);
            var relX = scaleTmp[0] < 0 ? -1 : 1;
            var relY = scaleTmp[1] < 0 ? -1 : 1;
            var sx = ((scaleTmp[0] - relX) * globalScaleRatio + relX) / scaleTmp[0] || 0;
            var sy = ((scaleTmp[1] - relY) * globalScaleRatio + relY) / scaleTmp[1] || 0;
            m[0] *= sx;
            m[1] *= sx;
            m[2] *= sy;
            m[3] *= sy;
        }
        this.invTransform = this.invTransform || matrix.create();
        matrix.invert(this.invTransform, m);
    };
    Transformable.prototype.getComputedTransform = function () {
        var transformNode = this;
        var ancestors = [];
        while (transformNode) {
            ancestors.push(transformNode);
            transformNode = transformNode.parent;
        }
        while (transformNode = ancestors.pop()) {
            transformNode.updateTransform();
        }
        return this.transform;
    };
    Transformable.prototype.setLocalTransform = function (m) {
        if (!m) {
            return;
        }
        var sx = m[0] * m[0] + m[1] * m[1];
        var sy = m[2] * m[2] + m[3] * m[3];
        var rotation = Math.atan2(m[1], m[0]);
        var shearX = Math.PI / 2 + rotation - Math.atan2(m[3], m[2]);
        sy = Math.sqrt(sy) * Math.cos(shearX);
        sx = Math.sqrt(sx);
        this.skewX = shearX;
        this.skewY = 0;
        this.rotation = -rotation;
        this.x = +m[4];
        this.y = +m[5];
        this.scaleX = sx;
        this.scaleY = sy;
        this.originX = 0;
        this.originY = 0;
    };
    Transformable.prototype.decomposeTransform = function () {
        if (!this.transform) {
            return;
        }
        var parent = this.parent;
        var m = this.transform;
        if (parent && parent.transform) {
            parent.invTransform = parent.invTransform || matrix.create();
            matrix.mul(tmpTransform, parent.invTransform, m);
            m = tmpTransform;
        }
        var ox = this.originX;
        var oy = this.originY;
        if (ox || oy) {
            originTransform[4] = ox;
            originTransform[5] = oy;
            matrix.mul(tmpTransform, m, originTransform);
            tmpTransform[4] -= ox;
            tmpTransform[5] -= oy;
            m = tmpTransform;
        }
        this.setLocalTransform(m);
    };
    Transformable.prototype.getGlobalScale = function (out) {
        var m = this.transform;
        out = out || [];
        if (!m) {
            out[0] = 1;
            out[1] = 1;
            return out;
        }
        out[0] = Math.sqrt(m[0] * m[0] + m[1] * m[1]);
        out[1] = Math.sqrt(m[2] * m[2] + m[3] * m[3]);
        if (m[0] < 0) {
            out[0] = -out[0];
        }
        if (m[3] < 0) {
            out[1] = -out[1];
        }
        return out;
    };
    Transformable.prototype.transformCoordToLocal = function (x, y) {
        var v2 = [x, y];
        var invTransform = this.invTransform;
        if (invTransform) {
            vector.applyTransform(v2, v2, invTransform);
        }
        return v2;
    };
    Transformable.prototype.transformCoordToGlobal = function (x, y) {
        var v2 = [x, y];
        var transform = this.transform;
        if (transform) {
            vector.applyTransform(v2, v2, transform);
        }
        return v2;
    };
    Transformable.prototype.getLineScale = function () {
        var m = this.transform;
        return m && abs(m[0] - 1) > 1e-10 && abs(m[3] - 1) > 1e-10
            ? Math.sqrt(abs(m[0] * m[3] - m[2] * m[1]))
            : 1;
    };
    Transformable.prototype.copyTransform = function (source) {
        copyTransform(this, source);
    };
    Transformable.getLocalTransform = function (target, m) {
        m = m || [];
        var ox = target.originX || 0;
        var oy = target.originY || 0;
        var sx = target.scaleX;
        var sy = target.scaleY;
        var ax = target.anchorX;
        var ay = target.anchorY;
        var rotation = target.rotation || 0;
        var x = target.x;
        var y = target.y;
        var skewX = target.skewX ? Math.tan(target.skewX) : 0;
        var skewY = target.skewY ? Math.tan(-target.skewY) : 0;
        if (ox || oy || ax || ay) {
            var dx = ox + ax;
            var dy = oy + ay;
            m[4] = -dx * sx - skewX * dy * sy;
            m[5] = -dy * sy - skewY * dx * sx;
        }
        else {
            m[4] = m[5] = 0;
        }
        m[0] = sx;
        m[3] = sy;
        m[1] = skewY * sx;
        m[2] = skewX * sy;
        rotation && matrix.rotate(m, m, rotation);
        m[4] += ox + x;
        m[5] += oy + y;
        return m;
    };
    Transformable.initDefaultProps = (function () {
        var proto = Transformable.prototype;
        proto.scaleX =
            proto.scaleY =
                proto.globalScaleRatio = 1;
        proto.x =
            proto.y =
                proto.originX =
                    proto.originY =
                        proto.skewX =
                            proto.skewY =
                                proto.rotation =
                                    proto.anchorX =
                                        proto.anchorY = 0;
    })();
    return Transformable;
}());
;
export var TRANSFORMABLE_PROPS = [
    'x', 'y', 'originX', 'originY', 'anchorX', 'anchorY', 'rotation', 'scaleX', 'scaleY', 'skewX', 'skewY'
];
export function copyTransform(target, source) {
    for (var i = 0; i < TRANSFORMABLE_PROPS.length; i++) {
        var propName = TRANSFORMABLE_PROPS[i];
        target[propName] = source[propName];
    }
}
export default Transformable;
