import { __extends } from "tslib";
import Path from '../Path.js';
import * as vec2 from '../../core/vector.js';
import { quadraticSubdivide, cubicSubdivide, quadraticAt, cubicAt, quadraticDerivativeAt, cubicDerivativeAt } from '../../core/curve.js';
var out = [];
var BezierCurveShape = (function () {
    function BezierCurveShape() {
        this.x1 = 0;
        this.y1 = 0;
        this.x2 = 0;
        this.y2 = 0;
        this.cpx1 = 0;
        this.cpy1 = 0;
        this.percent = 1;
    }
    return BezierCurveShape;
}());
export { BezierCurveShape };
function someVectorAt(shape, t, isTangent) {
    var cpx2 = shape.cpx2;
    var cpy2 = shape.cpy2;
    if (cpx2 != null || cpy2 != null) {
        return [
            (isTangent ? cubicDerivativeAt : cubicAt)(shape.x1, shape.cpx1, shape.cpx2, shape.x2, t),
            (isTangent ? cubicDerivativeAt : cubicAt)(shape.y1, shape.cpy1, shape.cpy2, shape.y2, t)
        ];
    }
    else {
        return [
            (isTangent ? quadraticDerivativeAt : quadraticAt)(shape.x1, shape.cpx1, shape.x2, t),
            (isTangent ? quadraticDerivativeAt : quadraticAt)(shape.y1, shape.cpy1, shape.y2, t)
        ];
    }
}
var BezierCurve = (function (_super) {
    __extends(BezierCurve, _super);
    function BezierCurve(opts) {
        return _super.call(this, opts) || this;
    }
    BezierCurve.prototype.getDefaultStyle = function () {
        return {
            stroke: '#000',
            fill: null
        };
    };
    BezierCurve.prototype.getDefaultShape = function () {
        return new BezierCurveShape();
    };
    BezierCurve.prototype.buildPath = function (ctx, shape) {
        var x1 = shape.x1;
        var y1 = shape.y1;
        var x2 = shape.x2;
        var y2 = shape.y2;
        var cpx1 = shape.cpx1;
        var cpy1 = shape.cpy1;
        var cpx2 = shape.cpx2;
        var cpy2 = shape.cpy2;
        var percent = shape.percent;
        if (percent === 0) {
            return;
        }
        ctx.moveTo(x1, y1);
        if (cpx2 == null || cpy2 == null) {
            if (percent < 1) {
                quadraticSubdivide(x1, cpx1, x2, percent, out);
                cpx1 = out[1];
                x2 = out[2];
                quadraticSubdivide(y1, cpy1, y2, percent, out);
                cpy1 = out[1];
                y2 = out[2];
            }
            ctx.quadraticCurveTo(cpx1, cpy1, x2, y2);
        }
        else {
            if (percent < 1) {
                cubicSubdivide(x1, cpx1, cpx2, x2, percent, out);
                cpx1 = out[1];
                cpx2 = out[2];
                x2 = out[3];
                cubicSubdivide(y1, cpy1, cpy2, y2, percent, out);
                cpy1 = out[1];
                cpy2 = out[2];
                y2 = out[3];
            }
            ctx.bezierCurveTo(cpx1, cpy1, cpx2, cpy2, x2, y2);
        }
    };
    BezierCurve.prototype.pointAt = function (t) {
        return someVectorAt(this.shape, t, false);
    };
    BezierCurve.prototype.tangentAt = function (t) {
        var p = someVectorAt(this.shape, t, true);
        return vec2.normalize(p, p);
    };
    return BezierCurve;
}(Path));
;
BezierCurve.prototype.type = 'bezier-curve';
export default BezierCurve;
