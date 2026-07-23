import { __extends } from "tslib";
import Path from '../Path.js';
var cos = Math.cos;
var sin = Math.sin;
var TrochoidShape = (function () {
    function TrochoidShape() {
        this.cx = 0;
        this.cy = 0;
        this.r = 0;
        this.r0 = 0;
        this.d = 0;
        this.location = 'out';
    }
    return TrochoidShape;
}());
export { TrochoidShape };
var Trochoid = (function (_super) {
    __extends(Trochoid, _super);
    function Trochoid(opts) {
        return _super.call(this, opts) || this;
    }
    Trochoid.prototype.getDefaultStyle = function () {
        return {
            stroke: '#000',
            fill: null
        };
    };
    Trochoid.prototype.getDefaultShape = function () {
        return new TrochoidShape();
    };
    Trochoid.prototype.buildPath = function (ctx, shape) {
        var R = shape.r;
        var r = shape.r0;
        var d = shape.d;
        var offsetX = shape.cx;
        var offsetY = shape.cy;
        var delta = shape.location === 'out' ? 1 : -1;
        var x1;
        var y1;
        var x2;
        var y2;
        if (shape.location && R <= r) {
            return;
        }
        var num = 0;
        var i = 1;
        var theta;
        x1 = (R + delta * r) * cos(0)
            - delta * d * cos(0) + offsetX;
        y1 = (R + delta * r) * sin(0)
            - d * sin(0) + offsetY;
        ctx.moveTo(x1, y1);
        do {
            num++;
        } while ((r * num) % (R + delta * r) !== 0);
        do {
            theta = Math.PI / 180 * i;
            x2 = (R + delta * r) * cos(theta)
                - delta * d * cos((R / r + delta) * theta)
                + offsetX;
            y2 = (R + delta * r) * sin(theta)
                - d * sin((R / r + delta) * theta)
                + offsetY;
            ctx.lineTo(x2, y2);
            i++;
        } while (i <= (r * num) / (R + delta * r) * 360);
    };
    return Trochoid;
}(Path));
Trochoid.prototype.type = 'trochoid';
export default Trochoid;
