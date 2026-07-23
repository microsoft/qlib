import { __extends } from "tslib";
import Path from '../Path.js';
var PI = Math.PI;
var cos = Math.cos;
var sin = Math.sin;
var StarShape = (function () {
    function StarShape() {
        this.cx = 0;
        this.cy = 0;
        this.n = 3;
        this.r = 0;
    }
    return StarShape;
}());
export { StarShape };
var Star = (function (_super) {
    __extends(Star, _super);
    function Star(opts) {
        return _super.call(this, opts) || this;
    }
    Star.prototype.getDefaultShape = function () {
        return new StarShape();
    };
    Star.prototype.buildPath = function (ctx, shape) {
        var n = shape.n;
        if (!n || n < 2) {
            return;
        }
        var x = shape.cx;
        var y = shape.cy;
        var r = shape.r;
        var r0 = shape.r0;
        if (r0 == null) {
            r0 = n > 4
                ? r * cos(2 * PI / n) / cos(PI / n)
                : r / 3;
        }
        var dStep = PI / n;
        var deg = -PI / 2;
        var xStart = x + r * cos(deg);
        var yStart = y + r * sin(deg);
        deg += dStep;
        ctx.moveTo(xStart, yStart);
        for (var i = 0, end = n * 2 - 1, ri = void 0; i < end; i++) {
            ri = i % 2 === 0 ? r0 : r;
            ctx.lineTo(x + ri * cos(deg), y + ri * sin(deg));
            deg += dStep;
        }
        ctx.closePath();
    };
    return Star;
}(Path));
Star.prototype.type = 'star';
export default Star;
