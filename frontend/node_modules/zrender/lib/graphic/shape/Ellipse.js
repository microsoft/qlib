import { __extends } from "tslib";
import Path from '../Path.js';
var EllipseShape = (function () {
    function EllipseShape() {
        this.cx = 0;
        this.cy = 0;
        this.rx = 0;
        this.ry = 0;
    }
    return EllipseShape;
}());
export { EllipseShape };
var Ellipse = (function (_super) {
    __extends(Ellipse, _super);
    function Ellipse(opts) {
        return _super.call(this, opts) || this;
    }
    Ellipse.prototype.getDefaultShape = function () {
        return new EllipseShape();
    };
    Ellipse.prototype.buildPath = function (ctx, shape) {
        var k = 0.5522848;
        var x = shape.cx;
        var y = shape.cy;
        var a = shape.rx;
        var b = shape.ry;
        var ox = a * k;
        var oy = b * k;
        ctx.moveTo(x - a, y);
        ctx.bezierCurveTo(x - a, y - oy, x - ox, y - b, x, y - b);
        ctx.bezierCurveTo(x + ox, y - b, x + a, y - oy, x + a, y);
        ctx.bezierCurveTo(x + a, y + oy, x + ox, y + b, x, y + b);
        ctx.bezierCurveTo(x - ox, y + b, x - a, y + oy, x - a, y);
        ctx.closePath();
    };
    return Ellipse;
}(Path));
Ellipse.prototype.type = 'ellipse';
export default Ellipse;
