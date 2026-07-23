import { __extends } from "tslib";
import Path from '../Path.js';
var PI = Math.PI;
var sin = Math.sin;
var cos = Math.cos;
var IsogonShape = (function () {
    function IsogonShape() {
        this.x = 0;
        this.y = 0;
        this.r = 0;
        this.n = 0;
    }
    return IsogonShape;
}());
export { IsogonShape };
var Isogon = (function (_super) {
    __extends(Isogon, _super);
    function Isogon(opts) {
        return _super.call(this, opts) || this;
    }
    Isogon.prototype.getDefaultShape = function () {
        return new IsogonShape();
    };
    Isogon.prototype.buildPath = function (ctx, shape) {
        var n = shape.n;
        if (!n || n < 2) {
            return;
        }
        var x = shape.x;
        var y = shape.y;
        var r = shape.r;
        var dStep = 2 * PI / n;
        var deg = -PI / 2;
        ctx.moveTo(x + r * cos(deg), y + r * sin(deg));
        for (var i = 0, end = n - 1; i < end; i++) {
            deg += dStep;
            ctx.lineTo(x + r * cos(deg), y + r * sin(deg));
        }
        ctx.closePath();
        return;
    };
    return Isogon;
}(Path));
Isogon.prototype.type = 'isogon';
export default Isogon;
