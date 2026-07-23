import { __extends } from "tslib";
import Path from '../Path.js';
var sin = Math.sin;
var cos = Math.cos;
var radian = Math.PI / 180;
var RoseShape = (function () {
    function RoseShape() {
        this.cx = 0;
        this.cy = 0;
        this.r = [];
        this.k = 0;
        this.n = 1;
    }
    return RoseShape;
}());
export { RoseShape };
var Rose = (function (_super) {
    __extends(Rose, _super);
    function Rose(opts) {
        return _super.call(this, opts) || this;
    }
    Rose.prototype.getDefaultStyle = function () {
        return {
            stroke: '#000',
            fill: null
        };
    };
    Rose.prototype.getDefaultShape = function () {
        return new RoseShape();
    };
    Rose.prototype.buildPath = function (ctx, shape) {
        var R = shape.r;
        var k = shape.k;
        var n = shape.n;
        var x0 = shape.cx;
        var y0 = shape.cy;
        var x;
        var y;
        var r;
        ctx.moveTo(x0, y0);
        for (var i = 0, len = R.length; i < len; i++) {
            r = R[i];
            for (var j = 0; j <= 360 * n; j++) {
                x = r
                    * sin(k / n * j % 360 * radian)
                    * cos(j * radian)
                    + x0;
                y = r
                    * sin(k / n * j % 360 * radian)
                    * sin(j * radian)
                    + y0;
                ctx.lineTo(x, y);
            }
        }
    };
    return Rose;
}(Path));
Rose.prototype.type = 'rose';
export default Rose;
