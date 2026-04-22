import { __extends } from "tslib";
import Path from '../Path.js';
var HeartShape = (function () {
    function HeartShape() {
        this.cx = 0;
        this.cy = 0;
        this.width = 0;
        this.height = 0;
    }
    return HeartShape;
}());
export { HeartShape };
var Heart = (function (_super) {
    __extends(Heart, _super);
    function Heart(opts) {
        return _super.call(this, opts) || this;
    }
    Heart.prototype.getDefaultShape = function () {
        return new HeartShape();
    };
    Heart.prototype.buildPath = function (ctx, shape) {
        var x = shape.cx;
        var y = shape.cy;
        var a = shape.width;
        var b = shape.height;
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(x + a / 2, y - b * 2 / 3, x + a * 2, y + b / 3, x, y + b);
        ctx.bezierCurveTo(x - a * 2, y + b / 3, x - a / 2, y - b * 2 / 3, x, y);
    };
    return Heart;
}(Path));
Heart.prototype.type = 'heart';
export default Heart;
