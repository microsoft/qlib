import { __extends } from "tslib";
import Path from '../Path.js';
var DropletShape = (function () {
    function DropletShape() {
        this.cx = 0;
        this.cy = 0;
        this.width = 0;
        this.height = 0;
    }
    return DropletShape;
}());
export { DropletShape };
var Droplet = (function (_super) {
    __extends(Droplet, _super);
    function Droplet(opts) {
        return _super.call(this, opts) || this;
    }
    Droplet.prototype.getDefaultShape = function () {
        return new DropletShape();
    };
    Droplet.prototype.buildPath = function (ctx, shape) {
        var x = shape.cx;
        var y = shape.cy;
        var a = shape.width;
        var b = shape.height;
        ctx.moveTo(x, y + a);
        ctx.bezierCurveTo(x + a, y + a, x + a * 3 / 2, y - a / 3, x, y - b);
        ctx.bezierCurveTo(x - a * 3 / 2, y - a / 3, x - a, y + a, x, y + a);
        ctx.closePath();
    };
    return Droplet;
}(Path));
Droplet.prototype.type = 'droplet';
export default Droplet;
