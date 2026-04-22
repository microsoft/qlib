import { __extends } from "tslib";
import Path from '../Path.js';
var ArcShape = (function () {
    function ArcShape() {
        this.cx = 0;
        this.cy = 0;
        this.r = 0;
        this.startAngle = 0;
        this.endAngle = Math.PI * 2;
        this.clockwise = true;
    }
    return ArcShape;
}());
export { ArcShape };
var Arc = (function (_super) {
    __extends(Arc, _super);
    function Arc(opts) {
        return _super.call(this, opts) || this;
    }
    Arc.prototype.getDefaultStyle = function () {
        return {
            stroke: '#000',
            fill: null
        };
    };
    Arc.prototype.getDefaultShape = function () {
        return new ArcShape();
    };
    Arc.prototype.buildPath = function (ctx, shape) {
        var x = shape.cx;
        var y = shape.cy;
        var r = Math.max(shape.r, 0);
        var startAngle = shape.startAngle;
        var endAngle = shape.endAngle;
        var clockwise = shape.clockwise;
        var unitX = Math.cos(startAngle);
        var unitY = Math.sin(startAngle);
        ctx.moveTo(unitX * r + x, unitY * r + y);
        ctx.arc(x, y, r, startAngle, endAngle, !clockwise);
    };
    return Arc;
}(Path));
Arc.prototype.type = 'arc';
export default Arc;
