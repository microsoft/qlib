import { __extends } from "tslib";
import Path from '../Path.js';
var CircleShape = (function () {
    function CircleShape() {
        this.cx = 0;
        this.cy = 0;
        this.r = 0;
    }
    return CircleShape;
}());
export { CircleShape };
var Circle = (function (_super) {
    __extends(Circle, _super);
    function Circle(opts) {
        return _super.call(this, opts) || this;
    }
    Circle.prototype.getDefaultShape = function () {
        return new CircleShape();
    };
    Circle.prototype.buildPath = function (ctx, shape) {
        ctx.moveTo(shape.cx + shape.r, shape.cy);
        ctx.arc(shape.cx, shape.cy, shape.r, 0, Math.PI * 2);
    };
    return Circle;
}(Path));
;
Circle.prototype.type = 'circle';
export default Circle;
