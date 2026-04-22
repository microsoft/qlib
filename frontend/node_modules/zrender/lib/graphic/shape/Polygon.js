import { __extends } from "tslib";
import Path from '../Path.js';
import * as polyHelper from '../helper/poly.js';
var PolygonShape = (function () {
    function PolygonShape() {
        this.points = null;
        this.smooth = 0;
        this.smoothConstraint = null;
    }
    return PolygonShape;
}());
export { PolygonShape };
var Polygon = (function (_super) {
    __extends(Polygon, _super);
    function Polygon(opts) {
        return _super.call(this, opts) || this;
    }
    Polygon.prototype.getDefaultShape = function () {
        return new PolygonShape();
    };
    Polygon.prototype.buildPath = function (ctx, shape) {
        polyHelper.buildPath(ctx, shape, true);
    };
    return Polygon;
}(Path));
;
Polygon.prototype.type = 'polygon';
export default Polygon;
