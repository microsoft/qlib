import { __extends } from "tslib";
import Path from '../Path.js';
import * as polyHelper from '../helper/poly.js';
var PolylineShape = (function () {
    function PolylineShape() {
        this.points = null;
        this.percent = 1;
        this.smooth = 0;
        this.smoothConstraint = null;
    }
    return PolylineShape;
}());
export { PolylineShape };
var Polyline = (function (_super) {
    __extends(Polyline, _super);
    function Polyline(opts) {
        return _super.call(this, opts) || this;
    }
    Polyline.prototype.getDefaultStyle = function () {
        return {
            stroke: '#000',
            fill: null
        };
    };
    Polyline.prototype.getDefaultShape = function () {
        return new PolylineShape();
    };
    Polyline.prototype.buildPath = function (ctx, shape) {
        polyHelper.buildPath(ctx, shape, false);
    };
    return Polyline;
}(Path));
Polyline.prototype.type = 'polyline';
export default Polyline;
