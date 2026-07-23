import { __extends } from "tslib";
import Gradient from './Gradient.js';
var RadialGradient = (function (_super) {
    __extends(RadialGradient, _super);
    function RadialGradient(x, y, r, colorStops, globalCoord) {
        var _this = _super.call(this, colorStops) || this;
        _this.x = x == null ? 0.5 : x;
        _this.y = y == null ? 0.5 : y;
        _this.r = r == null ? 0.5 : r;
        _this.type = 'radial';
        _this.global = globalCoord || false;
        return _this;
    }
    return RadialGradient;
}(Gradient));
export default RadialGradient;
