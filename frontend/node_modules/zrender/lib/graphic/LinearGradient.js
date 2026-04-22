import { __extends } from "tslib";
import Gradient from './Gradient.js';
var LinearGradient = (function (_super) {
    __extends(LinearGradient, _super);
    function LinearGradient(x, y, x2, y2, colorStops, globalCoord) {
        var _this = _super.call(this, colorStops) || this;
        _this.x = x == null ? 0 : x;
        _this.y = y == null ? 0 : y;
        _this.x2 = x2 == null ? 1 : x2;
        _this.y2 = y2 == null ? 0 : y2;
        _this.type = 'linear';
        _this.global = globalCoord || false;
        return _this;
    }
    return LinearGradient;
}(Gradient));
export default LinearGradient;
;
