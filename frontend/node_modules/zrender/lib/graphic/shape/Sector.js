import { __extends } from "tslib";
import Path from '../Path.js';
import * as roundSectorHelper from '../helper/roundSector.js';
var SectorShape = (function () {
    function SectorShape() {
        this.cx = 0;
        this.cy = 0;
        this.r0 = 0;
        this.r = 0;
        this.startAngle = 0;
        this.endAngle = Math.PI * 2;
        this.clockwise = true;
        this.cornerRadius = 0;
    }
    return SectorShape;
}());
export { SectorShape };
var Sector = (function (_super) {
    __extends(Sector, _super);
    function Sector(opts) {
        return _super.call(this, opts) || this;
    }
    Sector.prototype.getDefaultShape = function () {
        return new SectorShape();
    };
    Sector.prototype.buildPath = function (ctx, shape) {
        roundSectorHelper.buildPath(ctx, shape);
    };
    Sector.prototype.isZeroArea = function () {
        return this.shape.startAngle === this.shape.endAngle
            || this.shape.r === this.shape.r0;
    };
    return Sector;
}(Path));
Sector.prototype.type = 'sector';
export default Sector;
