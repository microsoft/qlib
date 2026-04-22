import { __extends } from "tslib";
import Path from './Path.js';
var CompoundPath = (function (_super) {
    __extends(CompoundPath, _super);
    function CompoundPath() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.type = 'compound';
        return _this;
    }
    CompoundPath.prototype._updatePathDirty = function () {
        var paths = this.shape.paths;
        var dirtyPath = this.shapeChanged();
        for (var i = 0; i < paths.length; i++) {
            dirtyPath = dirtyPath || paths[i].shapeChanged();
        }
        if (dirtyPath) {
            this.dirtyShape();
        }
    };
    CompoundPath.prototype.beforeBrush = function () {
        this._updatePathDirty();
        var paths = this.shape.paths || [];
        var scale = this.getGlobalScale();
        for (var i = 0; i < paths.length; i++) {
            if (!paths[i].path) {
                paths[i].createPathProxy();
            }
            paths[i].path.setScale(scale[0], scale[1], paths[i].segmentIgnoreThreshold);
        }
    };
    CompoundPath.prototype.buildPath = function (ctx, shape) {
        var paths = shape.paths || [];
        for (var i = 0; i < paths.length; i++) {
            paths[i].buildPath(ctx, paths[i].shape, true);
        }
    };
    CompoundPath.prototype.afterBrush = function () {
        var paths = this.shape.paths || [];
        for (var i = 0; i < paths.length; i++) {
            paths[i].pathUpdated();
        }
    };
    CompoundPath.prototype.getBoundingRect = function () {
        this._updatePathDirty.call(this);
        return Path.prototype.getBoundingRect.call(this);
    };
    return CompoundPath;
}(Path));
export default CompoundPath;
