import Point from './Point.js';
var extent = [0, 0];
var extent2 = [0, 0];
var minTv = new Point();
var maxTv = new Point();
var OrientedBoundingRect = (function () {
    function OrientedBoundingRect(rect, transform) {
        this._corners = [];
        this._axes = [];
        this._origin = [0, 0];
        for (var i = 0; i < 4; i++) {
            this._corners[i] = new Point();
        }
        for (var i = 0; i < 2; i++) {
            this._axes[i] = new Point();
        }
        if (rect) {
            this.fromBoundingRect(rect, transform);
        }
    }
    OrientedBoundingRect.prototype.fromBoundingRect = function (rect, transform) {
        var corners = this._corners;
        var axes = this._axes;
        var x = rect.x;
        var y = rect.y;
        var x2 = x + rect.width;
        var y2 = y + rect.height;
        corners[0].set(x, y);
        corners[1].set(x2, y);
        corners[2].set(x2, y2);
        corners[3].set(x, y2);
        if (transform) {
            for (var i = 0; i < 4; i++) {
                corners[i].transform(transform);
            }
        }
        Point.sub(axes[0], corners[1], corners[0]);
        Point.sub(axes[1], corners[3], corners[0]);
        axes[0].normalize();
        axes[1].normalize();
        for (var i = 0; i < 2; i++) {
            this._origin[i] = axes[i].dot(corners[0]);
        }
    };
    OrientedBoundingRect.prototype.intersect = function (other, mtv) {
        var overlapped = true;
        var noMtv = !mtv;
        minTv.set(Infinity, Infinity);
        maxTv.set(0, 0);
        if (!this._intersectCheckOneSide(this, other, minTv, maxTv, noMtv, 1)) {
            overlapped = false;
            if (noMtv) {
                return overlapped;
            }
        }
        if (!this._intersectCheckOneSide(other, this, minTv, maxTv, noMtv, -1)) {
            overlapped = false;
            if (noMtv) {
                return overlapped;
            }
        }
        if (!noMtv) {
            Point.copy(mtv, overlapped ? minTv : maxTv);
        }
        return overlapped;
    };
    OrientedBoundingRect.prototype._intersectCheckOneSide = function (self, other, minTv, maxTv, noMtv, inverse) {
        var overlapped = true;
        for (var i = 0; i < 2; i++) {
            var axis = this._axes[i];
            this._getProjMinMaxOnAxis(i, self._corners, extent);
            this._getProjMinMaxOnAxis(i, other._corners, extent2);
            if (extent[1] < extent2[0] || extent[0] > extent2[1]) {
                overlapped = false;
                if (noMtv) {
                    return overlapped;
                }
                var dist0 = Math.abs(extent2[0] - extent[1]);
                var dist1 = Math.abs(extent[0] - extent2[1]);
                if (Math.min(dist0, dist1) > maxTv.len()) {
                    if (dist0 < dist1) {
                        Point.scale(maxTv, axis, -dist0 * inverse);
                    }
                    else {
                        Point.scale(maxTv, axis, dist1 * inverse);
                    }
                }
            }
            else if (minTv) {
                var dist0 = Math.abs(extent2[0] - extent[1]);
                var dist1 = Math.abs(extent[0] - extent2[1]);
                if (Math.min(dist0, dist1) < minTv.len()) {
                    if (dist0 < dist1) {
                        Point.scale(minTv, axis, dist0 * inverse);
                    }
                    else {
                        Point.scale(minTv, axis, -dist1 * inverse);
                    }
                }
            }
        }
        return overlapped;
    };
    OrientedBoundingRect.prototype._getProjMinMaxOnAxis = function (dim, corners, out) {
        var axis = this._axes[dim];
        var origin = this._origin;
        var proj = corners[0].dot(axis) + origin[dim];
        var min = proj;
        var max = proj;
        for (var i = 1; i < corners.length; i++) {
            var proj_1 = corners[i].dot(axis) + origin[dim];
            min = Math.min(proj_1, min);
            max = Math.max(proj_1, max);
        }
        out[0] = min;
        out[1] = max;
    };
    return OrientedBoundingRect;
}());
export default OrientedBoundingRect;
