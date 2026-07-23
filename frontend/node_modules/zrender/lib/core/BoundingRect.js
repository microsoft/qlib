import * as matrix from './matrix.js';
import Point from './Point.js';
var mathMin = Math.min;
var mathMax = Math.max;
var lt = new Point();
var rb = new Point();
var lb = new Point();
var rt = new Point();
var minTv = new Point();
var maxTv = new Point();
var BoundingRect = (function () {
    function BoundingRect(x, y, width, height) {
        if (width < 0) {
            x = x + width;
            width = -width;
        }
        if (height < 0) {
            y = y + height;
            height = -height;
        }
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
    }
    BoundingRect.prototype.union = function (other) {
        var x = mathMin(other.x, this.x);
        var y = mathMin(other.y, this.y);
        if (isFinite(this.x) && isFinite(this.width)) {
            this.width = mathMax(other.x + other.width, this.x + this.width) - x;
        }
        else {
            this.width = other.width;
        }
        if (isFinite(this.y) && isFinite(this.height)) {
            this.height = mathMax(other.y + other.height, this.y + this.height) - y;
        }
        else {
            this.height = other.height;
        }
        this.x = x;
        this.y = y;
    };
    BoundingRect.prototype.applyTransform = function (m) {
        BoundingRect.applyTransform(this, this, m);
    };
    BoundingRect.prototype.calculateTransform = function (b) {
        var a = this;
        var sx = b.width / a.width;
        var sy = b.height / a.height;
        var m = matrix.create();
        matrix.translate(m, m, [-a.x, -a.y]);
        matrix.scale(m, m, [sx, sy]);
        matrix.translate(m, m, [b.x, b.y]);
        return m;
    };
    BoundingRect.prototype.intersect = function (b, mtv) {
        if (!b) {
            return false;
        }
        if (!(b instanceof BoundingRect)) {
            b = BoundingRect.create(b);
        }
        var a = this;
        var ax0 = a.x;
        var ax1 = a.x + a.width;
        var ay0 = a.y;
        var ay1 = a.y + a.height;
        var bx0 = b.x;
        var bx1 = b.x + b.width;
        var by0 = b.y;
        var by1 = b.y + b.height;
        var overlap = !(ax1 < bx0 || bx1 < ax0 || ay1 < by0 || by1 < ay0);
        if (mtv) {
            var dMin = Infinity;
            var dMax = 0;
            var d0 = Math.abs(ax1 - bx0);
            var d1 = Math.abs(bx1 - ax0);
            var d2 = Math.abs(ay1 - by0);
            var d3 = Math.abs(by1 - ay0);
            var dx = Math.min(d0, d1);
            var dy = Math.min(d2, d3);
            if (ax1 < bx0 || bx1 < ax0) {
                if (dx > dMax) {
                    dMax = dx;
                    if (d0 < d1) {
                        Point.set(maxTv, -d0, 0);
                    }
                    else {
                        Point.set(maxTv, d1, 0);
                    }
                }
            }
            else {
                if (dx < dMin) {
                    dMin = dx;
                    if (d0 < d1) {
                        Point.set(minTv, d0, 0);
                    }
                    else {
                        Point.set(minTv, -d1, 0);
                    }
                }
            }
            if (ay1 < by0 || by1 < ay0) {
                if (dy > dMax) {
                    dMax = dy;
                    if (d2 < d3) {
                        Point.set(maxTv, 0, -d2);
                    }
                    else {
                        Point.set(maxTv, 0, d3);
                    }
                }
            }
            else {
                if (dx < dMin) {
                    dMin = dx;
                    if (d2 < d3) {
                        Point.set(minTv, 0, d2);
                    }
                    else {
                        Point.set(minTv, 0, -d3);
                    }
                }
            }
        }
        if (mtv) {
            Point.copy(mtv, overlap ? minTv : maxTv);
        }
        return overlap;
    };
    BoundingRect.prototype.contain = function (x, y) {
        var rect = this;
        return x >= rect.x
            && x <= (rect.x + rect.width)
            && y >= rect.y
            && y <= (rect.y + rect.height);
    };
    BoundingRect.prototype.clone = function () {
        return new BoundingRect(this.x, this.y, this.width, this.height);
    };
    BoundingRect.prototype.copy = function (other) {
        BoundingRect.copy(this, other);
    };
    BoundingRect.prototype.plain = function () {
        return {
            x: this.x,
            y: this.y,
            width: this.width,
            height: this.height
        };
    };
    BoundingRect.prototype.isFinite = function () {
        return isFinite(this.x)
            && isFinite(this.y)
            && isFinite(this.width)
            && isFinite(this.height);
    };
    BoundingRect.prototype.isZero = function () {
        return this.width === 0 || this.height === 0;
    };
    BoundingRect.create = function (rect) {
        return new BoundingRect(rect.x, rect.y, rect.width, rect.height);
    };
    BoundingRect.copy = function (target, source) {
        target.x = source.x;
        target.y = source.y;
        target.width = source.width;
        target.height = source.height;
    };
    BoundingRect.applyTransform = function (target, source, m) {
        if (!m) {
            if (target !== source) {
                BoundingRect.copy(target, source);
            }
            return;
        }
        if (m[1] < 1e-5 && m[1] > -1e-5 && m[2] < 1e-5 && m[2] > -1e-5) {
            var sx = m[0];
            var sy = m[3];
            var tx = m[4];
            var ty = m[5];
            target.x = source.x * sx + tx;
            target.y = source.y * sy + ty;
            target.width = source.width * sx;
            target.height = source.height * sy;
            if (target.width < 0) {
                target.x += target.width;
                target.width = -target.width;
            }
            if (target.height < 0) {
                target.y += target.height;
                target.height = -target.height;
            }
            return;
        }
        lt.x = lb.x = source.x;
        lt.y = rt.y = source.y;
        rb.x = rt.x = source.x + source.width;
        rb.y = lb.y = source.y + source.height;
        lt.transform(m);
        rt.transform(m);
        rb.transform(m);
        lb.transform(m);
        target.x = mathMin(lt.x, rb.x, lb.x, rt.x);
        target.y = mathMin(lt.y, rb.y, lb.y, rt.y);
        var maxX = mathMax(lt.x, rb.x, lb.x, rt.x);
        var maxY = mathMax(lt.y, rb.y, lb.y, rt.y);
        target.width = maxX - target.x;
        target.height = maxY - target.y;
    };
    return BoundingRect;
}());
export default BoundingRect;
