var Point = (function () {
    function Point(x, y) {
        this.x = x || 0;
        this.y = y || 0;
    }
    Point.prototype.copy = function (other) {
        this.x = other.x;
        this.y = other.y;
        return this;
    };
    Point.prototype.clone = function () {
        return new Point(this.x, this.y);
    };
    Point.prototype.set = function (x, y) {
        this.x = x;
        this.y = y;
        return this;
    };
    Point.prototype.equal = function (other) {
        return other.x === this.x && other.y === this.y;
    };
    Point.prototype.add = function (other) {
        this.x += other.x;
        this.y += other.y;
        return this;
    };
    Point.prototype.scale = function (scalar) {
        this.x *= scalar;
        this.y *= scalar;
    };
    Point.prototype.scaleAndAdd = function (other, scalar) {
        this.x += other.x * scalar;
        this.y += other.y * scalar;
    };
    Point.prototype.sub = function (other) {
        this.x -= other.x;
        this.y -= other.y;
        return this;
    };
    Point.prototype.dot = function (other) {
        return this.x * other.x + this.y * other.y;
    };
    Point.prototype.len = function () {
        return Math.sqrt(this.x * this.x + this.y * this.y);
    };
    Point.prototype.lenSquare = function () {
        return this.x * this.x + this.y * this.y;
    };
    Point.prototype.normalize = function () {
        var len = this.len();
        this.x /= len;
        this.y /= len;
        return this;
    };
    Point.prototype.distance = function (other) {
        var dx = this.x - other.x;
        var dy = this.y - other.y;
        return Math.sqrt(dx * dx + dy * dy);
    };
    Point.prototype.distanceSquare = function (other) {
        var dx = this.x - other.x;
        var dy = this.y - other.y;
        return dx * dx + dy * dy;
    };
    Point.prototype.negate = function () {
        this.x = -this.x;
        this.y = -this.y;
        return this;
    };
    Point.prototype.transform = function (m) {
        if (!m) {
            return;
        }
        var x = this.x;
        var y = this.y;
        this.x = m[0] * x + m[2] * y + m[4];
        this.y = m[1] * x + m[3] * y + m[5];
        return this;
    };
    Point.prototype.toArray = function (out) {
        out[0] = this.x;
        out[1] = this.y;
        return out;
    };
    Point.prototype.fromArray = function (input) {
        this.x = input[0];
        this.y = input[1];
    };
    Point.set = function (p, x, y) {
        p.x = x;
        p.y = y;
    };
    Point.copy = function (p, p2) {
        p.x = p2.x;
        p.y = p2.y;
    };
    Point.len = function (p) {
        return Math.sqrt(p.x * p.x + p.y * p.y);
    };
    Point.lenSquare = function (p) {
        return p.x * p.x + p.y * p.y;
    };
    Point.dot = function (p0, p1) {
        return p0.x * p1.x + p0.y * p1.y;
    };
    Point.add = function (out, p0, p1) {
        out.x = p0.x + p1.x;
        out.y = p0.y + p1.y;
    };
    Point.sub = function (out, p0, p1) {
        out.x = p0.x - p1.x;
        out.y = p0.y - p1.y;
    };
    Point.scale = function (out, p0, scalar) {
        out.x = p0.x * scalar;
        out.y = p0.y * scalar;
    };
    Point.scaleAndAdd = function (out, p0, p1, scalar) {
        out.x = p0.x + p1.x * scalar;
        out.y = p0.y + p1.y * scalar;
    };
    Point.lerp = function (out, p0, p1, t) {
        var onet = 1 - t;
        out.x = onet * p0.x + t * p1.x;
        out.y = onet * p0.y + t * p1.y;
    };
    return Point;
}());
export default Point;
