import * as vec2 from './vector.js';
import BoundingRect from './BoundingRect.js';
import { devicePixelRatio as dpr } from '../config.js';
import { fromLine, fromCubic, fromQuadratic, fromArc } from './bbox.js';
import { cubicLength, cubicSubdivide, quadraticLength, quadraticSubdivide } from './curve.js';
var CMD = {
    M: 1,
    L: 2,
    C: 3,
    Q: 4,
    A: 5,
    Z: 6,
    R: 7
};
var tmpOutX = [];
var tmpOutY = [];
var min = [];
var max = [];
var min2 = [];
var max2 = [];
var mathMin = Math.min;
var mathMax = Math.max;
var mathCos = Math.cos;
var mathSin = Math.sin;
var mathAbs = Math.abs;
var PI = Math.PI;
var PI2 = PI * 2;
var hasTypedArray = typeof Float32Array !== 'undefined';
var tmpAngles = [];
function modPI2(radian) {
    var n = Math.round(radian / PI * 1e8) / 1e8;
    return (n % 2) * PI;
}
export function normalizeArcAngles(angles, anticlockwise) {
    var newStartAngle = modPI2(angles[0]);
    if (newStartAngle < 0) {
        newStartAngle += PI2;
    }
    var delta = newStartAngle - angles[0];
    var newEndAngle = angles[1];
    newEndAngle += delta;
    if (!anticlockwise && newEndAngle - newStartAngle >= PI2) {
        newEndAngle = newStartAngle + PI2;
    }
    else if (anticlockwise && newStartAngle - newEndAngle >= PI2) {
        newEndAngle = newStartAngle - PI2;
    }
    else if (!anticlockwise && newStartAngle > newEndAngle) {
        newEndAngle = newStartAngle + (PI2 - modPI2(newStartAngle - newEndAngle));
    }
    else if (anticlockwise && newStartAngle < newEndAngle) {
        newEndAngle = newStartAngle - (PI2 - modPI2(newEndAngle - newStartAngle));
    }
    angles[0] = newStartAngle;
    angles[1] = newEndAngle;
}
var PathProxy = (function () {
    function PathProxy(notSaveData) {
        this.dpr = 1;
        this._xi = 0;
        this._yi = 0;
        this._x0 = 0;
        this._y0 = 0;
        this._len = 0;
        if (notSaveData) {
            this._saveData = false;
        }
        if (this._saveData) {
            this.data = [];
        }
    }
    PathProxy.prototype.increaseVersion = function () {
        this._version++;
    };
    PathProxy.prototype.getVersion = function () {
        return this._version;
    };
    PathProxy.prototype.setScale = function (sx, sy, segmentIgnoreThreshold) {
        segmentIgnoreThreshold = segmentIgnoreThreshold || 0;
        if (segmentIgnoreThreshold > 0) {
            this._ux = mathAbs(segmentIgnoreThreshold / dpr / sx) || 0;
            this._uy = mathAbs(segmentIgnoreThreshold / dpr / sy) || 0;
        }
    };
    PathProxy.prototype.setDPR = function (dpr) {
        this.dpr = dpr;
    };
    PathProxy.prototype.setContext = function (ctx) {
        this._ctx = ctx;
    };
    PathProxy.prototype.getContext = function () {
        return this._ctx;
    };
    PathProxy.prototype.beginPath = function () {
        this._ctx && this._ctx.beginPath();
        this.reset();
        return this;
    };
    PathProxy.prototype.reset = function () {
        if (this._saveData) {
            this._len = 0;
        }
        if (this._pathSegLen) {
            this._pathSegLen = null;
            this._pathLen = 0;
        }
        this._version++;
    };
    PathProxy.prototype.moveTo = function (x, y) {
        this._drawPendingPt();
        this.addData(CMD.M, x, y);
        this._ctx && this._ctx.moveTo(x, y);
        this._x0 = x;
        this._y0 = y;
        this._xi = x;
        this._yi = y;
        return this;
    };
    PathProxy.prototype.lineTo = function (x, y) {
        var dx = mathAbs(x - this._xi);
        var dy = mathAbs(y - this._yi);
        var exceedUnit = dx > this._ux || dy > this._uy;
        this.addData(CMD.L, x, y);
        if (this._ctx && exceedUnit) {
            this._ctx.lineTo(x, y);
        }
        if (exceedUnit) {
            this._xi = x;
            this._yi = y;
            this._pendingPtDist = 0;
        }
        else {
            var d2 = dx * dx + dy * dy;
            if (d2 > this._pendingPtDist) {
                this._pendingPtX = x;
                this._pendingPtY = y;
                this._pendingPtDist = d2;
            }
        }
        return this;
    };
    PathProxy.prototype.bezierCurveTo = function (x1, y1, x2, y2, x3, y3) {
        this._drawPendingPt();
        this.addData(CMD.C, x1, y1, x2, y2, x3, y3);
        if (this._ctx) {
            this._ctx.bezierCurveTo(x1, y1, x2, y2, x3, y3);
        }
        this._xi = x3;
        this._yi = y3;
        return this;
    };
    PathProxy.prototype.quadraticCurveTo = function (x1, y1, x2, y2) {
        this._drawPendingPt();
        this.addData(CMD.Q, x1, y1, x2, y2);
        if (this._ctx) {
            this._ctx.quadraticCurveTo(x1, y1, x2, y2);
        }
        this._xi = x2;
        this._yi = y2;
        return this;
    };
    PathProxy.prototype.arc = function (cx, cy, r, startAngle, endAngle, anticlockwise) {
        this._drawPendingPt();
        tmpAngles[0] = startAngle;
        tmpAngles[1] = endAngle;
        normalizeArcAngles(tmpAngles, anticlockwise);
        startAngle = tmpAngles[0];
        endAngle = tmpAngles[1];
        var delta = endAngle - startAngle;
        this.addData(CMD.A, cx, cy, r, r, startAngle, delta, 0, anticlockwise ? 0 : 1);
        this._ctx && this._ctx.arc(cx, cy, r, startAngle, endAngle, anticlockwise);
        this._xi = mathCos(endAngle) * r + cx;
        this._yi = mathSin(endAngle) * r + cy;
        return this;
    };
    PathProxy.prototype.arcTo = function (x1, y1, x2, y2, radius) {
        this._drawPendingPt();
        if (this._ctx) {
            this._ctx.arcTo(x1, y1, x2, y2, radius);
        }
        return this;
    };
    PathProxy.prototype.rect = function (x, y, w, h) {
        this._drawPendingPt();
        this._ctx && this._ctx.rect(x, y, w, h);
        this.addData(CMD.R, x, y, w, h);
        return this;
    };
    PathProxy.prototype.closePath = function () {
        this._drawPendingPt();
        this.addData(CMD.Z);
        var ctx = this._ctx;
        var x0 = this._x0;
        var y0 = this._y0;
        if (ctx) {
            ctx.closePath();
        }
        this._xi = x0;
        this._yi = y0;
        return this;
    };
    PathProxy.prototype.fill = function (ctx) {
        ctx && ctx.fill();
        this.toStatic();
    };
    PathProxy.prototype.stroke = function (ctx) {
        ctx && ctx.stroke();
        this.toStatic();
    };
    PathProxy.prototype.len = function () {
        return this._len;
    };
    PathProxy.prototype.setData = function (data) {
        var len = data.length;
        if (!(this.data && this.data.length === len) && hasTypedArray) {
            this.data = new Float32Array(len);
        }
        for (var i = 0; i < len; i++) {
            this.data[i] = data[i];
        }
        this._len = len;
    };
    PathProxy.prototype.appendPath = function (path) {
        if (!(path instanceof Array)) {
            path = [path];
        }
        var len = path.length;
        var appendSize = 0;
        var offset = this._len;
        for (var i = 0; i < len; i++) {
            appendSize += path[i].len();
        }
        if (hasTypedArray && (this.data instanceof Float32Array)) {
            this.data = new Float32Array(offset + appendSize);
        }
        for (var i = 0; i < len; i++) {
            var appendPathData = path[i].data;
            for (var k = 0; k < appendPathData.length; k++) {
                this.data[offset++] = appendPathData[k];
            }
        }
        this._len = offset;
    };
    PathProxy.prototype.addData = function (cmd, a, b, c, d, e, f, g, h) {
        if (!this._saveData) {
            return;
        }
        var data = this.data;
        if (this._len + arguments.length > data.length) {
            this._expandData();
            data = this.data;
        }
        for (var i = 0; i < arguments.length; i++) {
            data[this._len++] = arguments[i];
        }
    };
    PathProxy.prototype._drawPendingPt = function () {
        if (this._pendingPtDist > 0) {
            this._ctx && this._ctx.lineTo(this._pendingPtX, this._pendingPtY);
            this._pendingPtDist = 0;
        }
    };
    PathProxy.prototype._expandData = function () {
        if (!(this.data instanceof Array)) {
            var newData = [];
            for (var i = 0; i < this._len; i++) {
                newData[i] = this.data[i];
            }
            this.data = newData;
        }
    };
    PathProxy.prototype.toStatic = function () {
        if (!this._saveData) {
            return;
        }
        this._drawPendingPt();
        var data = this.data;
        if (data instanceof Array) {
            data.length = this._len;
            if (hasTypedArray && this._len > 11) {
                this.data = new Float32Array(data);
            }
        }
    };
    PathProxy.prototype.getBoundingRect = function () {
        min[0] = min[1] = min2[0] = min2[1] = Number.MAX_VALUE;
        max[0] = max[1] = max2[0] = max2[1] = -Number.MAX_VALUE;
        var data = this.data;
        var xi = 0;
        var yi = 0;
        var x0 = 0;
        var y0 = 0;
        var i;
        for (i = 0; i < this._len;) {
            var cmd = data[i++];
            var isFirst = i === 1;
            if (isFirst) {
                xi = data[i];
                yi = data[i + 1];
                x0 = xi;
                y0 = yi;
            }
            switch (cmd) {
                case CMD.M:
                    xi = x0 = data[i++];
                    yi = y0 = data[i++];
                    min2[0] = x0;
                    min2[1] = y0;
                    max2[0] = x0;
                    max2[1] = y0;
                    break;
                case CMD.L:
                    fromLine(xi, yi, data[i], data[i + 1], min2, max2);
                    xi = data[i++];
                    yi = data[i++];
                    break;
                case CMD.C:
                    fromCubic(xi, yi, data[i++], data[i++], data[i++], data[i++], data[i], data[i + 1], min2, max2);
                    xi = data[i++];
                    yi = data[i++];
                    break;
                case CMD.Q:
                    fromQuadratic(xi, yi, data[i++], data[i++], data[i], data[i + 1], min2, max2);
                    xi = data[i++];
                    yi = data[i++];
                    break;
                case CMD.A:
                    var cx = data[i++];
                    var cy = data[i++];
                    var rx = data[i++];
                    var ry = data[i++];
                    var startAngle = data[i++];
                    var endAngle = data[i++] + startAngle;
                    i += 1;
                    var anticlockwise = !data[i++];
                    if (isFirst) {
                        x0 = mathCos(startAngle) * rx + cx;
                        y0 = mathSin(startAngle) * ry + cy;
                    }
                    fromArc(cx, cy, rx, ry, startAngle, endAngle, anticlockwise, min2, max2);
                    xi = mathCos(endAngle) * rx + cx;
                    yi = mathSin(endAngle) * ry + cy;
                    break;
                case CMD.R:
                    x0 = xi = data[i++];
                    y0 = yi = data[i++];
                    var width = data[i++];
                    var height = data[i++];
                    fromLine(x0, y0, x0 + width, y0 + height, min2, max2);
                    break;
                case CMD.Z:
                    xi = x0;
                    yi = y0;
                    break;
            }
            vec2.min(min, min, min2);
            vec2.max(max, max, max2);
        }
        if (i === 0) {
            min[0] = min[1] = max[0] = max[1] = 0;
        }
        return new BoundingRect(min[0], min[1], max[0] - min[0], max[1] - min[1]);
    };
    PathProxy.prototype._calculateLength = function () {
        var data = this.data;
        var len = this._len;
        var ux = this._ux;
        var uy = this._uy;
        var xi = 0;
        var yi = 0;
        var x0 = 0;
        var y0 = 0;
        if (!this._pathSegLen) {
            this._pathSegLen = [];
        }
        var pathSegLen = this._pathSegLen;
        var pathTotalLen = 0;
        var segCount = 0;
        for (var i = 0; i < len;) {
            var cmd = data[i++];
            var isFirst = i === 1;
            if (isFirst) {
                xi = data[i];
                yi = data[i + 1];
                x0 = xi;
                y0 = yi;
            }
            var l = -1;
            switch (cmd) {
                case CMD.M:
                    xi = x0 = data[i++];
                    yi = y0 = data[i++];
                    break;
                case CMD.L: {
                    var x2 = data[i++];
                    var y2 = data[i++];
                    var dx = x2 - xi;
                    var dy = y2 - yi;
                    if (mathAbs(dx) > ux || mathAbs(dy) > uy || i === len - 1) {
                        l = Math.sqrt(dx * dx + dy * dy);
                        xi = x2;
                        yi = y2;
                    }
                    break;
                }
                case CMD.C: {
                    var x1 = data[i++];
                    var y1 = data[i++];
                    var x2 = data[i++];
                    var y2 = data[i++];
                    var x3 = data[i++];
                    var y3 = data[i++];
                    l = cubicLength(xi, yi, x1, y1, x2, y2, x3, y3, 10);
                    xi = x3;
                    yi = y3;
                    break;
                }
                case CMD.Q: {
                    var x1 = data[i++];
                    var y1 = data[i++];
                    var x2 = data[i++];
                    var y2 = data[i++];
                    l = quadraticLength(xi, yi, x1, y1, x2, y2, 10);
                    xi = x2;
                    yi = y2;
                    break;
                }
                case CMD.A:
                    var cx = data[i++];
                    var cy = data[i++];
                    var rx = data[i++];
                    var ry = data[i++];
                    var startAngle = data[i++];
                    var delta = data[i++];
                    var endAngle = delta + startAngle;
                    i += 1;
                    if (isFirst) {
                        x0 = mathCos(startAngle) * rx + cx;
                        y0 = mathSin(startAngle) * ry + cy;
                    }
                    l = mathMax(rx, ry) * mathMin(PI2, Math.abs(delta));
                    xi = mathCos(endAngle) * rx + cx;
                    yi = mathSin(endAngle) * ry + cy;
                    break;
                case CMD.R: {
                    x0 = xi = data[i++];
                    y0 = yi = data[i++];
                    var width = data[i++];
                    var height = data[i++];
                    l = width * 2 + height * 2;
                    break;
                }
                case CMD.Z: {
                    var dx = x0 - xi;
                    var dy = y0 - yi;
                    l = Math.sqrt(dx * dx + dy * dy);
                    xi = x0;
                    yi = y0;
                    break;
                }
            }
            if (l >= 0) {
                pathSegLen[segCount++] = l;
                pathTotalLen += l;
            }
        }
        this._pathLen = pathTotalLen;
        return pathTotalLen;
    };
    PathProxy.prototype.rebuildPath = function (ctx, percent) {
        var d = this.data;
        var ux = this._ux;
        var uy = this._uy;
        var len = this._len;
        var x0;
        var y0;
        var xi;
        var yi;
        var x;
        var y;
        var drawPart = percent < 1;
        var pathSegLen;
        var pathTotalLen;
        var accumLength = 0;
        var segCount = 0;
        var displayedLength;
        var pendingPtDist = 0;
        var pendingPtX;
        var pendingPtY;
        if (drawPart) {
            if (!this._pathSegLen) {
                this._calculateLength();
            }
            pathSegLen = this._pathSegLen;
            pathTotalLen = this._pathLen;
            displayedLength = percent * pathTotalLen;
            if (!displayedLength) {
                return;
            }
        }
        lo: for (var i = 0; i < len;) {
            var cmd = d[i++];
            var isFirst = i === 1;
            if (isFirst) {
                xi = d[i];
                yi = d[i + 1];
                x0 = xi;
                y0 = yi;
            }
            if (cmd !== CMD.L && pendingPtDist > 0) {
                ctx.lineTo(pendingPtX, pendingPtY);
                pendingPtDist = 0;
            }
            switch (cmd) {
                case CMD.M:
                    x0 = xi = d[i++];
                    y0 = yi = d[i++];
                    ctx.moveTo(xi, yi);
                    break;
                case CMD.L: {
                    x = d[i++];
                    y = d[i++];
                    var dx = mathAbs(x - xi);
                    var dy = mathAbs(y - yi);
                    if (dx > ux || dy > uy) {
                        if (drawPart) {
                            var l = pathSegLen[segCount++];
                            if (accumLength + l > displayedLength) {
                                var t = (displayedLength - accumLength) / l;
                                ctx.lineTo(xi * (1 - t) + x * t, yi * (1 - t) + y * t);
                                break lo;
                            }
                            accumLength += l;
                        }
                        ctx.lineTo(x, y);
                        xi = x;
                        yi = y;
                        pendingPtDist = 0;
                    }
                    else {
                        var d2 = dx * dx + dy * dy;
                        if (d2 > pendingPtDist) {
                            pendingPtX = x;
                            pendingPtY = y;
                            pendingPtDist = d2;
                        }
                    }
                    break;
                }
                case CMD.C: {
                    var x1 = d[i++];
                    var y1 = d[i++];
                    var x2 = d[i++];
                    var y2 = d[i++];
                    var x3 = d[i++];
                    var y3 = d[i++];
                    if (drawPart) {
                        var l = pathSegLen[segCount++];
                        if (accumLength + l > displayedLength) {
                            var t = (displayedLength - accumLength) / l;
                            cubicSubdivide(xi, x1, x2, x3, t, tmpOutX);
                            cubicSubdivide(yi, y1, y2, y3, t, tmpOutY);
                            ctx.bezierCurveTo(tmpOutX[1], tmpOutY[1], tmpOutX[2], tmpOutY[2], tmpOutX[3], tmpOutY[3]);
                            break lo;
                        }
                        accumLength += l;
                    }
                    ctx.bezierCurveTo(x1, y1, x2, y2, x3, y3);
                    xi = x3;
                    yi = y3;
                    break;
                }
                case CMD.Q: {
                    var x1 = d[i++];
                    var y1 = d[i++];
                    var x2 = d[i++];
                    var y2 = d[i++];
                    if (drawPart) {
                        var l = pathSegLen[segCount++];
                        if (accumLength + l > displayedLength) {
                            var t = (displayedLength - accumLength) / l;
                            quadraticSubdivide(xi, x1, x2, t, tmpOutX);
                            quadraticSubdivide(yi, y1, y2, t, tmpOutY);
                            ctx.quadraticCurveTo(tmpOutX[1], tmpOutY[1], tmpOutX[2], tmpOutY[2]);
                            break lo;
                        }
                        accumLength += l;
                    }
                    ctx.quadraticCurveTo(x1, y1, x2, y2);
                    xi = x2;
                    yi = y2;
                    break;
                }
                case CMD.A:
                    var cx = d[i++];
                    var cy = d[i++];
                    var rx = d[i++];
                    var ry = d[i++];
                    var startAngle = d[i++];
                    var delta = d[i++];
                    var psi = d[i++];
                    var anticlockwise = !d[i++];
                    var r = (rx > ry) ? rx : ry;
                    var isEllipse = mathAbs(rx - ry) > 1e-3;
                    var endAngle = startAngle + delta;
                    var breakBuild = false;
                    if (drawPart) {
                        var l = pathSegLen[segCount++];
                        if (accumLength + l > displayedLength) {
                            endAngle = startAngle + delta * (displayedLength - accumLength) / l;
                            breakBuild = true;
                        }
                        accumLength += l;
                    }
                    if (isEllipse && ctx.ellipse) {
                        ctx.ellipse(cx, cy, rx, ry, psi, startAngle, endAngle, anticlockwise);
                    }
                    else {
                        ctx.arc(cx, cy, r, startAngle, endAngle, anticlockwise);
                    }
                    if (breakBuild) {
                        break lo;
                    }
                    if (isFirst) {
                        x0 = mathCos(startAngle) * rx + cx;
                        y0 = mathSin(startAngle) * ry + cy;
                    }
                    xi = mathCos(endAngle) * rx + cx;
                    yi = mathSin(endAngle) * ry + cy;
                    break;
                case CMD.R:
                    x0 = xi = d[i];
                    y0 = yi = d[i + 1];
                    x = d[i++];
                    y = d[i++];
                    var width = d[i++];
                    var height = d[i++];
                    if (drawPart) {
                        var l = pathSegLen[segCount++];
                        if (accumLength + l > displayedLength) {
                            var d_1 = displayedLength - accumLength;
                            ctx.moveTo(x, y);
                            ctx.lineTo(x + mathMin(d_1, width), y);
                            d_1 -= width;
                            if (d_1 > 0) {
                                ctx.lineTo(x + width, y + mathMin(d_1, height));
                            }
                            d_1 -= height;
                            if (d_1 > 0) {
                                ctx.lineTo(x + mathMax(width - d_1, 0), y + height);
                            }
                            d_1 -= width;
                            if (d_1 > 0) {
                                ctx.lineTo(x, y + mathMax(height - d_1, 0));
                            }
                            break lo;
                        }
                        accumLength += l;
                    }
                    ctx.rect(x, y, width, height);
                    break;
                case CMD.Z:
                    if (drawPart) {
                        var l = pathSegLen[segCount++];
                        if (accumLength + l > displayedLength) {
                            var t = (displayedLength - accumLength) / l;
                            ctx.lineTo(xi * (1 - t) + x0 * t, yi * (1 - t) + y0 * t);
                            break lo;
                        }
                        accumLength += l;
                    }
                    ctx.closePath();
                    xi = x0;
                    yi = y0;
            }
        }
    };
    PathProxy.prototype.clone = function () {
        var newProxy = new PathProxy();
        var data = this.data;
        newProxy.data = data.slice ? data.slice()
            : Array.prototype.slice.call(data);
        newProxy._len = this._len;
        return newProxy;
    };
    PathProxy.CMD = CMD;
    PathProxy.initDefaultProps = (function () {
        var proto = PathProxy.prototype;
        proto._saveData = true;
        proto._ux = 0;
        proto._uy = 0;
        proto._pendingPtDist = 0;
        proto._version = 0;
    })();
    return PathProxy;
}());
export default PathProxy;
