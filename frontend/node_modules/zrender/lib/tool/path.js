import { __extends } from "tslib";
import Path from '../graphic/Path.js';
import PathProxy from '../core/PathProxy.js';
import transformPath from './transformPath.js';
import { extend } from '../core/util.js';
var mathSqrt = Math.sqrt;
var mathSin = Math.sin;
var mathCos = Math.cos;
var PI = Math.PI;
function vMag(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
}
;
function vRatio(u, v) {
    return (u[0] * v[0] + u[1] * v[1]) / (vMag(u) * vMag(v));
}
;
function vAngle(u, v) {
    return (u[0] * v[1] < u[1] * v[0] ? -1 : 1)
        * Math.acos(vRatio(u, v));
}
;
function processArc(x1, y1, x2, y2, fa, fs, rx, ry, psiDeg, cmd, path) {
    var psi = psiDeg * (PI / 180.0);
    var xp = mathCos(psi) * (x1 - x2) / 2.0
        + mathSin(psi) * (y1 - y2) / 2.0;
    var yp = -1 * mathSin(psi) * (x1 - x2) / 2.0
        + mathCos(psi) * (y1 - y2) / 2.0;
    var lambda = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry);
    if (lambda > 1) {
        rx *= mathSqrt(lambda);
        ry *= mathSqrt(lambda);
    }
    var f = (fa === fs ? -1 : 1)
        * mathSqrt((((rx * rx) * (ry * ry))
            - ((rx * rx) * (yp * yp))
            - ((ry * ry) * (xp * xp))) / ((rx * rx) * (yp * yp)
            + (ry * ry) * (xp * xp))) || 0;
    var cxp = f * rx * yp / ry;
    var cyp = f * -ry * xp / rx;
    var cx = (x1 + x2) / 2.0
        + mathCos(psi) * cxp
        - mathSin(psi) * cyp;
    var cy = (y1 + y2) / 2.0
        + mathSin(psi) * cxp
        + mathCos(psi) * cyp;
    var theta = vAngle([1, 0], [(xp - cxp) / rx, (yp - cyp) / ry]);
    var u = [(xp - cxp) / rx, (yp - cyp) / ry];
    var v = [(-1 * xp - cxp) / rx, (-1 * yp - cyp) / ry];
    var dTheta = vAngle(u, v);
    if (vRatio(u, v) <= -1) {
        dTheta = PI;
    }
    if (vRatio(u, v) >= 1) {
        dTheta = 0;
    }
    if (dTheta < 0) {
        var n = Math.round(dTheta / PI * 1e6) / 1e6;
        dTheta = PI * 2 + (n % 2) * PI;
    }
    path.addData(cmd, cx, cy, rx, ry, theta, dTheta, psi, fs);
}
var commandReg = /([mlvhzcqtsa])([^mlvhzcqtsa]*)/ig;
var numberReg = /-?([0-9]*\.)?[0-9]+([eE]-?[0-9]+)?/g;
function createPathProxyFromString(data) {
    var path = new PathProxy();
    if (!data) {
        return path;
    }
    var cpx = 0;
    var cpy = 0;
    var subpathX = cpx;
    var subpathY = cpy;
    var prevCmd;
    var CMD = PathProxy.CMD;
    var cmdList = data.match(commandReg);
    if (!cmdList) {
        return path;
    }
    for (var l = 0; l < cmdList.length; l++) {
        var cmdText = cmdList[l];
        var cmdStr = cmdText.charAt(0);
        var cmd = void 0;
        var p = cmdText.match(numberReg) || [];
        var pLen = p.length;
        for (var i = 0; i < pLen; i++) {
            p[i] = parseFloat(p[i]);
        }
        var off = 0;
        while (off < pLen) {
            var ctlPtx = void 0;
            var ctlPty = void 0;
            var rx = void 0;
            var ry = void 0;
            var psi = void 0;
            var fa = void 0;
            var fs = void 0;
            var x1 = cpx;
            var y1 = cpy;
            var len = void 0;
            var pathData = void 0;
            switch (cmdStr) {
                case 'l':
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'L':
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'm':
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.M;
                    path.addData(cmd, cpx, cpy);
                    subpathX = cpx;
                    subpathY = cpy;
                    cmdStr = 'l';
                    break;
                case 'M':
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.M;
                    path.addData(cmd, cpx, cpy);
                    subpathX = cpx;
                    subpathY = cpy;
                    cmdStr = 'L';
                    break;
                case 'h':
                    cpx += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'H':
                    cpx = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'v':
                    cpy += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'V':
                    cpy = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'C':
                    cmd = CMD.C;
                    path.addData(cmd, p[off++], p[off++], p[off++], p[off++], p[off++], p[off++]);
                    cpx = p[off - 2];
                    cpy = p[off - 1];
                    break;
                case 'c':
                    cmd = CMD.C;
                    path.addData(cmd, p[off++] + cpx, p[off++] + cpy, p[off++] + cpx, p[off++] + cpy, p[off++] + cpx, p[off++] + cpy);
                    cpx += p[off - 2];
                    cpy += p[off - 1];
                    break;
                case 'S':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.C) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cmd = CMD.C;
                    x1 = p[off++];
                    y1 = p[off++];
                    cpx = p[off++];
                    cpy = p[off++];
                    path.addData(cmd, ctlPtx, ctlPty, x1, y1, cpx, cpy);
                    break;
                case 's':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.C) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cmd = CMD.C;
                    x1 = cpx + p[off++];
                    y1 = cpy + p[off++];
                    cpx += p[off++];
                    cpy += p[off++];
                    path.addData(cmd, ctlPtx, ctlPty, x1, y1, cpx, cpy);
                    break;
                case 'Q':
                    x1 = p[off++];
                    y1 = p[off++];
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, x1, y1, cpx, cpy);
                    break;
                case 'q':
                    x1 = p[off++] + cpx;
                    y1 = p[off++] + cpy;
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, x1, y1, cpx, cpy);
                    break;
                case 'T':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.Q) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, ctlPtx, ctlPty, cpx, cpy);
                    break;
                case 't':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.Q) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, ctlPtx, ctlPty, cpx, cpy);
                    break;
                case 'A':
                    rx = p[off++];
                    ry = p[off++];
                    psi = p[off++];
                    fa = p[off++];
                    fs = p[off++];
                    x1 = cpx, y1 = cpy;
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.A;
                    processArc(x1, y1, cpx, cpy, fa, fs, rx, ry, psi, cmd, path);
                    break;
                case 'a':
                    rx = p[off++];
                    ry = p[off++];
                    psi = p[off++];
                    fa = p[off++];
                    fs = p[off++];
                    x1 = cpx, y1 = cpy;
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.A;
                    processArc(x1, y1, cpx, cpy, fa, fs, rx, ry, psi, cmd, path);
                    break;
            }
        }
        if (cmdStr === 'z' || cmdStr === 'Z') {
            cmd = CMD.Z;
            path.addData(cmd);
            cpx = subpathX;
            cpy = subpathY;
        }
        prevCmd = cmd;
    }
    path.toStatic();
    return path;
}
var SVGPath = (function (_super) {
    __extends(SVGPath, _super);
    function SVGPath() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    SVGPath.prototype.applyTransform = function (m) { };
    return SVGPath;
}(Path));
function isPathProxy(path) {
    return path.setData != null;
}
function createPathOptions(str, opts) {
    var pathProxy = createPathProxyFromString(str);
    var innerOpts = extend({}, opts);
    innerOpts.buildPath = function (path) {
        if (isPathProxy(path)) {
            path.setData(pathProxy.data);
            var ctx = path.getContext();
            if (ctx) {
                path.rebuildPath(ctx, 1);
            }
        }
        else {
            var ctx = path;
            pathProxy.rebuildPath(ctx, 1);
        }
    };
    innerOpts.applyTransform = function (m) {
        transformPath(pathProxy, m);
        this.dirtyShape();
    };
    return innerOpts;
}
export function createFromString(str, opts) {
    return new SVGPath(createPathOptions(str, opts));
}
export function extendFromString(str, defaultOpts) {
    var innerOpts = createPathOptions(str, defaultOpts);
    var Sub = (function (_super) {
        __extends(Sub, _super);
        function Sub(opts) {
            var _this = _super.call(this, opts) || this;
            _this.applyTransform = innerOpts.applyTransform;
            _this.buildPath = innerOpts.buildPath;
            return _this;
        }
        return Sub;
    }(SVGPath));
    return Sub;
}
export function mergePath(pathEls, opts) {
    var pathList = [];
    var len = pathEls.length;
    for (var i = 0; i < len; i++) {
        var pathEl = pathEls[i];
        pathList.push(pathEl.getUpdatedPathProxy(true));
    }
    var pathBundle = new Path(opts);
    pathBundle.createPathProxy();
    pathBundle.buildPath = function (path) {
        if (isPathProxy(path)) {
            path.appendPath(pathList);
            var ctx = path.getContext();
            if (ctx) {
                path.rebuildPath(ctx, 1);
            }
        }
    };
    return pathBundle;
}
export function clonePath(sourcePath, opts) {
    opts = opts || {};
    var path = new Path();
    if (sourcePath.shape) {
        path.setShape(sourcePath.shape);
    }
    path.setStyle(sourcePath.style);
    if (opts.bakeTransform) {
        transformPath(path.path, sourcePath.getComputedTransform());
    }
    else {
        if (opts.toLocal) {
            path.setLocalTransform(sourcePath.getComputedTransform());
        }
        else {
            path.copyTransform(sourcePath);
        }
    }
    path.buildPath = sourcePath.buildPath;
    path.applyTransform = path.applyTransform;
    path.z = sourcePath.z;
    path.z2 = sourcePath.z2;
    path.zlevel = sourcePath.zlevel;
    return path;
}
