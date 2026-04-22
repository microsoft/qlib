import { isArray } from '../../core/util.js';
var PI = Math.PI;
var PI2 = PI * 2;
var mathSin = Math.sin;
var mathCos = Math.cos;
var mathACos = Math.acos;
var mathATan2 = Math.atan2;
var mathAbs = Math.abs;
var mathSqrt = Math.sqrt;
var mathMax = Math.max;
var mathMin = Math.min;
var e = 1e-4;
function intersect(x0, y0, x1, y1, x2, y2, x3, y3) {
    var dx10 = x1 - x0;
    var dy10 = y1 - y0;
    var dx32 = x3 - x2;
    var dy32 = y3 - y2;
    var t = dy32 * dx10 - dx32 * dy10;
    if (t * t < e) {
        return;
    }
    t = (dx32 * (y0 - y2) - dy32 * (x0 - x2)) / t;
    return [x0 + t * dx10, y0 + t * dy10];
}
function computeCornerTangents(x0, y0, x1, y1, radius, cr, clockwise) {
    var x01 = x0 - x1;
    var y01 = y0 - y1;
    var lo = (clockwise ? cr : -cr) / mathSqrt(x01 * x01 + y01 * y01);
    var ox = lo * y01;
    var oy = -lo * x01;
    var x11 = x0 + ox;
    var y11 = y0 + oy;
    var x10 = x1 + ox;
    var y10 = y1 + oy;
    var x00 = (x11 + x10) / 2;
    var y00 = (y11 + y10) / 2;
    var dx = x10 - x11;
    var dy = y10 - y11;
    var d2 = dx * dx + dy * dy;
    var r = radius - cr;
    var s = x11 * y10 - x10 * y11;
    var d = (dy < 0 ? -1 : 1) * mathSqrt(mathMax(0, r * r * d2 - s * s));
    var cx0 = (s * dy - dx * d) / d2;
    var cy0 = (-s * dx - dy * d) / d2;
    var cx1 = (s * dy + dx * d) / d2;
    var cy1 = (-s * dx + dy * d) / d2;
    var dx0 = cx0 - x00;
    var dy0 = cy0 - y00;
    var dx1 = cx1 - x00;
    var dy1 = cy1 - y00;
    if (dx0 * dx0 + dy0 * dy0 > dx1 * dx1 + dy1 * dy1) {
        cx0 = cx1;
        cy0 = cy1;
    }
    return {
        cx: cx0,
        cy: cy0,
        x0: -ox,
        y0: -oy,
        x1: cx0 * (radius / r - 1),
        y1: cy0 * (radius / r - 1)
    };
}
function normalizeCornerRadius(cr) {
    var arr;
    if (isArray(cr)) {
        var len = cr.length;
        if (!len) {
            return cr;
        }
        if (len === 1) {
            arr = [cr[0], cr[0], 0, 0];
        }
        else if (len === 2) {
            arr = [cr[0], cr[0], cr[1], cr[1]];
        }
        else if (len === 3) {
            arr = cr.concat(cr[2]);
        }
        else {
            arr = cr;
        }
    }
    else {
        arr = [cr, cr, cr, cr];
    }
    return arr;
}
export function buildPath(ctx, shape) {
    var _a;
    var radius = mathMax(shape.r, 0);
    var innerRadius = mathMax(shape.r0 || 0, 0);
    var hasRadius = radius > 0;
    var hasInnerRadius = innerRadius > 0;
    if (!hasRadius && !hasInnerRadius) {
        return;
    }
    if (!hasRadius) {
        radius = innerRadius;
        innerRadius = 0;
    }
    if (innerRadius > radius) {
        var tmp = radius;
        radius = innerRadius;
        innerRadius = tmp;
    }
    var startAngle = shape.startAngle, endAngle = shape.endAngle;
    if (isNaN(startAngle) || isNaN(endAngle)) {
        return;
    }
    var cx = shape.cx, cy = shape.cy;
    var clockwise = !!shape.clockwise;
    var arc = mathAbs(endAngle - startAngle);
    var mod = arc > PI2 && arc % PI2;
    mod > e && (arc = mod);
    if (!(radius > e)) {
        ctx.moveTo(cx, cy);
    }
    else if (arc > PI2 - e) {
        ctx.moveTo(cx + radius * mathCos(startAngle), cy + radius * mathSin(startAngle));
        ctx.arc(cx, cy, radius, startAngle, endAngle, !clockwise);
        if (innerRadius > e) {
            ctx.moveTo(cx + innerRadius * mathCos(endAngle), cy + innerRadius * mathSin(endAngle));
            ctx.arc(cx, cy, innerRadius, endAngle, startAngle, clockwise);
        }
    }
    else {
        var icrStart = void 0;
        var icrEnd = void 0;
        var ocrStart = void 0;
        var ocrEnd = void 0;
        var ocrs = void 0;
        var ocre = void 0;
        var icrs = void 0;
        var icre = void 0;
        var ocrMax = void 0;
        var icrMax = void 0;
        var limitedOcrMax = void 0;
        var limitedIcrMax = void 0;
        var xre = void 0;
        var yre = void 0;
        var xirs = void 0;
        var yirs = void 0;
        var xrs = radius * mathCos(startAngle);
        var yrs = radius * mathSin(startAngle);
        var xire = innerRadius * mathCos(endAngle);
        var yire = innerRadius * mathSin(endAngle);
        var hasArc = arc > e;
        if (hasArc) {
            var cornerRadius = shape.cornerRadius;
            if (cornerRadius) {
                _a = normalizeCornerRadius(cornerRadius), icrStart = _a[0], icrEnd = _a[1], ocrStart = _a[2], ocrEnd = _a[3];
            }
            var halfRd = mathAbs(radius - innerRadius) / 2;
            ocrs = mathMin(halfRd, ocrStart);
            ocre = mathMin(halfRd, ocrEnd);
            icrs = mathMin(halfRd, icrStart);
            icre = mathMin(halfRd, icrEnd);
            limitedOcrMax = ocrMax = mathMax(ocrs, ocre);
            limitedIcrMax = icrMax = mathMax(icrs, icre);
            if (ocrMax > e || icrMax > e) {
                xre = radius * mathCos(endAngle);
                yre = radius * mathSin(endAngle);
                xirs = innerRadius * mathCos(startAngle);
                yirs = innerRadius * mathSin(startAngle);
                if (arc < PI) {
                    var it_1 = intersect(xrs, yrs, xirs, yirs, xre, yre, xire, yire);
                    if (it_1) {
                        var x0 = xrs - it_1[0];
                        var y0 = yrs - it_1[1];
                        var x1 = xre - it_1[0];
                        var y1 = yre - it_1[1];
                        var a = 1 / mathSin(mathACos((x0 * x1 + y0 * y1) / (mathSqrt(x0 * x0 + y0 * y0) * mathSqrt(x1 * x1 + y1 * y1))) / 2);
                        var b = mathSqrt(it_1[0] * it_1[0] + it_1[1] * it_1[1]);
                        limitedOcrMax = mathMin(ocrMax, (radius - b) / (a + 1));
                        limitedIcrMax = mathMin(icrMax, (innerRadius - b) / (a - 1));
                    }
                }
            }
        }
        if (!hasArc) {
            ctx.moveTo(cx + xrs, cy + yrs);
        }
        else if (limitedOcrMax > e) {
            var crStart = mathMin(ocrStart, limitedOcrMax);
            var crEnd = mathMin(ocrEnd, limitedOcrMax);
            var ct0 = computeCornerTangents(xirs, yirs, xrs, yrs, radius, crStart, clockwise);
            var ct1 = computeCornerTangents(xre, yre, xire, yire, radius, crEnd, clockwise);
            ctx.moveTo(cx + ct0.cx + ct0.x0, cy + ct0.cy + ct0.y0);
            if (limitedOcrMax < ocrMax && crStart === crEnd) {
                ctx.arc(cx + ct0.cx, cy + ct0.cy, limitedOcrMax, mathATan2(ct0.y0, ct0.x0), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
            else {
                crStart > 0 && ctx.arc(cx + ct0.cx, cy + ct0.cy, crStart, mathATan2(ct0.y0, ct0.x0), mathATan2(ct0.y1, ct0.x1), !clockwise);
                ctx.arc(cx, cy, radius, mathATan2(ct0.cy + ct0.y1, ct0.cx + ct0.x1), mathATan2(ct1.cy + ct1.y1, ct1.cx + ct1.x1), !clockwise);
                crEnd > 0 && ctx.arc(cx + ct1.cx, cy + ct1.cy, crEnd, mathATan2(ct1.y1, ct1.x1), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
        }
        else {
            ctx.moveTo(cx + xrs, cy + yrs);
            ctx.arc(cx, cy, radius, startAngle, endAngle, !clockwise);
        }
        if (!(innerRadius > e) || !hasArc) {
            ctx.lineTo(cx + xire, cy + yire);
        }
        else if (limitedIcrMax > e) {
            var crStart = mathMin(icrStart, limitedIcrMax);
            var crEnd = mathMin(icrEnd, limitedIcrMax);
            var ct0 = computeCornerTangents(xire, yire, xre, yre, innerRadius, -crEnd, clockwise);
            var ct1 = computeCornerTangents(xrs, yrs, xirs, yirs, innerRadius, -crStart, clockwise);
            ctx.lineTo(cx + ct0.cx + ct0.x0, cy + ct0.cy + ct0.y0);
            if (limitedIcrMax < icrMax && crStart === crEnd) {
                ctx.arc(cx + ct0.cx, cy + ct0.cy, limitedIcrMax, mathATan2(ct0.y0, ct0.x0), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
            else {
                crEnd > 0 && ctx.arc(cx + ct0.cx, cy + ct0.cy, crEnd, mathATan2(ct0.y0, ct0.x0), mathATan2(ct0.y1, ct0.x1), !clockwise);
                ctx.arc(cx, cy, innerRadius, mathATan2(ct0.cy + ct0.y1, ct0.cx + ct0.x1), mathATan2(ct1.cy + ct1.y1, ct1.cx + ct1.x1), clockwise);
                crStart > 0 && ctx.arc(cx + ct1.cx, cy + ct1.cy, crStart, mathATan2(ct1.y1, ct1.x1), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
        }
        else {
            ctx.lineTo(cx + xire, cy + yire);
            ctx.arc(cx, cy, innerRadius, endAngle, startAngle, clockwise);
        }
    }
    ctx.closePath();
}
