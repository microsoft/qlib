import PathProxy from '../../core/PathProxy';
import { isArray } from '../../core/util';

const PI = Math.PI;
const PI2 = PI * 2;
const mathSin = Math.sin;
const mathCos = Math.cos;
const mathACos = Math.acos;
const mathATan2 = Math.atan2;
const mathAbs = Math.abs;
const mathSqrt = Math.sqrt;
const mathMax = Math.max;
const mathMin = Math.min;
const e = 1e-4;

function intersect(
    x0: number, y0: number,
    x1: number, y1: number,
    x2: number, y2: number,
    x3: number, y3: number
): [number, number] {
    const dx10 = x1 - x0;
    const dy10 = y1 - y0;
    const dx32 = x3 - x2;
    const dy32 = y3 - y2;
    let t = dy32 * dx10 - dx32 * dy10;
    if (t * t < e) {
        return;
    }
    t = (dx32 * (y0 - y2) - dy32 * (x0 - x2)) / t;
    return [x0 + t * dx10, y0 + t * dy10];
}

// Compute perpendicular offset line of length rc.
function computeCornerTangents(
    x0: number, y0: number,
    x1: number, y1: number,
    radius: number, cr: number,
    clockwise: boolean
) {
    const x01 = x0 - x1;
    const y01 = y0 - y1;
    const lo = (clockwise ? cr : -cr) / mathSqrt(x01 * x01 + y01 * y01);
    const ox = lo * y01;
    const oy = -lo * x01;
    const x11 = x0 + ox;
    const y11 = y0 + oy;
    const x10 = x1 + ox;
    const y10 = y1 + oy;
    const x00 = (x11 + x10) / 2;
    const y00 = (y11 + y10) / 2;
    const dx = x10 - x11;
    const dy = y10 - y11;
    const d2 = dx * dx + dy * dy;
    const r = radius - cr;
    const s = x11 * y10 - x10 * y11;
    const d = (dy < 0 ? -1 : 1) * mathSqrt(mathMax(0, r * r * d2 - s * s));
    let cx0 = (s * dy - dx * d) / d2;
    let cy0 = (-s * dx - dy * d) / d2;
    const cx1 = (s * dy + dx * d) / d2;
    const cy1 = (-s * dx + dy * d) / d2;
    const dx0 = cx0 - x00;
    const dy0 = cy0 - y00;
    const dx1 = cx1 - x00;
    const dy1 = cy1 - y00;

    // Pick the closer of the two intersection points
    // TODO: Is there a faster way to determine which intersection to use?
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

// For compatibility, don't use normalizeCssArray
// 5 represents [5, 5, 5, 5]
// [5] represents [5, 5, 0, 0]
// [5, 10] represents [5, 5, 10, 10]
// [5, 10, 15] represents [5, 10, 15, 15]
// [5, 10, 15, 20] represents [5, 10, 15, 20]
function normalizeCornerRadius(cr: number | number[]): number[] {
    let arr: number[];
    if (isArray(cr)) {
        const len = cr.length;
        if (!len) {
            return cr as number[];
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

export function buildPath(ctx: CanvasRenderingContext2D | PathProxy, shape: {
    cx: number
    cy: number
    startAngle: number
    endAngle: number
    clockwise?: boolean,
    r?: number,
    r0?: number,
    cornerRadius?: number | number[]
}) {
    let radius = mathMax(shape.r, 0);
    let innerRadius = mathMax(shape.r0 || 0, 0);
    const hasRadius = radius > 0;
    const hasInnerRadius = innerRadius > 0;

    if (!hasRadius && !hasInnerRadius) {
        return;
    }

    if (!hasRadius) {
        // use innerRadius as radius if no radius
        radius = innerRadius;
        innerRadius = 0;
    }

    if (innerRadius > radius) {
        // swap, ensure that radius is always larger than innerRadius
        const tmp = radius;
        radius = innerRadius;
        innerRadius = tmp;
    }

    const { startAngle, endAngle } = shape;
    if (isNaN(startAngle) || isNaN(endAngle)) {
        return;
    }

    const { cx, cy } = shape;
    const clockwise = !!shape.clockwise;

    let arc = mathAbs(endAngle - startAngle);
    const mod = arc > PI2 && arc % PI2;
    mod > e && (arc = mod);

    // is a point
    if (!(radius > e)) {
        ctx.moveTo(cx, cy);
    }
    // is a circle or annulus
    else if (arc > PI2 - e) {
        ctx.moveTo(
            cx + radius * mathCos(startAngle),
            cy + radius * mathSin(startAngle)
        );
        ctx.arc(cx, cy, radius, startAngle, endAngle, !clockwise);

        if (innerRadius > e) {
            ctx.moveTo(
                cx + innerRadius * mathCos(endAngle),
                cy + innerRadius * mathSin(endAngle)
            );
            ctx.arc(cx, cy, innerRadius, endAngle, startAngle, clockwise);
        }
    }
    // is a circular or annular sector
    else {
        let icrStart;
        let icrEnd;
        let ocrStart;
        let ocrEnd;

        let ocrs;
        let ocre;
        let icrs;
        let icre;

        let ocrMax;
        let icrMax;
        let limitedOcrMax;
        let limitedIcrMax;

        let xre;
        let yre;
        let xirs;
        let yirs;

        const xrs = radius * mathCos(startAngle);
        const yrs = radius * mathSin(startAngle);
        const xire = innerRadius * mathCos(endAngle);
        const yire = innerRadius * mathSin(endAngle);

        const hasArc = arc > e;
        if (hasArc) {
            const cornerRadius = shape.cornerRadius;
            if (cornerRadius) {
                [icrStart, icrEnd, ocrStart, ocrEnd] = normalizeCornerRadius(cornerRadius);
            }

            const halfRd = mathAbs(radius - innerRadius) / 2;
            ocrs = mathMin(halfRd, ocrStart);
            ocre = mathMin(halfRd, ocrEnd);
            icrs = mathMin(halfRd, icrStart);
            icre = mathMin(halfRd, icrEnd);

            limitedOcrMax = ocrMax = mathMax(ocrs, ocre);
            limitedIcrMax = icrMax = mathMax(icrs, icre);

            // draw corner radius
            if (ocrMax > e || icrMax > e) {
                xre = radius * mathCos(endAngle);
                yre = radius * mathSin(endAngle);
                xirs = innerRadius * mathCos(startAngle);
                yirs = innerRadius * mathSin(startAngle);

                // restrict the max value of corner radius
                if (arc < PI) {
                    const it = intersect(xrs, yrs, xirs, yirs, xre, yre, xire, yire);
                    if (it) {
                        const x0 = xrs - it[0];
                        const y0 = yrs - it[1];
                        const x1 = xre - it[0];
                        const y1 = yre - it[1];
                        const a = 1 / mathSin(
                            // eslint-disable-next-line max-len
                            mathACos((x0 * x1 + y0 * y1) / (mathSqrt(x0 * x0 + y0 * y0) * mathSqrt(x1 * x1 + y1 * y1))) / 2
                        );
                        const b = mathSqrt(it[0] * it[0] + it[1] * it[1]);
                        limitedOcrMax = mathMin(ocrMax, (radius - b) / (a + 1));
                        limitedIcrMax = mathMin(icrMax, (innerRadius - b) / (a - 1));
                    }
                }
            }
        }

        // the sector is collapsed to a line
        if (!hasArc) {
            ctx.moveTo(cx + xrs, cy + yrs);
        }
        // the outer ring has corners
        else if (limitedOcrMax > e) {
            const crStart = mathMin(ocrStart, limitedOcrMax);
            const crEnd = mathMin(ocrEnd, limitedOcrMax);
            const ct0 = computeCornerTangents(xirs, yirs, xrs, yrs, radius, crStart, clockwise);
            const ct1 = computeCornerTangents(xre, yre, xire, yire, radius, crEnd, clockwise);

            ctx.moveTo(cx + ct0.cx + ct0.x0, cy + ct0.cy + ct0.y0);

            // Have the corners merged?
            if (limitedOcrMax < ocrMax && crStart === crEnd) {
                // eslint-disable-next-line max-len
                ctx.arc(cx + ct0.cx, cy + ct0.cy, limitedOcrMax, mathATan2(ct0.y0, ct0.x0), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
            else {
                // draw the two corners and the ring
                // eslint-disable-next-line max-len
                crStart > 0 && ctx.arc(cx + ct0.cx, cy + ct0.cy, crStart, mathATan2(ct0.y0, ct0.x0), mathATan2(ct0.y1, ct0.x1), !clockwise);
                // eslint-disable-next-line max-len
                ctx.arc(cx, cy, radius, mathATan2(ct0.cy + ct0.y1, ct0.cx + ct0.x1), mathATan2(ct1.cy + ct1.y1, ct1.cx + ct1.x1), !clockwise);
                // eslint-disable-next-line max-len
                crEnd > 0 && ctx.arc(cx + ct1.cx, cy + ct1.cy, crEnd, mathATan2(ct1.y1, ct1.x1), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
        }
        // the outer ring is a circular arc
        else {
            ctx.moveTo(cx + xrs, cy + yrs);
            ctx.arc(cx, cy, radius, startAngle, endAngle, !clockwise);
        }

        // no inner ring, is a circular sector
        if (!(innerRadius > e) || !hasArc) {
            ctx.lineTo(cx + xire, cy + yire);
        }
        // the inner ring has corners
        else if (limitedIcrMax > e) {
            const crStart = mathMin(icrStart, limitedIcrMax);
            const crEnd = mathMin(icrEnd, limitedIcrMax);
            const ct0 = computeCornerTangents(xire, yire, xre, yre, innerRadius, -crEnd, clockwise);
            const ct1 = computeCornerTangents(xrs, yrs, xirs, yirs, innerRadius, -crStart, clockwise);
            ctx.lineTo(cx + ct0.cx + ct0.x0, cy + ct0.cy + ct0.y0);

            // Have the corners merged?
            if (limitedIcrMax < icrMax && crStart === crEnd) {
                // eslint-disable-next-line max-len
                ctx.arc(cx + ct0.cx, cy + ct0.cy, limitedIcrMax, mathATan2(ct0.y0, ct0.x0), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
            // draw the two corners and the ring
            else {
                // eslint-disable-next-line max-len
                crEnd > 0 && ctx.arc(cx + ct0.cx, cy + ct0.cy, crEnd, mathATan2(ct0.y0, ct0.x0), mathATan2(ct0.y1, ct0.x1), !clockwise);
                // eslint-disable-next-line max-len
                ctx.arc(cx, cy, innerRadius, mathATan2(ct0.cy + ct0.y1, ct0.cx + ct0.x1), mathATan2(ct1.cy + ct1.y1, ct1.cx + ct1.x1), clockwise);
                // eslint-disable-next-line max-len
                crStart > 0 && ctx.arc(cx + ct1.cx, cy + ct1.cy, crStart, mathATan2(ct1.y1, ct1.x1), mathATan2(ct1.y0, ct1.x0), !clockwise);
            }
        }
        // the inner ring is just a circular arc
        else {
            // FIXME: if no lineTo, svg renderer will perform an abnormal drawing behavior.
            ctx.lineTo(cx + xire, cy + yire);

            ctx.arc(cx, cy, innerRadius, endAngle, startAngle, clockwise);
        }
    }

    ctx.closePath();
}
