
import {normalizeRadian} from './util';

const PI2 = Math.PI * 2;

/**
 * 圆弧描边包含判断
 */
export function containStroke(
    cx: number, cy: number, r: number, startAngle: number, endAngle: number,
    anticlockwise: boolean,
    lineWidth: number, x: number, y: number
): boolean {

    if (lineWidth === 0) {
        return false;
    }
    const _l = lineWidth;

    x -= cx;
    y -= cy;
    const d = Math.sqrt(x * x + y * y);

    if ((d - _l > r) || (d + _l < r)) {
        return false;
    }
    // TODO
    if (Math.abs(startAngle - endAngle) % PI2 < 1e-4) {
        // Is a circle
        return true;
    }
    if (anticlockwise) {
        const tmp = startAngle;
        startAngle = normalizeRadian(endAngle);
        endAngle = normalizeRadian(tmp);
    }
    else {
        startAngle = normalizeRadian(startAngle);
        endAngle = normalizeRadian(endAngle);
    }
    if (startAngle > endAngle) {
        endAngle += PI2;
    }

    let angle = Math.atan2(y, x);
    if (angle < 0) {
        angle += PI2;
    }
    return (angle >= startAngle && angle <= endAngle)
        || (angle + PI2 >= startAngle && angle + PI2 <= endAngle);
}