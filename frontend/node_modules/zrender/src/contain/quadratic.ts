import {quadraticProjectPoint} from '../core/curve';

/**
 * 二次贝塞尔曲线描边包含判断
 */
export function containStroke(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    lineWidth: number, x: number, y: number
): boolean {
    if (lineWidth === 0) {
        return false;
    }
    const _l = lineWidth;
    // Quick reject
    if (
        (y > y0 + _l && y > y1 + _l && y > y2 + _l)
        || (y < y0 - _l && y < y1 - _l && y < y2 - _l)
        || (x > x0 + _l && x > x1 + _l && x > x2 + _l)
        || (x < x0 - _l && x < x1 - _l && x < x2 - _l)
    ) {
        return false;
    }
    const d = quadraticProjectPoint(
        x0, y0, x1, y1, x2, y2,
        x, y, null
    );
    return d <= _l / 2;
}
