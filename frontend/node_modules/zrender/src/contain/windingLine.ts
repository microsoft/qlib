
export default function windingLine(
    x0: number, y0: number, x1: number, y1: number, x: number, y: number
): number {
    if ((y > y0 && y > y1) || (y < y0 && y < y1)) {
        return 0;
    }
    // Ignore horizontal line
    if (y1 === y0) {
        return 0;
    }
    const t = (y - y0) / (y1 - y0);

    let dir = y1 < y0 ? 1 : -1;
    // Avoid winding error when intersection point is the connect point of two line of polygon
    if (t === 1 || t === 0) {
        dir = y1 < y0 ? 0.5 : -0.5;
    }

    const x_ = t * (x1 - x0) + x0;

    // If (x, y) on the line, considered as "contain".
    return x_ === x ? Infinity : x_ > x ? dir : 0;
}