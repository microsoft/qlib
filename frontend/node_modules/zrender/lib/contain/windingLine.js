export default function windingLine(x0, y0, x1, y1, x, y) {
    if ((y > y0 && y > y1) || (y < y0 && y < y1)) {
        return 0;
    }
    if (y1 === y0) {
        return 0;
    }
    var t = (y - y0) / (y1 - y0);
    var dir = y1 < y0 ? 1 : -1;
    if (t === 1 || t === 0) {
        dir = y1 < y0 ? 0.5 : -0.5;
    }
    var x_ = t * (x1 - x0) + x0;
    return x_ === x ? Infinity : x_ > x ? dir : 0;
}
