import { cubicAt, cubicRootAt } from '../core/curve.js';
import { trim } from '../core/util.js';
var regexp = /cubic-bezier\(([0-9,\.e ]+)\)/;
export function createCubicEasingFunc(cubicEasingStr) {
    var cubic = cubicEasingStr && regexp.exec(cubicEasingStr);
    if (cubic) {
        var points = cubic[1].split(',');
        var a_1 = +trim(points[0]);
        var b_1 = +trim(points[1]);
        var c_1 = +trim(points[2]);
        var d_1 = +trim(points[3]);
        if (isNaN(a_1 + b_1 + c_1 + d_1)) {
            return;
        }
        var roots_1 = [];
        return function (p) {
            return p <= 0
                ? 0 : p >= 1
                ? 1
                : cubicRootAt(0, a_1, c_1, 1, p, roots_1) && cubicAt(0, b_1, d_1, 1, roots_1[0]);
        };
    }
}
