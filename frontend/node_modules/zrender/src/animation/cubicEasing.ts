import { cubicAt, cubicRootAt } from '../core/curve';
import { trim } from '../core/util';

const regexp = /cubic-bezier\(([0-9,\.e ]+)\)/;

export function createCubicEasingFunc(cubicEasingStr: string) {
    const cubic = cubicEasingStr && regexp.exec(cubicEasingStr);
    if (cubic) {
        const points = cubic[1].split(',');
        const a = +trim(points[0]);
        const b = +trim(points[1]);
        const c = +trim(points[2]);
        const d = +trim(points[3]);

        if (isNaN(a + b + c + d)) {
            return;
        }

        const roots: number[] = [];
        return (p: number) => {
            return p <= 0
                ? 0 : p >= 1
                    ? 1
                    : cubicRootAt(0, a, c, 1, p, roots) && cubicAt(0, b, d, 1, roots[0]);
        };
    }
}