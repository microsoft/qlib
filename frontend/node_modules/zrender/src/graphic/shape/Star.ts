/**
 * n角星（n>3）
 * @module zrender/graphic/shape/Star
 */

import Path, { PathProps } from '../Path';

const PI = Math.PI;
const cos = Math.cos;
const sin = Math.sin;

export class StarShape {
    cx = 0
    cy = 0
    n = 3
    r0: number
    r = 0
}

export interface StarProps extends PathProps {
    shape?: Partial<StarShape>
}
class Star extends Path<StarProps> {

    shape: StarShape

    constructor(opts?: StarProps) {
        super(opts);
    }

    getDefaultShape() {
        return new StarShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: StarShape) {

        const n = shape.n;
        if (!n || n < 2) {
            return;
        }

        const x = shape.cx;
        const y = shape.cy;
        const r = shape.r;
        let r0 = shape.r0;

        // 如果未指定内部顶点外接圆半径，则自动计算
        if (r0 == null) {
            r0 = n > 4
                // 相隔的外部顶点的连线的交点，
                // 被取为内部交点，以此计算r0
                ? r * cos(2 * PI / n) / cos(PI / n)
                // 二三四角星的特殊处理
                : r / 3;
        }

        const dStep = PI / n;
        let deg = -PI / 2;
        const xStart = x + r * cos(deg);
        const yStart = y + r * sin(deg);
        deg += dStep;

        // 记录边界点，用于判断inside
        ctx.moveTo(xStart, yStart);
        for (let i = 0, end = n * 2 - 1, ri; i < end; i++) {
            ri = i % 2 === 0 ? r0 : r;
            ctx.lineTo(x + ri * cos(deg), y + ri * sin(deg));
            deg += dStep;
        }

        ctx.closePath();
    }
}

Star.prototype.type = 'star';
export default Star;