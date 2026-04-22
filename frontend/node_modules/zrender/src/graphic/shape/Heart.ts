/**
 * 心形
 */

import Path, { PathProps } from '../Path';

export class HeartShape {
    cx = 0
    cy = 0
    width = 0
    height = 0
}

export interface HeartProps extends PathProps {
    shape?: Partial<HeartShape>
}
class Heart extends Path<HeartProps> {

    shape: HeartShape

    constructor(opts?: HeartProps) {
        super(opts);
    }

    getDefaultShape() {
        return new HeartShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: HeartShape) {
        const x = shape.cx;
        const y = shape.cy;
        const a = shape.width;
        const b = shape.height;
        ctx.moveTo(x, y);
        ctx.bezierCurveTo(
            x + a / 2, y - b * 2 / 3,
            x + a * 2, y + b / 3,
            x, y + b
        );
        ctx.bezierCurveTo(
            x - a * 2, y + b / 3,
            x - a / 2, y - b * 2 / 3,
            x, y
        );
    }
}


Heart.prototype.type = 'heart';

export default Heart;