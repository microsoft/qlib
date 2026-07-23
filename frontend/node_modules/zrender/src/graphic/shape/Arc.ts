/**
 * 圆弧
 */

import Path, { PathProps } from '../Path';

export class ArcShape {
    cx = 0;
    cy = 0;
    r = 0;
    startAngle = 0;
    endAngle = Math.PI * 2
    clockwise? = true
}

export interface ArcProps extends PathProps {
    shape?: Partial<ArcShape>
}

class Arc extends Path<ArcProps> {

    shape: ArcShape

    constructor(opts?: ArcProps) {
        super(opts);
    }

    getDefaultStyle() {
        return {
            stroke: '#000',
            fill: null as string
        };
    }

    getDefaultShape() {
        return new ArcShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: ArcShape) {

        const x = shape.cx;
        const y = shape.cy;
        const r = Math.max(shape.r, 0);
        const startAngle = shape.startAngle;
        const endAngle = shape.endAngle;
        const clockwise = shape.clockwise;

        const unitX = Math.cos(startAngle);
        const unitY = Math.sin(startAngle);

        ctx.moveTo(unitX * r + x, unitY * r + y);
        ctx.arc(x, y, r, startAngle, endAngle, !clockwise);
    }
}

Arc.prototype.type = 'arc';

export default Arc;