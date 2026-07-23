/**
 * 玫瑰线
 * @module zrender/graphic/shape/Rose
 */

import Path, { PathProps } from '../Path';

const sin = Math.sin;
const cos = Math.cos;
const radian = Math.PI / 180;

export class RoseShape {
    cx = 0
    cy = 0
    r: number[] = []
    k = 0
    n = 1
}

export interface RoseProps extends PathProps {
    shape?: Partial<RoseShape>
}
class Rose extends Path<RoseProps> {

    shape: RoseShape

    constructor(opts?: RoseProps) {
        super(opts);
    }

    getDefaultStyle() {
        return {
            stroke: '#000',
            fill: null as string
        };
    }

    getDefaultShape() {
        return new RoseShape();
    }


    buildPath(ctx: CanvasRenderingContext2D, shape: RoseShape) {
        const R = shape.r;
        const k = shape.k;
        const n = shape.n;
        const x0 = shape.cx;
        const y0 = shape.cy;
        let x;
        let y;
        let r;


        ctx.moveTo(x0, y0);

        for (let i = 0, len = R.length; i < len; i++) {
            r = R[i];

            for (let j = 0; j <= 360 * n; j++) {
                x = r
                        * sin(k / n * j % 360 * radian)
                        * cos(j * radian)
                        + x0;
                y = r
                        * sin(k / n * j % 360 * radian)
                        * sin(j * radian)
                        + y0;
                ctx.lineTo(x, y);
            }
        }
    }
}

Rose.prototype.type = 'rose';

export default Rose;