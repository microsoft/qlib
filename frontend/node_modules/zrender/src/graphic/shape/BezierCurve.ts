/**
 * 贝塞尔曲线
 */

import Path, { PathProps } from '../Path';
import * as vec2 from '../../core/vector';
import {
    quadraticSubdivide,
    cubicSubdivide,
    quadraticAt,
    cubicAt,
    quadraticDerivativeAt,
    cubicDerivativeAt
} from '../../core/curve';

const out: number[] = [];

export class BezierCurveShape {
    x1 = 0
    y1 = 0
    x2 = 0
    y2 = 0
    cpx1 = 0
    cpy1 = 0
    cpx2?: number
    cpy2?: number
    // Curve show percent, for animating
    percent = 1
}

function someVectorAt(shape: BezierCurveShape, t: number, isTangent: boolean) {
    const cpx2 = shape.cpx2;
    const cpy2 = shape.cpy2;
    if (cpx2 != null || cpy2 != null) {
        return [
            (isTangent ? cubicDerivativeAt : cubicAt)(shape.x1, shape.cpx1, shape.cpx2, shape.x2, t),
            (isTangent ? cubicDerivativeAt : cubicAt)(shape.y1, shape.cpy1, shape.cpy2, shape.y2, t)
        ];
    }
    else {
        return [
            (isTangent ? quadraticDerivativeAt : quadraticAt)(shape.x1, shape.cpx1, shape.x2, t),
            (isTangent ? quadraticDerivativeAt : quadraticAt)(shape.y1, shape.cpy1, shape.y2, t)
        ];
    }
}

export interface BezierCurveProps extends PathProps {
    shape?: Partial<BezierCurveShape>
}
class BezierCurve extends Path<BezierCurveProps> {

    shape: BezierCurveShape

    constructor(opts?: BezierCurveProps) {
        super(opts);
    }

    getDefaultStyle() {
        return {
            stroke: '#000',
            fill: null as string
        };
    }

    getDefaultShape() {
        return new BezierCurveShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: BezierCurveShape) {
        let x1 = shape.x1;
        let y1 = shape.y1;
        let x2 = shape.x2;
        let y2 = shape.y2;
        let cpx1 = shape.cpx1;
        let cpy1 = shape.cpy1;
        let cpx2 = shape.cpx2;
        let cpy2 = shape.cpy2;
        let percent = shape.percent;
        if (percent === 0) {
            return;
        }

        ctx.moveTo(x1, y1);

        if (cpx2 == null || cpy2 == null) {
            if (percent < 1) {
                quadraticSubdivide(x1, cpx1, x2, percent, out);
                cpx1 = out[1];
                x2 = out[2];
                quadraticSubdivide(y1, cpy1, y2, percent, out);
                cpy1 = out[1];
                y2 = out[2];
            }

            ctx.quadraticCurveTo(
                cpx1, cpy1,
                x2, y2
            );
        }
        else {
            if (percent < 1) {
                cubicSubdivide(x1, cpx1, cpx2, x2, percent, out);
                cpx1 = out[1];
                cpx2 = out[2];
                x2 = out[3];
                cubicSubdivide(y1, cpy1, cpy2, y2, percent, out);
                cpy1 = out[1];
                cpy2 = out[2];
                y2 = out[3];
            }
            ctx.bezierCurveTo(
                cpx1, cpy1,
                cpx2, cpy2,
                x2, y2
            );
        }
    }

    /**
     * Get point at percent
     */
    pointAt(t: number) {
        return someVectorAt(this.shape, t, false);
    }

    /**
     * Get tangent at percent
     */
    tangentAt(t: number) {
        const p = someVectorAt(this.shape, t, true);
        return vec2.normalize(p, p);
    }
};

BezierCurve.prototype.type = 'bezier-curve';

export default BezierCurve;
