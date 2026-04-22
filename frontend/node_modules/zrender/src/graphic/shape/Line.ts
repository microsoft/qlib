/**
 * 直线
 * @module zrender/graphic/shape/Line
 */

import Path, { PathProps } from '../Path';
import {subPixelOptimizeLine} from '../helper/subPixelOptimize';
import { VectorArray } from '../../core/vector';

// Avoid create repeatly.
const subPixelOptimizeOutputShape = {};

export class LineShape {
    // Start point
    x1 = 0
    y1 = 0
    // End point
    x2 = 0
    y2 = 0

    percent = 1
}

export interface LineProps extends PathProps {
    shape?: Partial<LineShape>
}
class Line extends Path<LineProps> {

    shape: LineShape

    constructor(opts?: LineProps) {
        super(opts);
    }

    getDefaultStyle() {
        return {
            stroke: '#000',
            fill: null as string
        };
    }

    getDefaultShape() {
        return new LineShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: LineShape) {
        let x1;
        let y1;
        let x2;
        let y2;

        if (this.subPixelOptimize) {
            const optimizedShape = subPixelOptimizeLine(
                subPixelOptimizeOutputShape, shape, this.style
            );
            x1 = optimizedShape.x1;
            y1 = optimizedShape.y1;
            x2 = optimizedShape.x2;
            y2 = optimizedShape.y2;
        }
        else {
            x1 = shape.x1;
            y1 = shape.y1;
            x2 = shape.x2;
            y2 = shape.y2;
        }

        const percent = shape.percent;

        if (percent === 0) {
            return;
        }

        ctx.moveTo(x1, y1);

        if (percent < 1) {
            x2 = x1 * (1 - percent) + x2 * percent;
            y2 = y1 * (1 - percent) + y2 * percent;
        }
        ctx.lineTo(x2, y2);
    }

    /**
     * Get point at percent
     */
    pointAt(p: number): VectorArray {
        const shape = this.shape;
        return [
            shape.x1 * (1 - p) + shape.x2 * p,
            shape.y1 * (1 - p) + shape.y2 * p
        ];
    }
}

Line.prototype.type = 'line';
export default Line;