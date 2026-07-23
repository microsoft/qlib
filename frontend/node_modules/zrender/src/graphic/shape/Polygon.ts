/**
 * 多边形
 * @module zrender/shape/Polygon
 */

import Path, { PathProps } from '../Path';
import * as polyHelper from '../helper/poly';
import { VectorArray } from '../../core/vector';

export class PolygonShape {
    points: VectorArray[] = null
    smooth?: number = 0
    smoothConstraint?: VectorArray[] = null
}

export interface PolygonProps extends PathProps {
    shape?: Partial<PolygonShape>
}
class Polygon extends Path<PolygonProps> {

    shape: PolygonShape

    constructor(opts?: PolygonProps) {
        super(opts);
    }

    getDefaultShape() {
        return new PolygonShape();
    }

    buildPath(ctx: CanvasRenderingContext2D, shape: PolygonShape) {
        polyHelper.buildPath(ctx, shape, true);
    }
};

Polygon.prototype.type = 'polygon';

export default Polygon;