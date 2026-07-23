
import smoothBezier from './smoothBezier';
import { VectorArray } from '../../core/vector';
import PathProxy from '../../core/PathProxy';

export function buildPath(
    ctx: CanvasRenderingContext2D | PathProxy,
    shape: {
        points: VectorArray[],
        smooth?: number
        smoothConstraint?: VectorArray[]
    },
    closePath: boolean
) {
    const smooth = shape.smooth;
    let points = shape.points;
    if (points && points.length >= 2) {
        if (smooth) {
            const controlPoints = smoothBezier(
                points, smooth, closePath, shape.smoothConstraint
            );

            ctx.moveTo(points[0][0], points[0][1]);
            const len = points.length;
            for (let i = 0; i < (closePath ? len : len - 1); i++) {
                const cp1 = controlPoints[i * 2];
                const cp2 = controlPoints[i * 2 + 1];
                const p = points[(i + 1) % len];
                ctx.bezierCurveTo(
                    cp1[0], cp1[1], cp2[0], cp2[1], p[0], p[1]
                );
            }
        }
        else {
            ctx.moveTo(points[0][0], points[0][1]);
            for (let i = 1, l = points.length; i < l; i++) {
                ctx.lineTo(points[i][0], points[i][1]);
            }
        }

        closePath && ctx.closePath();
    }
}
