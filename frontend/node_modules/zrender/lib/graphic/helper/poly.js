import smoothBezier from './smoothBezier.js';
export function buildPath(ctx, shape, closePath) {
    var smooth = shape.smooth;
    var points = shape.points;
    if (points && points.length >= 2) {
        if (smooth) {
            var controlPoints = smoothBezier(points, smooth, closePath, shape.smoothConstraint);
            ctx.moveTo(points[0][0], points[0][1]);
            var len = points.length;
            for (var i = 0; i < (closePath ? len : len - 1); i++) {
                var cp1 = controlPoints[i * 2];
                var cp2 = controlPoints[i * 2 + 1];
                var p = points[(i + 1) % len];
                ctx.bezierCurveTo(cp1[0], cp1[1], cp2[0], cp2[1], p[0], p[1]);
            }
        }
        else {
            ctx.moveTo(points[0][0], points[0][1]);
            for (var i = 1, l = points.length; i < l; i++) {
                ctx.lineTo(points[i][0], points[i][1]);
            }
        }
        closePath && ctx.closePath();
    }
}
