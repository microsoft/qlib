/**
 * Catmull-Rom spline 插值折线
 */

import {distance as v2Distance, VectorArray} from '../../core/vector';

function interpolate(
    p0: number, p1: number, p2: number, p3: number, t: number, t2: number, t3: number
) {
    const v0 = (p2 - p0) * 0.5;
    const v1 = (p3 - p1) * 0.5;
    return (2 * (p1 - p2) + v0 + v1) * t3
            + (-3 * (p1 - p2) - 2 * v0 - v1) * t2
            + v0 * t + p1;
}

export default function smoothSpline(points: VectorArray[], isLoop?: boolean): VectorArray[] {
    const len = points.length;
    const ret = [];

    let distance = 0;
    for (let i = 1; i < len; i++) {
        distance += v2Distance(points[i - 1], points[i]);
    }

    let segs = distance / 2;
    segs = segs < len ? len : segs;
    for (let i = 0; i < segs; i++) {
        const pos = i / (segs - 1) * (isLoop ? len : len - 1);
        const idx = Math.floor(pos);

        const w = pos - idx;

        let p0;
        let p1 = points[idx % len];
        let p2;
        let p3;
        if (!isLoop) {
            p0 = points[idx === 0 ? idx : idx - 1];
            p2 = points[idx > len - 2 ? len - 1 : idx + 1];
            p3 = points[idx > len - 3 ? len - 1 : idx + 2];
        }
        else {
            p0 = points[(idx - 1 + len) % len];
            p2 = points[(idx + 1) % len];
            p3 = points[(idx + 2) % len];
        }

        const w2 = w * w;
        const w3 = w * w2;

        ret.push([
            interpolate(p0[0], p1[0], p2[0], p3[0], w, w2, w3),
            interpolate(p0[1], p1[1], p2[1], p3[1], w, w2, w3)
        ]);
    }
    return ret;
}