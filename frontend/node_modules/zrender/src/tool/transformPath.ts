import PathProxy from '../core/PathProxy';
import {applyTransform as v2ApplyTransform, VectorArray} from '../core/vector';
import { MatrixArray } from '../core/matrix';

const CMD = PathProxy.CMD;

const points: VectorArray[] = [[], [], []];
const mathSqrt = Math.sqrt;
const mathAtan2 = Math.atan2;

export default function transformPath(path: PathProxy, m: MatrixArray) {
    if (!m) {
        return;
    }

    let data = path.data;
    const len = path.len();
    let cmd;
    let nPoint: number;
    let i: number;
    let j: number;
    let k: number;
    let p: VectorArray;

    const M = CMD.M;
    const C = CMD.C;
    const L = CMD.L;
    const R = CMD.R;
    const A = CMD.A;
    const Q = CMD.Q;

    for (i = 0, j = 0; i < len;) {
        cmd = data[i++];
        j = i;
        nPoint = 0;

        switch (cmd) {
            case M:
                nPoint = 1;
                break;
            case L:
                nPoint = 1;
                break;
            case C:
                nPoint = 3;
                break;
            case Q:
                nPoint = 2;
                break;
            case A:
                const x = m[4];
                const y = m[5];
                const sx = mathSqrt(m[0] * m[0] + m[1] * m[1]);
                const sy = mathSqrt(m[2] * m[2] + m[3] * m[3]);
                const angle = mathAtan2(-m[1] / sy, m[0] / sx);
                // cx
                data[i] *= sx;
                data[i++] += x;
                // cy
                data[i] *= sy;
                data[i++] += y;
                // Scale rx and ry
                // FIXME Assume psi is 0 here
                data[i++] *= sx;
                data[i++] *= sy;

                // Start angle
                data[i++] += angle;
                // end angle
                data[i++] += angle;
                // FIXME psi
                i += 2;
                j = i;
                break;
            case R:
                // x0, y0
                p[0] = data[i++];
                p[1] = data[i++];
                v2ApplyTransform(p, p, m);
                data[j++] = p[0];
                data[j++] = p[1];
                // x1, y1
                p[0] += data[i++];
                p[1] += data[i++];
                v2ApplyTransform(p, p, m);
                data[j++] = p[0];
                data[j++] = p[1];
        }

        for (k = 0; k < nPoint; k++) {
            let p = points[k];
            p[0] = data[i++];
            p[1] = data[i++];

            v2ApplyTransform(p, p, m);
            // Write back
            data[j++] = p[0];
            data[j++] = p[1];
        }
    }

    path.increaseVersion();
}
