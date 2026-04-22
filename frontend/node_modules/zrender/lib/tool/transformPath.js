import PathProxy from '../core/PathProxy.js';
import { applyTransform as v2ApplyTransform } from '../core/vector.js';
var CMD = PathProxy.CMD;
var points = [[], [], []];
var mathSqrt = Math.sqrt;
var mathAtan2 = Math.atan2;
export default function transformPath(path, m) {
    if (!m) {
        return;
    }
    var data = path.data;
    var len = path.len();
    var cmd;
    var nPoint;
    var i;
    var j;
    var k;
    var p;
    var M = CMD.M;
    var C = CMD.C;
    var L = CMD.L;
    var R = CMD.R;
    var A = CMD.A;
    var Q = CMD.Q;
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
                var x = m[4];
                var y = m[5];
                var sx = mathSqrt(m[0] * m[0] + m[1] * m[1]);
                var sy = mathSqrt(m[2] * m[2] + m[3] * m[3]);
                var angle = mathAtan2(-m[1] / sy, m[0] / sx);
                data[i] *= sx;
                data[i++] += x;
                data[i] *= sy;
                data[i++] += y;
                data[i++] *= sx;
                data[i++] *= sy;
                data[i++] += angle;
                data[i++] += angle;
                i += 2;
                j = i;
                break;
            case R:
                p[0] = data[i++];
                p[1] = data[i++];
                v2ApplyTransform(p, p, m);
                data[j++] = p[0];
                data[j++] = p[1];
                p[0] += data[i++];
                p[1] += data[i++];
                v2ApplyTransform(p, p, m);
                data[j++] = p[0];
                data[j++] = p[1];
        }
        for (k = 0; k < nPoint; k++) {
            var p_1 = points[k];
            p_1[0] = data[i++];
            p_1[1] = data[i++];
            v2ApplyTransform(p_1, p_1, m);
            data[j++] = p_1[0];
            data[j++] = p_1[1];
        }
    }
    path.increaseVersion();
}
