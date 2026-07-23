import PathProxy from '../core/PathProxy';
import * as line from './line';
import * as cubic from './cubic';
import * as quadratic from './quadratic';
import * as arc from './arc';
import * as curve from '../core/curve';
import windingLine from './windingLine';

const CMD = PathProxy.CMD;
const PI2 = Math.PI * 2;

const EPSILON = 1e-4;

function isAroundEqual(a: number, b: number) {
    return Math.abs(a - b) < EPSILON;
}

// 临时数组
const roots = [-1, -1, -1];
const extrema = [-1, -1];

function swapExtrema() {
    const tmp = extrema[0];
    extrema[0] = extrema[1];
    extrema[1] = tmp;
}

function windingCubic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number, x3: number, y3: number,
    x: number, y: number
): number {
    // Quick reject
    if (
        (y > y0 && y > y1 && y > y2 && y > y3)
        || (y < y0 && y < y1 && y < y2 && y < y3)
    ) {
        return 0;
    }
    const nRoots = curve.cubicRootAt(y0, y1, y2, y3, y, roots);
    if (nRoots === 0) {
        return 0;
    }
    else {
        let w = 0;
        let nExtrema = -1;
        let y0_;
        let y1_;
        for (let i = 0; i < nRoots; i++) {
            let t = roots[i];

            // Avoid winding error when intersection point is the connect point of two line of polygon
            let unit = (t === 0 || t === 1) ? 0.5 : 1;

            let x_ = curve.cubicAt(x0, x1, x2, x3, t);
            if (x_ < x) { // Quick reject
                continue;
            }
            if (nExtrema < 0) {
                nExtrema = curve.cubicExtrema(y0, y1, y2, y3, extrema);
                if (extrema[1] < extrema[0] && nExtrema > 1) {
                    swapExtrema();
                }
                y0_ = curve.cubicAt(y0, y1, y2, y3, extrema[0]);
                if (nExtrema > 1) {
                    y1_ = curve.cubicAt(y0, y1, y2, y3, extrema[1]);
                }
            }
            if (nExtrema === 2) {
                // 分成三段单调函数
                if (t < extrema[0]) {
                    w += y0_ < y0 ? unit : -unit;
                }
                else if (t < extrema[1]) {
                    w += y1_ < y0_ ? unit : -unit;
                }
                else {
                    w += y3 < y1_ ? unit : -unit;
                }
            }
            else {
                // 分成两段单调函数
                if (t < extrema[0]) {
                    w += y0_ < y0 ? unit : -unit;
                }
                else {
                    w += y3 < y0_ ? unit : -unit;
                }
            }
        }
        return w;
    }
}

function windingQuadratic(
    x0: number, y0: number, x1: number, y1: number, x2: number, y2: number,
    x: number, y: number
): number {
    // Quick reject
    if (
        (y > y0 && y > y1 && y > y2)
        || (y < y0 && y < y1 && y < y2)
    ) {
        return 0;
    }
    const nRoots = curve.quadraticRootAt(y0, y1, y2, y, roots);
    if (nRoots === 0) {
        return 0;
    }
    else {
        const t = curve.quadraticExtremum(y0, y1, y2);
        if (t >= 0 && t <= 1) {
            let w = 0;
            let y_ = curve.quadraticAt(y0, y1, y2, t);
            for (let i = 0; i < nRoots; i++) {
                // Remove one endpoint.
                let unit = (roots[i] === 0 || roots[i] === 1) ? 0.5 : 1;

                let x_ = curve.quadraticAt(x0, x1, x2, roots[i]);
                if (x_ < x) {   // Quick reject
                    continue;
                }
                if (roots[i] < t) {
                    w += y_ < y0 ? unit : -unit;
                }
                else {
                    w += y2 < y_ ? unit : -unit;
                }
            }
            return w;
        }
        else {
            // Remove one endpoint.
            const unit = (roots[0] === 0 || roots[0] === 1) ? 0.5 : 1;

            const x_ = curve.quadraticAt(x0, x1, x2, roots[0]);
            if (x_ < x) {   // Quick reject
                return 0;
            }
            return y2 < y0 ? unit : -unit;
        }
    }
}
// TODO
// Arc 旋转
// startAngle, endAngle has been normalized by normalizeArcAngles
function windingArc(
    cx: number, cy: number, r: number, startAngle: number, endAngle: number, anticlockwise: boolean,
    x: number, y: number
) {
    y -= cy;
    if (y > r || y < -r) {
        return 0;
    }
    const tmp = Math.sqrt(r * r - y * y);
    roots[0] = -tmp;
    roots[1] = tmp;

    const dTheta = Math.abs(startAngle - endAngle);
    if (dTheta < 1e-4) {
        return 0;
    }
    if (dTheta >= PI2 - 1e-4) {
        // Is a circle
        startAngle = 0;
        endAngle = PI2;
        const dir = anticlockwise ? 1 : -1;
        if (x >= roots[0] + cx && x <= roots[1] + cx) {
            return dir;
        }
        else {
            return 0;
        }
    }

    if (startAngle > endAngle) {
        // Swap, make sure startAngle is smaller than endAngle.
        const tmp = startAngle;
        startAngle = endAngle;
        endAngle = tmp;
    }
    // endAngle - startAngle is normalized to 0 - 2*PI.
    // So following will normalize them to 0 - 4*PI
    if (startAngle < 0) {
        startAngle += PI2;
        endAngle += PI2;
    }

    let w = 0;
    for (let i = 0; i < 2; i++) {
        const x_ = roots[i];
        if (x_ + cx > x) {
            let angle = Math.atan2(y, x_);
            let dir = anticlockwise ? 1 : -1;
            if (angle < 0) {
                angle = PI2 + angle;
            }
            if (
                (angle >= startAngle && angle <= endAngle)
                || (angle + PI2 >= startAngle && angle + PI2 <= endAngle)
            ) {
                if (angle > Math.PI / 2 && angle < Math.PI * 1.5) {
                    dir = -dir;
                }
                w += dir;
            }
        }
    }
    return w;
}


function containPath(
    path: PathProxy, lineWidth: number, isStroke: boolean, x: number, y: number
): boolean {
    const data = path.data;
    const len = path.len();
    let w = 0;
    let xi = 0;
    let yi = 0;
    let x0 = 0;
    let y0 = 0;
    let x1;
    let y1;

    for (let i = 0; i < len;) {
        const cmd = data[i++];
        const isFirst = i === 1;
        // Begin a new subpath
        if (cmd === CMD.M && i > 1) {
            // Close previous subpath
            if (!isStroke) {
                w += windingLine(xi, yi, x0, y0, x, y);
            }
            // 如果被任何一个 subpath 包含
            // if (w !== 0) {
            //     return true;
            // }
        }

        if (isFirst) {
            // 如果第一个命令是 L, C, Q
            // 则 previous point 同绘制命令的第一个 point
            //
            // 第一个命令为 Arc 的情况下会在后面特殊处理
            xi = data[i];
            yi = data[i + 1];

            x0 = xi;
            y0 = yi;
        }

        switch (cmd) {
            case CMD.M:
                // moveTo 命令重新创建一个新的 subpath, 并且更新新的起点
                // 在 closePath 的时候使用
                x0 = data[i++];
                y0 = data[i++];
                xi = x0;
                yi = y0;
                break;
            case CMD.L:
                if (isStroke) {
                    if (line.containStroke(xi, yi, data[i], data[i + 1], lineWidth, x, y)) {
                        return true;
                    }
                }
                else {
                    // NOTE 在第一个命令为 L, C, Q 的时候会计算出 NaN
                    w += windingLine(xi, yi, data[i], data[i + 1], x, y) || 0;
                }
                xi = data[i++];
                yi = data[i++];
                break;
            case CMD.C:
                if (isStroke) {
                    if (cubic.containStroke(xi, yi,
                        data[i++], data[i++], data[i++], data[i++], data[i], data[i + 1],
                        lineWidth, x, y
                    )) {
                        return true;
                    }
                }
                else {
                    w += windingCubic(
                        xi, yi,
                        data[i++], data[i++], data[i++], data[i++], data[i], data[i + 1],
                        x, y
                    ) || 0;
                }
                xi = data[i++];
                yi = data[i++];
                break;
            case CMD.Q:
                if (isStroke) {
                    if (quadratic.containStroke(xi, yi,
                        data[i++], data[i++], data[i], data[i + 1],
                        lineWidth, x, y
                    )) {
                        return true;
                    }
                }
                else {
                    w += windingQuadratic(
                        xi, yi,
                        data[i++], data[i++], data[i], data[i + 1],
                        x, y
                    ) || 0;
                }
                xi = data[i++];
                yi = data[i++];
                break;
            case CMD.A:
                // TODO Arc 判断的开销比较大
                const cx = data[i++];
                const cy = data[i++];
                const rx = data[i++];
                const ry = data[i++];
                const theta = data[i++];
                const dTheta = data[i++];
                // TODO Arc 旋转
                i += 1;
                const anticlockwise = !!(1 - data[i++]);
                x1 = Math.cos(theta) * rx + cx;
                y1 = Math.sin(theta) * ry + cy;
                // 不是直接使用 arc 命令
                if (!isFirst) {
                    w += windingLine(xi, yi, x1, y1, x, y);
                }
                else {
                    // 第一个命令起点还未定义
                    x0 = x1;
                    y0 = y1;
                }
                // zr 使用scale来模拟椭圆, 这里也对x做一定的缩放
                const _x = (x - cx) * ry / rx + cx;
                if (isStroke) {
                    if (arc.containStroke(
                        cx, cy, ry, theta, theta + dTheta, anticlockwise,
                        lineWidth, _x, y
                    )) {
                        return true;
                    }
                }
                else {
                    w += windingArc(
                        cx, cy, ry, theta, theta + dTheta, anticlockwise,
                        _x, y
                    );
                }
                xi = Math.cos(theta + dTheta) * rx + cx;
                yi = Math.sin(theta + dTheta) * ry + cy;
                break;
            case CMD.R:
                x0 = xi = data[i++];
                y0 = yi = data[i++];
                const width = data[i++];
                const height = data[i++];
                x1 = x0 + width;
                y1 = y0 + height;
                if (isStroke) {
                    if (line.containStroke(x0, y0, x1, y0, lineWidth, x, y)
                        || line.containStroke(x1, y0, x1, y1, lineWidth, x, y)
                        || line.containStroke(x1, y1, x0, y1, lineWidth, x, y)
                        || line.containStroke(x0, y1, x0, y0, lineWidth, x, y)
                    ) {
                        return true;
                    }
                }
                else {
                    // FIXME Clockwise ?
                    w += windingLine(x1, y0, x1, y1, x, y);
                    w += windingLine(x0, y1, x0, y0, x, y);
                }
                break;
            case CMD.Z:
                if (isStroke) {
                    if (line.containStroke(
                        xi, yi, x0, y0, lineWidth, x, y
                    )) {
                        return true;
                    }
                }
                else {
                    // Close a subpath
                    w += windingLine(xi, yi, x0, y0, x, y);
                    // 如果被任何一个 subpath 包含
                    // FIXME subpaths may overlap
                    // if (w !== 0) {
                    //     return true;
                    // }
                }
                xi = x0;
                yi = y0;
                break;
        }
    }
    if (!isStroke && !isAroundEqual(yi, y0)) {
        w += windingLine(xi, yi, x0, y0, x, y) || 0;
    }
    return w !== 0;
}

export function contain(pathProxy: PathProxy, x: number, y: number): boolean {
    return containPath(pathProxy, 0, false, x, y);
}

export function containStroke(pathProxy: PathProxy, lineWidth: number, x: number, y: number): boolean {
    return containPath(pathProxy, lineWidth, true, x, y);
}