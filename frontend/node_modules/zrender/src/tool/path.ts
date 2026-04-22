import Path, { PathProps } from '../graphic/Path';
import PathProxy from '../core/PathProxy';
import transformPath from './transformPath';
import { VectorArray } from '../core/vector';
import { MatrixArray } from '../core/matrix';
import { extend } from '../core/util';

// command chars
// const cc = [
//     'm', 'M', 'l', 'L', 'v', 'V', 'h', 'H', 'z', 'Z',
//     'c', 'C', 'q', 'Q', 't', 'T', 's', 'S', 'a', 'A'
// ];

const mathSqrt = Math.sqrt;
const mathSin = Math.sin;
const mathCos = Math.cos;
const PI = Math.PI;

function vMag(v: VectorArray): number {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1]);
};
function vRatio(u: VectorArray, v: VectorArray): number {
    return (u[0] * v[0] + u[1] * v[1]) / (vMag(u) * vMag(v));
};
function vAngle(u: VectorArray, v: VectorArray): number {
    return (u[0] * v[1] < u[1] * v[0] ? -1 : 1)
            * Math.acos(vRatio(u, v));
};

function processArc(
    x1: number, y1: number, x2: number, y2: number, fa: number, fs: number,
    rx: number, ry: number, psiDeg: number, cmd: number, path: PathProxy
) {
    // https://www.w3.org/TR/SVG11/implnote.html#ArcImplementationNotes
    const psi = psiDeg * (PI / 180.0);
    const xp = mathCos(psi) * (x1 - x2) / 2.0
                + mathSin(psi) * (y1 - y2) / 2.0;
    const yp = -1 * mathSin(psi) * (x1 - x2) / 2.0
                + mathCos(psi) * (y1 - y2) / 2.0;

    const lambda = (xp * xp) / (rx * rx) + (yp * yp) / (ry * ry);

    if (lambda > 1) {
        rx *= mathSqrt(lambda);
        ry *= mathSqrt(lambda);
    }

    const f = (fa === fs ? -1 : 1)
        * mathSqrt((((rx * rx) * (ry * ry))
                - ((rx * rx) * (yp * yp))
                - ((ry * ry) * (xp * xp))) / ((rx * rx) * (yp * yp)
                + (ry * ry) * (xp * xp))
            ) || 0;

    const cxp = f * rx * yp / ry;
    const cyp = f * -ry * xp / rx;

    const cx = (x1 + x2) / 2.0
                + mathCos(psi) * cxp
                - mathSin(psi) * cyp;
    const cy = (y1 + y2) / 2.0
            + mathSin(psi) * cxp
            + mathCos(psi) * cyp;

    const theta = vAngle([ 1, 0 ], [ (xp - cxp) / rx, (yp - cyp) / ry ]);
    const u = [ (xp - cxp) / rx, (yp - cyp) / ry ];
    const v = [ (-1 * xp - cxp) / rx, (-1 * yp - cyp) / ry ];
    let dTheta = vAngle(u, v);

    if (vRatio(u, v) <= -1) {
        dTheta = PI;
    }
    if (vRatio(u, v) >= 1) {
        dTheta = 0;
    }

    if (dTheta < 0) {
        const n = Math.round(dTheta / PI * 1e6) / 1e6;
        // Convert to positive
        dTheta = PI * 2 + (n % 2) * PI;
    }

    path.addData(cmd, cx, cy, rx, ry, theta, dTheta, psi, fs);
}


const commandReg = /([mlvhzcqtsa])([^mlvhzcqtsa]*)/ig;
// Consider case:
// (1) delimiter can be comma or space, where continuous commas
// or spaces should be seen as one comma.
// (2) value can be like:
// '2e-4', 'l.5.9' (ignore 0), 'M-10-10', 'l-2.43e-1,34.9983',
// 'l-.5E1,54', '121-23-44-11' (no delimiter)
const numberReg = /-?([0-9]*\.)?[0-9]+([eE]-?[0-9]+)?/g;
// const valueSplitReg = /[\s,]+/;

function createPathProxyFromString(data: string) {
    const path = new PathProxy();

    if (!data) {
        return path;
    }

    // const data = data.replace(/-/g, ' -')
    //     .replace(/  /g, ' ')
    //     .replace(/ /g, ',')
    //     .replace(/,,/g, ',');

    // const n;
    // create pipes so that we can split the data
    // for (n = 0; n < cc.length; n++) {
    //     cs = cs.replace(new RegExp(cc[n], 'g'), '|' + cc[n]);
    // }

    // data = data.replace(/-/g, ',-');

    // create array
    // const arr = cs.split('|');
    // init context point
    let cpx = 0;
    let cpy = 0;
    let subpathX = cpx;
    let subpathY = cpy;
    let prevCmd;

    const CMD = PathProxy.CMD;

    // commandReg.lastIndex = 0;
    // const cmdResult;
    // while ((cmdResult = commandReg.exec(data)) != null) {
    //     const cmdStr = cmdResult[1];
    //     const cmdContent = cmdResult[2];

    const cmdList = data.match(commandReg);
    if (!cmdList) {
        // Invalid svg path.
        return path;
    }

    for (let l = 0; l < cmdList.length; l++) {
        const cmdText = cmdList[l];
        let cmdStr = cmdText.charAt(0);

        let cmd;

        // String#split is faster a little bit than String#replace or RegExp#exec.
        // const p = cmdContent.split(valueSplitReg);
        // const pLen = 0;
        // for (let i = 0; i < p.length; i++) {
        //     // '' and other invalid str => NaN
        //     const val = parseFloat(p[i]);
        //     !isNaN(val) && (p[pLen++] = val);
        // }


        // Following code will convert string to number. So convert type to number here
        const p = cmdText.match(numberReg) as any[] as number[] || [];
        const pLen = p.length;
        for (let i = 0; i < pLen; i++) {
            p[i] = parseFloat(p[i] as any as string);
        }

        let off = 0;
        while (off < pLen) {
            let ctlPtx;
            let ctlPty;

            let rx;
            let ry;
            let psi;
            let fa;
            let fs;

            let x1 = cpx;
            let y1 = cpy;

            let len: number;
            let pathData: number[] | Float32Array;
            // convert l, H, h, V, and v to L
            switch (cmdStr) {
                case 'l':
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'L':
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'm':
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.M;
                    path.addData(cmd, cpx, cpy);
                    subpathX = cpx;
                    subpathY = cpy;
                    cmdStr = 'l';
                    break;
                case 'M':
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.M;
                    path.addData(cmd, cpx, cpy);
                    subpathX = cpx;
                    subpathY = cpy;
                    cmdStr = 'L';
                    break;
                case 'h':
                    cpx += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'H':
                    cpx = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'v':
                    cpy += p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'V':
                    cpy = p[off++];
                    cmd = CMD.L;
                    path.addData(cmd, cpx, cpy);
                    break;
                case 'C':
                    cmd = CMD.C;
                    path.addData(
                        cmd, p[off++], p[off++], p[off++], p[off++], p[off++], p[off++]
                    );
                    cpx = p[off - 2];
                    cpy = p[off - 1];
                    break;
                case 'c':
                    cmd = CMD.C;
                    path.addData(
                        cmd,
                        p[off++] + cpx, p[off++] + cpy,
                        p[off++] + cpx, p[off++] + cpy,
                        p[off++] + cpx, p[off++] + cpy
                    );
                    cpx += p[off - 2];
                    cpy += p[off - 1];
                    break;
                case 'S':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.C) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cmd = CMD.C;
                    x1 = p[off++];
                    y1 = p[off++];
                    cpx = p[off++];
                    cpy = p[off++];
                    path.addData(cmd, ctlPtx, ctlPty, x1, y1, cpx, cpy);
                    break;
                case 's':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.C) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cmd = CMD.C;
                    x1 = cpx + p[off++];
                    y1 = cpy + p[off++];
                    cpx += p[off++];
                    cpy += p[off++];
                    path.addData(cmd, ctlPtx, ctlPty, x1, y1, cpx, cpy);
                    break;
                case 'Q':
                    x1 = p[off++];
                    y1 = p[off++];
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, x1, y1, cpx, cpy);
                    break;
                case 'q':
                    x1 = p[off++] + cpx;
                    y1 = p[off++] + cpy;
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, x1, y1, cpx, cpy);
                    break;
                case 'T':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.Q) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, ctlPtx, ctlPty, cpx, cpy);
                    break;
                case 't':
                    ctlPtx = cpx;
                    ctlPty = cpy;
                    len = path.len();
                    pathData = path.data;
                    if (prevCmd === CMD.Q) {
                        ctlPtx += cpx - pathData[len - 4];
                        ctlPty += cpy - pathData[len - 3];
                    }
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.Q;
                    path.addData(cmd, ctlPtx, ctlPty, cpx, cpy);
                    break;
                case 'A':
                    rx = p[off++];
                    ry = p[off++];
                    psi = p[off++];
                    fa = p[off++];
                    fs = p[off++];

                    x1 = cpx, y1 = cpy;
                    cpx = p[off++];
                    cpy = p[off++];
                    cmd = CMD.A;
                    processArc(
                        x1, y1, cpx, cpy, fa, fs, rx, ry, psi, cmd, path
                    );
                    break;
                case 'a':
                    rx = p[off++];
                    ry = p[off++];
                    psi = p[off++];
                    fa = p[off++];
                    fs = p[off++];

                    x1 = cpx, y1 = cpy;
                    cpx += p[off++];
                    cpy += p[off++];
                    cmd = CMD.A;
                    processArc(
                        x1, y1, cpx, cpy, fa, fs, rx, ry, psi, cmd, path
                    );
                    break;
            }
        }

        if (cmdStr === 'z' || cmdStr === 'Z') {
            cmd = CMD.Z;
            path.addData(cmd);
            // z may be in the middle of the path.
            cpx = subpathX;
            cpy = subpathY;
        }

        prevCmd = cmd;
    }

    path.toStatic();

    return path;
}

type SVGPathOption = Omit<PathProps, 'shape' | 'buildPath'>
interface InnerSVGPathOption extends PathProps {
    applyTransform?: (m: MatrixArray) => void
}
class SVGPath extends Path {
    applyTransform(m: MatrixArray) {}
}

function isPathProxy(path: PathProxy | CanvasRenderingContext2D): path is PathProxy {
    return (path as PathProxy).setData != null;
}
// TODO Optimize double memory cost problem
function createPathOptions(str: string, opts: SVGPathOption): InnerSVGPathOption {
    const pathProxy = createPathProxyFromString(str);
    const innerOpts: InnerSVGPathOption = extend({}, opts);
    innerOpts.buildPath = function (path: PathProxy | CanvasRenderingContext2D) {
        if (isPathProxy(path)) {
            path.setData(pathProxy.data);
            // Svg and vml renderer don't have context
            const ctx = path.getContext();
            if (ctx) {
                path.rebuildPath(ctx, 1);
            }
        }
        else {
            const ctx = path;
            pathProxy.rebuildPath(ctx, 1);
        }
    };

    innerOpts.applyTransform = function (this: SVGPath, m: MatrixArray) {
        transformPath(pathProxy, m);
        this.dirtyShape();
    };

    return innerOpts;
}

/**
 * Create a Path object from path string data
 * http://www.w3.org/TR/SVG/paths.html#PathData
 * @param  opts Other options
 */
export function createFromString(str: string, opts?: SVGPathOption): SVGPath {
    // PENDING
    return new SVGPath(createPathOptions(str, opts));
}

/**
 * Create a Path class from path string data
 * @param  str
 * @param  opts Other options
 */
export function extendFromString(str: string, defaultOpts?: SVGPathOption): typeof SVGPath {
    const innerOpts = createPathOptions(str, defaultOpts);
    class Sub extends SVGPath {
        constructor(opts: InnerSVGPathOption) {
            super(opts);
            this.applyTransform = innerOpts.applyTransform;
            this.buildPath = innerOpts.buildPath;
        }
    }
    return Sub;
}

/**
 * Merge multiple paths
 */
// TODO Apply transform
// TODO stroke dash
// TODO Optimize double memory cost problem
export function mergePath(pathEls: Path[], opts: PathProps) {
    const pathList: PathProxy[] = [];
    const len = pathEls.length;
    for (let i = 0; i < len; i++) {
        const pathEl = pathEls[i];
        pathList.push(pathEl.getUpdatedPathProxy(true));
    }

    const pathBundle = new Path(opts);
    // Need path proxy.
    pathBundle.createPathProxy();
    pathBundle.buildPath = function (path: PathProxy | CanvasRenderingContext2D) {
        if (isPathProxy(path)) {
            path.appendPath(pathList);
            // Svg and vml renderer don't have context
            const ctx = path.getContext();
            if (ctx) {
                // Path bundle not support percent draw.
                path.rebuildPath(ctx, 1);
            }
        }
    };

    return pathBundle;
}

/**
 * Clone a path.
 */
export function clonePath(sourcePath: Path, opts?: {
    /**
     * If bake global transform to path.
     */
    bakeTransform?: boolean
    /**
     * Convert global transform to local.
     */
    toLocal?: boolean
}) {
    opts = opts || {};
    const path = new Path();
    if (sourcePath.shape) {
        path.setShape(sourcePath.shape);
    }
    path.setStyle(sourcePath.style);

    if (opts.bakeTransform) {
        transformPath(path.path, sourcePath.getComputedTransform());
    }
    else {
        // TODO Copy getLocalTransform, updateTransform since they can be changed.
        if (opts.toLocal) {
            path.setLocalTransform(sourcePath.getComputedTransform());
        }
        else {
            path.copyTransform(sourcePath);
        }
    }

    // These methods may be overridden
    path.buildPath = sourcePath.buildPath;
    (path as SVGPath).applyTransform = (path as SVGPath).applyTransform;

    path.z = sourcePath.z;
    path.z2 = sourcePath.z2;
    path.zlevel = sourcePath.zlevel;

    return path;
}
