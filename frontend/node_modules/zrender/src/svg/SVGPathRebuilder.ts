import { PathRebuilder } from '../core/PathProxy';
import { isAroundZero } from './helper';

const mathSin = Math.sin;
const mathCos = Math.cos;
const PI = Math.PI;
const PI2 = Math.PI * 2;
const degree = 180 / PI;


export default class SVGPathRebuilder implements PathRebuilder {
    private _d: (string | number)[]
    private _str: string
    private _invalid: boolean

    // If is start of subpath
    private _start: boolean
    private _p: number

    reset(precision?: number) {
        this._start = true;
        this._d = [];
        this._str = '';

        this._p = Math.pow(10, precision || 4);
    }
    moveTo(x: number, y: number) {
        this._add('M', x, y);
    }
    lineTo(x: number, y: number) {
        this._add('L', x, y);
    }
    bezierCurveTo(x: number, y: number, x2: number, y2: number, x3: number, y3: number) {
        this._add('C', x, y, x2, y2, x3, y3);
    }
    quadraticCurveTo(x: number, y: number, x2: number, y2: number) {
        this._add('Q', x, y, x2, y2);
    }
    arc(cx: number, cy: number, r: number, startAngle: number, endAngle: number, anticlockwise: boolean) {
        this.ellipse(cx, cy, r, r, 0, startAngle, endAngle, anticlockwise);
    }
    ellipse(
        cx: number, cy: number,
        rx: number, ry: number,
        psi: number,
        startAngle: number,
        endAngle: number,
        anticlockwise: boolean
    ) {
        let dTheta = endAngle - startAngle;
        const clockwise = !anticlockwise;

        const dThetaPositive = Math.abs(dTheta);
        const isCircle = isAroundZero(dThetaPositive - PI2)
            || (clockwise ? dTheta >= PI2 : -dTheta >= PI2);

        // Mapping to 0~2PI
        const unifiedTheta = dTheta > 0 ? dTheta % PI2 : (dTheta % PI2 + PI2);

        let large = false;
        if (isCircle) {
            large = true;
        }
        else if (isAroundZero(dThetaPositive)) {
            large = false;
        }
        else {
            large = (unifiedTheta >= PI) === !!clockwise;
        }

        const x0 = cx + rx * mathCos(startAngle);
        const y0 = cy + ry * mathSin(startAngle);

        if (this._start) {
            // Move to (x0, y0) only when CMD.A comes at the
            // first position of a shape.
            // For instance, when drawing a ring, CMD.A comes
            // after CMD.M, so it's unnecessary to move to
            // (x0, y0).
            this._add('M', x0, y0);
        }

        const xRot = Math.round(psi * degree);
        // It will not draw if start point and end point are exactly the same
        // We need to add two arcs
        if (isCircle) {
            const p = 1 / this._p;
            const dTheta = (clockwise ? 1 : -1) * (PI2 - p);
            this._add(
                'A', rx, ry, xRot, 1, +clockwise,
                cx + rx * mathCos(startAngle + dTheta),
                cy + ry * mathSin(startAngle + dTheta)
            );
            // TODO.
            // Usually we can simply divide the circle into two halfs arcs.
            // But it will cause slightly diff with previous screenshot.
            // We can't tell it but visual regression test can. To avoid too much breaks.
            // We keep the logic on the browser as before.
            // But in SSR mode wich has lower precision. We close the circle by adding another arc.
            if (p > 1e-2) {
                this._add('A', rx, ry, xRot, 0, +clockwise, x0, y0);
            }
        }
        else {
            const x = cx + rx * mathCos(endAngle);
            const y = cy + ry * mathSin(endAngle);

            // FIXME Ellipse
            this._add('A', rx, ry, xRot, +large, +clockwise, x, y);
        }

    }
    rect(x: number, y: number, w: number, h: number) {
        this._add('M', x, y);
        // Use relative coordinates to reduce the size.
        this._add('l', w, 0);
        this._add('l', 0, h);
        this._add('l', -w, 0);
        // this._add('L', x, y);
        this._add('Z');
    }
    closePath() {
        // Not use Z as first command
        if (this._d.length > 0) {
            this._add('Z');
        }
    }

    _add(cmd: string, a?: number, b?: number, c?: number, d?: number, e?: number, f?: number, g?: number, h?: number) {
        const vals = [];
        const p = this._p;
        for (let i = 1; i < arguments.length; i++) {
            const val = arguments[i];
            if (isNaN(val)) {
                this._invalid = true;
                return;
            }
            vals.push(Math.round(val * p) / p);
        }
        this._d.push(cmd + vals.join(' '));
        this._start = cmd === 'Z';
    }

    generateStr() {
        this._str = this._invalid ? '' : this._d.join('');
        this._d = [];
    }
    getStr() {
        return this._str;
    }
}