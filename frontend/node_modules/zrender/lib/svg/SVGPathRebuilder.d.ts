import { PathRebuilder } from '../core/PathProxy';
export default class SVGPathRebuilder implements PathRebuilder {
    private _d;
    private _str;
    private _invalid;
    private _start;
    private _p;
    reset(precision?: number): void;
    moveTo(x: number, y: number): void;
    lineTo(x: number, y: number): void;
    bezierCurveTo(x: number, y: number, x2: number, y2: number, x3: number, y3: number): void;
    quadraticCurveTo(x: number, y: number, x2: number, y2: number): void;
    arc(cx: number, cy: number, r: number, startAngle: number, endAngle: number, anticlockwise: boolean): void;
    ellipse(cx: number, cy: number, rx: number, ry: number, psi: number, startAngle: number, endAngle: number, anticlockwise: boolean): void;
    rect(x: number, y: number, w: number, h: number): void;
    closePath(): void;
    _add(cmd: string, a?: number, b?: number, c?: number, d?: number, e?: number, f?: number, g?: number, h?: number): void;
    generateStr(): void;
    getStr(): string;
}
