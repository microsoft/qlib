import Path, { PathProps } from 'zrender/lib/graphic/Path.js';
declare class PointerShape {
    angle: number;
    width: number;
    r: number;
    x: number;
    y: number;
}
interface PointerPathProps extends PathProps {
    shape?: Partial<PointerShape>;
}
export default class PointerPath extends Path<PointerPathProps> {
    readonly type = "pointer";
    shape: PointerShape;
    constructor(opts?: PointerPathProps);
    getDefaultShape(): PointerShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: PointerShape): void;
}
export {};
