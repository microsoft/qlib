import Path, { PathProps } from '../Path';
export declare class CircleShape {
    cx: number;
    cy: number;
    r: number;
}
export interface CircleProps extends PathProps {
    shape?: Partial<CircleShape>;
}
declare class Circle extends Path<CircleProps> {
    shape: CircleShape;
    constructor(opts?: CircleProps);
    getDefaultShape(): CircleShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: CircleShape): void;
}
export default Circle;
