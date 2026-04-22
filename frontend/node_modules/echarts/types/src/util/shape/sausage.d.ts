import { Path } from '../graphic.js';
import { PathProps } from 'zrender/lib/graphic/Path.js';
/**
 * Sausage: similar to sector, but have half circle on both sides
 */
declare class SausageShape {
    cx: number;
    cy: number;
    r0: number;
    r: number;
    startAngle: number;
    endAngle: number;
    clockwise: boolean;
}
interface SausagePathProps extends PathProps {
    shape?: SausageShape;
}
declare class SausagePath extends Path<SausagePathProps> {
    type: string;
    constructor(opts?: SausagePathProps);
    getDefaultShape(): SausageShape;
    buildPath(ctx: CanvasRenderingContext2D, shape: SausageShape): void;
}
export default SausagePath;
