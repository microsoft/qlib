import Displayable from '../graphic/Displayable';
import { ImagePatternObject } from '../graphic/Pattern';
import Path from '../graphic/Path';
export declare function createCanvasPattern(this: void, ctx: CanvasRenderingContext2D, pattern: ImagePatternObject, el: {
    dirty: () => void;
}): CanvasPattern;
export declare type BrushScope = {
    inHover: boolean;
    viewWidth: number;
    viewHeight: number;
    prevElClipPaths?: Path[];
    prevEl?: Displayable;
    allClipped?: boolean;
    batchFill?: string;
    batchStroke?: string;
    lastDrawType?: number;
};
export declare function brushSingle(ctx: CanvasRenderingContext2D, el: Displayable): void;
export declare function brush(ctx: CanvasRenderingContext2D, el: Displayable, scope: BrushScope, isLast: boolean): void;
