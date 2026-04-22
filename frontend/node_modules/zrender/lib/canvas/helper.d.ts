import { LinearGradientObject } from '../graphic/LinearGradient';
import { RadialGradientObject } from '../graphic/RadialGradient';
import { GradientObject } from '../graphic/Gradient';
import { RectLike } from '../core/BoundingRect';
import Path from '../graphic/Path';
export declare function createLinearGradient(this: void, ctx: CanvasRenderingContext2D, obj: LinearGradientObject, rect: RectLike): CanvasGradient;
export declare function createRadialGradient(this: void, ctx: CanvasRenderingContext2D, obj: RadialGradientObject, rect: RectLike): CanvasGradient;
export declare function getCanvasGradient(this: void, ctx: CanvasRenderingContext2D, obj: GradientObject, rect: RectLike): CanvasGradient;
export declare function isClipPathChanged(clipPaths: Path[], prevClipPaths: Path[]): boolean;
export declare function getSize(root: HTMLElement, whIdx: number, opts: {
    width?: number | string;
    height?: number | string;
}): number;
