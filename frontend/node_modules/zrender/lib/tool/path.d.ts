import Path, { PathProps } from '../graphic/Path';
import { MatrixArray } from '../core/matrix';
declare type SVGPathOption = Omit<PathProps, 'shape' | 'buildPath'>;
declare class SVGPath extends Path {
    applyTransform(m: MatrixArray): void;
}
export declare function createFromString(str: string, opts?: SVGPathOption): SVGPath;
export declare function extendFromString(str: string, defaultOpts?: SVGPathOption): typeof SVGPath;
export declare function mergePath(pathEls: Path[], opts: PathProps): Path<PathProps>;
export declare function clonePath(sourcePath: Path, opts?: {
    bakeTransform?: boolean;
    toLocal?: boolean;
}): Path<PathProps>;
export {};
