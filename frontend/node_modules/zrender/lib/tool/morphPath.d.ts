import Path from '../graphic/Path';
import Element, { ElementAnimateConfig } from '../Element';
import { split } from './dividePath';
export declare function alignBezierCurves(array1: number[][], array2: number[][]): number[][][];
export interface CombineMorphingPath extends Path {
    childrenRef(): (CombineMorphingPath | Path)[];
    __isCombineMorphing: boolean;
}
export declare function centroid(array: number[]): number[];
export declare function isCombineMorphing(path: Element): path is CombineMorphingPath;
export declare function isMorphing(el: Element): boolean;
export declare function morphPath(fromPath: Path, toPath: Path, animationOpts: ElementAnimateConfig): Path;
export interface DividePathParams {
    path: Path;
    count: number;
}
export interface DividePath {
    (params: DividePathParams): Path[];
}
export interface IndividualDelay {
    (index: number, count: number, fromPath: Path, toPath: Path): number;
}
export interface CombineConfig extends ElementAnimateConfig {
    dividePath?: DividePath;
    individualDelay?: IndividualDelay;
}
export declare function combineMorph(fromList: (CombineMorphingPath | Path)[], toPath: Path, animationOpts: CombineConfig): {
    fromIndividuals: Path<import("../graphic/Path").PathProps>[];
    toIndividuals: Path<import("../graphic/Path").PathProps>[];
    count: number;
};
export interface SeparateConfig extends ElementAnimateConfig {
    dividePath?: DividePath;
    individualDelay?: IndividualDelay;
}
export declare function separateMorph(fromPath: Path, toPathList: Path[], animationOpts: SeparateConfig): {
    fromIndividuals: Path<import("../graphic/Path").PathProps>[];
    toIndividuals: Path<import("../graphic/Path").PathProps>[];
    count: number;
};
export { split as defaultDividePath };
