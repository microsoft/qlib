import { Path } from '../util/graphic.js';
import SeriesModel from '../model/Series.js';
import Element, { ElementAnimateConfig } from 'zrender/lib/Element.js';
import { UniversalTransitionOption } from '../util/types.js';
declare type DescendentPaths = Path[];
export declare function applyMorphAnimation(from: DescendentPaths | DescendentPaths[], to: DescendentPaths | DescendentPaths[], divideShape: UniversalTransitionOption['divideShape'], seriesModel: SeriesModel, dataIndex: number, animateOtherProps: (fromIndividual: Path, toIndividual: Path, rawFrom: Path, rawTo: Path, animationCfg: ElementAnimateConfig) => void): void;
export declare function getPathList(elements: Element): DescendentPaths;
export declare function getPathList(elements: Element[]): DescendentPaths[];
export {};
