import { Point, Polyline } from '../util/graphic.js';
import Element from 'zrender/lib/Element.js';
import { LabelLineOption, DisplayState, StatesOptionMixin } from '../util/types.js';
import Model from '../model/Model.js';
import * as vector from 'zrender/lib/core/vector.js';
/**
 * Calculate a proper guide line based on the label position and graphic element definition
 * @param label
 * @param labelRect
 * @param target
 * @param targetRect
 */
export declare function updateLabelLinePoints(target: Element, labelLineModel: Model<LabelLineOption>): void;
/**
 * Reduce the line segment attached to the label to limit the turn angle between two segments.
 * @param linePoints
 * @param minTurnAngle Radian of minimum turn angle. 0 - 180
 */
export declare function limitTurnAngle(linePoints: number[][], minTurnAngle: number): void;
/**
 * Limit the angle of line and the surface
 * @param maxSurfaceAngle Radian of minimum turn angle. 0 - 180. 0 is same direction to normal. 180 is opposite
 */
export declare function limitSurfaceAngle(linePoints: vector.VectorArray[], surfaceNormal: Point, maxSurfaceAngle: number): void;
declare type LabelLineModel = Model<LabelLineOption>;
/**
 * Create a label line if necessary and set it's style.
 */
export declare function setLabelLineStyle(targetEl: Element, statesModels: Record<DisplayState, LabelLineModel>, defaultStyle?: Polyline['style']): void;
export declare function getLabelLineStatesModels<LabelName extends string = 'labelLine'>(itemModel: Model<StatesOptionMixin<any, any> & Partial<Record<LabelName, any>>>, labelLineName?: LabelName): Record<DisplayState, LabelLineModel>;
export {};
