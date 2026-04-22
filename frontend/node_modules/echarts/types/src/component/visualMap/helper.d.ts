import VisualMapModel from './VisualMapModel.js';
import ExtensionAPI from '../../core/ExtensionAPI.js';
import { Payload } from '../../util/types.js';
declare const paramsSet: readonly [readonly ["left", "right", "width"], readonly ["top", "bottom", "height"]];
export declare type ItemHorizontalAlign = typeof paramsSet[0][number];
export declare type ItemVerticalAlign = typeof paramsSet[1][number];
export declare type ItemAlign = ItemVerticalAlign | ItemHorizontalAlign;
/**
 * @param visualMapModel
 * @param api
 * @param itemSize always [short, long]
 * @return {string} 'left' or 'right' or 'top' or 'bottom'
 */
export declare function getItemAlign(visualMapModel: VisualMapModel, api: ExtensionAPI, itemSize: number[]): ItemAlign;
/**
 * Prepare dataIndex for outside usage, where dataIndex means rawIndex, and
 * dataIndexInside means filtered index.
 */
export declare function makeHighDownBatch(batch: Payload['batch'], visualMapModel: VisualMapModel): Payload['batch'];
export {};
