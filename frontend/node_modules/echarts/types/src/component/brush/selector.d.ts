import BoundingRect, { RectLike } from 'zrender/lib/core/BoundingRect.js';
import { BrushType } from '../helper/BrushController.js';
import { BrushAreaParamInternal } from './BrushModel.js';
export interface BrushSelectableArea extends BrushAreaParamInternal {
    boundingRect: BoundingRect;
    selectors: BrushCommonSelectorsForSeries;
}
/**
 * Key of the first level is brushType: `line`, `rect`, `polygon`.
 * See moudule:echarts/component/helper/BrushController
 * function param:
 *      {Object} itemLayout fetch from data.getItemLayout(dataIndex)
 *      {Object} selectors {point: selector, rect: selector, ...}
 *      {Object} area {range: [[], [], ..], boudingRect}
 * function return:
 *      {boolean} Whether in the given brush.
 */
interface BrushSelectorOnBrushType {
    point(itemLayout: number[], selectors: BrushCommonSelectorsForSeries, area: BrushSelectableArea): boolean;
    rect(itemLayout: RectLike, selectors: BrushCommonSelectorsForSeries, area: BrushSelectableArea): boolean;
}
/**
 * This methods are corresponding to `BrushSelectorOnBrushType`,
 * but `area: BrushSelectableArea` is binded to each method.
 */
export interface BrushCommonSelectorsForSeries {
    point(itemLayout: number[]): boolean;
    rect(itemLayout: RectLike): boolean;
}
export declare function makeBrushCommonSelectorForSeries(area: BrushSelectableArea): BrushCommonSelectorsForSeries;
declare const selector: Record<BrushType, BrushSelectorOnBrushType>;
export default selector;
