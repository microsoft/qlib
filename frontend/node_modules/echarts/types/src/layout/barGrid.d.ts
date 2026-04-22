import BarSeriesModel from '../chart/bar/BarSeries.js';
import Axis2D from '../coord/cartesian/Axis2D.js';
import GlobalModel from '../model/Global.js';
import { StageHandler, Dictionary } from '../util/types.js';
/**
 * {
 *  [coordSysId]: {
 *      [stackId]: {bandWidth, offset, width}
 *  }
 * }
 */
declare type BarWidthAndOffset = Dictionary<Dictionary<{
    bandWidth: number;
    offset: number;
    offsetCenter: number;
    width: number;
}>>;
export interface BarGridLayoutOptionForCustomSeries {
    count: number;
    barWidth?: number | string;
    barMaxWidth?: number | string;
    barMinWidth?: number | string;
    barGap?: number | string;
    barCategoryGap?: number | string;
}
interface LayoutOption extends BarGridLayoutOptionForCustomSeries {
    axis: Axis2D;
}
export declare type BarGridLayoutResult = BarWidthAndOffset[string][string][];
/**
 * @return {Object} {width, offset, offsetCenter} If axis.type is not 'category', return undefined.
 */
export declare function getLayoutOnAxis(opt: LayoutOption): BarGridLayoutResult;
export declare function prepareLayoutBarSeries(seriesType: string, ecModel: GlobalModel): BarSeriesModel[];
export declare function makeColumnLayout(barSeries: BarSeriesModel[]): BarWidthAndOffset;
/**
 * @param barWidthAndOffset The result of makeColumnLayout
 * @param seriesModel If not provided, return all.
 * @return {stackId: {offset, width}} or {offset, width} if seriesModel provided.
 */
declare function retrieveColumnLayout(barWidthAndOffset: BarWidthAndOffset, axis: Axis2D): typeof barWidthAndOffset[string];
declare function retrieveColumnLayout(barWidthAndOffset: BarWidthAndOffset, axis: Axis2D, seriesModel: BarSeriesModel): typeof barWidthAndOffset[string][string];
export { retrieveColumnLayout };
export declare function layout(seriesType: string, ecModel: GlobalModel): void;
export declare function createProgressiveLayout(seriesType: string): StageHandler;
