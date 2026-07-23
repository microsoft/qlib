import Model from '../../model/Model.js';
import { getLayoutRect } from '../../util/layout.js';
import { enableDataStack, isDimensionStacked, getStackedDimension } from '../../data/helper/dataStackHelper.js';
import SeriesModel from '../../model/Series.js';
import { AxisBaseModel } from '../../coord/AxisBaseModel.js';
import { getECData } from '../../util/innerStore.js';
import { DisplayState, TextCommonOption } from '../../util/types.js';
/**
 * Create a multi dimension List structure from seriesModel.
 */
export declare function createList(seriesModel: SeriesModel): import("../../data/SeriesData").default<Model<any>, import("../../data/SeriesData").DefaultDataVisual>;
export { getLayoutRect };
export { createDimensions } from '../../data/helper/createDimensions.js';
export declare const dataStack: {
    isDimensionStacked: typeof isDimensionStacked;
    enableDataStack: typeof enableDataStack;
    getStackedDimension: typeof getStackedDimension;
};
/**
 * Create a symbol element with given symbol configuration: shape, x, y, width, height, color
 * @param {string} symbolDesc
 * @param {number} x
 * @param {number} y
 * @param {number} w
 * @param {number} h
 * @param {string} color
 */
export { createSymbol } from '../../util/symbol.js';
/**
 * Create scale
 * @param {Array.<number>} dataExtent
 * @param {Object|module:echarts/Model} option If `optoin.type`
 *        is secified, it can only be `'value'` currently.
 */
export declare function createScale(dataExtent: number[], option: object | AxisBaseModel): import("../../scale/Scale").default<import("zrender/lib/core/types").Dictionary<unknown>>;
/**
 * Mixin common methods to axis model,
 *
 * Include methods
 * `getFormattedLabels() => Array.<string>`
 * `getCategories() => Array.<string>`
 * `getMin(origin: boolean) => number`
 * `getMax(origin: boolean) => number`
 * `getNeedCrossZero() => boolean`
 */
export declare function mixinAxisModelCommonMethods(Model: Model): void;
export { getECData };
export { enableHoverEmphasis } from '../../util/states.js';
export declare function createTextStyle(textStyleModel: Model<TextCommonOption>, opts?: {
    state?: DisplayState;
}): import("zrender/lib/graphic/Text").TextStyleProps;
