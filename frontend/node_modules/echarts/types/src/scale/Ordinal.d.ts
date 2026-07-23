/**
 * Linear continuous scale
 * http://en.wikipedia.org/wiki/Level_of_measurement
 */
import Scale from './Scale.js';
import OrdinalMeta from '../data/OrdinalMeta.js';
import SeriesData from '../data/SeriesData.js';
import { OrdinalRawValue, OrdinalNumber, DimensionLoose, OrdinalSortInfo, OrdinalScaleTick, ScaleTick } from '../util/types.js';
import { CategoryAxisBaseOption } from '../coord/axisCommonTypes.js';
declare type OrdinalScaleSetting = {
    ordinalMeta?: OrdinalMeta | CategoryAxisBaseOption['data'];
    extent?: [number, number];
};
declare class OrdinalScale extends Scale<OrdinalScaleSetting> {
    static type: string;
    readonly type = "ordinal";
    private _ordinalMeta;
    /**
     * For example:
     * Given original ordinal data:
     * ```js
     * option = {
     *     xAxis: {
     *         // Their raw ordinal numbers are:
     *         //      0    1    2    3    4    5
     *         data: ['a', 'b', 'c', 'd', 'e', 'f']
     *     },
     *     yAxis: {}
     *     series: {
     *         type: 'bar',
     *         data: [
     *             ['d', 110], // ordinalNumber: 3
     *             ['c', 660], // ordinalNumber: 2
     *             ['f', 220], // ordinalNumber: 5
     *             ['e', 550]  // ordinalNumber: 4
     *         ],
     *         realtimeSort: true
     *     }
     * };
     * ```
     * After realtime sorted (order by yValue desc):
     * ```js
     * _ordinalNumbersByTick: [
     *     2, // tick: 0, yValue: 660
     *     5, // tick: 1, yValue: 220
     *     3, // tick: 2, yValue: 110
     *     4, // tick: 3, yValue: 550
     *     0, // tick: 4, yValue: -
     *     1, // tick: 5, yValue: -
     * ],
     * _ticksByOrdinalNumber: [
     *     4, // ordinalNumber: 0, yValue: -
     *     5, // ordinalNumber: 1, yValue: -
     *     0, // ordinalNumber: 2, yValue: 660
     *     2, // ordinalNumber: 3, yValue: 110
     *     3, // ordinalNumber: 4, yValue: 550
     *     1, // ordinalNumber: 5, yValue: 220
     * ]
     * ```
     * The index of this array is from `0` to `ordinalMeta.categories.length`.
     *
     * @see `Ordinal['getRawOrdinalNumber']`
     * @see `OrdinalSortInfo`
     */
    private _ordinalNumbersByTick;
    /**
     * This is the inverted map of `_ordinalNumbersByTick`.
     * The index of this array is from `0` to `ordinalMeta.categories.length`.
     *
     * @see `Ordinal['_ordinalNumbersByTick']`
     * @see `Ordinal['_getTickNumber']`
     * @see `OrdinalSortInfo`
     */
    private _ticksByOrdinalNumber;
    constructor(setting?: OrdinalScaleSetting);
    parse(val: OrdinalRawValue | OrdinalNumber): OrdinalNumber;
    contain(rank: OrdinalRawValue | OrdinalNumber): boolean;
    /**
     * Normalize given rank or name to linear [0, 1]
     * @param val raw ordinal number.
     * @return normalized value in [0, 1].
     */
    normalize(val: OrdinalRawValue | OrdinalNumber): number;
    /**
     * @param val normalized value in [0, 1].
     * @return raw ordinal number.
     */
    scale(val: number): OrdinalNumber;
    getTicks(): OrdinalScaleTick[];
    getMinorTicks(splitNumber: number): number[][];
    /**
     * @see `Ordinal['_ordinalNumbersByTick']`
     */
    setSortInfo(info: OrdinalSortInfo): void;
    private _getTickNumber;
    /**
     * @usage
     * ```js
     * const ordinalNumber = ordinalScale.getRawOrdinalNumber(tickVal);
     *
     * // case0
     * const rawOrdinalValue = axisModel.getCategories()[ordinalNumber];
     * // case1
     * const rawOrdinalValue = this._ordinalMeta.categories[ordinalNumber];
     * // case2
     * const coord = axis.dataToCoord(ordinalNumber);
     * ```
     *
     * @param {OrdinalNumber} tickNumber index of display
     */
    getRawOrdinalNumber(tickNumber: number): OrdinalNumber;
    /**
     * Get item on tick
     */
    getLabel(tick: ScaleTick): string;
    count(): number;
    unionExtentFromData(data: SeriesData, dim: DimensionLoose): void;
    /**
     * @override
     * If value is in extent range
     */
    isInExtentRange(value: OrdinalNumber): boolean;
    getOrdinalMeta(): OrdinalMeta;
    calcNiceTicks(): void;
    calcNiceExtent(): void;
}
export default OrdinalScale;
