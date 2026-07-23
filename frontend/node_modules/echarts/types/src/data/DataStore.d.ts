import { DimensionIndex, DimensionName, OptionDataItem, ParsedValue, ParsedValueNumeric } from '../util/types.js';
import { DataProvider } from './helper/dataProvider.js';
import OrdinalMeta from './OrdinalMeta.js';
import { Source } from './Source.js';
export declare const CtorUint32Array: ArrayConstructor | Uint32ArrayConstructor;
export declare const CtorUint16Array: ArrayConstructor | Uint16ArrayConstructor;
export declare const CtorInt32Array: ArrayConstructor | Int32ArrayConstructor;
export declare const CtorFloat64Array: ArrayConstructor | Float64ArrayConstructor;
/**
 * Multi dimensional data store
 */
declare const dataCtors: {
    readonly float: ArrayConstructor | Float64ArrayConstructor;
    readonly int: ArrayConstructor | Int32ArrayConstructor;
    readonly ordinal: ArrayConstructor;
    readonly number: ArrayConstructor;
    readonly time: ArrayConstructor | Float64ArrayConstructor;
};
export declare type DataStoreDimensionType = keyof typeof dataCtors;
declare type EachCb = (...args: any) => void;
declare type FilterCb = (...args: any) => boolean;
declare type MapCb = (...args: any) => ParsedValue | ParsedValue[];
export declare type DimValueGetter = (this: DataStore, dataItem: any, property: string, dataIndex: number, dimIndex: DimensionIndex) => ParsedValue;
export interface DataStoreDimensionDefine {
    /**
     * Default to be float.
     */
    type?: DataStoreDimensionType;
    /**
     * Only used in SOURCE_FORMAT_OBJECT_ROWS and SOURCE_FORMAT_KEYED_COLUMNS to retrieve value
     * by "object property".
     * For example, in `[{bb: 124, aa: 543}, ...]`, "aa" and "bb" is "object property".
     *
     * Deliberately name it as "property" rather than "name" to prevent it from been used in
     * SOURCE_FORMAT_ARRAY_ROWS, because if it comes from series, it probably
     * can not be shared by different series.
     */
    property?: string;
    /**
     * When using category axis.
     * Category strings will be collected and stored in ordinalMeta.categories.
     * And store will store the index of categories.
     */
    ordinalMeta?: OrdinalMeta;
    /**
     * Offset for ordinal parsing and collect
     */
    ordinalOffset?: number;
}
/**
 * Basically, DataStore API keep immutable.
 */
declare class DataStore {
    private _chunks;
    private _provider;
    private _rawExtent;
    private _extent;
    private _indices;
    private _count;
    private _rawCount;
    private _dimensions;
    private _dimValueGetter;
    private _calcDimNameToIdx;
    defaultDimValueGetter: DimValueGetter;
    /**
     * Initialize from data
     */
    initData(provider: DataProvider, inputDimensions: DataStoreDimensionDefine[], dimValueGetter?: DimValueGetter): void;
    getProvider(): DataProvider;
    /**
     * Caution: even when a `source` instance owned by a series, the created data store
     * may still be shared by different sereis (the source hash does not use all `source`
     * props, see `sourceManager`). In this case, the `source` props that are not used in
     * hash (like `source.dimensionDefine`) probably only belongs to a certain series and
     * thus should not be fetch here.
     */
    getSource(): Source;
    /**
     * @caution Only used in dataStack.
     */
    ensureCalculationDimension(dimName: DimensionName, type: DataStoreDimensionType): DimensionIndex;
    collectOrdinalMeta(dimIdx: number, ordinalMeta: OrdinalMeta): void;
    getOrdinalMeta(dimIdx: number): OrdinalMeta;
    getDimensionProperty(dimIndex: DimensionIndex): DataStoreDimensionDefine['property'];
    /**
     * Caution: Can be only called on raw data (before `this._indices` created).
     */
    appendData(data: ArrayLike<any>): number[];
    appendValues(values: any[][], minFillLen?: number): {
        start: number;
        end: number;
    };
    private _initDataFromProvider;
    count(): number;
    /**
     * Get value. Return NaN if idx is out of range.
     */
    get(dim: DimensionIndex, idx: number): ParsedValue;
    getValues(idx: number): ParsedValue[];
    getValues(dimensions: readonly DimensionIndex[], idx?: number): ParsedValue[];
    /**
     * @param dim concrete dim
     */
    getByRawIndex(dim: DimensionIndex, rawIdx: number): ParsedValue;
    /**
     * Get sum of data in one dimension
     */
    getSum(dim: DimensionIndex): number;
    /**
     * Get median of data in one dimension
     */
    getMedian(dim: DimensionIndex): number;
    /**
     * Retrieve the index with given raw data index.
     */
    indexOfRawIndex(rawIndex: number): number;
    /**
     * Retrieve the index of nearest value.
     * @param dim
     * @param value
     * @param [maxDistance=Infinity]
     * @return If and only if multiple indices have
     *         the same value, they are put to the result.
     */
    indicesOfNearest(dim: DimensionIndex, value: number, maxDistance?: number): number[];
    getIndices(): ArrayLike<number>;
    /**
     * Data filter.
     */
    filter(dims: DimensionIndex[], cb: FilterCb): DataStore;
    /**
     * Select data in range. (For optimization of filter)
     * (Manually inline code, support 5 million data filtering in data zoom.)
     */
    selectRange(range: {
        [dimIdx: number]: [number, number];
    }): DataStore;
    /**
     * Data mapping to a new List with given dimensions
     */
    map(dims: DimensionIndex[], cb: MapCb): DataStore;
    /**
     * @caution Danger!! Only used in dataStack.
     */
    modify(dims: DimensionIndex[], cb: MapCb): void;
    private _updateDims;
    /**
     * Large data down sampling using largest-triangle-three-buckets
     * @param {string} valueDimension
     * @param {number} targetCount
     */
    lttbDownSample(valueDimension: DimensionIndex, rate: number): DataStore;
    /**
     * Large data down sampling using min-max
     * @param {string} valueDimension
     * @param {number} rate
     */
    minmaxDownSample(valueDimension: DimensionIndex, rate: number): DataStore;
    /**
     * Large data down sampling on given dimension
     * @param sampleIndex Sample index for name and id
     */
    downSample(dimension: DimensionIndex, rate: number, sampleValue: (frameValues: ArrayLike<ParsedValue>) => ParsedValueNumeric, sampleIndex: (frameValues: ArrayLike<ParsedValue>, value: ParsedValueNumeric) => number): DataStore;
    /**
     * Data iteration
     * @param ctx default this
     * @example
     *  list.each('x', function (x, idx) {});
     *  list.each(['x', 'y'], function (x, y, idx) {});
     *  list.each(function (idx) {})
     */
    each(dims: DimensionIndex[], cb: EachCb): void;
    /**
     * Get extent of data in one dimension
     */
    getDataExtent(dim: DimensionIndex): [number, number];
    /**
     * Get raw data index.
     * Do not initialize.
     * Default `getRawIndex`. And it can be changed.
     */
    getRawIndex: (idx: number) => number;
    /**
     * Get raw data item
     */
    getRawDataItem(idx: number): OptionDataItem;
    /**
     * Clone shallow.
     *
     * @param clonedDims Determine which dims to clone. Will share the data if not specified.
     */
    clone(clonedDims?: DimensionIndex[], ignoreIndices?: boolean): DataStore;
    private _copyCommonProps;
    private _cloneIndices;
    private _getRawIdxIdentity;
    private _getRawIdx;
    private _updateGetRawIdx;
    private static internalField;
}
export default DataStore;
