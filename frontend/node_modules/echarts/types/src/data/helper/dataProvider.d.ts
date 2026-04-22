import { Source } from '../Source.js';
import { ArrayLike } from 'zrender/lib/core/types.js';
import { DimensionName, DimensionIndex, OptionSourceData, OptionDataItem, OptionDataValue, SourceFormat, SeriesLayoutBy, ParsedValue, DimensionLoose, NullUndefined } from '../../util/types.js';
import SeriesData from '../SeriesData.js';
export interface DataProvider {
    /**
     * true: all of the value are in primitive type (in type `OptionDataValue`).
     * false: Not sure whether any of them is non primitive type (in type `OptionDataItemObject`).
     *     Like `data: [ { value: xx, itemStyle: {...} }, ...]`
     *     At present it only happen in `SOURCE_FORMAT_ORIGINAL`.
     */
    pure?: boolean;
    /**
     * If data is persistent and will not be released after use.
     */
    persistent?: boolean;
    getSource(): Source;
    count(): number;
    getItem(idx: number, out?: OptionDataItem): OptionDataItem;
    fillStorage?(start: number, end: number, out: ArrayLike<ParsedValue>[], extent: number[][]): void;
    appendData?(newData: ArrayLike<OptionDataItem>): void;
    clean?(): void;
}
export interface DefaultDataProvider {
    fillStorage?(start: number, end: number, out: ArrayLike<ParsedValue>[], extent: number[][]): void;
}
/**
 * If normal array used, mutable chunk size is supported.
 * If typed array used, chunk size must be fixed.
 */
export declare class DefaultDataProvider implements DataProvider {
    private _source;
    private _data;
    private _offset;
    private _dimSize;
    pure: boolean;
    persistent: boolean;
    static protoInitialize: void;
    constructor(sourceParam: Source | OptionSourceData, dimSize?: number);
    getSource(): Source;
    count(): number;
    getItem(idx: number, out?: ArrayLike<OptionDataValue>): OptionDataItem;
    appendData(newData: OptionSourceData): void;
    clean(): void;
    private static internalField;
}
declare type RawSourceItemGetter = (rawData: OptionSourceData, startIndex: number, dimsDef: {
    name?: DimensionName;
}[], idx: number, out?: ArrayLike<OptionDataValue>) => OptionDataItem | ArrayLike<OptionDataValue>;
export declare function getRawSourceItemGetter(sourceFormat: SourceFormat, seriesLayoutBy: SeriesLayoutBy): RawSourceItemGetter;
declare type RawSourceDataCounter = (rawData: OptionSourceData, startIndex: number, dimsDef: {
    name?: DimensionName;
}[]) => number;
export declare function getRawSourceDataCounter(sourceFormat: SourceFormat, seriesLayoutBy: SeriesLayoutBy): RawSourceDataCounter;
declare type RawSourceValueGetter = (dataItem: OptionDataItem, dimIndex: DimensionIndex, property: DimensionName) => OptionDataValue;
export declare function getRawSourceValueGetter(sourceFormat: SourceFormat): RawSourceValueGetter;
export declare function retrieveRawValue(data: SeriesData, dataIndex: number, dim?: DimensionLoose | NullUndefined): OptionDataValue | OptionDataItem;
/**
 * Compatible with some cases (in pie, map) like:
 * data: [{name: 'xx', value: 5, selected: true}, ...]
 * where only sourceFormat is 'original' and 'objectRows' supported.
 *
 * // TODO
 * Supported detail options in data item when using 'arrayRows'.
 *
 * @param data
 * @param dataIndex
 * @param attr like 'selected'
 */
export declare function retrieveRawAttr(data: SeriesData, dataIndex: number, attr: string): any;
export {};
